#!/usr/bin/env bash
# Administrator-only, non-forced repair; never kills GPU jobs or reboots.
set -euo pipefail
export LC_ALL=C

maintenance_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
update_config=/etc/apt/apt.conf.d/99zz-hkbz-disable-automatic-updates.conf
timer_units=(apt-daily.timer apt-daily-upgrade.timer)
apt_units=(apt-daily.service apt-daily-upgrade.service)

fail() { echo "ERROR: $*" >&2; return 1; }

assert_no_open_files() {
    local listing path status=0
    if (( $# == 0 )); then
        fail "No files/devices supplied for the idle check."
        return 1
    fi
    for path in "$@"; do
        if [[ "$path" != /* ]]; then
            fail "Idle checks require absolute paths: $path"
            return 1
        fi
    done
    # PSmisc 23.4 rejects GNU-style '--' here. Absolute paths cannot be
    # interpreted as options, so no option separator is needed.
    listing=$(fuser -v "$@" 2>&1) || status=$?
    if (( status == 0 )); then
        echo "$listing" >&2
        fail "Files/devices are in use; nothing will be force-stopped."
    elif (( status != 1 )) || [[ -n "$listing" ]]; then
        echo "$listing" >&2
        fail "Cannot reliably establish that the files/devices are idle."
    fi
}

assert_no_package_work() {
    local unit state path
    local locks=()
    for unit in "${apt_units[@]}"; do
        state=$(systemctl show "$unit" --property=ActiveState --value)
        [[ "$state" == inactive || "$state" == failed ]] ||
            fail "$unit is $state; wait for package maintenance to finish."
    done
    for path in /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend \
        /var/lib/apt/lists/lock /var/cache/apt/archives/lock; do
        [[ ! -e "$path" ]] || locks+=("$path")
    done
    (( ${#locks[@]} > 0 )) || fail "APT lock files are missing."
    assert_no_open_files "${locks[@]}"
}

disable_updates() {
    local backup_dir unit value
    local template="$maintenance_dir/99zz-hkbz-disable-automatic-updates.conf"
    [[ -f "$template" ]] || fail "Missing APT configuration template."
    [[ ! -L "$update_config" ]] || fail "Refusing to overwrite a symlink at $update_config."
    assert_no_package_work
    backup_dir=$(mktemp -d /var/backups/hkbz-gpu-maintenance.XXXXXXXX)
    if [[ -e "$update_config" || -L "$update_config" ]]; then
        cp -a -- "$update_config" "$backup_dir/"
    fi
    systemctl show "${timer_units[@]}" "${apt_units[@]}" unattended-upgrades.service \
        --property=Id,ActiveState,UnitFileState | tee "$backup_dir/systemd-before.txt"
    echo "Backup: $backup_dir"
    # Stop timers, not installers. Masking services without --now does not kill
    # an installer that happened to start just after the first idle check.
    systemctl disable --now "${timer_units[@]}"
    systemctl mask "${timer_units[@]}" "${apt_units[@]}"
    assert_no_package_work
    install -o root -g root -m 0644 -- "$template" "$update_config"
    # This unit is the shutdown-wait helper; the installer idle check is above.
    systemctl disable --now unattended-upgrades.service
    systemctl mask unattended-upgrades.service
    for unit in "${timer_units[@]}" "${apt_units[@]}" unattended-upgrades.service; do
        value=$(systemctl show "$unit" --property=UnitFileState --value)
        [[ "$value" == masked ]] || fail "$unit was not masked."
        value=$(systemctl show "$unit" --property=ActiveState --value)
        [[ "$value" == inactive || "$value" == failed ]] || fail "$unit is still active."
    done
    for value in Enable Update-Package-Lists Download-Upgradeable-Packages \
        AutocleanInterval Unattended-Upgrade; do
        [[ "$(apt-config shell VALUE "APT::Periodic::$value")" == "VALUE='0'" ]] ||
            fail "APT periodic setting was overridden: $value"
    done
    [[ "$(apt-config shell VALUE Unattended-Upgrade::Automatic-Reboot)" == "VALUE='false'" ]] ||
        fail "Automatic reboot setting was overridden."
    echo "APT automatic updates DISABLED. Schedule manual security updates."
}

reload_driver() {
    local loaded disk nvml unit state module holder refcount child_count
    local previous_modules=() gpu_nodes=() devices=()
    shopt -s nullglob
    loaded=$(< /sys/module/nvidia/version)
    disk=$(modinfo -F version nvidia)
    nvml=$(readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1)
    echo "Loaded NVIDIA: $loaded; installed module: $disk; NVML: $nvml"
    [[ "${nvml##*.so.}" == "$disk" ]] || fail "Installed module and NVML still differ."
    if [[ "$loaded" == "$disk" ]]; then
        nvidia-smi
        echo "Driver versions already match; no module reload needed."
        return
    fi
    for unit in display-manager.service nvidia-persistenced.service; do
        state=$(systemctl show "$unit" --property=ActiveState --value)
        [[ "$state" == inactive || "$state" == failed ]] || fail "$unit is $state."
    done
    gpu_nodes=(/dev/nvidia[0-9]*)
    (( ${#gpu_nodes[@]} == 8 )) || fail "Expected all eight GPU device nodes."
    devices=("${gpu_nodes[@]}" /dev/nvidiactl /dev/nvidia-uvm* /dev/dri/*)
    assert_no_open_files "${devices[@]}"
    for module in nvidia nvidia_uvm nvidia_modeset nvidia_drm; do
        [[ -d "/sys/module/$module" ]] || continue
        previous_modules+=("$module")
        child_count=0
        for holder in /sys/module/"$module"/holders/*; do
            case "${holder##*/}" in
                nvidia_uvm|nvidia_modeset|nvidia_drm) ((child_count+=1)) ;;
                *) fail "Unexpected module depending on $module: $holder" ;;
            esac
        done
        refcount=$(< "/sys/module/$module/refcnt")
        (( refcount == child_count )) || fail "$module has live references ($refcount)."
    done
    echo "All eight GPUs idle. Reloading installed driver without forcing or rebooting."
    # Non-forced removal also refuses new clients that race the checks above.
    for ((i=${#previous_modules[@]}-1; i>=0; i--)); do
        # modprobe may already have removed an unused dependency.
        [[ -d "/sys/module/${previous_modules[i]}" ]] || continue
        if ! modprobe -r "${previous_modules[i]}"; then
            echo "Unload refused; attempting to restore the previously loaded module set." >&2
            for module in "${previous_modules[@]}"; do
                modprobe "$module" || echo "Restore failed: $module" >&2
            done
            fail "Driver reload stopped; no processes were killed."
        fi
    done
    for module in "${previous_modules[@]}"; do
        modprobe "$module" || fail "Loading $module failed; no automatic reboot attempted."
    done
    [[ "$(< /sys/module/nvidia/version)" == "$disk" ]] || fail "Wrong kernel driver loaded."
    nvidia-smi --query-gpu=index,name,driver_version --format=csv
    [[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -eq 8 ]] ||
        fail "NVML did not enumerate all eight GPUs."
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits
    echo "Driver/NVML repair PASSED. No experiments were started."
}

main() {
    local action=${1:---help}
    case "$action" in
        --apply|--disable-updates|--reload-driver) ;;
        *) echo "Usage: sudo bash $0 {--apply|--disable-updates|--reload-driver}"
           echo "--apply disables APT automation, then attempts an idle-only GPU driver reload."
           return 0 ;;
    esac
    (( EUID == 0 )) || fail "Administrator permission required; run sudo in your terminal."
    if [[ "$action" == --apply || "$action" == --disable-updates ]]; then
        disable_updates
    fi
    if [[ "$action" == --apply || "$action" == --reload-driver ]]; then
        reload_driver
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi

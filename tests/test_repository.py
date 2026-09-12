from pathlib import Path
import subprocess
import sys

import yaml

from onpolicy.envs.HKBZ.data_generator import AirportScenarioGenerator, PROFILES, _write_case


ROOT = Path(__file__).resolve().parents[1]


def test_generator_import_without_site_packages():
    """A preinstalled optional dependency must not hide import-time coupling."""
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "from onpolicy.envs.HKBZ.data_generator import AirportScenarioGenerator"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_default_environment_paths_are_portable():
    config = yaml.safe_load((ROOT / "onpolicy/config/env.yaml").read_text(encoding="utf-8"))
    path_keys = (
        "dataset_dir",
        "eval_dataset_dir",
        "jobs_path",
        "fixed_res_path",
        "mobile_res_path",
        "sites_path",
        "flights_path",
    )
    assert all(not Path(config[key]).is_absolute() for key in path_keys)
    assert config["jobs_path"].endswith("job.json")


def test_generator_is_reproducible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        case, metadata = AirportScenarioGenerator(
            profile=PROFILES['balanced'], seed=123, split='train', case_id='smoke'
        ).generate()
        _write_case(output / 'case_01', case, metadata)

    filenames = (
        "job.json",
        "fixed_resources.json",
        "mobile_resources.json",
        "sites.json",
        "flights.json",
    )
    for filename in filenames:
        assert (first / "case_01" / filename).read_bytes() == (
            second / "case_01" / filename
        ).read_bytes()

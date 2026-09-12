"""CPU-only lifecycle tests: exercise real scheduling/publication, fake compute."""
from pathlib import Path

import pytest

from onpolicy.scripts.train import run_stage3_representation as controller
from onpolicy.utils.stage3_representation import ARMS, allocate_resources
from onpolicy.utils.stage3_research import atomic_json, read_json, digest_file, digest_json
from onpolicy.utils import stage3_representation_recovery as recovery


def summary(gain):
    return {"gain_fraction":gain, "completion_rate":1., "delta_seconds":-100*gain,
        "regression_over_5pct_fraction":0., "distributions":{"ood_stress":{"regression_fraction":0.}},
        "profiles":{"stress_joint":{"regression_fraction":0.}}, "tail_makespan":100., "source_tail_makespan":100.}


def suite_fixture(tmp_path, monkeypatch, screen_gain=0., pilot_gain=0.):
    manifest_path = tmp_path/"recovery/test/manifest.json"
    manifest = {"root":str(tmp_path), "source":{"sha256":"C0"}, "source_costs":{"case":100.},
        "mechanisms":{f"{mode}_g{group:04d}":{} for mode in ("R","J") for group in (1,121,181)},
        "resources":{"gpus":[{"index":i,"uuid":f"gpu{i}"} for i in range(8)], "core_groups":[[i,i+64] for i in range(64)]},
        "execution":{"python":"python", "workspace_root":str(tmp_path)},
        "timeouts_seconds":{"progress_warning":900,"validator_request":21600},
        "recovery":{"attempt_id":"test", "audit_path":"audit"}}
    atomic_json(manifest_path, manifest)
    calls = []
    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 900000+len(calls)
            self.phase = command[command.index("--phase")+1]
            self.output = Path(command[command.index("--output")+1])
            self.root = tmp_path
            calls.append((self.phase,self.output.name,command))
            kwargs["stdout"].write(self.phase+"/"+self.output.name+"\n")
            atomic_json(self.output/"status.json", {"status":"running" if self.phase == "validator" else "completed"})
            if self.phase == "contract":
                atomic_json(self.output/"result.json", {"passed":True})
                if self.output.name == "E0":
                    publish("C0",0,0.)
            elif self.phase == "mechanism":
                atomic_json(self.output/"result.json", {"completed":True})
            elif self.phase == "fit":
                atomic_json(self.output/"result.json", {"completed":True})
            elif self.phase == "train":
                until = int(command[command.index("--until")+1])
                arm = command[command.index("--arm")+1]
                atomic_json(self.output/"result.json", {"completed":True,"training_episodes":until})
                publish(arm,until,screen_gain if until == 960 else pilot_gain)
        def poll(self):
            return None if self.phase == "validator" and not (tmp_path/"validator/STOP").exists() else 0
        def wait(self, timeout=None): return 0
    def publish(arm,episodes,gain):
        request = f"{arm}_e{episodes:06d}"
        atomic_json(tmp_path/f"validator/results/{request}.json", {"request_id":request,
            "evaluation":{"summary":summary(gain),"cases":[{"case_id":"case","makespan":100.}]}})
    monkeypatch.setattr(controller.subprocess,"Popen",FakeProcess)
    monkeypatch.setattr(controller.time,"sleep",lambda _:None)
    monkeypatch.setattr(controller,"verify",lambda _: {})
    plan = allocate_resources(manifest["resources"]["core_groups"],[],list(range(8)))
    suite = controller.Suite(manifest,manifest_path=manifest_path)
    monkeypatch.setattr(suite,"resources",lambda:plan)
    return suite,calls,publish


def test_same_name_across_phases_preserves_commands_logs_and_jobs(tmp_path,monkeypatch):
    suite,calls,_ = suite_fixture(tmp_path,monkeypatch)
    atomic_json(tmp_path/"commands/E0.json", {"legacy":"preserved"})
    (tmp_path/"logs").mkdir()
    (tmp_path/"logs/E0.log").write_text("legacy log\n")
    placement = {"gpu":0,"cpus":[8,72],"cuda_memory_fraction":.65}
    suite.start("E0","contract",placement,encoder="E0")
    suite.start("E0","fit",placement,encoder="E0")
    assert set(suite.jobs) == {"contract/E0","fit/E0"}
    for phase in ("contract","fit"):
        command = read_json(tmp_path/f"commands/{phase}/E0.json")
        assert command["task_key"] == f"{phase}/E0"
        assert str(suite.manifest_path) in command["command"]
        assert (tmp_path/f"logs/{phase}/E0.log").read_text() == f"{phase}/E0\n"
    assert read_json(tmp_path/"commands/E0.json") == {"legacy":"preserved"}
    assert (tmp_path/"logs/E0.log").read_text() == "legacy log\n"
    with pytest.raises(FileExistsError): suite.start("E0","fit",placement)
    assert len(calls) == 2


def test_wait_tracks_running_fit_separately_from_completed_contract(tmp_path,monkeypatch):
    suite,calls,_ = suite_fixture(tmp_path,monkeypatch)
    original = controller.subprocess.Popen
    class DelayedProcess(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.polls_left = 5 if self.phase == "fit" else 0
        def poll(self):
            if self.polls_left:
                self.polls_left -= 1
                return None
            return super().poll()
    monkeypatch.setattr(controller.subprocess,"Popen",DelayedProcess)
    class Heartbeat:
        def update(self,**kwargs): pass
    suite.run_tasks([{"phase":"contract","name":"E0","options":{"encoder":"E0"}}],Heartbeat())
    suite.run_tasks([{"phase":"fit","name":"E0","options":{"encoder":"E0"}}],Heartbeat())
    assert suite.jobs["fit/E0"]["process"].polls_left == 0
    assert suite.jobs["contract/E0"]["process"].poll() == 0


def test_old_snapshot_reproduces_the_cross_phase_collision(tmp_path,monkeypatch):
    import importlib.util
    import sys
    root = Path(__file__).resolve().parents[4]
    # The historical file is read-only evidence, never an execution target.
    source = root/"result/hkbz_train_logs/stage3_representation_all_20260908_r1/source/onpolicy/scripts/train/run_stage3_representation.py"
    if not source.exists():
        pytest.skip("Historical failure snapshot not included in this checkout")
    suite,_,_ = suite_fixture(tmp_path,monkeypatch)
    spec=importlib.util.spec_from_file_location("stage3_original_failure",source)
    module=importlib.util.module_from_spec(spec)
    before=list(sys.path)
    try: spec.loader.exec_module(module)
    finally: sys.path[:]=before
    old=module.Suite(suite.m)
    placement={"gpu":0,"cpus":[8,72],"cuda_memory_fraction":.65}
    old.start("E0","contract",placement,encoder="E0")
    with pytest.raises(FileExistsError,match="commands/E0.json"):
        old.start("E0","fit",placement,encoder="E0")


@pytest.mark.parametrize("screen,pilot,expected_endpoints", [(0.,0.,{960}),(.006,.005,{960,1440}),(.006,.011,{960,1440,1920})])
def test_real_scheduler_crosses_diagnosis_fit_screen_and_gated_pilot(tmp_path,monkeypatch,screen,pilot,expected_endpoints):
    suite,calls,_ = suite_fixture(tmp_path,monkeypatch,screen,pilot)
    suite.run()
    phases = [phase for phase,_,_ in calls]
    assert phases.count("contract") == 4 and phases.count("mechanism") == 6 and phases.count("fit") == 4
    assert phases.count("validator") == 2  # ONE persistent global pool across phases.
    trains = [command for phase,_,command in calls if phase == "train"]
    endpoints = [int(cmd[cmd.index("--until")+1]) for cmd in trains]
    assert set(endpoints) == expected_endpoints and endpoints.count(960) == 8
    assert endpoints.count(1440) <= 2 and endpoints.count(1920) <= 2
    assert read_json(tmp_path/"result.json")["completed"]
    assert len(suite.jobs) == len(calls)  # No overwritten in-memory jobs.
    for cmd in trains:
        if int(cmd[cmd.index("--until")+1]) > 960:
            assert "--resume-checkpoint" in cmd


def test_resume_fit_skips_completed_diagnostics_and_uses_new_pool_names(tmp_path,monkeypatch):
    suite,calls,publish = suite_fixture(tmp_path,monkeypatch)
    suite.resume_fit = True
    for encoder in ("E0","E1","E2","E3"):
        atomic_json(tmp_path/f"contract/{encoder}/result.json",{"passed":True})
    publish("C0",0,0.)
    for i in range(2):
        atomic_json(tmp_path/f"validator/workers/pool{i}/status.json",{"status":"failed","legacy":True})
    suite.run()
    assert not any(phase in ("contract","mechanism") for phase,_,_ in calls)
    assert {name for phase,name,_ in calls if phase == "validator"} == {"pool0_test","pool1_test"}
    assert all(read_json(tmp_path/f"validator/workers/pool{i}/status.json")["legacy"] for i in range(2))
    assert read_json(tmp_path/"training_admission.json")["eligible_arms"] == list(ARMS)
    assert read_json(tmp_path/"status.json")["active_manifest"] == str(suite.manifest_path)


@pytest.mark.parametrize("phase,name", [("fit","../E0"),("../fit","E0"),("train","x/y"),("unknown","E0")])
def test_task_identity_rejects_unsafe_paths(phase,name):
    with pytest.raises(ValueError): controller.Suite.task_key(phase,name)


def recovery_fixture(tmp_path):
    root,workspace = tmp_path/"run",tmp_path/"workspace"
    root.mkdir(); workspace.mkdir()
    relative = "onpolicy/runner/shared/scientific_engine.py"
    source = root/"source"/relative; source.parent.mkdir(parents=True)
    source.write_text("unchanged scientific implementation\n")
    for name in recovery.RECOVERY_OVERLAYS:
        p=workspace/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text("reviewed administrative fix\n")
    files={relative:digest_file(source)}
    manifest={"root":str(root),"created_unix":1.,"schema":"study","arms":ARMS,
        "code":{"files":files,"sha256":digest_json(files)},
        "execution":{"code_root":str(root/"source"),"workspace_root":str(workspace),"python":"python","packages":{}},
        "source_costs":{"case":100.},"source":{"sha256":"C0"},"splits":{"tune":[{"path":"case"}]},
        "contract_sha256":"contract","input_files":{},"training":{"seed":42,"lr":5e-6},"material_passport":{},
        "mechanisms":{f"{m}_g{g:04d}":{} for m in ("R","J") for g in (1,121,181)}}
    manifest["manifest_sha256"]=controller.identity(manifest)
    atomic_json(root/"manifest.json",manifest)
    for phase,names in (("contract",("E0","E1","E2","E3")),("mechanism",manifest["mechanisms"])):
        for name in names:
            atomic_json(root/phase/name/"status.json",{"status":"completed","pid":0})
            row={"passed":True,"representation":{"variant":name}} if phase=="contract" else {"completed":True,"historical_trace_exact":True}
            atomic_json(root/phase/name/"result.json",row)
    checkpoint=root/"contract/E0/initial_c0.pt"; checkpoint.write_text("validated C0")
    atomic_json(root/"validator/results/C0_e000000.json",{"ok":True,"training_episodes":0,"request_id":"C0_e000000",
        "code_sha256":manifest["code"]["sha256"],"cases_sha256":digest_json(manifest["splits"]["tune"]),
        "contract_sha256":"contract","checkpoint_sha256":digest_file(checkpoint),
        "evaluation":{"cases":[{"case_id":"case","completed":True,"makespan":100.}]}})
    atomic_json(root/"status.json",{"status":"failed","phase":"teacher_fit_diagnostic","pid":0,
        "error":"File exists: commands/E0.json"})
    atomic_json(root/"resource_leases.json",{"old":True})
    atomic_json(root/"commands/E0.json",{"old":True})
    return root,workspace,manifest


def test_recovery_preserves_old_snapshot_artifacts_and_scientific_protocol(tmp_path):
    root,workspace,prior=recovery_fixture(tmp_path)
    before={str(p.relative_to(root)):digest_file(p) for p in root.rglob("*") if p.is_file()}
    recovery.prepare_recovery(root/"manifest.json","fit1",workspace)
    path=root/"recovery/fit1/manifest.json"; manifest=read_json(path)
    assert all(digest_file(root/name)==sha for name,sha in before.items())
    assert manifest["training"] == prior["training"] and manifest["source"] == prior["source"]
    assert manifest["root"]==prior["root"] and manifest["execution"]["code_root"] != prior["execution"]["code_root"]
    recovery.validate_recovery(manifest)
    recovery.activate_recovery(manifest,path)
    assert read_json(root/"active_execution.json")["manifest"]==str(path)
    assert digest_file(root/"status.json")==before["status.json"]  # Heartbeat replacement happens only after activation.
    with pytest.raises(FileExistsError): recovery.activate_recovery(manifest,path)


@pytest.mark.parametrize("change", ["training","engine","evidence"])
def test_recovery_rejects_changed_science_or_evidence(tmp_path,change):
    root,workspace,_=recovery_fixture(tmp_path)
    recovery.prepare_recovery(root/"manifest.json","fit1",workspace)
    manifest=read_json(root/"recovery/fit1/manifest.json")
    if change=="training": manifest["training"]["lr"]*=2
    elif change=="engine": manifest["code"]["files"]["onpolicy/runner/shared/scientific_engine.py"]="changed"
    else: atomic_json(root/"contract/E2/result.json",{"passed":False})
    with pytest.raises(ValueError): recovery.validate_recovery(manifest)


def test_recovery_refuses_to_discard_partial_fit(tmp_path):
    root,workspace,_=recovery_fixture(tmp_path)
    recovery.prepare_recovery(root/"manifest.json","fit1",workspace)
    path=root/"recovery/fit1/manifest.json"
    atomic_json(root/"fit/E0/updates/000.json",{"precious":"partial update"})
    with pytest.raises(ValueError,match="fit already has artifacts"):
        recovery.activate_recovery(read_json(path),path)
    assert not (root/"recovery/fit1/started.json").exists()


def test_failed_admission_cannot_be_reused(tmp_path):
    root,workspace,_=recovery_fixture(tmp_path)
    atomic_json(root/"contract/E2/result.json",{"passed":False})
    with pytest.raises(ValueError,match="Failed architecture"):
        recovery.prepare_recovery(root/"manifest.json","fit1",workspace)
    assert not (root/"recovery/fit1").exists()

"""Resource scheduling tests; these do not constitute algorithm admission."""
import copy
from types import SimpleNamespace

import pytest

from onpolicy.utils.stage3_local_resources import (SHARED_VALIDATOR, cpu_set,
    memory_fraction, pilot_resource_plan, validate_coexistence, validate_worker_placement)


def resources(shared=True):
    value = {"physical_gpus": [0, 1, 2, 3], "parent_cpuset": "0-31,64-95",
             "logical_cpu_count": 64, "physical_cpu_cores": 32,
             "lanes": ["0-7,64-71", "8-15,72-79", "16-23,80-87", "24-31,88-95"]}
    if shared:
        value["pilot_validator"] = copy.deepcopy(SHARED_VALIDATOR)
    return value


def test_four_arms_share_only_gpu3_without_increasing_cpu_budget():
    original = resources()
    before = copy.deepcopy(original)
    plan = pilot_resource_plan(original, 4)
    assert original == before
    assert list(plan["trainers"]) == [0, 1, 2, 3]
    workers = list(plan["trainers"].values()) + [plan["validator"]]
    cpus = [cpu_set(w["cpuset"]) for w in workers]
    assert sum(map(len, cpus)) == 64
    assert set.union(*cpus) == cpu_set(original["parent_cpuset"])
    assert all((cpu + 64 in group) for group in cpus for cpu in group if cpu < 64)
    assert plan["trainers"][3]["cuda_memory_fraction"] == .60
    assert plan["validator"]["cuda_memory_fraction"] == .20
    assert len(cpu_set(plan["trainers"][3]["cpuset"])) == 12
    assert len(cpu_set(plan["validator"]["cpuset"])) == 4


@pytest.mark.parametrize("count", [1, 2, 3])
def test_fewer_arms_keep_dedicated_validator(count):
    plan = pilot_resource_plan(resources(), count)
    assert plan["profile"] == "dedicated_validator"
    assert len(plan["trainers"]) == count
    assert plan["validator"]["cpuset"] == "24-31,88-95"
    assert plan["validator"]["cuda_memory_fraction"] == .80


def test_old_manifest_preserves_three_training_lanes():
    plan = pilot_resource_plan(resources(shared=False), 4)
    assert plan["profile"] == "dedicated_validator"
    assert list(plan["trainers"]) == [0, 1, 2]


@pytest.mark.parametrize("key,value", [("trainer_cpus", "24-31,88-95"),
    ("validator_cpus", "32-33,96-97"), ("validator_cuda_memory_fraction", .30),
    ("trainer_cuda_memory_fraction", float("nan")), ("gpu_lane", 2), ("mode", "arbitrary")])
def test_invalid_shared_resource_contract_is_rejected(key, value):
    r = resources()
    r["pilot_validator"][key] = value
    with pytest.raises(ValueError):
        pilot_resource_plan(r, 4)


@pytest.mark.parametrize("value", [0, -1, .81, float("nan"), float("inf")])
def test_allocator_fraction_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        memory_fraction(value)
    from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
    with pytest.raises(ValueError, match="allocator fraction"):
        ResearchEngine(cuda_memory_fraction=value)


@pytest.mark.parametrize("value", ["0-2,2", "-1", "4-2", "1-2-3", ""])
def test_malformed_cpu_sets_are_rejected(value):
    with pytest.raises(ValueError):
        cpu_set(value)


def test_cpu_partition_and_gpu_budget_cannot_expand():
    r = resources()
    r["lanes"][0] = "0-8,64-71"
    with pytest.raises(ValueError, match="partition"):
        pilot_resource_plan(r, 4)
    r = resources()
    r["physical_gpus"] = [0, 1, 2, 4]
    with pytest.raises(ValueError, match="GPU0-3"):
        pilot_resource_plan(r, 4)


def test_only_one_trainer_and_one_validator_can_coexist():
    plan = pilot_resource_plan(resources(), 4)
    trainer = {**plan["trainers"][3], "phase": "train"}
    validator = {**plan["validator"], "phase": "validator"}
    validate_coexistence(plan, "train", trainer, [validator])
    validate_coexistence(plan, "validator", validator, [trainer])
    for phase, target, peers in (("train", trainer, [trainer]),
            ("validator", validator, [validator]), ("fit", trainer, [validator]),
            ("train", trainer, [trainer, validator])):
        with pytest.raises(RuntimeError):
            validate_coexistence(plan, phase, target, peers)
    with pytest.raises(RuntimeError):
        validate_coexistence(None, "train", trainer, [validator])


def test_actual_launcher_places_four_trainers_and_single_validator(tmp_path, monkeypatch):
    from onpolicy.scripts.train import run_stage3_local_exploration as controller
    from onpolicy.utils.stage3_research import read_json
    manifest = {"root": str(tmp_path), "resources": resources(),
                "execution": {"workspace_root": "workspace", "python": "python"},
                "timeouts_seconds": {"suite": 1000}}
    suite = controller.Suite(manifest)
    suite.pilot_plan = pilot_resource_plan(manifest["resources"], 4)
    launched = []
    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    suite.start("validator", "validator", 3)
    for lane, arm in enumerate(controller.ARMS):
        suite.start(f"pilot_{arm}", "train", lane, arm=arm)
    assert len(launched) == 5
    assert [kw["env"]["CUDA_VISIBLE_DEVICES"] for _, kw in launched] == ["3", "0", "1", "2", "3"]
    assert launched[0][0][2] == "30-31,94-95"
    assert launched[-1][0][2] == "24-29,88-93"
    assert launched[0][1]["env"]["HKBZ_STAGE3_CUDA_MEMORY_FRACTION"] == "0.2"
    assert launched[-1][1]["env"]["HKBZ_STAGE3_CUDA_MEMORY_FRACTION"] == "0.6"
    assert read_json(tmp_path / "commands/validator.json")["cuda_memory_fraction"] == .2
    with pytest.raises(RuntimeError):
        suite.start("second_validator", "validator", 3)


def test_worker_forwards_fraction_without_changing_rollout_width(monkeypatch):
    from onpolicy.scripts.train import stage3_local_worker as worker
    from onpolicy.runner.shared import stage3_research_engine as module
    captured = []
    monkeypatch.setattr(module, "ResearchEngine", lambda *args, **kwargs: captured.append((args, kwargs)))
    monkeypatch.setenv("HKBZ_STAGE3_CUDA_MEMORY_FRACTION", ".20")
    worker.engine({"source": {"path": "checkpoint"}})
    assert captured == [(("checkpoint",), {"width": 8, "exploration": None, "diagnostics": True,
                                           "cuda_memory_fraction": .20})]
    monkeypatch.delenv("HKBZ_STAGE3_CUDA_MEMORY_FRACTION")
    worker.engine({"source": {"path": "checkpoint"}}, "J")
    assert captured[-1][1]["cuda_memory_fraction"] == .80
    assert captured[-1][1]["exploration"] == "J"


def test_worker_checks_real_affinity_device_and_allocator_limit():
    placement = pilot_resource_plan(resources(), 4)["validator"]
    actual = {"affinity": cpu_set(placement["cpuset"]), "visible_devices": "3", "cuda_memory_fraction": ".20"}
    validate_worker_placement(placement, **actual)
    for key, bad in (("affinity", cpu_set("24-31,88-95")), ("visible_devices", "0,1,2,3"),
                     ("cuda_memory_fraction", ".80")):
        with pytest.raises(ValueError):
            validate_worker_placement(placement, **{**actual, key: bad})

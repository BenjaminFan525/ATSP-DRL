"""Full Stage1 coverage, immutable two-arm contracts and validation requests."""
from collections import Counter
from pathlib import Path

from onpolicy.utils.stage3_research import (ROOT, TRAIN_ROOT, atomic_json, read_json,
    digest_file, digest_json, verify_cases, code_changes)
from onpolicy.utils.stage3_representation import RepresentationQueue
from onpolicy.utils.stage3_local_exploration import MODES

ARMS = {
    "B_SHARED": {"encoder": "F_SHARED", "gpus": [0, 1, 2, 3]},
    "C_PRIVATE": {"encoder": "F_PRIVATE", "gpus": [4, 5, 6, 7]},
}
COUNTS = {"iid": 480, "ood_stress": 108, "ood_scale": 12}
QUOTAS = {"iid": 480, "ood_stress": 432, "ood_scale": 48}
HISTORY = "stable_request_identity_v1"
EVAL_PROTOCOL = {"decoder": "single_greedy", "history": HISTORY,
                 "environment": "H2/F4/soft", "tau": .3, "seed": 42}


def identity(manifest):
    return digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def schedule(cases, seed, batch_size=32, epochs=8):
    if Counter(c["distribution"] for c in cases) != COUNTS or len({c["path"] for c in cases}) != 600:
        raise ValueError("Full-data training requires exactly the unique Stage1 train600")
    if batch_size != 32 or not 1 <= epochs <= 8:
        raise ValueError("Full-data protocol fixes batch32 and at most eight coverage epochs")
    result = []
    for epoch in range(epochs):
        visits = []
        for distribution, quota in QUOTAS.items():
            pool = [c for c in cases if c["distribution"] == distribution]
            for repetition in range(quota // len(pool)):
                visits.extend((c, repetition) for c in pool)
        visits.sort(key=lambda x: digest_json([seed, "full-data-order", epoch, x[0]["content_sha256"], x[1]]))
        assert len(visits) == 960 and len({c["path"] for c, _ in visits}) == 600
        for first in range(0, 960, batch_size):
            batch = visits[first:first + batch_size]
            result.append({"cases": [c for c, _ in batch], "visit": epoch, "data_epoch": epoch + 1,
                "seeds": [int(digest_json([seed, "full-data-trajectory", epoch, c["content_sha256"], r])[:15], 16)
                          % (2**31 - 1) for c, r in batch],
                "visit_ids": [f"{epoch}:{first+i}" for i in range(batch_size)],
                "group": len(result) + 1, "training_episodes": (len(result) + 1) * batch_size})
    return result


def verify(manifest, inputs=True):
    from onpolicy.utils.stage3_numerics import RUNTIME
    if (manifest.get("schema") != "stage3-full-data-two-arm-v1"
            or identity(manifest) != manifest["manifest_sha256"] or manifest["arms"] != ARMS
            or manifest["history"] != HISTORY or manifest["numerics"] != RUNTIME):
        raise ValueError("Full-data study identity mismatch")
    if Path(manifest["execution"]["code_root"]).resolve() != ROOT.resolve():
        raise ValueError("Full-data jobs must use their immutable source snapshot")
    if code_changes(manifest["code"]["files"], ROOT):
        raise ValueError("Frozen full-data source changed")
    if digest_file(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("Original C0 changed")
    if digest_json(manifest["contract"]) != manifest["contract_sha256"]:
        raise ValueError("Full-data environment contract changed")
    if (manifest["contract"]["device_future_intent_horizon"] != 2
            or manifest["contract"]["device_frontier_max_requests"] != 4
            or manifest["contract"]["device_lookahead_reservation_mode"] != "soft"):
        raise ValueError("Only H2/F4/soft is authorized")
    if inputs:
        import importlib.metadata
        for path, checksum in manifest["input_files"].items():
            if digest_file(path) != checksum:
                raise ValueError(f"Full-data input changed: {path}")
        for name, cases in manifest["splits"].items():
            verify_cases(cases, training=name == "train_full600")
        for name, version in manifest["execution"]["packages"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Runtime package changed: {name}")
        plan = schedule(manifest["splits"]["train_full600"], manifest["training"]["seed"])
        if digest_json(plan) != manifest["training"]["schedule_sha256"]:
            raise ValueError("Full-data sampling changed")
    return code_changes(manifest["code"]["files"], manifest["execution"]["workspace_root"])


def source_costs(manifest, *, confirmation=False):
    path = Path(manifest["root"]) / ("confirmation_baseline.json" if confirmation else "baselines.json")
    admission = Path(manifest["root"]) / "training_admission.json"
    if not confirmation and admission.exists() and read_json(admission)["baseline_sha256"] != digest_file(path):
        raise ValueError("Admitted source baseline file changed")
    payload = read_json(path)
    if (payload["protocol_sha256"] != manifest["manifest_sha256"]
            or payload["source_sha256"] != manifest["source"]["sha256"]
            or payload["contract_sha256"] != manifest["contract_sha256"]
            or payload["history"] != HISTORY
            or digest_json(payload["costs"]) != payload["costs_sha256"]):
        raise ValueError("C0 costs do not belong to this full-data contract")
    names = ("confirmation",) if confirmation else ("train_full600", "validation", "tune")
    if set(payload["costs"]) != {c["path"] for name in names for c in manifest["splits"][name]}:
        raise ValueError("C0 baseline case coverage mismatch")
    import math
    if any(not math.isfinite(v) or v <= 0 for v in payload["costs"].values()):
        raise ValueError("Invalid C0 baseline costs")
    return payload["costs"]


def publish(manifest, checkpoint, arm, episodes, variant, *, split, start=0, count=None, baseline=False):
    cases = manifest["splits"][split]
    count = len(cases) - start if count is None else count
    selected = cases[start:start + count]
    if not selected or len(selected) != count:
        raise ValueError("Invalid validation slice")
    request_id = f"{arm}_{split}_e{episodes:06d}_s{start:04d}"
    queue = RepresentationQueue(Path(manifest["root"]) / "validator")
    return queue.submit({"request_id": request_id, "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest_file(checkpoint), "cases": selected, "cases_sha256": digest_json(selected),
        "contract_sha256": manifest["contract_sha256"], "code_sha256": manifest["code"]["sha256"],
        "protocol_sha256": manifest["manifest_sha256"], "tau": .3, "seed": 42,
        "evaluation_protocol": EVAL_PROTOCOL, "training_exploration": MODES["J"],
        "training_episodes": episodes, "representation": variant, "arm": arm,
        "split": split, "case_start": start, "case_count": count, "baseline": baseline})


def publish_epoch(manifest, checkpoint, arm, episodes, variant):
    return [publish(manifest, checkpoint, arm, episodes, variant, split=s) for s in ("validation", "tune")]


def validate_request(manifest, request):
    split = request["split"]
    if split not in ("train_full600", "validation", "tune", "confirmation"):
        raise ValueError("Unregistered evaluation split")
    start, count = request["case_start"], request["case_count"]
    if not isinstance(start, int) or not isinstance(count, int) or start < 0 or count < 1:
        raise ValueError("Invalid evaluation bounds")
    cases = manifest["splits"][split][start:start + count]
    if (request["cases"] != cases or len(cases) != count or digest_json(cases) != request["cases_sha256"]
            or request["evaluation_protocol"] != EVAL_PROTOCOL or request["tau"] != .3 or request["seed"] != 42
            or request["training_exploration"] != MODES["J"]
            or request["protocol_sha256"] != manifest["manifest_sha256"]
            or request["code_sha256"] != manifest["code"]["sha256"]
            or request["contract_sha256"] != manifest["contract_sha256"]
            or request["cache_key"] != RepresentationQueue.identity(request)
            or digest_file(request["checkpoint"]) != request["checkpoint_sha256"]):
        raise ValueError("Evaluation request contract mismatch")
    if request["request_id"] != f"{request['arm']}_{split}_e{request['training_episodes']:06d}_s{start:04d}":
        raise ValueError("Noncanonical request ID")
    if split == "confirmation":
        locked = read_json(Path(manifest["root"]) / "selection_locked.json")
        if locked["protocol_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Confirmation requires locked selection")
        if not request["baseline"] and request["checkpoint_sha256"] != locked["selected"][request["arm"]]["checkpoint_sha256"]:
            raise ValueError("Unselected checkpoint attempted confirmation")
    if request["baseline"]:
        if (request["arm"] != "C0" or request["training_episodes"] != 0
                or request["checkpoint_sha256"] != manifest["source"]["sha256"]
                or request["representation"] != "F_SHARED"):
            raise ValueError("Invalid source baseline request")
    elif (request["arm"] not in ARMS or ARMS[request["arm"]]["encoder"] != request["representation"]
          or split == "train_full600" or start != 0 or count != len(manifest["splits"][split])):
        raise ValueError("Invalid trained-policy evaluation")

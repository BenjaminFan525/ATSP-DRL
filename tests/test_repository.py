from pathlib import Path

import yaml

from onpolicy.envs.HKBZ.data_generator import build_dataset


ROOT = Path(__file__).resolve().parents[1]


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
    kwargs = dict(
        num_cases=1,
        num_stands=10,
        num_planes=6,
        seed=123,
        save_layouts=False,
    )
    build_dataset(base_dir=first, **kwargs)
    build_dataset(base_dir=second, **kwargs)

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

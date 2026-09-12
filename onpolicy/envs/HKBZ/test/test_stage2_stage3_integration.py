"""Regression boundary between frozen Stage2 and the continuing Stage3 code."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import RequestReadyTimeHead
from onpolicy.config.config import get_config
from onpolicy.utils.stage2_matching import solve_resource_matching


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / 'artifacts/stage2_frozen/20260912/manifest.json'


def test_stage3_options_survive_and_frozen_source_is_opt_in():
    args = get_config().parse_args([
        '--training_stage', 'joint_finetune',
        '--role_event_credit_mode', 'critical_path_v2',
        '--actor_grad_clip_mode', 'per_group',
        '--counterfactual_baseline_mix', '0.4',
        '--counterfactual_baseline_mix_schedule', '0.4,0.8',
        '--bc_reference_checkpoint', 'explicit-c0.pt',
    ])
    assert args.stage2_frozen_manifest == ''
    assert args.role_event_credit_mode == 'critical_path_v2'
    assert args.actor_grad_clip_mode == 'per_group'
    assert args.counterfactual_baseline_mix == .4
    assert args.bc_reference_checkpoint == 'explicit-c0.pt'


def test_manifest_environment_is_default_only_and_stage3_budget_is_preserved():
    from onpolicy.scripts.train.train_hkbz import parse_args
    from onpolicy.utils.stage2_frozen import load_frozen_stage2

    args = parse_args(['--stage2_frozen_manifest', str(MANIFEST),
                       '--num_episodes', '17', '--lr', '0.000012'], get_config())
    assert Path(args.env_config) == MANIFEST.parent / 'config/env_resource_joint.yaml'
    _, args, _ = load_frozen_stage2(MANIFEST, args)
    assert args.num_episodes == 17 and args.lr == .000012
    assert args.training_stage == 'joint_finetune'
    assert args.stage3_handoff_mode == 'strict'
    assert Path(args.checkpoint_dir) == MANIFEST.parent / 'checkpoints/b0.pt'

    explicit = ROOT / 'onpolicy/config/env_joint_finetune.yaml'
    args = parse_args(['--stage2_frozen_manifest', str(MANIFEST),
                       '--env_config=' + str(explicit)], get_config())
    assert Path(args.env_config) == explicit


def test_frozen_source_and_resume_cannot_be_combined():
    from onpolicy.scripts.train import train_hkbz

    with patch.object(train_hkbz, 'apply_formal_safe_pipeline_manifest') as configure:
        with pytest.raises(ValueError, match='either --stage2_frozen_manifest'):
            train_hkbz.main(['--stage2_frozen_manifest', str(MANIFEST),
                             '--checkpoint_dir', 'existing-stage3.pt'])
        configure.assert_not_called()


def test_default_ready_head_keeps_legacy_tensors_output_and_rng():
    torch.manual_seed(912)
    head = RequestReadyTimeHead(8, 3600., device='cpu', dtype=torch.float32)
    assert set(head.state_dict()) == {
        f'network.{layer}.{kind}' for layer in (0, 2, 3) for kind in ('weight', 'bias')
    }
    nodes = [torch.randn(2, 5, 8) for _ in range(4)]
    dag = torch.rand(2, 5) * 600.
    context = torch.cat([*nodes, torch.log1p(dag / 3600.).unsqueeze(-1)], dim=-1)
    legacy_residual = 3600. * F.softplus(head.network(context).squeeze(-1))
    rng = torch.get_rng_state().clone()
    prediction, residual = head(*nodes, dag)
    assert torch.equal(prediction, dag + legacy_residual)
    assert torch.equal(residual, legacy_residual)
    assert torch.equal(torch.get_rng_state(), rng)


def _legacy_matching(scores, lookahead):
    rows, requests = scores.shape
    real = requests - 1
    matrix = np.full((rows, real + rows), -1.e30, dtype=np.float64)
    if real:
        matrix[:, :real] = scores[:, 1:]
        matrix[:, :real][np.isfinite(scores[:, 1:]) & ~lookahead[None, 1:]] += 1.e6
    for row in range(rows):
        matrix[row, real + row] = scores[row, 0]
    matrix[~np.isfinite(matrix)] = -1.e30
    chosen_rows, columns = linear_sum_assignment(matrix, maximize=True)
    assignments = np.full(rows, -1, dtype=np.int64)
    assignments[chosen_rows] = columns
    return np.where(assignments < real, assignments + 1, 0)


@pytest.mark.parametrize('requests', [1, 2, 9, 25])
def test_stage2_matching_extraction_preserves_stage3_greedy_and_ties(requests):
    rng = np.random.default_rng(912)
    for rows in (1, 2, 8):
        for tied in (False, True):
            scores = np.zeros((rows, requests)) if tied else rng.normal(size=(rows, requests))
            scores[rng.random(scores.shape) < .4] = -np.inf
            scores[:, 0] = 0.  # Every device has its private legal no-op.
            lookahead = rng.random(requests) < .5
            assert np.array_equal(solve_resource_matching(scores, lookahead),
                                  _legacy_matching(scores, lookahead))

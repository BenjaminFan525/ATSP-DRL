import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from onpolicy.algorithms.gnn_mappo.algorithm.MAPPOPolicy import (
    GNN_MAPPOPolicy,
)
from onpolicy.config.config import get_config
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT / 'onpolicy/envs/HKBZ/dataset/'
    'fjsp_v3_t600_v120_test60/train/case_0001'
)


def policy(mode, *, resource_adapter=False, ready_injection="learned"):
    args = get_config().parse_args([])
    args.max_agent_num = 24
    args.max_device_num = 80
    args.resource_policy = 'drl'
    args.device_policy_head_mode = mode
    args.ordinary_device_type_count = 10
    args.device_resource_adapter = resource_adapter
    args.request_ready_prediction = True
    args.request_ready_policy_injection = ready_injection
    with (ROOT / 'onpolicy/config/ac.yaml').open(
        'r', encoding='utf-8'
    ) as handle:
        config = yaml.safe_load(handle)
    return GNN_MAPPOPolicy(args, config)


class TestStage2NextRound(unittest.TestCase):
    def test_every_head_family_routes_a_real_resource_decision(self):
        config = {
            'jobs_path': str(CASE_DIR / 'job.json'),
            'fixed_res_path': str(CASE_DIR / 'fixed_resources.json'),
            'mobile_res_path': str(CASE_DIR / 'mobile_resources.json'),
            'sites_path': str(CASE_DIR / 'sites.json'),
            'flights_path': str(CASE_DIR / 'flights.json'),
            'resource_policy': 'drl',
            'n_agents': 24,
            'max_device_num': 80,
            'global_feature_mode': 'f1f2',
        }
        for mode in ('shared', 'type_adapter', 'per_type'):
            torch.manual_seed(7)
            env = AircraftScheduleEnv(config)
            try:
                obs, _, info = env.reset()
                model = policy(mode)
                n_agents = env.n_agents
                recurrent_n = 1
                hidden_size = 64
                rnn = np.zeros(
                    (1, n_agents, recurrent_n, hidden_size),
                    dtype=np.float32,
                )
                last_op = -np.ones((1, n_agents), dtype=np.float32)
                last_site = -np.ones((1, n_agents), dtype=np.float32)
                saw_resource_decision = False
                for _ in range(8):
                    active = info['active_agents'][None, :, None].astype(
                        np.float32
                    )
                    outputs = model.get_actions(
                        Batch.from_data_list([obs]),
                        rnn,
                        active,
                        last_op,
                        last_site,
                        agent_types=info['agent_types'][None, :],
                        return_decision_mask=True,
                    )
                    self.assertTrue(all(
                        torch.isfinite(value).all()
                        for value in outputs if torch.is_tensor(value)
                    ))
                    actions = outputs[1].detach().cpu().numpy()[0]
                    saw_resource_decision |= bool(np.any(
                        info['active_agents'] & (info['agent_types'] > 0)
                    ))
                    obs, _, done, info = env.step(actions)
                    rnn = outputs[3].detach().cpu().numpy()
                    last_op = actions[:, 0][None, :]
                    last_site = actions[:, 1][None, :]
                    if saw_resource_decision or bool(np.asarray(done).all()):
                        break
                self.assertTrue(saw_resource_decision, mode)
                self.assertEqual(int(obs.device_type_ids.min()), -1)
                self.assertEqual(int(obs.device_type_ids.max()), 9)
            finally:
                env.close()

    def test_type_adapter_is_zero_initialized_and_checkpoint_compatible(self):
        shared = policy('shared')
        adapter = policy('type_adapter')
        adapter.load_model_state(shared.ac.state_dict())
        self.assertTrue(torch.equal(
            adapter.ac.device_type_adapter.weight,
            torch.zeros_like(adapter.ac.device_type_adapter.weight),
        ))
        self.assertEqual(
            adapter.ac.device_policy_head_mode, 'type_adapter'
        )

    def test_resource_adapter_is_zero_initialized_and_checkpoint_compatible(self):
        shared = policy('shared')
        adapted = policy(
            'shared', resource_adapter=True, ready_injection='none'
        )
        adapted.load_model_state(shared.ac.state_dict())
        final = adapted.ac.resource_residual_adapter.network[-1]
        self.assertTrue(torch.equal(final.weight, torch.zeros_like(final.weight)))
        self.assertTrue(torch.equal(final.bias, torch.zeros_like(final.bias)))
        self.assertEqual(adapted.ac.request_ready_policy_injection, 'none')

    def test_per_type_heads_copy_the_shared_device_backend(self):
        shared = policy('shared')
        separated = policy('per_type')
        separated.load_model_state(shared.ac.state_dict())
        source_encoder = shared.ac.device_sel_enc.state_dict()
        source_actor = shared.ac.device_actor.state_dict()
        for module in separated.ac.device_type_sel_encs:
            for name, value in module.state_dict().items():
                self.assertTrue(torch.equal(value, source_encoder[name]))
        for module in separated.ac.device_type_actors:
            for name, value in module.state_dict().items():
                self.assertTrue(torch.equal(value, source_actor[name]))

    def test_resource_fitted_potential_schema_is_strict(self):
        payload = {
            'potential_schema_version': (
                AircraftScheduleEnv.RESOURCE_POTENTIAL_SCHEMA_VERSION
            ),
            'environment_semantics_version': AircraftScheduleEnv.SEMANTICS_VERSION,
            'feature_names': list(
                AircraftScheduleEnv.RESOURCE_POTENTIAL_FEATURES
            ),
            'weights': {
                name: 1.0
                for name in AircraftScheduleEnv.RESOURCE_POTENTIAL_FEATURES
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'potential.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            env = AircraftScheduleEnv.__new__(AircraftScheduleEnv)
            env.hindsight_reward_mode = 'team_time_resource_fitted_potential'
            env._load_iga_potential_config({
                'iga_potential_beta': 0.1,
                'iga_potential_gamma': 1.0,
                'iga_potential_weights_path': str(path),
            })
            self.assertEqual(
                set(env.iga_potential_weights),
                set(AircraftScheduleEnv.RESOURCE_POTENTIAL_FEATURES),
            )


if __name__ == '__main__':
    unittest.main()

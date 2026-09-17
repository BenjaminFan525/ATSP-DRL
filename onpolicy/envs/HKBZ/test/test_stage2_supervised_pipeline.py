"""Focused tests for the dense, supervised-only Stage2 contract."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Batch
import yaml

from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import (
    GNN_Actor_Critic,
    RequestReadyTimeHead,
)
from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.test.test_device_lookahead_dispatch import (
    _commit_plane_to_future_mobile_job,
    _make_env,
)
from onpolicy.runner.shared.hkbz_runner import HKBZ_Runner
from onpolicy.utils.training_stage import STAGE2_SUPERVISION_CONTRACT


class RequestReadyPredictionHeadTest(unittest.TestCase):
    def test_prediction_is_differentiable_and_never_below_dag_bound(self):
        head = RequestReadyTimeHead(
            8,
            3600.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        request = torch.randn(4, 8)
        operation = torch.randn(4, 8)
        site = torch.randn(4, 8)
        global_embedding = torch.randn(4, 8)
        dag_bound = torch.tensor([0.0, 1.0, 60.0, 900.0])

        prediction, residual = head(
            request,
            operation,
            site,
            global_embedding,
            dag_bound,
        )

        self.assertTrue(torch.all(torch.isfinite(prediction)))
        self.assertTrue(torch.all(residual >= 0.0))
        self.assertTrue(torch.all(prediction >= dag_bound))
        prediction.sum().backward()
        self.assertIsNotNone(head.network[-1].bias.grad)
        self.assertGreater(
            float(head.network[-1].bias.grad.abs().sum().item()),
            0.0,
        )

    def test_reliable_head_routes_blocking_and_orders_quantiles(self):
        head = RequestReadyTimeHead(
            8,
            3600.0,
            context_features=True,
            head_mode="horizon_split",
            quantile_head=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        shape = (2, 3, 8)
        request = torch.randn(*shape)
        operation = torch.randn(*shape)
        site = torch.randn(*shape)
        global_embedding = torch.randn(*shape)
        dag_bound = torch.tensor([
            [30.0, 120.0, 0.0],
            [15.0, 240.0, 0.0],
        ])
        kinds = torch.tensor([[1, 2, 3], [1, 2, 3]])
        context = torch.zeros(
            2, 3, RequestReadyTimeHead.EXPLICIT_CONTEXT_DIM
        )

        prediction, residual, quantiles = head(
            request,
            operation,
            site,
            global_embedding,
            dag_bound,
            explicit_context=context,
            kind_ids=kinds,
            hard_blocking=True,
            return_quantiles=True,
        )

        blocking = kinds == RequestReadyTimeHead.BLOCKING_KIND
        self.assertTrue(torch.equal(
            prediction[blocking], torch.zeros_like(prediction[blocking])
        ))
        self.assertTrue(torch.equal(
            residual[blocking], torch.zeros_like(residual[blocking])
        ))
        self.assertTrue(torch.equal(
            quantiles[blocking], torch.zeros_like(quantiles[blocking])
        ))
        future = ~blocking
        self.assertTrue(torch.all(prediction[future] >= dag_bound[future]))
        self.assertTrue(torch.all(
            quantiles[..., 0][future] <= quantiles[..., 1][future]
        ))
        self.assertTrue(torch.all(
            quantiles[..., 1][future] <= quantiles[..., 2][future]
        ))
        prediction[future].sum().backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum().item())
            for parameter in head.parameters()
            if parameter.grad is not None
        ), 0.0)

    def test_batched_environment_graph_exposes_predictions_to_resource_actor(self):
        env = _make_env(True)
        try:
            _commit_plane_to_future_mobile_job(env)
            observation = env._get_obs()
            info = env._get_info()
            root = Path(__file__).resolve().parents[4]
            with (root / "onpolicy/config/ac.yaml").open(
                "r", encoding="utf-8"
            ) as stream:
                ac_config = yaml.safe_load(stream)
            policy = GNN_Actor_Critic(
                **ac_config,
                max_plane_agents=env.n_plane_agents,
                max_device_agents=env.max_device_num,
                request_ready_prediction=True,
                request_ready_time_scale=3600.0,
            )
            batch_size = 2
            data = {
                "graph": Batch.from_data_list([
                    copy.deepcopy(observation) for _ in range(batch_size)
                ]),
                "hidden_states": torch.zeros(
                    batch_size, env.n_agents, 1, 64
                ),
            }
            policy_info = {
                "active_agents": torch.as_tensor(
                    np.repeat(
                        info["active_agents"][None, :], batch_size, axis=0
                    ),
                    dtype=torch.bool,
                ),
                "last_op_indices": torch.as_tensor(
                    np.repeat(
                        info["last_op_indices"][None, :], batch_size, axis=0
                    ),
                    dtype=torch.long,
                ),
                "last_site_indices": torch.as_tensor(
                    np.repeat(
                        info["last_site_indices"][None, :], batch_size, axis=0
                    ),
                    dtype=torch.long,
                ),
            }
            with torch.no_grad():
                _, actions, _, _ = policy(
                    data, policy_info, deterministic=True
                )
                _, _, _, components = policy(
                    data,
                    policy_info,
                    chosen_op=actions[..., 0],
                    chosen_site=actions[..., 1],
                    eval_action=True,
                    return_decision_mask=True,
                    return_log_prob_components=True,
                )
            predictions = components["request_ready_prediction_seconds"]
            dag_bounds = components["request_ready_dag_seconds"]
            valid = components["request_ready_prediction_valid"]
            self.assertEqual(
                tuple(predictions.shape),
                (batch_size, env.max_request_num),
            )
            self.assertTrue(bool(valid.any().item()))
            self.assertTrue(torch.all(predictions[valid] >= dag_bounds[valid]))
            self.assertEqual(
                tuple(observation.request_ready_context_values.shape),
                (env.max_request_num, 6),
            )
            self.assertEqual(
                int(observation.request_kind_ids.numel()),
                env.max_request_num,
            )
        finally:
            env.close()

    def test_inactive_agents_carry_recurrent_selection_history(self):
        """Sparse global events must not erase an agent's last decision."""

        env = _make_env(True)
        try:
            observation = env._get_obs()
            root = Path(__file__).resolve().parents[4]
            with (root / "onpolicy/config/ac.yaml").open(
                "r", encoding="utf-8"
            ) as stream:
                ac_config = yaml.safe_load(stream)
            policy = GNN_Actor_Critic(
                **ac_config,
                max_plane_agents=env.n_plane_agents,
                max_device_agents=env.max_device_num,
                request_ready_prediction=True,
                request_ready_time_scale=3600.0,
            )
            policy.eval()

            batch_size = 2
            last_op = -torch.ones(
                batch_size, env.n_agents, dtype=torch.long
            )
            last_site = -torch.ones_like(last_op)
            last_op[:, 0] = 3
            last_site[:, 0] = 2
            last_op[:, -1] = 1
            last_site[:, -1] = 0
            data = {
                "graph": Batch.from_data_list([
                    copy.deepcopy(observation) for _ in range(batch_size)
                ]),
                "hidden_states": torch.zeros(
                    batch_size, env.n_agents, 1, 64
                ),
            }
            policy_info = {
                "active_agents": torch.zeros(
                    batch_size, env.n_agents, dtype=torch.bool
                ),
                "last_op_indices": last_op,
                "last_site_indices": last_site,
            }

            with torch.no_grad():
                actions, _ = policy(
                    data,
                    policy_info,
                    deterministic=True,
                    criticize=False,
                )

            torch.testing.assert_close(actions[..., 0], last_op)
            torch.testing.assert_close(actions[..., 1], last_site)
        finally:
            env.close()


class IntrinsicReadyLabelContractTest(unittest.TestCase):
    def setUp(self):
        self.predictions = torch.zeros((1, 3), dtype=torch.float32)
        self.dag_bounds = torch.tensor(
            [[0.0, 5.0, 20.0]], dtype=torch.float32
        )
        self.valid = torch.tensor([[False, True, True]])

    @staticmethod
    def _record(**overrides):
        record = {
            "request_id": 1,
            "intrinsic_ready_time": 30.0,
            "observation_time": 10.0,
            "ready_lead_seconds": 20.0,
            "label_schema_version": 1,
            "target_semantics": STAGE2_SUPERVISION_CONTRACT[
                "ready_target_semantics"
            ],
        }
        record.update(overrides)
        return record

    def _parse(self, records):
        metadata = {"request_ready_targets": [records]}
        return HKBZ_Runner._request_ready_supervision_tensors(
            None,
            metadata,
            self.predictions,
            self.dag_bounds,
            self.valid,
        )

    def test_exact_absolute_label_is_converted_to_observation_relative_lead(self):
        targets, mask = self._parse([self._record()])
        self.assertEqual(mask.tolist(), [[False, True, False]])
        self.assertEqual(float(targets[0, 1].item()), 20.0)

    def test_ready_label_context_marks_only_blocking_wait(self):
        metadata = {"request_ready_targets": [[
            self._record(request_id=1, request_kind="blocking_wait"),
            self._record(
                request_id=2,
                intrinsic_ready_time=35.0,
                observation_time=10.0,
                ready_lead_seconds=25.0,
                request_kind="bounded_mobile_frontier_h2",
            ),
        ]]}
        _, mask, blocking = HKBZ_Runner._request_ready_supervision_tensors(
            None,
            metadata,
            self.predictions,
            self.dag_bounds,
            self.valid,
            return_context=True,
        )
        self.assertEqual(mask.tolist(), [[False, True, True]])
        self.assertEqual(blocking.tolist(), [[False, True, False]])

    def test_ready_label_context_exposes_horizon_kind(self):
        metadata = {"request_ready_targets": [[
            self._record(
                request_id=1,
                request_kind="bounded_mobile_frontier_h1",
            ),
            self._record(
                request_id=2,
                intrinsic_ready_time=35.0,
                observation_time=10.0,
                ready_lead_seconds=25.0,
                request_kind="bounded_mobile_frontier_h2",
            ),
        ]]}
        _, mask, _, kinds = HKBZ_Runner._request_ready_supervision_tensors(
            None,
            metadata,
            self.predictions,
            self.dag_bounds,
            self.valid,
            return_context=True,
            return_kind=True,
        )
        self.assertEqual(mask.tolist(), [[False, True, True]])
        self.assertEqual(
            kinds.tolist(),
            [[
                HKBZ_Runner.REQUEST_READY_KIND_OTHER,
                HKBZ_Runner.REQUEST_READY_KIND_H1,
                HKBZ_Runner.REQUEST_READY_KIND_H2,
            ]],
        )

    def test_ready_epoch_metrics_are_label_weighted(self):
        predictions = torch.tensor([[0.0, 100.0, 700.0]])
        targets = torch.tensor([[0.0, 200.0, 400.0]])
        mask = torch.tensor([[False, True, True]])
        kinds = torch.tensor([[
            HKBZ_Runner.REQUEST_READY_KIND_OTHER,
            HKBZ_Runner.REQUEST_READY_KIND_H1,
            HKBZ_Runner.REQUEST_READY_KIND_H2,
        ]])
        sums = HKBZ_Runner._request_ready_metric_sums(
            predictions, targets, mask, kinds
        )
        HKBZ_Runner._finalize_request_ready_epoch_metrics(sums)
        self.assertEqual(sums["request_ready_all_count"], 2)
        self.assertEqual(sums["request_ready_all_mae_seconds"], 200.0)
        self.assertEqual(sums["request_ready_h1_mae_seconds"], 100.0)
        self.assertEqual(sums["request_ready_h2_mae_seconds"], 300.0)
        self.assertEqual(
            sums["request_ready_all_underprediction_rate"], 0.5
        )
        self.assertEqual(
            sums["request_ready_nonblocking_mae_seconds"], 200.0
        )
        self.assertEqual(sums["request_ready_all_late_120_rate"], 0.5)
        self.assertEqual(sums["request_ready_all_late_300_rate"], 0.0)

    def test_blocking_is_removed_from_nonblocking_metrics(self):
        predictions = torch.tensor([[0.0, 150.0, 900.0]])
        targets = torch.tensor([[0.0, 0.0, 300.0]])
        mask = torch.tensor([[False, True, True]])
        kinds = torch.tensor([[
            HKBZ_Runner.REQUEST_READY_KIND_OTHER,
            HKBZ_Runner.REQUEST_READY_KIND_BLOCKING,
            HKBZ_Runner.REQUEST_READY_KIND_H2,
        ]])
        sums = HKBZ_Runner._request_ready_metric_sums(
            predictions, targets, mask, kinds
        )
        HKBZ_Runner._finalize_request_ready_epoch_metrics(sums)
        self.assertEqual(sums["request_ready_all_mae_seconds"], 375.0)
        self.assertEqual(
            sums["request_ready_nonblocking_mae_seconds"], 600.0
        )
        self.assertEqual(
            sums["request_ready_nonblocking_late_300_rate"], 1.0
        )

    def test_case_macro_metrics_do_not_weight_long_cases_more(self):
        predictions = torch.tensor([
            [0.0, 100.0, 100.0],
            [0.0, 1000.0, 0.0],
        ])
        targets = torch.zeros_like(predictions)
        mask = torch.tensor([
            [False, True, True],
            [False, True, False],
        ])
        blocking = torch.zeros_like(mask)
        case_stats = HKBZ_Runner._request_ready_case_metric_sums(
            predictions,
            targets,
            mask,
            blocking,
            ["case_a", "case_b"],
        )
        metrics = {}
        HKBZ_Runner._finalize_request_ready_case_metrics(
            metrics, case_stats
        )
        self.assertEqual(metrics["request_ready_case_count"], 2)
        self.assertEqual(
            metrics["request_ready_case_macro_mae_seconds"], 550.0
        )

    def test_case_holdout_fold_is_stable(self):
        first = HKBZ_Runner._request_ready_case_fold(
            "/tmp/data/case_0001", 5
        )
        second = HKBZ_Runner._request_ready_case_fold("case_0001", 5)
        self.assertEqual(first, second)
        self.assertIn(first, range(5))

    def test_ready_only_scope_exposes_only_regression_head(self):
        runner = object.__new__(HKBZ_Runner)
        runner.device_bc_training_scope = "ready_only"
        head = torch.nn.Linear(4, 1)
        runner.policy = SimpleNamespace(ac=SimpleNamespace(
            request_ready_prediction=True,
            request_ready_head=head,
        ))
        self.assertEqual(runner._device_bc_trainable_modules(), [head])

    def test_old_scalar_or_lead_only_labels_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "absolute intrinsic_ready_time"):
            self._parse({1: 20.0})
        with self.assertRaisesRegex(RuntimeError, "require request_id"):
            self._parse([{
                "request_id": 1,
                "ready_lead_seconds": 20.0,
            }])

    def test_schema_mismatch_and_dag_violation_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "schema/semantics mismatch"):
            self._parse([self._record(label_schema_version=0)])
        with self.assertRaisesRegex(RuntimeError, "dependency lower bound"):
            self._parse([self._record(
                request_id=2,
                intrinsic_ready_time=25.0,
                observation_time=10.0,
                ready_lead_seconds=15.0,
            )])

    def test_float32_dag_rounding_does_not_reject_an_equal_label(self):
        target = 394.3333333333333
        self.dag_bounds[0, 1] = target

        targets, mask = self._parse([self._record(
            intrinsic_ready_time=target,
            observation_time=0.0,
            ready_lead_seconds=target,
        )])

        self.assertTrue(mask[0, 1].item())
        self.assertEqual(
            float(targets[0, 1].item()),
            float(self.dag_bounds[0, 1].item()),
        )

    def test_float32_tolerance_still_rejects_a_real_dag_violation(self):
        lower_bound = 394.3333333333333
        self.dag_bounds[0, 1] = lower_bound
        target = lower_bound - 0.01

        with self.assertRaisesRegex(
            RuntimeError,
            "dependency lower bound",
        ):
            self._parse([self._record(
                intrinsic_ready_time=target,
                observation_time=0.0,
                ready_lead_seconds=target,
            )])

    def test_teacher_index_requires_explicit_schema_and_semantics(self):
        labels = [{
            "observation_time": 12.0,
            "intrinsic_ready_time": 42.0,
            "dag_lower_bound_seconds": 20.0,
            "request_kind": "bounded_mobile_frontier_h1",
            "plane_id": "P1",
            "job_code": "J1",
            "site_code": "S1",
        }]
        with self.assertRaisesRegex(ValueError, "schema version"):
            AircraftScheduleEnv._intrinsic_ready_time_label_index({
                "intrinsic_ready_time_labels": labels,
            })

        teacher = {
            "intrinsic_ready_time_label_schema_version": 1,
            "intrinsic_ready_time_semantics": (
                AircraftScheduleEnv.INTRINSIC_READY_TIME_SEMANTICS
            ),
            "intrinsic_ready_time_labels": labels,
        }
        index = AircraftScheduleEnv._intrinsic_ready_time_label_index(teacher)
        key = AircraftScheduleEnv._intrinsic_ready_time_label_key(labels[0])
        self.assertEqual(index[key], 42.0)

    def test_near_equal_event_times_do_not_alias_different_dag_frontiers(self):
        teacher = {
            "intrinsic_ready_time_label_schema_version": 1,
            "intrinsic_ready_time_semantics": (
                AircraftScheduleEnv.INTRINSIC_READY_TIME_SEMANTICS
            ),
            "intrinsic_ready_time_labels": [{
                "observation_time": 3496.9999999999986,
                "intrinsic_ready_time": 3796.9999999999977,
                "dag_lower_bound_seconds": 0.0,
                "request_kind": "bounded_mobile_frontier_h1",
                "plane_id": "Plane_0_19",
                "job_code": "ZY07",
                "site_code": "34",
            }],
        }
        env = object.__new__(AircraftScheduleEnv)
        env.total_time = 3496.9999999999973
        env.request_list = [{"id": 0}, {
            "id": 14,
            "plane_id": "Plane_0_19",
            "job_code": "ZY07",
            "site_code": "34",
            "lead_time": 420.0,
            "request_kind": "bounded_mobile_frontier_h2",
        }]
        index = AircraftScheduleEnv._intrinsic_ready_time_label_index(teacher)

        targets = env._request_ready_supervision_targets(
            "iga", {"intrinsic_ready_time_index": index}
        )

        self.assertEqual(targets, [])

        env.request_list[1].update({
            "lead_time": 0.0,
            "request_kind": "bounded_mobile_frontier_h1",
        })
        targets = env._request_ready_supervision_targets(
            "iga", {"intrinsic_ready_time_index": index}
        )
        self.assertEqual(len(targets), 1)
        self.assertAlmostEqual(targets[0]["ready_lead_seconds"], 300.0)

    def test_blocking_request_without_live_lead_time_matches_zero_dag_label(self):
        label = {
            "observation_time": 4000.0,
            "intrinsic_ready_time": 3990.0,
            "dag_lower_bound_seconds": 0.0,
            "request_kind": "blocking_wait",
            "plane_id": "Plane_0_1",
            "job_code": "ZY03",
            "site_code": "12",
        }
        teacher = {
            "intrinsic_ready_time_label_schema_version": 1,
            "intrinsic_ready_time_semantics": (
                AircraftScheduleEnv.INTRINSIC_READY_TIME_SEMANTICS
            ),
            "intrinsic_ready_time_labels": [label],
        }
        env = object.__new__(AircraftScheduleEnv)
        env.total_time = 4000.0
        env.request_list = [{"id": 0}, {
            "id": 1,
            "plane_id": "Plane_0_1",
            "job_code": "ZY03",
            "site_code": "12",
            "waiting_time": 10.0,
            "request_kind": "blocking_wait",
        }]

        targets = env._request_ready_supervision_targets(
            "iga",
            {
                "intrinsic_ready_time_index": (
                    AircraftScheduleEnv._intrinsic_ready_time_label_index(
                        teacher
                    )
                )
            },
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["ready_lead_seconds"], 0.0)
        self.assertEqual(targets[0]["dag_lower_bound_seconds"], 0.0)

    def test_environment_accepts_verified_schema2_teacher_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            path.write_text(json.dumps({
                "schema_version": 2,
                "teacher_scope": "stage2_resource_policy",
                "teacher_method": "resource_iga_all",
                "environment_semantics_version": (
                    AircraftScheduleEnv.SEMANTICS_VERSION
                ),
                "case_count": 1,
                "entries": {"case_0001": {}},
                "intrinsic_ready_time_label_schema_version": 1,
                "intrinsic_ready_time_semantics": (
                    AircraftScheduleEnv.INTRINSIC_READY_TIME_SEMANTICS
                ),
                "intrinsic_ready_time_observed_request_count": 100,
                "intrinsic_ready_time_labeled_request_count": 99,
                "intrinsic_ready_time_label_coverage": 0.99,
            }), encoding="utf-8")
            env = object.__new__(AircraftScheduleEnv)
            observed = env._load_resource_iga_teacher_index(path)
            self.assertEqual(observed["schema_version"], 2)

    def test_completed_replay_backfills_only_factual_exact_targets(self):
        env = object.__new__(AircraftScheduleEnv)
        env.intrinsic_ready_time_occurrences = {
            ("P1", "J1", "S1"): [35.0],
        }
        trace = [{
            "time_before": 10.0,
            "requests": [
                {
                    "id": 1,
                    "plane_id": "P1",
                    "job_code": "J1",
                    "site_code": "S1",
                    "lead_time": 20.0,
                    "waiting_time": 0.0,
                    "is_lookahead": True,
                    "is_noop": False,
                    "request_kind": "bounded_mobile_frontier",
                },
                {
                    "id": 2,
                    "plane_id": "P2",
                    "job_code": "J2",
                    "site_code": "S2",
                    "lead_time": 0.0,
                    "waiting_time": 7.0,
                    "is_lookahead": False,
                    "is_noop": False,
                    "request_kind": "blocking_wait",
                },
                {
                    "id": 3,
                    "plane_id": "P3",
                    "job_code": "J3",
                    "site_code": "S3",
                    "lead_time": 5.0,
                    "waiting_time": 0.0,
                    "is_lookahead": True,
                    "is_noop": False,
                    "request_kind": "speculative_branch",
                },
            ],
        }]

        annotated, labels, coverage = (
            env.attach_intrinsic_ready_time_labels(trace)
        )

        self.assertEqual(len(labels), 2)
        self.assertEqual(
            annotated[0]["requests"][0]["intrinsic_ready_time"], 35.0
        )
        self.assertEqual(
            annotated[0]["requests"][1]["intrinsic_ready_time"], 3.0
        )
        self.assertNotIn(
            "intrinsic_ready_time", annotated[0]["requests"][2]
        )
        self.assertEqual(coverage["observed_request_count"], 3)
        self.assertEqual(coverage["labeled_request_count"], 2)


class FinalMatchingSupervisionTest(unittest.TestCase):
    @staticmethod
    def _runner():
        runner = object.__new__(HKBZ_Runner)
        runner.device = torch.device("cpu")
        runner.device_bc_assignment_margin = 0.2
        runner.policy = SimpleNamespace(ac=SimpleNamespace(
            max_plane_agents=1,
            AGENT_TYPE_DEVICE=1,
            AGENT_TYPE_TRANSPORTER=2,
        ))
        return runner

    def test_loss_is_invariant_to_equivalent_device_permutation(self):
        runner = self._runner()
        graph = SimpleNamespace(device_type_ids=torch.tensor([3, 3, 3]))
        raw = torch.tensor([[
            [1.0, -4.0, -4.0],
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
            [4.0, 0.0, 0.0],
        ]], requires_grad=True)
        logits = torch.log_softmax(raw, dim=-1)
        active = np.array([[False, True, True, True]])
        types = np.array([[0, 1, 1, 1]])
        labels = np.zeros((1, 4, 2), dtype=np.int64)
        labels[0, 1, 0] = 1
        labels[0, 2, 0] = 2
        first = runner._device_assignment_supervision(
            logits, active, labels, [graph], types
        )

        permutation = torch.tensor([0, 3, 1, 2])
        permuted_logits = logits[:, permutation].detach().requires_grad_(True)
        permuted_labels = labels[:, permutation.numpy()].copy()
        second = runner._device_assignment_supervision(
            permuted_logits, active, permuted_labels, [graph], types
        )
        torch.testing.assert_close(first["loss"], second["loss"])
        torch.testing.assert_close(first["margin_loss"], second["margin_loss"])
        self.assertEqual(first["labels"], 1)
        self.assertEqual(first["exact"], 1.0)
        self.assertEqual(first["f1"], 1.0)
        (first["loss"] + first["margin_loss"]).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_noop_causes_survive_teacher_merge(self):
        runner = self._runner()
        runner.num_agents = 4
        policy_actions = np.zeros((1, 4, 2), dtype=np.int64)
        result = {
            "actions": policy_actions[0].copy(),
            "info": {
                "capacity_unmatched": 1,
                "decisions": [{
                    "agent_id": 3,
                    "selected_request_id": 0,
                    "noop_cause": "capacity_unmatched",
                }],
            },
        }
        _, _, stats, metadata = runner._merge_device_bc_actions(
            policy_actions, [result], return_metadata=True
        )
        self.assertEqual(stats["capacity_unmatched"], 1)
        self.assertEqual(
            metadata["teacher_noop_causes"][0, 3],
            "capacity_unmatched",
        )

    def test_legacy_timing_mode_reproduces_noop_conflation(self):
        lookahead = np.asarray([False, False, True])
        self.assertEqual(
            HKBZ_Runner._device_bc_timing_class(
                0, lookahead, "capacity_unmatched", False
            ),
            -1,
        )
        self.assertEqual(
            HKBZ_Runner._device_bc_timing_class(
                0, lookahead, "capacity_unmatched", True
            ),
            2,
        )
        self.assertEqual(
            HKBZ_Runner._device_bc_timing_class(
                0, lookahead, "temporal_defer", False
            ),
            2,
        )
        self.assertEqual(
            HKBZ_Runner._device_bc_timing_class(
                2, lookahead, "", False
            ),
            1,
        )
        self.assertEqual(
            HKBZ_Runner._device_bc_timing_class(
                1, lookahead, "", False
            ),
            0,
        )


class PolicyHistoryContractTest(unittest.TestCase):
    def test_plane_history_comes_from_environment_and_device_history_persists(self):
        previous = np.asarray([[
            [20, 30],
            [-1, -1],
            [7, 0],
        ]], dtype=np.int64)
        infos = {
            "last_op_indices": np.asarray([[2, 4, -1]]),
            "last_site_indices": np.asarray([[8, 9, -1]]),
        }

        history = HKBZ_Runner._authoritative_policy_history(
            infos, previous, plane_count=2
        )

        np.testing.assert_array_equal(
            history,
            np.asarray([[
                [2, 8],
                [4, 9],
                [7, 0],
            ]], dtype=np.int64),
        )

    def test_missing_or_malformed_environment_history_fails_closed(self):
        previous = -np.ones((1, 3, 2), dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "last_site_indices"):
            HKBZ_Runner._authoritative_policy_history(
                {"last_op_indices": np.zeros((1, 3))},
                previous,
                plane_count=2,
            )
        with self.assertRaisesRegex(RuntimeError, "expected"):
            HKBZ_Runner._authoritative_policy_history(
                {
                    "last_op_indices": np.zeros((1, 2)),
                    "last_site_indices": np.zeros((1, 3)),
                },
                previous,
                plane_count=2,
            )


if __name__ == "__main__":
    unittest.main()

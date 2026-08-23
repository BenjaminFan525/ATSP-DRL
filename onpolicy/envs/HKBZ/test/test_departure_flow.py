from pathlib import Path
import math
import unittest

import numpy as np

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = (
    ROOT
    / "onpolicy/envs/HKBZ/dataset/"
    / "fjsp_v3_resource_joint_eval_s20260811/joint/tune/case_0001"
)


def _make_env(resource_policy="heuristic", **overrides):
    config = {
        "jobs_path": str(CASE_DIR / "job.json"),
        "fixed_res_path": str(CASE_DIR / "fixed_resources.json"),
        "mobile_res_path": str(CASE_DIR / "mobile_resources.json"),
        "sites_path": str(CASE_DIR / "sites.json"),
        "flights_path": str(CASE_DIR / "flights.json"),
        "seed": 42,
        "use_domain_rand": False,
        "resource_policy": resource_policy,
        "n_agents": 24,
        "max_device_num": 80,
    }
    config.update(overrides)
    env = AircraftScheduleEnv(config)
    env.reset()
    return env


def _move_plane(plane, destination):
    if plane.site.plane == plane:
        plane.site.remove_plane()
    plane.site = destination
    destination.add_plane(plane)


def _add_plane(env, pid, site_code):
    env.add_planes([{
        "batch": 0,
        "idx": pid,
        "velocity": 5,
        "site": env.sites[site_code],
        "fuel": 30,
        "jobs": env._build_plane_jobs(drop_optional=False),
    }])
    return env.planes[f"Plane_0_{pid}"]


def _complete_service(env, plane):
    service = [
        code for code in env.service_job_code_list if code in plane.jobs
    ]
    plane.finished_jobs = list(service)
    plane.ever_finished_jobs = set(service)
    plane.left_jobs = list(env.departure_job_code_list)
    plane.current_jobs = []
    plane.is_busy = False
    plane.is_transporting = False
    plane.is_waiting = False
    plane.choosed_job = None
    plane.destination = None
    plane.pending_job = None
    plane.departure_staging_decided = False


def _leave_only_final_service_job(env, plane):
    final_service = env.service_job_code_list[-1]
    completed = [
        code for code in env.service_job_code_list if code != final_service
    ]
    plane.finished_jobs = completed
    plane.ever_finished_jobs = set(completed)
    plane.left_jobs = [final_service] + list(env.departure_job_code_list)
    plane.current_jobs = []
    plane.is_busy = False
    plane.is_transporting = False
    plane.is_waiting = False
    plane.choosed_job = None
    plane.destination = None
    plane.pending_job = None


def _prepare_planes(env, count, completed=True):
    first = env.planes["Plane_0_0"]
    _move_plane(first, env.sites["1"])
    planes = [first]
    for pid in range(1, count):
        planes.append(_add_plane(env, pid, str(pid + 1)))
    env.flights_data = env.flights_data[:count]
    env.landing_list = []
    env.departure_barrier_open = False
    env.departure_ready_since = {}
    env.departure_transporter_by_plane = {}
    env.departure_plane_by_transporter = {}
    env._departure_runway_plan = {}
    env.departed_agent_ids = set()
    if completed:
        for plane in planes:
            _complete_service(env, plane)
    return planes


def _actions_for_active_planes(env, info):
    actions = np.full((env.n_agents, 2), -1, dtype=np.int64)
    n_jobs = len(env.job_code_list)
    claimed_sites = set()
    priority = {
        env.departure_job_code_list[-1]: 0,
        env.departure_job_code_list[0]: 1,
        env.TRANSFER_JOB_CODE: 2,
    }
    for pid in range(env.n_plane_agents):
        if not info["active_agents"][pid]:
            continue
        own_ops = env.agent_op_mask[
            pid, pid * n_jobs:(pid + 1) * n_jobs
        ]
        pairs = np.argwhere(
            own_ops[:, None] & env.agent_job_site_mask_matrix[pid]
        )
        candidates = []
        plane = next(
            plane for plane in env.planes.values()
            if int(plane.code.split("_")[-1]) == pid
        )
        current_idx = env.site_code_list.index(plane.site.code)
        for job_idx, site_idx in pairs:
            site_idx = int(site_idx)
            if site_idx in claimed_sites:
                continue
            job_code = env.job_code_list[int(job_idx)]
            hold_rank = 0 if (
                job_code == env.TRANSFER_JOB_CODE
                and site_idx == current_idx
            ) else 1
            candidates.append((
                priority.get(job_code, 3),
                hold_rank,
                int(job_idx),
                site_idx,
            ))
        if not candidates:
            raise AssertionError(f"No legal pair for active plane {pid}")
        _, _, job_idx, site_idx = min(candidates)
        actions[pid] = [pid * n_jobs + job_idx, site_idx]
        claimed_sites.add(site_idx)
    return actions


class DepartureFlowTest(unittest.TestCase):
    def test_runtime_contract_keeps_site_width_and_adds_existing_jobs(self):
        env = _make_env()
        observation = env._get_obs()
        self.assertEqual(observation["site"].x.shape[1], 22)
        self.assertEqual(len(env.service_job_code_list), 17)
        self.assertEqual(
            env.job_code_list[-3:], ["ZY-T", "ZY-S", "ZY-F"]
        )
        plane = next(iter(env.planes.values()))
        self.assertNotIn("ZY-T", plane.left_jobs)
        self.assertEqual(plane.left_jobs[-2:], ["ZY-S", "ZY-F"])
        self.assertEqual(
            set(plane.jobs["ZY-S"].predecessor),
            set(env.service_job_code_list),
        )
        self.assertEqual(set(plane.jobs["ZY-F"].predecessor), {"ZY-S"})
        self.assertEqual(env.takeoff_site_code_list, ["41", "42", "43"])

    def test_departure_observation_and_potential_expose_ready_queue(self):
        env = _make_env()
        try:
            plane = _prepare_planes(env, 1)[0]
            env._sync_departure_pipeline_state()
            observation = env._get_obs()
            features = env.get_iga_potential_features()
            pid = int(plane.code.split('_')[-1])

            self.assertEqual(
                observation.job_phase_codes.tolist()[-3:], [1, 2, 2]
            )
            self.assertEqual(int(observation.agent_departure_phase[pid]), 1)
            self.assertEqual(
                float(observation.agent_service_progress[pid]), 1.0
            )
            self.assertEqual(features['remaining_service_jobs'], 0.0)
            self.assertGreater(features['remaining_departure_jobs'], 0.0)
            self.assertEqual(features['departure_ready_count'], 1.0)
            self.assertGreaterEqual(features['departure_tow_work'], 60.0)
            self.assertTrue(all(
                math.isfinite(value) and value >= 0.0
                for value in features.values()
            ))
        finally:
            env.close()

    def test_departure_is_per_plane_not_global(self):
        env = _make_env()
        first, second = _prepare_planes(env, 2, completed=False)
        _complete_service(env, first)
        _leave_only_final_service_job(env, second)
        second.jobs[env.service_job_code_list[-1]].time = 10000.0
        env.flights_data.append(dict(env.flights_data[-1]))
        env.landing_list = [(20000.0, 0, 2, 30)]

        observation = env._get_obs()
        self.assertTrue(env._departure_barrier_is_open())
        self.assertEqual(env._ready_job_codes(first), ["ZY-T"])
        pid = 0
        n_jobs = len(env.job_code_list)
        ready_indices = np.flatnonzero(
            env.agent_op_mask[pid, pid * n_jobs:(pid + 1) * n_jobs]
        )
        self.assertEqual(
            [env.job_code_list[idx] for idx in ready_indices], ["ZY-T"]
        )
        transfer_idx = env.job_code_list.index("ZY-T")
        legal_sites = {
            env.site_code_list[idx]
            for idx in np.flatnonzero(
                env.agent_job_site_mask_matrix[pid, transfer_idx]
            )
        }
        self.assertIn(first.site.code, legal_sites)
        self.assertTrue(legal_sites.isdisjoint(env.runway_code_list))
        self.assertFalse(bool(observation.op_mask[
            pid, pid * n_jobs + env.job_code_list.index("ZY-S")
        ]))

        info = env._get_info()
        for _ in range(8):
            actions = _actions_for_active_planes(env, info)
            _, _, _, info = env.step(actions)
            if 0 in env.departed_agent_ids:
                break

        self.assertIn(0, env.departed_agent_ids)
        first_departure_time = env.departure_log[0]["time"]
        second_service = next(
            record for record in env.trajectory_log
            if record.get("plane_id") == second.code
            and record.get("target_job_code")
            == env.service_job_code_list[-1]
        )
        self.assertLess(first_departure_time, second_service["end_time"])
        self.assertGreater(env.landing_list[0][0], first_departure_time)
        first_departures = [
            record for record in env.trajectory_log
            if record.get("plane_id") == first.code
            and record.get("action_phase") == "departure"
        ]
        self.assertTrue(first_departures)
        self.assertTrue(all(
            record.get("departure_eligible")
            for record in first_departures
        ))

    def test_departure_pickup_does_not_claim_runway_early(self):
        env = _make_env()
        plane = _prepare_planes(env, 1, completed=True)[0]
        plane.mark_departure_staging_decided()

        remote_site = env.sites["40"]
        for device in env.mobile_devices["R014"]:
            if device.site != remote_site:
                device.start_transport(remote_site)
                device.finish_transport()

        env._refresh_request_pool()
        pickup = next(
            request for request in env.request_list
            if request.get("request_kind") == "departure_pickup"
        )
        transporter = env.mobile_devices["R014"][0]
        self.assertTrue(env._start_device_request(
            transporter, pickup, pickup["id"]
        ))
        self.assertTrue(transporter.is_transporting)
        self.assertTrue(all(
            not env.sites[code].is_occupied
            for code in env.takeoff_site_code_list
        ))
        self.assertEqual(env._compute_departure_runway_plan(), {})

        transporter.update(transporter.left_trans_time)
        observation = env._get_obs()
        assigned = env._departure_runway_plan[plane.code]
        departure_idx = env.job_code_list.index("ZY-S")
        legal_runways = {
            env.site_code_list[index]
            for index in np.flatnonzero(
                env.agent_job_site_mask_matrix[0, departure_idx]
            )
        }
        self.assertEqual(legal_runways, {assigned})
        self.assertTrue(bool(observation.op_mask[0, departure_idx]))
        self.assertTrue(all(
            not env.sites[code].is_occupied
            for code in env.takeoff_site_code_list
        ))

    def test_runway_wave_uses_dynamic_distance_matching_not_plane_id(self):
        env = _make_env()
        planes = _prepare_planes(env, 4, completed=True)
        target_sites = ["1", "2", "3", "40"]
        transporters = env.mobile_devices["R014"][:4]
        for plane, site_code, transporter in zip(
            planes, target_sites, transporters
        ):
            _move_plane(plane, env.sites[site_code])
            plane.mark_departure_staging_decided()
            if transporter.site != plane.site:
                transporter.start_transport(plane.site)
                transporter.finish_transport()
            self.assertTrue(env._reserve_departure_transporter(
                plane, transporter
            ))

        plan = env._compute_departure_runway_plan()
        self.assertEqual(len(plan), 3)
        self.assertIn("Plane_0_3", plan)
        self.assertNotEqual(
            set(plan), {"Plane_0_0", "Plane_0_1", "Plane_0_2"}
        )

    def test_r014_uses_transporter_velocity(self):
        env = _make_env()
        self.assertTrue(env.mobile_devices["R014"])
        self.assertTrue(all(
            device.velocity == 5
            for device in env.mobile_devices["R014"]
        ))

    def test_drl_r014_dispatch_wakes_plane_without_deadlock(self):
        env = _make_env(resource_policy="drl")
        plane = _prepare_planes(env, 1, completed=True)[0]
        plane.mark_departure_staging_decided()
        env._get_obs()
        info = env._get_info()

        pickup = next(
            request for request in env.request_list
            if request.get("request_kind") == "departure_pickup"
        )
        transporter = next(
            device for device in env.device_list[:env.max_device_num]
            if env._is_transporter_device(device)
            and env._device_can_dispatch(device, pickup)
        )
        dev_idx = env.device_list.index(transporter)
        agent_id = env.n_plane_agents + dev_idx
        actions = np.zeros((env.n_agents, 2), dtype=np.int64)
        actions[:, 0] = -1
        actions[env.n_plane_agents:, 0] = 0
        actions[agent_id] = [pickup["id"], 0]

        _, _, _, info = env.step(actions)
        self.assertFalse(env.cycle_terminated, env.cycle_reason)
        self.assertTrue(info["active_agents"][0])
        self.assertIsNotNone(env._departure_transporter_for_plane(
            plane, ready_only=True
        ))
        self.assertIn(plane.code, env._departure_runway_plan)

    def test_staging_promotes_local_lookahead_r014_before_event_deadlock(self):
        """A zero-time lease promotion must wake the ZY-S plane decision.

        This is the exact ordering that failed in the Wave-4 natural BC
        rollout: the observation offers ZY-T while a pre-positioned R014 has
        an unpromoted lease; executing the hold/staging action makes the lease
        promotable without creating any future timed event.
        """
        for reservation_mode in ("soft", "hard"):
            with self.subTest(reservation_mode=reservation_mode):
                env = _make_env(
                    resource_policy="drl",
                    device_lookahead_dispatch=True,
                    device_future_intent_horizon=1,
                    device_departure_lookahead=True,
                    device_lookahead_reservation_mode=reservation_mode,
                    device_reservation_grace_seconds=300.0,
                )
                try:
                    plane = _prepare_planes(env, 1, completed=True)[0]
                    transporter = env.mobile_devices["R014"][0]
                    if transporter.site != plane.site:
                        transporter.start_transport(plane.site)
                        transporter.finish_transport()
                    transporter.lookahead_reservation = {
                        "identity": [
                            plane.code,
                            env.TRANSFER_JOB_CODE,
                            plane.site.code,
                        ],
                        "plane_id": plane.code,
                        "job_code": env.TRANSFER_JOB_CODE,
                        "site_code": plane.site.code,
                        "mode": reservation_mode,
                        "created_time": float(env.total_time),
                        "needed_time": float(env.total_time),
                        "expires_at": float(env.total_time + 300.0),
                    }

                    env._get_obs()
                    info = env._get_info()
                    self.assertFalse(plane.departure_staging_decided)
                    self.assertNotIn(
                        plane.code, env.departure_transporter_by_plane
                    )

                    actions = np.zeros((env.n_agents, 2), dtype=np.int64)
                    transfer_idx = env.job_code_list.index(
                        env.TRANSFER_JOB_CODE
                    )
                    current_site_idx = env.site_code_list.index(
                        plane.site.code
                    )
                    actions[0] = [transfer_idx, current_site_idx]

                    _, _, dones, next_info = env.step(actions)

                    self.assertFalse(bool(np.all(dones)))
                    self.assertTrue(plane.departure_staging_decided)
                    self.assertIsNone(transporter.lookahead_reservation)
                    self.assertEqual(
                        env.departure_transporter_by_plane[plane.code],
                        transporter.code,
                    )
                    self.assertTrue(bool(next_info["active_agents"][0]))
                    self.assertIn(plane.code, env._departure_runway_plan)
                finally:
                    env.close()

    def test_hold_action_fast_forwards_to_future_arrival_without_zero_time_loop(self):
        env = _make_env()
        plane = next(iter(env.planes.values()))
        _move_plane(plane, env.sites["1"])
        _complete_service(env, plane)
        env.flights_data = env.flights_data[:2]
        env.landing_list = [(60.0, 0, 1, 30)]
        env.departure_barrier_open = False

        observation = env._get_obs()
        info = env._get_info()
        actions = _actions_for_active_planes(env, info)
        _, _, _, next_info = env.step(actions)

        self.assertEqual(env.total_time, 60.0)
        self.assertTrue(plane.departure_staging_decided)
        holds = [
            record for record in env.trajectory_log
            if record.get("decision_type") == "hold_at_current_stand"
        ]
        self.assertEqual(len(holds), 1)
        self.assertEqual(holds[0]["start_time"], holds[0]["end_time"])
        self.assertIn("Plane_0_1", env.planes)
        self.assertFalse(bool(next_info["cycle_terminated"]))

    def test_takeoff_transport_requires_r014_and_uses_split_duration(self):
        env = _make_env()
        plane = next(iter(env.planes.values()))
        _move_plane(plane, env.sites["1"])
        _complete_service(env, plane)
        runway = env.sites["41"]
        with self.assertRaisesRegex(RuntimeError, "without an R014"):
            plane.start_transport(runway, None, purpose="departure")

        transporter = env.mobile_devices["R014"][0]
        if transporter.site != plane.site:
            transporter.start_transport(plane.site)
            transporter.finish_transport()
        origin = plane.site
        distance = abs(origin.pos[0] - runway.pos[0]) + abs(
            origin.pos[1] - runway.pos[1]
        )
        expected = math.ceil(distance / plane.velocity) + 60
        duration = plane.start_transport(
            runway, transporter, purpose="departure"
        )
        self.assertEqual(duration, expected)
        self.assertEqual(plane.left_trans_time, expected)
        self.assertEqual(transporter.left_trans_time, expected)

    def test_completed_plane_can_relocate_to_free_a_specific_stand(self):
        env = _make_env()
        first, second = _prepare_planes(env, 2, completed=False)
        _complete_service(env, first)
        _leave_only_final_service_job(env, second)
        # Keep the barrier closed while the first plane stages elsewhere.
        env.landing_list = [(20000.0, 0, 2, 30)]
        second.jobs[env.service_job_code_list[-1]].time = 10000.0

        env._get_obs()
        info = env._get_info()
        actions = np.full((env.n_agents, 2), -1, dtype=np.int64)
        n_jobs = len(env.job_code_list)
        transfer_idx = env.job_code_list.index("ZY-T")
        final_service_idx = len(env.service_job_code_list) - 1
        actions[0] = [transfer_idx, env.site_code_list.index("3")]
        actions[1] = [
            n_jobs + final_service_idx,
            env.site_code_list.index(second.site.code),
        ]
        env.step(actions)

        self.assertEqual(first.site.code, "3")
        self.assertFalse(env.sites["1"].is_occupied)
        self.assertTrue(first.departure_staging_decided)
        self.assertTrue(first.has_completed_service_jobs())
        self.assertTrue(
            first.LONG_OCCUPANCY_JOBS.issubset(first.finished_jobs)
        )
        records = [
            record for record in env.trajectory_log
            if record.get("target_job_code") == "ZY-T"
            and record.get("plane_id") == first.code
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_site_code"], "3")
        self.assertTrue(records[0]["device_ids"])
        self.assertFalse(env.cycle_terminated, env.cycle_reason)

    def test_six_planes_depart_in_runway_capacity_waves_without_deadlock(self):
        env = _make_env()
        planes = _prepare_planes(env, 6, completed=True)
        observation = env._get_obs()
        info = env._get_info()
        initial_s_agents = [
            plane.code for plane in planes
            if "ZY-S" in env._ready_job_codes(plane)
        ]
        self.assertEqual(initial_s_agents, [])

        done = False
        steps = 0
        while not done and steps < 100:
            actions = _actions_for_active_planes(env, info)
            observation, _, dones, info = env.step(actions)
            done = bool(np.all(dones))
            steps += 1

        self.assertTrue(done, env._deadlock_snapshot())
        self.assertLess(steps, 100)
        self.assertFalse(env.cycle_terminated, env.cycle_reason)
        self.assertEqual(len(env.planes), 0)
        self.assertEqual(len(env.departure_log), 6)
        self.assertEqual(int(info["total_relocations"]), 6)
        departure_records = [
            record for record in env.trajectory_log
            if record.get("action_phase") == "departure"
        ]
        self.assertEqual(
            sum(record["target_job_code"] == "ZY-S"
                for record in departure_records),
            6,
        )
        self.assertEqual(
            sum(record["target_job_code"] == "ZY-F"
                for record in departure_records),
            6,
        )
        for record in departure_records:
            self.assertIn(record["target_site_code"], {"41", "42", "43"})
            self.assertTrue(record["departure_eligible"])
            self.assertEqual(
                record["departure_mode"], "progressive_per_aircraft"
            )
        tow_records = [
            record for record in departure_records
            if record["target_job_code"] == "ZY-S"
        ]
        self.assertTrue(all(record["device_ids"] for record in tow_records))

    def test_direct_iga_decoder_uses_new_masks_through_departure(self):
        from onpolicy.envs.HKBZ.experiment.valid_iga import GA_Policy

        env = _make_env()
        _prepare_planes(env, 6, completed=True)
        observation = env._get_obs()
        info = env._get_info()
        job_priorities = np.zeros(
            (env.n_agents, len(env.job_code_list)), dtype=np.float64
        )
        site_priorities = np.zeros(
            (env.n_agents, len(env.site_code_list)), dtype=np.float64
        )

        done = False
        steps = 0
        while not done and steps < 100:
            actions = GA_Policy(
                env, info, job_priorities, site_priorities
            )
            observation, _, dones, info = env.step(actions)
            done = bool(np.all(dones))
            steps += 1

        self.assertTrue(done, env._deadlock_snapshot())
        self.assertLess(steps, 100)
        self.assertFalse(env.cycle_terminated, env.cycle_reason)
        self.assertEqual(len(env.departure_log), 6)
        self.assertEqual(
            sum(
                record.get("target_job_code") == "ZY-S"
                for record in env.trajectory_log
            ),
            6,
        )


if __name__ == "__main__":
    unittest.main()

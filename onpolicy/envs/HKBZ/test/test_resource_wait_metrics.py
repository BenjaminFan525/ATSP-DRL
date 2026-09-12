import unittest

from onpolicy.envs.HKBZ.experiment.resource_wait_metrics import (
    summarize_aircraft_resource_wait,
)


class AircraftResourceWaitMetricsTest(unittest.TestCase):
    def test_service_transfer_and_departure_waits_are_non_overlapping(self):
        plane_trajectory = [
            {
                "plane_id": "Plane_0_0",
                "target_job_code": "ZY02",
                "action_phase": "service",
                "origin_site_code": "1",
                "target_site_code": "1",
                "waiting_time": 30,
                "start_time": 10,
                "end_time": 100,
            },
            {
                "plane_id": "Plane_0_0",
                "target_job_code": "ZY-T",
                "action_phase": "post_service_relocation",
                "origin_site_code": "1",
                "target_site_code": "2",
                "waiting_time": 10,
                "start_time": 100,
                "end_time": 150,
            },
            {
                "plane_id": "Plane_0_1",
                "target_job_code": "ZY03",
                "action_phase": "service",
                "origin_site_code": "Z",
                "target_site_code": "3",
                "waiting_time": 0,
                "start_time": 0,
                "end_time": 80,
            },
            {
                "plane_id": "Plane_0_1",
                "target_job_code": "ZY-T",
                "action_phase": "post_service_relocation",
                "origin_site_code": "3",
                "target_site_code": "3",
                "waiting_time": 0,
                "start_time": 170,
                "end_time": 180,
            },
        ]
        resource_trajectory = [
            {
                "plane_id": "Plane_0_1",
                "job_code": "ZY-T",
                "request_kind": "departure_pickup",
                # This field is measured from service completion and includes
                # staging.  The metric must instead use ZY-T end=180.
                "waiting_time_at_dispatch": 100,
                "trans_time": 40,
                "start_time": 200,
            }
        ]
        result = summarize_aircraft_resource_wait(
            plane_trajectory,
            resource_trajectory,
            {"ZY02": ["R008"], "ZY03": ["R002"], "ZY-T": ["R014"]},
            aircraft_count=2,
            include_events=True,
        )

        self.assertEqual(result["total_wait_seconds"], 100.0)
        self.assertEqual(result["mean_wait_seconds_per_aircraft"], 50.0)
        self.assertEqual(result["max_wait_seconds_per_aircraft"], 60.0)
        self.assertEqual(result["aircraft_with_positive_wait_count"], 2)
        self.assertEqual(result["positive_wait_event_count"], 3)
        self.assertEqual(result["resource_wait_opportunity_count"], 5)
        self.assertEqual(
            result["by_phase"]["departure_pickup"]["total_wait_seconds"],
            60.0,
        )
        self.assertEqual(
            result["by_category"]["ordinary"]["total_wait_seconds"],
            30.0,
        )
        self.assertFalse(result["fully_eliminated"])

    def test_landing_site_transfer_is_not_misclassified_as_r014_wait(self):
        result = summarize_aircraft_resource_wait(
            [
                {
                    "plane_id": "Plane_0_0",
                    "target_job_code": "ZY01",
                    "origin_site_code": "Z",
                    "target_site_code": "5",
                    "waiting_time": 25,
                }
            ],
            [],
            {"ZY01": []},
            aircraft_count=1,
        )
        self.assertEqual(result["resource_wait_opportunity_count"], 0)
        self.assertEqual(result["total_wait_seconds"], 0.0)
        self.assertTrue(result["fully_eliminated"])

    def test_departure_prepositioning_before_readiness_is_not_wait(self):
        result = summarize_aircraft_resource_wait(
            [
                {
                    "plane_id": "Plane_0_0",
                    "target_job_code": "ZY-T",
                    "action_phase": "post_service_relocation",
                    "origin_site_code": "3",
                    "target_site_code": "3",
                    "waiting_time": 0,
                    "start_time": 900,
                    "end_time": 1000,
                }
            ],
            [
                {
                    "plane_id": "Plane_0_0",
                    "job_code": "ZY-T",
                    "request_kind": "departure_pickup",
                    # R014 leaves 100 seconds before the aircraft is ready and
                    # arrives 20 seconds afterwards.  Only those 20 seconds
                    # are observable aircraft wait, not the full 120-second
                    # pre-positioning trip.
                    "start_time": 900,
                    "trans_time": 120,
                }
            ],
            {"ZY-T": ["R014"]},
            aircraft_count=1,
            include_events=True,
        )
        self.assertEqual(result["total_wait_seconds"], 20.0)
        event = next(
            item for item in result["events"]
            if item["source"] == "departure_pickup"
        )
        self.assertEqual(event["waiting_before_dispatch_seconds"], 0.0)
        self.assertEqual(event["travel_after_dispatch_seconds"], 20.0)

    def test_non_resource_wait_is_excluded_at_the_same_site(self):
        result = summarize_aircraft_resource_wait(
            [
                {
                    "plane_id": "Plane_0_0",
                    "target_job_code": "ZY14",
                    "origin_site_code": "5",
                    "target_site_code": "5",
                    "waiting_time": 90,
                }
            ],
            [],
            {"ZY14": []},
            aircraft_count=1,
        )
        self.assertEqual(result["resource_wait_opportunity_count"], 0)
        self.assertEqual(result["total_wait_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()

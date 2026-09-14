import copy
import unittest
from onpolicy.scripts.train.capture_stage3_boundary import certify_steps


class CaptureTests(unittest.TestCase):
    def states(self):
        parent = {"policy_updates": 4, "actor_optim": {"param_groups": [1], "state": {1: {"step": 4}, 2: {"step": 4}}},
                  "critic_optim": {"param_groups": [2], "state": {3: {"step": 4}}}}
        exported = copy.deepcopy(parent)
        exported["policy_updates"] = 6
        for key in ("actor_optim", "critic_optim"):
            for value in exported[key]["state"].values():
                value["step"] = 6
        updates = [{"update": {"epochs": [{"actor_step_applied": True}, {"actor_step_applied": True}]}}]
        return parent, exported, updates

    def test_complete_state_accepted(self):
        self.assertEqual(certify_steps(*self.states())["actor_steps_since_parent"], 2)

    def test_partial_or_extra_update_rejected(self):
        for key, param in (("actor_optim", 1), ("actor_optim", 2), ("critic_optim", 3)):
            with self.subTest(key=key, param=param):
                parent, exported, updates = self.states()
                exported[key]["state"][param]["step"] += 1
                with self.assertRaises(ValueError):
                    certify_steps(parent, exported, updates)

    def test_wrong_policy_counter_or_ownership_rejected(self):
        parent, exported, updates = self.states()
        exported["policy_updates"] -= 1
        with self.assertRaises(ValueError):
            certify_steps(parent, exported, updates)
        parent, exported, updates = self.states()
        exported["actor_optim"]["param_groups"] = [9]
        with self.assertRaises(ValueError):
            certify_steps(parent, exported, updates)


if __name__ == "__main__":
    unittest.main()

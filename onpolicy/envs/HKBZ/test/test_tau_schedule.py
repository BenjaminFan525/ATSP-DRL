import unittest
from types import SimpleNamespace

from onpolicy.utils.util import update_linear_anneal


class TestTauSchedule(unittest.TestCase):

    def test_legacy_schedule_is_unchanged(self):
        model = SimpleNamespace(tau=None)
        update_linear_anneal(model, 1.0, 0.2, 4, 8)
        self.assertAlmostEqual(model.tau, 0.6)

    def test_exploration_then_final_tau_plateau(self):
        model = SimpleNamespace(tau=None)
        observed = []
        for epoch in range(5):
            update_linear_anneal(
                model, 0.5, 0.3, epoch, 5, tau_anneal_epochs=3
            )
            observed.append(model.tau)
        self.assertEqual(observed, [0.5, 0.4, 0.3, 0.3, 0.3])

    def test_one_epoch_schedule_immediately_uses_final_tau(self):
        model = SimpleNamespace(tau=None)
        update_linear_anneal(
            model, 0.5, 0.3, 0, 4, tau_anneal_epochs=1
        )
        self.assertAlmostEqual(model.tau, 0.3)

    def test_negative_schedule_is_rejected(self):
        model = SimpleNamespace(tau=None)
        with self.assertRaises(ValueError):
            update_linear_anneal(
                model, 0.5, 0.3, 0, 4, tau_anneal_epochs=-1
            )


if __name__ == '__main__':
    unittest.main()

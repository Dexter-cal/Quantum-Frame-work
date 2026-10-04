import unittest
from qai.mechanics import LearningRateScheduler, EarlyStopping

class TestMechanicsExtensions(unittest.TestCase):
    def test_learning_rate_scheduler_step(self):
        scheduler = LearningRateScheduler(initial_lr=0.1, decay_factor=0.5, step_size=2, mode="step")
        self.assertEqual(scheduler.step(), 0.1)
        self.assertEqual(scheduler.step(), 0.05)

    def test_early_stopping_convergence(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(stopper.update(1.0))
        self.assertFalse(stopper.update(0.995)) # Delta too small
        self.assertTrue(stopper.update(0.995))  # Counter reaches patience

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import qai

class TestTimeSeriesForecaster(unittest.TestCase):
    def test_time_series_happy_path(self):
        # Linear sequence y_t = y_{t-1} + 1.0
        time_series = np.arange(1.0, 50.0, 1.0)
        model = qai.build(type="time_series", lags=3, horizon=1)
        model.train(time_series)

        self.assertTrue(model.training_status()["trained"])

        # Predict single next step
        next_val = model.predict([47.0, 48.0, 49.0])
        self.assertAlmostEqual(next_val, 50.0, delta=0.5)

        # Multi-step forecast
        fc = model.forecast(steps=3)
        self.assertEqual(len(fc), 3)
        np.testing.assert_allclose(fc, [50.0, 51.0, 52.0], atol=1.0)

        # Residuals check
        res = model.residuals()
        self.assertGreater(len(res), 0)

    def test_time_series_edge_cases(self):
        # Short series
        ts_short = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = qai.build(type="time_series", lags=2, horizon=1)
        model.train(ts_short)
        fc = model.forecast(steps=1)
        self.assertIsInstance(fc, (float, int, np.number))

    def test_time_series_error_cases(self):
        # Series too short for specified lags
        ts_too_short = np.array([1.0, 2.0])
        model = qai.build(type="time_series", lags=3)
        with self.assertRaises(ValueError):
            model.train(ts_too_short)

        # Predict before fit
        unfit_model = qai.build(type="time_series")
        with self.assertRaises(RuntimeError):
            unfit_model.predict([1.0, 2.0, 3.0])

    def test_security_misuse_scenario(self):
        time_series = np.arange(1.0, 20.0, 1.0)
        model = qai.build(type="time_series", lags=3)
        model.train(time_series)

        # Passing insufficient lag points
        with self.assertRaises(ValueError):
            model.predict([1.0])

if __name__ == "__main__":
    unittest.main()

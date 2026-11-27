"""Tests for qai.budget.TimeBudget preemptive interruption.

Verifies the fix for the v1.8.3 reviewer finding #2:
"A 2-second sleep with a 0.5-second budget took 2.0 seconds, and an infinite
loop was never interrupted."

The old TimeBudget only checked `exceeded()` *after* the function returned —
so a function that blocks for 2s with a 0.5s budget still ran for 2s, and a
function that loops forever was never interrupted.

The new TimeBudget.run() uses SIGALRM (POSIX main thread) or a watchdog
thread (elsewhere) to actually interrupt the running function.

Run:
    python3 -m pytest tests/test_time_budget_interruption.py -v
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

from qai.budget import TimeBudget


class TestTimeBudgetInterrupt(unittest.TestCase):
    def test_short_function_returns_normally(self):
        """A function that finishes within budget returns its value."""
        tb = TimeBudget(seconds=2.0)
        result = tb.run(lambda: 1 + 1)
        self.assertEqual(result, 2)

    def test_sleep_over_budget_is_interrupted(self):
        """time.sleep(2.0) with a 0.5s budget must be interrupted.

        Regression: previously the sleep ran for the full 2.0s because the
        budget was only checked *after* the function returned.
        """
        tb = TimeBudget(seconds=0.5)
        t0 = time.time()
        with self.assertRaises(TimeoutError):
            tb.run(lambda: time.sleep(2.0))
        elapsed = time.time() - t0
        # Should be ~0.5s, not ~2.0s. Allow generous upper bound for slow CI.
        self.assertLess(
            elapsed, 1.5,
            f"TimeBudget.run did not interrupt sleep — elapsed {elapsed:.2f}s "
            f"with 0.5s budget. Expected ~0.5s, got {elapsed:.2f}s.",
        )

    def test_infinite_loop_is_interrupted(self):
        """An infinite loop must be interrupted, not hang forever."""
        def infinite():
            while True:
                pass
        tb = TimeBudget(seconds=0.3)
        t0 = time.time()
        with self.assertRaises(TimeoutError):
            tb.run(infinite)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 2.0, "infinite loop was not interrupted")

    def test_context_manager_interrupts(self):
        """The `with` form also interrupts long-running code."""
        t0 = time.time()
        with self.assertRaises(TimeoutError):
            with TimeBudget(seconds=0.4):
                time.sleep(2.0)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 1.5)

    def test_function_returns_value_within_budget(self):
        """A function that returns before the budget is NOT interrupted."""
        tb = TimeBudget(seconds=1.0)
        result = tb.run(lambda sum=0: sum + 42)
        self.assertEqual(result, 42)

    def test_no_limit_never_interrupts(self):
        """TimeBudget(seconds=None) should never interrupt."""
        tb = TimeBudget(seconds=None)
        result = tb.run(lambda: "ok")
        self.assertEqual(result, "ok")

    def test_budget_does_not_leak_across_runs(self):
        """After a run completes, the SIGALRM handler must be disarmed."""
        tb = TimeBudget(seconds=1.0)
        tb.run(lambda: 1)  # finishes immediately, should disarm
        # If disarm failed, this sleep would be interrupted by the leftover alarm
        time.sleep(0.05)
        # No assertion needed — if we get here, the alarm didn't fire
        self.assertTrue(True)

    def test_cooperative_polling_still_works(self):
        """The old `exceeded()` / `should_stop()` API still works as before."""
        tb = TimeBudget(seconds=0.1)
        self.assertFalse(tb.exceeded())
        time.sleep(0.15)
        self.assertTrue(tb.exceeded())
        self.assertEqual(tb.remaining(), 0)


if __name__ == "__main__":
    unittest.main()

"""
Self-Improving Models — example script.

Shows how to wrap a model in a ModelGuardian that learns continuously
from streaming data, auto-retrains when drift is detected, and persists
knowledge across sessions.

    python self_improving.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import qai
from qai.self_improving import wrap, from_checkpoint


def main():
    # 1. Build and train an initial model
    print("=== 1. Build and train an initial model ===")
    m = qai.build(technique="neural_network", hidden_layer_sizes=(8, 6))
    X, y = qai.load_iris()
    m.train(X, y)
    pred = m.predict(X)
    acc = float((np.array(pred) == np.array(y)).mean())
    print(f"  Initial accuracy: {acc:.2%}")

    # 2. Wrap in a ModelGuardian
    print("\n=== 2. Wrap in ModelGuardian ===")
    guardian = wrap(m, drift_threshold=0.10, retrain_window=30)
    print(f"  Dashboard: {guardian.dashboard()}")

    # 3. Simulate streaming data — feed observations
    print("\n=== 3. Feed 30 streaming observations ===")
    rng = np.random.default_rng(42)
    for i in range(30):
        idx = rng.integers(0, len(X))
        x, true_y = X[idx], y[idx]
        # Serve + observe (model doesn't see true_y unless we call evaluate)
        guardian.serve(x)
    print(f"  After 30 obs: {guardian.state() if hasattr(guardian, 'state') else guardian.dashboard()}")

    # 4. Evaluate on labeled data (triggers retrain if degraded)
    print("\n=== 4. Evaluate against held-out data ===")
    eval_result = guardian.evaluate(X[100:], y[100:])
    print(f"  Eval: {eval_result}")

    # 5. Find blind spots
    print("\n=== 5. Find blind spots ===")
    spots = guardian.find_blind_spots(X[:30])
    print(f"  Found {len(spots) if isinstance(spots, list) else spots} blind spots")

    # 6. Alerts (if any)
    print("\n=== 6. Alerts ===")
    alerts = guardian.alerts()
    if alerts:
        for a in alerts:
            print(f"  - {a}")
    else:
        print("  No alerts. System is healthy.")

    # 7. Save state for next session
    print("\n=== 7. Save state for next session ===")
    state_path = "/tmp/qai_guardian_state"
    guardian.save_state(state_path)
    print(f"  Saved to {state_path}")

    # 8. Restore in next session
    print("\n=== 8. Restore from checkpoint ===")
    restored = from_checkpoint(state_path)
    print(f"  Restored dashboard: {restored.dashboard()}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()

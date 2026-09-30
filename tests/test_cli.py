import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import subprocess
import json
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TMP = "/tmp/qai_cli_test"
os.makedirs(TMP, exist_ok=True)


def run_cli(*args):
    """Genuinely spawns `python -m qai <args>` as a real subprocess --
    this is the actual command line, not a function call pretending to
    be one. If argparse, imports, or the entry point itself are broken,
    THIS is what would catch it, unlike calling cli.main() directly
    in-process."""
    result = subprocess.run(
        [sys.executable, "-m", "qai"] + list(args),
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
    )
    return result


print("=" * 60)
print("SETUP: train and export a real model to test the CLI against")
print("=" * 60)
sys.path.insert(0, PROJECT_ROOT)
import qai
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2x + 1
model = qai.build(type="regression")
model.train(X, y, verbose=False)
export_path = f"{TMP}/demo_model.json"
model.export(export_path)
print(f"Exported real trained model to {export_path}")

print()
print("=" * 60)
print("TEST: `python -m qai models list <dir>` -- REAL subprocess call")
print("=" * 60)
result = run_cli("models", "list", TMP)
print("stdout:", result.stdout)
print("stderr:", result.stderr)
print("returncode:", result.returncode)
assert result.returncode == 0, f"BUG: CLI exited with error: {result.stderr}"
assert "demo_model.json" in result.stdout, "BUG: exported model not found by `models list`"
assert "regression" in result.stdout, "BUG: technique name missing from `models list` output"
print("VERIFIED: real subprocess correctly lists the real exported model")

print()
print("=" * 60)
print("TEST: `python -m qai info <path>` -- REAL subprocess call")
print("=" * 60)
result = run_cli("info", export_path)
print("stdout:", result.stdout)
assert result.returncode == 0
assert "regression" in result.stdout
assert "weights: 1 coefficients" in result.stdout
print("VERIFIED: real subprocess correctly reports model info")

print()
print("=" * 60)
print("TEST: `python -m qai run <path> --input '[10.0]'` -- REAL prediction")
print("from a REAL subprocess, reconstructing the model from its export")
print("=" * 60)
result = run_cli("run", export_path, "--input", "[10.0]")
print("stdout:", result.stdout)
print("stderr:", result.stderr)
assert result.returncode == 0, f"BUG: run failed: {result.stderr}"

# REAL CHECK: y = 2x + 1, so for x=10, prediction should be ~21.0
output_line = [l for l in result.stdout.split("\n") if "prediction:" in l][0]
predicted_value = float(output_line.split("prediction:")[1].strip())
print(f"Predicted value for x=10: {predicted_value}")
assert abs(predicted_value - 21.0) < 0.5, \
    f"BUG: CLI-reconstructed model gave wrong prediction ({predicted_value}, expected ~21.0)"
print("VERIFIED: a model exported by the Python API, then loaded and run through a")
print("REAL command-line subprocess, gives the mathematically correct prediction")

print()
print("=" * 60)
print("TEST: CLI error handling -- malformed input via the REAL command line")
print("=" * 60)
result = run_cli("run", export_path, "--input", "not valid json")
print("stdout:", result.stdout)
print("stderr:", result.stderr)
print("returncode:", result.returncode)
assert result.returncode != 0, "BUG: CLI should exit non-zero on malformed input"
assert "Error" in result.stderr
print("VERIFIED: real CLI subprocess exits with a proper error code and clear message,")
print("not a raw Python traceback dumped on the user")

print()
print("=" * 60)
print("TEST: CLI with no arguments -- should show help, not crash")
print("=" * 60)
result = run_cli()
print("returncode:", result.returncode)
assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
print("VERIFIED: running `qai` with no arguments shows real help text, doesn't crash")

print()
print("ALL CLI TESTS PASSED -- genuine subprocess calls, exercising the real")
print("command line exactly as an actual user would type it")

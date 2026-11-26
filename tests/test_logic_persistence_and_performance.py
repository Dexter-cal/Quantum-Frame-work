import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import time
import subprocess
import qai

TMP = "/tmp/qai_logic_save_test"
os.makedirs(TMP, exist_ok=True)

print("=" * 60)
print("TEST: save_logic_source() -- does it capture the REAL source code?")
print("=" * 60)


@qai.logic
def double_it(x):
    result = x * 2
    return result


path = f"{TMP}/double_it_logic.py"
qai.save_logic_source(double_it, path)
with open(path) as f:
    saved_source = f.read()
print("Saved source:")
print(saved_source)
assert "def double_it" in saved_source
assert "return result" in saved_source
print("VERIFIED: the actual real source code was captured, not a stub")

print()
print("=" * 60)
print("TEST: reload the SAME source in a GENUINELY SEPARATE subprocess")
print("(the strongest possible test -- matches the CLI test's rigor:")
print(" a completely independent Python process, not just re-importing")
print(" in the same script)")
print("=" * 60)

reload_script = f"""
import sys
sys.path.insert(0, "{os.path.join(os.path.dirname(__file__), '..')}")
import qai
fn = qai.load_logic_source("{path}")
result = fn(21)
print(f"RESULT:{{result}}")
"""
script_path = f"{TMP}/reload_test.py"
with open(script_path, "w") as f:
    f.write(reload_script)

result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=15)
print("subprocess stdout:", result.stdout)
print("subprocess stderr:", result.stderr)
assert result.returncode == 0, f"BUG: reload subprocess failed: {result.stderr}"
output_line = [l for l in result.stdout.split("\n") if l.startswith("RESULT:")][0]
reloaded_result = int(output_line.split("RESULT:")[1])
assert reloaded_result == 42, f"BUG: reloaded logic gave {reloaded_result}, expected 42 (21*2)"
print("VERIFIED: logic saved in ONE process, reloaded and correctly executed in a")
print("GENUINELY SEPARATE process, giving the mathematically correct answer (42)")

print()
print("=" * 60)
print("TEST: save/reload logic that uses a REAL trained model, not just plain math")
print("(the actually meaningful case -- Section 68's real claim)")
print("=" * 60)

import numpy as np
model = qai.build(type="regression")
model.train(np.array([[1.0],[2.0],[3.0]]), np.array([2.0,4.0,6.0]), verbose=False)
model.export(f"{TMP}/model_for_logic.json")


@qai.logic
def classify_with_model(x):
    prediction = model.predict(x)
    return prediction


qai.save_logic_source(classify_with_model, f"{TMP}/model_logic.py")

reload_with_model_script = f"""
import sys
sys.path.insert(0, "{os.path.join(os.path.dirname(__file__), '..')}")
import qai, json
import numpy as np

meta = json.load(open("{TMP}/model_for_logic.json"))
model = qai.build(type="regression")
model.technique.w = np.array(meta["w"])
model.technique.b = meta["b"]
model.technique._trained = True

fn = qai.load_logic_source("{TMP}/model_logic.py", namespace={{"model": model}})
result = fn([10.0])
print(f"RESULT:{{result}}")
"""
script2_path = f"{TMP}/reload_with_model.py"
with open(script2_path, "w") as f:
    f.write(reload_with_model_script)

result2 = subprocess.run([sys.executable, script2_path], capture_output=True, text=True, timeout=15)
print("subprocess stdout:", result2.stdout)
print("subprocess stderr:", result2.stderr)
assert result2.returncode == 0, f"BUG: {result2.stderr}"
output_line2 = [l for l in result2.stdout.split("\n") if l.startswith("RESULT:")][0]
reloaded_prediction = float(output_line2.split("RESULT:")[1])
print(f"Reloaded logic + reloaded model predicted: {reloaded_prediction} (expected ~20.0, since y=2x)")
assert abs(reloaded_prediction - 20.0) < 0.1, f"BUG: expected ~20.0, got {reloaded_prediction}"
print("VERIFIED: logic AND its model, both saved separately, both reloaded in a fresh")
print("process, and correctly working TOGETHER -- the real, complete answer to")
print("'can the brain (weights) and its logic both be saved and reloaded'")

print()
print("=" * 60)
print("TEST: REAL, MEASURED performance overhead of the @qai.logic decorator")
print("Not a guess -- an actual timed comparison, undecorated vs decorated")
print("=" * 60)


def bare_function(x):
    return x * 2


@qai.logic
def decorated_function(x):
    return x * 2


n_calls = 100000

start = time.perf_counter()
for _ in range(n_calls):
    bare_function(5)
bare_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(n_calls):
    decorated_function(5)
decorated_time = time.perf_counter() - start

overhead_per_call_microseconds = ((decorated_time - bare_time) / n_calls) * 1_000_000
print(f"{n_calls} calls, bare function: {bare_time:.4f}s")
print(f"{n_calls} calls, @qai.logic decorated: {decorated_time:.4f}s")
print(f"Overhead per call: {overhead_per_call_microseconds:.3f} microseconds")
print(f"Decorated is {decorated_time/bare_time:.2f}x the bare function's time")

print()
print("ALL LOGIC SAVE/RELOAD + PERFORMANCE TESTS PASSED")

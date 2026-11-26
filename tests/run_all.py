"""Run all qai tests sequentially and report pass/fail counts."""
import os, sys, subprocess, time
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Pass project root via env so child processes can find qai
env = os.environ.copy()
env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

test_files = sorted(f for f in os.listdir(TEST_DIR) if f.startswith("test_") and f.endswith(".py"))

passed, failed, errored = [], [], []
start = time.time()

for tf in test_files:
    path = os.path.join(TEST_DIR, tf)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=120,
            cwd=TEST_DIR, env=env,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            passed.append((tf, elapsed))
            print(f"  PASS  {tf:50s}  {elapsed:5.1f}s")
        else:
            # If the test exits with non-zero, it failed
            failed.append((tf, result.stderr[-500:] if result.stderr else result.stdout[-500:]))
            print(f"  FAIL  {tf:50s}  {elapsed:5.1f}s")
            # Show last bit of stderr for diagnosis
            err_lines = (result.stderr or result.stdout).splitlines()[-5:]
            for line in err_lines:
                print(f"        {line}")
    except subprocess.TimeoutExpired:
        errored.append((tf, "TIMEOUT (>120s)"))
        print(f"  TMO   {tf:50s}  >120s")
    except Exception as e:
        errored.append((tf, str(e)[:100]))
        print(f"  ERR   {tf:50s}  {e}")

print()
print("=" * 70)
print(f"TEST SUMMARY")
print(f"  Passed:  {len(passed)}")
print(f"  Failed:  {len(failed)}")
print(f"  Errors:  {len(errored)}")
print(f"  Total:   {len(test_files)}")
print(f"  Time:    {time.time()-start:.1f}s")
print("=" * 70)

if failed:
    print("\nFAILURES:")
    for tf, err in failed:
        print(f"\n{tf}:")
        print(f"  {err}")

sys.exit(0 if not failed and not errored else 1)

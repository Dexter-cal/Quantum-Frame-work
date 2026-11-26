"""
Benchmark Suite — example script.

Runs the synthetic MMLU/HumanEval/GSM8K-style benchmarks on a stub model.

    python benchmark_suite.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import qai


class StubModel:
    """A trivial baseline model — always answers 'A'."""
    def chat(self, prompt): return "A"
    def predict(self, x): return "A"
    def generate(self, x): return "A"


class RandomModel:
    """A random-guessing baseline."""
    def __init__(self):
        import random
        self._r = random
    def chat(self, prompt):
        return self._r.choice(["A", "B", "C", "D"])


def main():
    print("=== qai Benchmark Suite ===\n")
    print("Available benchmarks:", qai.list_benchmarks())
    print()

    # 1. Run a single benchmark
    print("--- Single benchmark (MMLU, 10 questions) ---")
    result = qai.benchmark_suite.run_benchmark("mmlu", StubModel(), n=10)
    print(f"  MMLU accuracy: {result['accuracy']:.2%} ({result['n_correct']}/{result['n_total']})")
    print()

    # 2. Run full suite on StubModel
    print("--- Full suite on StubModel (always answers 'A') ---")
    suite = qai.benchmark_suite.BenchmarkSuite()
    report = suite.run(StubModel(), n=5)
    print(f"  Weighted average: {report.weighted_average:.2%}")
    for name, r in report.results.items():
        if "error" in r:
            print(f"    {name}: ERROR {r['error'][:50]}")
        else:
            print(f"    {name}: {r.get('accuracy', 0):.2%}")
    print()

    # 3. Compare two models
    print("--- Compare two models ---")
    leaderboard = suite.compare([
        ("StubModel (always A)", StubModel()),
        ("RandomModel (uniform)", RandomModel()),
    ])
    print(leaderboard.to_markdown())
    print()

    # 4. Benchmark info
    print("--- Benchmark metadata ---")
    for name in qai.list_benchmarks():
        info = qai.benchmark_info(name)
        print(f"  {info['name']:15} task_type={info.get('task_type', '?'):18} "
              f"items={info.get('n_samples', '?')}")
    print()
    print("=== DONE ===")


if __name__ == "__main__":
    main()

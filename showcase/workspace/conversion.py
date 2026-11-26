"""
Model Conversion — example script.

Shows how to export a qai model to multiple formats and benchmark them.

    python conversion.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import qai


def main():
    # 1. Build and train a model
    print("=== 1. Build and train a model ===")
    m = qai.build(technique="neural_network", hidden_layer_sizes=(8, 6))
    X, y = qai.load_iris()
    m.train(X, y)
    pred = m.predict(X)
    acc = float((np.array(pred) == np.array(y)).mean())
    print(f"  Trained: {m.technique_name}, accuracy: {acc:.2%}")

    # 2. List supported formats
    print("\n=== 2. Supported export formats ===")
    for fmt in qai.supported_formats():
        avail = qai.available_backends().get(fmt, False)
        print(f"  {fmt:14} {'available' if avail else '(optional dep needed)'}")

    # 3. Export to multiple formats
    print("\n=== 3. Export to multiple formats ===")
    tmp = tempfile.mkdtemp()
    exports = {}
    for fmt in ["pickle", "qai_native", "torchscript"]:
        try:
            path = qai.export(m, fmt, os.path.join(tmp, f"model.{fmt}"))
            size = os.path.getsize(path)
            print(f"  {fmt:14} → {path} ({size:,} bytes)")
            exports[fmt] = path
        except Exception as e:
            print(f"  {fmt:14} → FAILED: {e}")

    # 4. Compare predictions: original vs each export
    print("\n=== 4. Compare predictions ===")
    sample = X[0]
    orig_pred = m.predict(sample)
    print(f"  Original prediction: {orig_pred}")
    for fmt, path in exports.items():
        try:
            loaded = qai.import_model(path)
            new_pred = loaded.predict(sample) if hasattr(loaded, "predict") else None
            match = "MATCH" if str(orig_pred) == str(new_pred) else "DIFF"
            print(f"  {fmt:14} → {new_pred}  [{match}]")
        except Exception as e:
            print(f"  {fmt:14} → load error: {e}")

    # 5. Benchmark formats (load + inference time)
    print("\n=== 5. Benchmark formats (load + inference time) ===")
    results = qai.conversion.benchmark_formats(m, formats=list(exports.keys()),
                                                 sample_input=sample)
    for r in sorted(results, key=lambda x: x.get("total_time_s", float("inf"))):
        if r.get("error"):
            print(f"  {r['format']:14} → ERROR: {str(r['error'])[:60]}")
        else:
            print(f"  {r['format']:14} → load: {r.get('load_time_s', 0):.3f}s, "
                  f"infer: {r.get('inference_time_s', 0)*1000:.2f}ms, "
                  f"size: {r.get('size_bytes', 0):,}B")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()

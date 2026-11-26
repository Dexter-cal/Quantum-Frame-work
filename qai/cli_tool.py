"""
qai.cli_tool -- command-line interface for the qai framework.

Usage:
    qai build --technique regression --fit data.csv --save model.qmodel
    qai predict --model model.qmodel --input 1.0,2.0,3.0,4.0
    qai inspect model.qmodel
    qai convert model.qmodel --to safetensors
    qai serve model.qmodel --port 8080
    qai list-techniques
    qai list-formats
    qai benchmark model.qmodel --data test.csv
    qai shelf ./models --list
    qai info
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qai",
        description="Quantum AI Framework — command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    # build
    p_build = subparsers.add_parser("build", help="Build and train a model")
    p_build.add_argument("--technique", "-t", default="regression", help="technique name")
    p_build.add_argument("--data", "-d", help="CSV data file")
    p_build.add_argument("--target", "-y", help="target column name")
    p_build.add_argument("--save", "-s", help="save path (.qmodel)")
    p_build.add_argument("--params", "-p", help="JSON params", default="{}")

    # predict
    p_predict = subparsers.add_parser("predict", help="Run prediction")
    p_predict.add_argument("--model", "-m", required=True, help="model file path")
    p_predict.add_argument("--input", "-i", required=True, help="comma-separated input values")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a model file")
    p_inspect.add_argument("path", help="model file path")

    # convert
    p_convert = subparsers.add_parser("convert", help="Convert between formats")
    p_convert.add_argument("input", help="input file")
    p_convert.add_argument("--to", "-t", required=True, help="output format (.safetensors, .pt, .pkl, .weights)")

    # serve
    p_serve = subparsers.add_parser("serve", help="Serve a model via HTTP")
    p_serve.add_argument("--model", "-m", required=True, help="model file path")
    p_serve.add_argument("--port", "-p", type=int, default=8080, help="port number")

    # list-techniques
    subparsers.add_parser("list-techniques", help="List all available techniques")

    # list-formats
    subparsers.add_parser("list-formats", help="List all supported file formats")

    # shelf
    p_shelf = subparsers.add_parser("shelf", help="Browse model shelf")
    p_shelf.add_argument("directory", help="models directory")
    p_shelf.add_argument("--list", "-l", action="store_true", help="list all models")
    p_shelf.add_argument("--search", "-s", help="search query")

    # info
    subparsers.add_parser("info", help="Show framework info")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark a model")
    p_bench.add_argument("--model", "-m", required=True, help="model file path")
    p_bench.add_argument("--data", "-d", help="test data CSV")
    p_bench.add_argument("--repeats", "-r", type=int, default=10, help="number of repeats")

    # weights
    p_weights = subparsers.add_parser("weights", help="Inspect/edit model weights")
    p_weights.add_argument("path", help="weights file path")
    p_weights.add_argument("--inspect", action="store_true", help="inspect weights")
    p_weights.add_argument("--validate", action="store_true", help="validate weights")
    p_weights.add_argument("--layer", help="specific layer to inspect")

    # automl
    p_automl = subparsers.add_parser("automl", help="Run AutoML")
    p_automl.add_argument("--data", "-d", required=True, help="CSV data file")
    p_automl.add_argument("--target", "-y", required=True, help="target column")
    p_automl.add_argument("--task", "-t", default="classification", help="classification|regression")
    p_automl.add_argument("--time", type=int, default=60, help="time budget in seconds")

    # bundle -- subcommand for .qmodel bundle operations
    p_bundle = subparsers.add_parser("bundle", help="Bundle operations (inspect/run/serve)")
    bundle_sub = p_bundle.add_subparsers(dest="bundle_command", help="bundle subcommand")
    # bundle inspect
    p_bi = bundle_sub.add_parser("inspect", help="Inspect a .qmodel bundle (no loading)")
    p_bi.add_argument("path", help="path to .qmodel file")
    p_bi.add_argument("--readme", action="store_true", help="print the README")
    p_bi.add_argument("--logic", action="store_true", help="print the bundled logic.py source")
    # bundle run
    p_br = bundle_sub.add_parser("run", help="Load a bundle and run a single prediction")
    p_br.add_argument("path", help="path to .qmodel file")
    p_br.add_argument("--input", "-i", required=True, help="comma-separated input values")
    p_br.add_argument("--sign-key", "-k", help="HMAC signing key (if bundle is signed)")
    # bundle serve
    p_bs = bundle_sub.add_parser("serve", help="Serve a bundle via HTTP")
    p_bs.add_argument("path", help="path to .qmodel file")
    p_bs.add_argument("--port", "-p", type=int, default=8080, help="port number")
    p_bs.add_argument("--sign-key", "-k", help="HMAC signing key (if bundle is signed)")
    # bundle verify
    p_bv = bundle_sub.add_parser("verify", help="Verify a bundle's HMAC signature")
    p_bv.add_argument("path", help="path to .qmodel file")
    p_bv.add_argument("--sign-key", "-k", required=True, help="HMAC signing key")
    # bundle diff
    p_bd = bundle_sub.add_parser("diff", help="Diff two bundles")
    p_bd.add_argument("path1", help="first .qmodel file")
    p_bd.add_argument("path2", help="second .qmodel file")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list-techniques":
        cmd_list_techniques()
    elif args.command == "list-formats":
        cmd_list_formats()
    elif args.command == "shelf":
        cmd_shelf(args)
    elif args.command == "info":
        cmd_info()
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "weights":
        cmd_weights(args)
    elif args.command == "automl":
        cmd_automl(args)
    elif args.command == "bundle":
        cmd_bundle(args)
    else:
        parser.print_help()


def cmd_build(args):
    """Build and train a model."""
    import qai
    import numpy as np
    import pandas as pd

    model = qai.build(technique=args.technique, **json.loads(args.params))
    if args.data:
        df = pd.read_csv(args.data)
        if args.target and args.target in df.columns:
            X = df.drop(columns=[args.target]).values.astype(np.float32)
            y = df[args.target].values
            model.fit(X, y)
            print(f"Trained {args.technique} on {len(X)} samples")
        else:
            X = df.values.astype(np.float32)
            model.fit(X)
            print(f"Trained {args.technique} on {len(X)} samples (unsupervised)")
    if args.save:
        path = qai.save_trained(model, os.path.dirname(args.save) or ".")
        print(f"Saved to {path}")
    print(f"Model: {model.help()}")


def cmd_predict(args):
    """Run prediction."""
    import qai
    import numpy as np
    model = qai.load_model_file(args.model)
    x = np.array([float(v) for v in args.input.split(",")], dtype=np.float32)
    result = model.forward(x)
    pred = float(result) if np.isscalar(result) else result.tolist()
    print(json.dumps({"prediction": pred}))


def cmd_inspect(args):
    """Inspect a model file."""
    import qai
    info = qai.inspect_model(args.path)
    print(json.dumps(info, indent=2, default=str))


def cmd_convert(args):
    """Convert between formats."""
    import qai
    output = args.input.rsplit(".", 1)[0] + args.to
    qai.convert(args.input, output)
    print(f"Converted {args.input} → {output}")


def cmd_serve(args):
    """Serve a model."""
    import qai
    model = qai.load_model_file(args.model)
    server = qai.ModelServer(model, port=args.port)
    print(f"Serving {args.model} on port {args.port}")
    server.start(block=True)


def cmd_list_techniques():
    """List all techniques."""
    import qai
    techniques = sorted(qai.techniques._REGISTRY.keys())
    print(f"{len(techniques)} techniques available:")
    for t in techniques:
        cls = qai.techniques._REGISTRY[t]
        family = getattr(cls, "family", "unknown")
        print(f"  {t:30s}  ({family})")


def cmd_list_formats():
    """List all formats."""
    import qai
    formats = qai.list_formats()
    print(f"{len(formats)} formats supported:")
    for f in formats:
        safe = "✓" if f["is_safe"] else "✗"
        print(f"  {safe} {f['extension']:15s}  {f['name']:30s}  [{f['category']}]")


def cmd_shelf(args):
    """Browse model shelf."""
    import qai
    shelf = qai.ModelShelf(args.directory)
    if args.search:
        results = shelf.search(args.search)
        print(f"Search '{args.search}': {len(results)} models")
        for m in results:
            print(f"  {m.name:40s} {m.size_human:>8}  {m.technique}")
    else:
        print(shelf.tree())


def cmd_info():
    """Show framework info."""
    import qai
    print("Quantum AI Framework")
    print(f"  Techniques: {len(qai.techniques._REGISTRY)}")
    print(f"  Creative ideas: {len(qai.creative.__all__)}")
    print(f"  Formats: {len(qai.formats.FORMATS)}")
    print(f"  C bridge methods: {len(qai.CBridgedMethods.all_methods())}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")


def cmd_benchmark(args):
    """Benchmark a model."""
    import qai
    import numpy as np
    model = qai.load_model_file(args.model)
    if args.data:
        import pandas as pd
        df = pd.read_csv(args.data)
        X = df.values.astype(np.float32)
    else:
        X = [np.random.randn(4).astype(np.float32) for _ in range(10)]
    bench = qai.ModelBenchmark(model)
    result = bench.benchmark_inference(X, repeats=args.repeats)
    print(json.dumps({k: v for k, v in result.items()
                       if k != "memory_before_mb" and k != "memory_after_mb"},
                      indent=2))


def cmd_weights(args):
    """Inspect/edit weights."""
    import qai
    if args.inspect:
        info = qai.inspect_weights(args.path)
        print(json.dumps(info, indent=2, default=str))
    elif args.validate:
        result = qai.validate_weights(args.path)
        print(json.dumps(result, indent=2, default=str))
    elif args.layer:
        info = qai.inspect_layer(args.path, args.layer)
        print(json.dumps(info, indent=2, default=str))
    else:
        info = qai.inspect_weights(args.path)
        print(f"Weights file: {info['name']}")
        print(f"  Layers: {info['n_layers']}")
        print(f"  Params: {info['n_params']}")
        print(f"  Fingerprint: {info['fingerprint']}")
        print(f"  Size: {info['size']}")


def cmd_automl(args):
    """Run AutoML."""
    import qai
    import numpy as np
    import pandas as pd
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target]).values.astype(np.float32)
    y = df[args.target].values
    aml = qai.AutoML(task=args.task, time_budget_s=args.time, verbose=True)
    result = aml.run(X, y)
    print(f"\nAutoML Results:")
    print(f"  Best technique: {result['best_technique']}")
    print(f"  Best score: {result['best_score']:.4f}")
    print(f"  Time: {result['elapsed_s']:.1f}s")
    print(f"  Top 3:")
    for r in result["model_rankings"][:3]:
        print(f"    {r['technique']:30s} {r.get('score', 0):.4f}")


# =====================================================================
# Bundle operations: qai bundle inspect|run|serve|verify|diff ...
# =====================================================================
def cmd_bundle(args):
    """Dispatch `qai bundle <subcommand>`."""
    sub = getattr(args, "bundle_command", None)
    if sub is None:
        print("Usage: qai bundle {inspect|run|serve|verify|diff} ...")
        return
    if sub == "inspect":
        _bundle_inspect(args)
    elif sub == "run":
        _bundle_run(args)
    elif sub == "serve":
        _bundle_serve(args)
    elif sub == "verify":
        _bundle_verify(args)
    elif sub == "diff":
        _bundle_diff(args)
    else:
        print(f"Unknown bundle subcommand: {sub}")


def _bundle_inspect(args):
    """qai bundle inspect <path> [--readme] [--logic]"""
    import qai
    if args.readme:
        # Print README straight from the zip (no model load)
        import zipfile
        with zipfile.ZipFile(args.path, "r") as zf:
            if "README.md" in zf.namelist():
                print(zf.read("README.md").decode("utf-8"))
            else:
                print("(no README in this bundle)")
        return
    if args.logic:
        import zipfile
        with zipfile.ZipFile(args.path, "r") as zf:
            if "logic.py" in zf.namelist():
                print(zf.read("logic.py").decode("utf-8"))
            else:
                print("(no logic.py in this bundle)")
        return
    info = qai.inspect_bundle(args.path)
    m = info["manifest"]
    print(f"=== Bundle: {args.path} ===")
    print(f"  format_version: {m.get('format_version', '?')}")
    print(f"  technique:      {m.get('technique', '?')}")
    print(f"  total_params:   {m.get('total_params', 0):,}")
    print(f"  weight_format:  {m.get('weight_format', '?')}")
    print(f"  signed:         {m.get('signed', False)}")
    print(f"  logic_count:    {m.get('logic_count', 0)}")
    print(f"  has_tokenizer:  {m.get('has_tokenizer', False)}")
    if m.get("description"):
        print(f"  description:    {m['description'][:80]}")
    print(f"  bundle_size:    {info['size_bytes']:,} bytes")
    print(f"  uncompressed:   {info['size_uncompressed_bytes']:,} bytes")
    print(f"  contents:")
    for c in info["contents"]:
        print(f"    {c['name']:25s} {c['size']:>10,} bytes")


def _bundle_run(args):
    """qai bundle run <path> --input 1.0,2.0,3.0"""
    import qai
    b = qai.load_bundle(args.path, sign_key=args.sign_key)
    # Parse input
    parts = args.input.split(",")
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        vals = parts  # treat as strings
    import numpy as np
    x = np.array(vals, dtype=float) if all(isinstance(v, float) for v in vals) else vals
    pred = b.model.predict(x)
    print(f"Prediction: {pred}")


def _bundle_serve(args):
    """qai bundle serve <path> --port 8080"""
    import qai
    b = qai.load_bundle(args.path, sign_key=args.sign_key)
    print(f"Loaded bundle: {b}")
    print(f"Serving on port {args.port}... (Ctrl+C to stop)")
    try:
        from qai.core.serve import build_flask_app
        app = build_flask_app(b.model)
        app.run(host="0.0.0.0", port=args.port)
    except ImportError:
        print("Flask not installed -- pip install flask")


def _bundle_verify(args):
    """qai bundle verify <path> --sign-key KEY"""
    import qai
    valid = qai.verify_bundle(args.path, args.sign_key)
    print(f"Signature valid: {valid}")
    sys.exit(0 if valid else 1)


def _bundle_diff(args):
    """qai bundle diff <path1> <path2>"""
    import qai
    diff = qai.diff_bundles(args.path1, args.path2)
    print(f"=== Bundle diff: {args.path1} vs {args.path2} ===")
    print(f"  technique_changed: {diff['technique_changed']} "
          f"({diff['technique1']} -> {diff['technique2']})")
    print(f"  weights_changed:   {diff['weights_changed']}")
    print(f"  logic_changed:     {diff['logic_changed']}")
    print(f"  tokenizer_changed: {diff['tokenizer_changed']}")
    if diff["params_changed"]:
        print(f"  params_changed:")
        for k, (v1, v2) in diff["params_changed"].items():
            print(f"    {k}: {v1!r} -> {v2!r}")
    if diff["weight_diffs"]:
        print(f"  weight_diffs:")
        for k, d in diff["weight_diffs"].items():
            if "l2_distance" in d:
                print(f"    {k}: L2={d['l2_distance']:.4f} "
                      f"max_abs={d['max_abs_change']:.4f} "
                      f"n_changed={d['n_changed']}")


if __name__ == "__main__":
    main()

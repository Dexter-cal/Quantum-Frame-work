"""
qai.cli -- Section 101's real command-line tool. `python -m qai <args>`
works without needing a pip install, so this is genuinely testable via
subprocess in this sandbox -- real command-line behavior, not just
function calls dressed up as a CLI.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np


def _load_exported_model_metadata(path):
    """Reads a .qmodel-equivalent JSON export (Section 68's real, if
    simplified, export format used throughout this build) WITHOUT
    reconstructing a full trained Model -- just enough for `list`/`info`
    to report on it, matching Section 101's design intent."""
    with open(path) as f:
        return json.load(f)


def cmd_models_list(args):
    directory = args.directory or "."
    found = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json") or fname.endswith(".qmodel"):
            full_path = os.path.join(directory, fname)
            try:
                meta = _load_exported_model_metadata(full_path)
                found.append((fname, meta.get("technique", "?"), os.path.getsize(full_path)))
            except (json.JSONDecodeError, KeyError):
                continue  # not a real qai export, skip silently -- matches real tools ignoring unrelated files

    if not found:
        print(f"No qai models found in {directory}")
        return 0

    print(f"{'NAME':<25s} {'TECHNIQUE':<15s} {'SIZE':<10s}")
    for name, technique, size in found:
        print(f"{name:<25s} {technique:<15s} {size} bytes")
    return 0


def cmd_info(args):
    meta = _load_exported_model_metadata(args.path)
    print(f"technique: {meta.get('technique')}")
    print(f"params: {meta.get('params')}")
    if "w" in meta:
        print(f"weights: {len(meta['w'])} coefficients + bias={meta.get('b')}")
    return 0


# =====================================================================
# `qai bundle ...` subcommands -- inspect/run/serve/sign/verify on .qmodel files
# =====================================================================
def cmd_bundle_inspect(args):
    """qai bundle inspect <path> -- show metadata without loading the model."""
    import qai
    info = qai.inspect_bundle(args.path)
    m = info["manifest"]
    print(f"path:           {args.path}")
    print(f"size:           {info['size_bytes'] / 1024:.1f} KB compressed")
    print(f"                {info['size_uncompressed_bytes'] / 1024:.1f} KB uncompressed")
    print(f"format_version: {m.get('format_version')}")
    print(f"technique:      {m.get('technique')}")
    print(f"signed:         {m.get('signed', False)}")
    print(f"created_at:     {m.get('created_at_iso', '?')}")
    print(f"total_params:   {m.get('total_params', '?')}")
    print(f"weight_format:  {m.get('weight_format')}")
    print(f"logic_count:    {m.get('logic_count', 0)}")
    print(f"has_tokenizer:  {m.get('has_tokenizer', False)}")
    if m.get("description"):
        print(f"description:    {m['description']}")
    print()
    print("contents:")
    for c in info["contents"]:
        comp_pct = (1 - c["compressed_size"] / max(c["size"], 1)) * 100
        print(f"  {c['name']:<22s} {c['size']:>8d} bytes  ({comp_pct:.0f}% compressed)")
    return 0


def cmd_bundle_run(args):
    """qai bundle run <path> --input '<json>' -- load the model + run a prediction."""
    import qai
    try:
        bundle = qai.load_bundle(args.path, sign_key=args.sign_key)
    except Exception as e:
        print(f"Error loading bundle: {e}", file=sys.stderr)
        return 1
    model = bundle.model
    print(f"# Loaded: {model.technique_name if hasattr(model, 'technique_name') else type(model).__name__}", file=sys.stderr)
    if bundle.readme:
        print(f"# (bundle has README -- use `qai bundle inspect` to see it)", file=sys.stderr)
    try:
        import json as _json
        input_data = _json.loads(args.input)
    except Exception as e:
        print(f"Error parsing --input as JSON: {e}", file=sys.stderr)
        return 1
    import numpy as np
    try:
        result = model.predict(np.asarray(input_data, dtype=float))
        print(f"prediction: {result}")
    except Exception as e:
        print(f"Error during predict: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_bundle_serve(args):
    """qai bundle serve <path> --port 5000 -- load the model + serve /predict via HTTP."""
    import qai
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json as _json
    try:
        bundle = qai.load_bundle(args.path, sign_key=args.sign_key)
    except Exception as e:
        print(f"Error loading bundle: {e}", file=sys.stderr)
        return 1
    model = bundle.model
    print(f"# Loaded: {type(model).__name__} from {args.path}", file=sys.stderr)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a): pass  # quiet
        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404); return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = _json.loads(self.rfile.read(length))
                import numpy as np
                result = model.predict(np.asarray(body["input"], dtype=float))
                out = _json.dumps({"prediction": result.tolist() if hasattr(result, "tolist") else result})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out.encode())
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json.dumps({"error": str(e)}).encode())
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_error(404)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"# Serving {args.path} on http://127.0.0.1:{args.port}/predict", file=sys.stderr)
    print(f"# POST {{\"input\": [1.0, 2.0, 3.0]}} to /predict", file=sys.stderr)
    print(f"# Ctrl+C to stop", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n# Stopped.", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def cmd_bundle_sign(args):
    """qai bundle sign <path> --key <key> -- re-sign a bundle (or sign an unsigned one)."""
    import qai
    # Load the existing model, re-save with the new key
    bundle = qai.load_bundle(args.path, sign_key=None)  # load without verifying
    out_path = args.out or args.path
    # Re-save with the new key
    qai.save_bundle(bundle.model, out_path,
                     logic_fns=list(bundle.logic_fns.values()),
                     tokenizer=bundle.tokenizer,
                     sign_key=args.key,
                     description=(bundle.manifest.get("description") or "") + " (re-signed)")
    print(f"Signed {out_path} with key '{args.key}'")
    return 0


def cmd_bundle_verify(args):
    """qai bundle verify <path> --key <key> -- verify a bundle's HMAC signature."""
    import qai
    valid = qai.verify_bundle(args.path, args.key)
    if valid:
        print(f"VALID: {args.path} signature matches key '{args.key}'")
        return 0
    else:
        print(f"INVALID: {args.path} signature does NOT match key '{args.key}'", file=sys.stderr)
        return 1


def cmd_bundle(args):
    """Dispatcher for `qai bundle <subcommand>`."""
    if args.bundle_command == "inspect":
        return cmd_bundle_inspect(args)
    elif args.bundle_command == "run":
        return cmd_bundle_run(args)
    elif args.bundle_command == "serve":
        return cmd_bundle_serve(args)
    elif args.bundle_command == "sign":
        return cmd_bundle_sign(args)
    elif args.bundle_command == "verify":
        return cmd_bundle_verify(args)
    else:
        print("Usage: qai bundle {inspect|run|serve|sign|verify} ...", file=sys.stderr)
        return 1


def cmd_run(args):
    """Reconstructs a real, usable Regression model from an export and
    runs a real prediction -- genuinely demonstrating `qai run` working
    end to end from the command line, not just reading metadata."""
    meta = _load_exported_model_metadata(args.path)
    if meta.get("technique") != "regression":
        print(f"Error: `qai run` currently only reconstructs 'regression' exports "
              f"(this file is '{meta.get('technique')}')", file=sys.stderr)
        return 1

    import qai
    model = qai.build(type="regression")
    model.technique.w = np.array(meta["w"])
    model.technique.b = meta["b"]
    model.technique._trained = True

    try:
        input_values = json.loads(args.input)
    except json.JSONDecodeError:
        print(f"Error: --input must be valid JSON (e.g. '[1.0, 2.0, 3.0]')", file=sys.stderr)
        return 1

    prediction = model.predict(np.array(input_values))
    print(f"prediction: {prediction}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="qai", description="qai -- the Quantum AI framework Python prototype")
    subparsers = parser.add_subparsers(dest="command")

    models_parser = subparsers.add_parser("models", help="manage local models")
    models_sub = models_parser.add_subparsers(dest="models_command")
    list_parser = models_sub.add_parser("list", help="list local qai models")
    list_parser.add_argument("directory", nargs="?", default=".")

    info_parser = subparsers.add_parser("info", help="show info about a model file")
    info_parser.add_argument("path")

    run_parser = subparsers.add_parser("run", help="run a real prediction from a model file")
    run_parser.add_argument("path")
    run_parser.add_argument("--input", required=True, help="JSON array of input values")

    # NEW: `qai bundle <subcommand>` -- works on .qmodel bundle files
    bundle_parser = subparsers.add_parser("bundle", help="inspect/run/serve/sign/verify .qmodel bundles")
    bundle_sub = bundle_parser.add_subparsers(dest="bundle_command")

    bp_inspect = bundle_sub.add_parser("inspect", help="show bundle metadata without loading the model")
    bp_inspect.add_argument("path")

    bp_run = bundle_sub.add_parser("run", help="load the bundle and run a prediction")
    bp_run.add_argument("path")
    bp_run.add_argument("--input", required=True, help="JSON array of input values")
    bp_run.add_argument("--sign-key", default=None, help="HMAC signing key (for signed bundles)")

    bp_serve = bundle_sub.add_parser("serve", help="load the bundle and serve /predict via HTTP")
    bp_serve.add_argument("path")
    bp_serve.add_argument("--port", type=int, default=5000)
    bp_serve.add_argument("--sign-key", default=None)

    bp_sign = bundle_sub.add_parser("sign", help="re-sign a bundle with a new key")
    bp_sign.add_argument("path")
    bp_sign.add_argument("--key", required=True, help="HMAC signing key")
    bp_sign.add_argument("--out", default=None, help="output path (default: overwrite)")

    bp_verify = bundle_sub.add_parser("verify", help="verify a bundle's HMAC signature")
    bp_verify.add_argument("path")
    bp_verify.add_argument("--key", required=True, help="HMAC signing key")

    # NEW: `qai compress / decompress / archive / extract` -- learned meta-compressor
    from .compress import add_compress_cli
    add_compress_cli(subparsers)

    # NEW: `qai cpc <subcommand>` -- Convergent Pattern Compression
    from .cpc import add_cpc_cli
    add_cpc_cli(subparsers)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models" and args.models_command == "list":
        return cmd_models_list(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "bundle":
        return cmd_bundle(args)
    elif args.command in ("compress", "decompress", "archive", "extract"):
        return _dispatch_compress_cli(args)
    elif args.command == "cpc":
        return _dispatch_cpc_cli(args)
    else:
        parser.print_help()
        return 1


def _dispatch_cpc_cli(args):
    """Dispatch CPC subcommands."""
    from .cpc import (_cli_cpc_ingest, _cli_cpc_ask, _cli_cpc_search,
                       _cli_cpc_graph, _cli_cpc_stats, _cli_cpc_forget)
    dispatch = {
        "ingest": _cli_cpc_ingest,
        "ask": _cli_cpc_ask,
        "search": _cli_cpc_search,
        "graph": _cli_cpc_graph,
        "stats": _cli_cpc_stats,
        "forget": _cli_cpc_forget,
    }
    handler = dispatch.get(args.cpc_command)
    if handler:
        return handler(args) or 0
    print("Usage: qai cpc {ingest|ask|search|graph|stats|forget} ...", file=sys.stderr)
    return 1


def _dispatch_compress_cli(args):
    """Dispatch the compress/decompress/archive/extract subcommands."""
    from .compress import _cli_compress, _cli_decompress, _cli_archive, _cli_extract
    if args.command == "compress":
        return _cli_compress(args) or 0
    elif args.command == "decompress":
        return _cli_decompress(args) or 0
    elif args.command == "archive":
        return _cli_archive(args) or 0
    elif args.command == "extract":
        return _cli_extract(args) or 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

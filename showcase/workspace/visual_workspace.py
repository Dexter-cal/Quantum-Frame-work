"""
Visual Workspace — example script.

Launches the interactive "Model Microscope" with a trained model.

Try:
    python visual_workspace.py                  # launches on port 7788
    python visual_workspace.py --port 8000      # different port
    python visual_workspace.py --no-browser     # don't open browser
    python visual_workspace.py --static         # generate static HTML instead
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import qai
import numpy as np


def build_demo_model():
    """Build a small iris classifier to demo the workspace."""
    print("[qai] Building demo model: neural_network (4-8-6-3)…")
    m = qai.build(technique="neural_network", hidden_layer_sizes=(8, 6))
    X, y = qai.load_iris()
    m.train(X, y)
    pred = m.predict(X)
    acc = float((np.array(pred) == np.array(y)).mean())
    print(f"[qai] Trained on {len(X)} iris samples. Accuracy: {acc:.2%}")
    return m


def main():
    p = argparse.ArgumentParser(description="Launch the qai Visual Workspace")
    p.add_argument("--port", type=int, default=7788, help="port (default 7788)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser")
    p.add_argument("--static", action="store_true",
                   help="generate a static HTML file instead of launching a server")
    p.add_argument("--output", default="qai_workspace.html",
                   help="output file for --static (default: qai_workspace.html)")
    p.add_argument("--no-model", action="store_true",
                   help="don't load a model — use DEMO mode (synthetic data)")
    args = p.parse_args()

    model = None if args.no_model else build_demo_model()

    if args.static:
        path = qai.generate_workspace(model, args.output)
        print(f"\n[qai] Static workspace written to: {path}")
        print(f"[qai] Open in any browser: file://{os.path.abspath(path)}")
    else:
        print(f"\n[qai] Launching Visual Workspace at http://127.0.0.1:{args.port}/")
        print(f"[qai] Press Ctrl+C to stop the server.")
        qai.serve_workspace(model, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

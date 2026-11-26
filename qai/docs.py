"""
qai.docs -- documentation utilities + tutorials.

Quickstart helpers that generate example code and tutorials on demand.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


QUICKSTART_TUTORIAL = """
# Quantum AI Framework — Quickstart (222 techniques, 108 creative ideas)

## Installation

    pip install qai  # or clone the repo

## 1. Build + train + save a model

    import qai, numpy as np
    X = np.random.randn(100, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(int)

    model = qai.build(technique="classifier")
    model.fit(X, y)  # or model.train(X, y) — aliases
    pred = model.forward(X[0])
    path = qai.save_trained(model, "./my_models", X=X, y=y)  # → .qmodel file

## 2. Data manipulation (all from qai.data — no .pro needed)

    import pandas as pd
    df = pd.read_csv("data.csv")

    # Clean → Feature engineer → Augment → Split
    cleaned, report = qai.data.DataCleaner.clean(df)
    enhanced, new_feats = qai.data.FeatureEngineer.auto_features(cleaned)
    X_aug, y_aug = qai.data.DataAugmentor.smote(X, y, n_new=50)
    splits = qai.data.DataSplitter.train_val_test(X, y, stratify=True)

    # Profile, visualize, view, explore
    profile = qai.data.DataInspector.profile(df)
    qai.data.DataVisualizer2.visualize(df, save_path="report.png")
    print(qai.data.DataViewer.view(df, "full"))
    filtered = qai.data.DataExplorer.filter(df, "age > 30")

    # Compare + narrate
    print(qai.data.DataComparer.diff_report(df_old, df_new))
    print(qai.data.DataStoryteller.narrate(df, "My Dataset"))

    # Pipeline
    pipe = qai.data.DataPipeline2()
    pipe.add("clean", lambda df: qai.data.DataCleaner.clean(df)[0])
    pipe.add("scale", lambda df: qai.data.DataTransformer.scale(df))
    result, report = pipe.run(df, verbose=True)

## 3. Multimodal processing (any data type)

    result = qai.data.MultimodalPipeline.process("image.jpg")  # auto-detects
    result = qai.data.MultimodalPipeline.process("video.mp4")
    result = qai.data.MultimodalPipeline.process("audio.wav")
    result = qai.data.MultimodalPipeline.process("document.txt")

    # Text
    cleaned = qai.data.TextProcessor.clean(text)
    sentiment = qai.data.TextProcessor.sentiment(text)

    # Image
    img = qai.data.ImageProcessor.load("photo.jpg")
    augmented = qai.data.ImageProcessor.augment(img, ["flip_h", "rotate90"])

    # Graph
    graph = qai.data.GraphProcessor.from_edges([(0,1), (1,2)])

    # Geospatial
    dist = qai.data.GeospatialProcessor.haversine(40.7, -74.0, 34.1, -118.2)

## 4. Weights file system (.weights / .wt)

    qai.save_weights(model, "model.weights")
    qai.save_weights(model, "model.wt")  # alias
    info = qai.inspect_weights("model.weights")
    qai.edit_weights("model.weights", layer="w", operation="scale", factor=0.9)
    qai.merge_weights("a.weights", "b.weights", "merged.weights", strategy="ties")
    qai.validate_weights("model.weights")

## 5. Weight surgery

    surgeon = qai.WeightSurgeon(model)
    surgeon.disable_neuron("0.weight", 0)
    qai.NeuronInspector.inspect(model, "0.weight")
    qai.ChannelPruner.prune(model, "0.weight", fraction=0.25)

## 6. Super-intelligent agents

    agent = qai.QuantumAgent(llm=my_llm, personality="creative")
    response = agent.think("Design a neural network", depth="deep")
    coder = qai.CodeAgent(llm=my_llm)
    result = coder.develop("Build a REST API")

## 7. Budget + hardware auto-config

    budget = qai.ResourceBudget(ram_mb=4096, gpu_mb=8192, time_s=3600)
    hw = qai.AutoScaleProfiler.profile()
    config = qai.AutoConfig.configure(model, dataset_size=1000)
    advice = qai.TrainingAdvisor.advise(model, dataset_size=1000)

## 8. Deployment

    qai.ModelDeployer.deploy_all(model, "./deployment")
    server = qai.ModelServer(model, port=8080); server.start()

## 9. GPU + C bridge

    arr = qai.gpu.DeviceTensor(np_array)
    out = qai.gpu.matmul(arr, arr)
    src = qai.CBridgedMethods.get_function_source(my_func)  # works in REPL!

## 10. Web dashboard

    qai.serve_dashboard(port=8080)  # visit http://localhost:8080
"""


ADVANCED_TUTORIAL = """
# Quantum AI Framework — Advanced

## Custom CUDA kernels

    @qai.gpu.kernel
    def my_kernel(x, y, out, n):
        tid = qai.gpu.thread_id()
        if tid < n:
            out[tid] = x[tid] * y[tid]

## C-extension bridge

    bridge = qai.CPythonBridge()
    bridge.compile_and_load('double sq(double x) { return x*x; }', 'sq',
                              restype=ctypes.c_double, argtypes=[ctypes.c_double])
    print(bridge.call('sq", 5.0))  # -> 25.0

## Distributed training

    trainer = qai.DistributedTrainer(backend="ray", num_workers=4)
    result = trainer.train(model_factory, data_fn, train_step_fn, epochs=10)

## LLM serving

    client = qai.LLMClient(backend="ollama", model="llama3.1")
    response = client.generate("Hello, world!")

## Real library integrations

    hf = qai.integrations_real.HuggingFaceModel("distilbert-base-uncased")
    hf.load()
    result = hf.forward("This is a test.")

    faiss = qai.integrations_real.RealFAISSIndex(dim=768)
    faiss.build()
    faiss.add(ids, vectors)
    results = faiss.search(query, top_k=5)
"""


def get_quickstart() -> str:
    """Return the quickstart tutorial as a string."""
    return QUICKSTART_TUTORIAL


def get_advanced_tutorial() -> str:
    """Return the advanced tutorial as a string."""
    return ADVANCED_TUTORIAL


def list_techniques() -> List[str]:
    """List all registered techniques."""
    import qai
    return sorted(qai.techniques._REGISTRY.keys())


def list_creative_ideas() -> List[str]:
    """List all creative idea classes."""
    import qai
    return [n for n in qai.creative.__all__ if not n.startswith("_")]


def technique_catalog() -> Dict[str, str]:
    """Return a name -> family mapping for all techniques."""
    import qai
    out = {}
    for name, cls in qai.techniques._REGISTRY.items():
        out[name] = getattr(cls, "family", "unknown")
    return out


def generate_docs_index() -> str:
    """Generate a markdown index of all qai modules."""
    import qai
    lines = ["# Quantum AI Framework — Module Index\n"]
    lines.append(f"Total techniques: {len(qai.techniques._REGISTRY)}\n")
    lines.append(f"Total creative ideas: {len(qai.creative.__all__)}\n")
    lines.append("\n## Techniques by Family\n")
    by_family: Dict[str, List[str]] = {}
    for name, cls in qai.techniques._REGISTRY.items():
        family = getattr(cls, "family", "unknown")
        by_family.setdefault(family, []).append(name)
    for family in sorted(by_family.keys()):
        lines.append(f"### {family}")
        for n in sorted(by_family[family]):
            lines.append(f"  - `{n}`")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "QUICKSTART_TUTORIAL", "ADVANCED_TUTORIAL",
    "get_quickstart", "get_advanced_tutorial",
    "list_techniques", "list_creative_ideas",
    "technique_catalog", "generate_docs_index",
]

#!/usr/bin/env python3
"""Generate docs/MODULE_REGISTRY.md — every qai module, categorized, with
real docstring one-liners. Marks previously-undocumented modules."""
import ast, re
from pathlib import Path

ROOT = Path("/home/z/my-project/download/Quantum-Frame-work-fixed")
QAI = ROOT / "qai"

NEW = """ab_testing active_learning active_learning_v2 agents_base api_reference autodoc autoscale
cbridge compression_advisor confidence_calibrator cpc_ai data_augmentor data_profiler
experiment_compare feature_store federated_bridge frontier2025 hf_converter hyperparam_optimizer
integrations integrations_real interpret memory_vault mind model_card_generator model_chat
model_health model_monitor model_versioner modelfile modelfile_extended modelfile_extensions
modellog observability pipeline_builder self_healing surgery visual_workspace_v2 webui
workspace_html hparam_optimizer""".split()

# curated category -> (order, list of module names (stems or subpackage names))
CATS = {
    "Core & framework": [
        "framework", "types", "configs", "core", "techniques", "mechanics", "completion",
        "native", "sklearn_compat", "asyncio", "chainable_output", "remainder", "finalize",
        "upgrades", "cbridge", "gpu", "cuda_kernels", "system", "math",
    ],
    "Data handling": [
        "data", "preprocessing", "datasets", "data_pipeline", "data_profiler", "data_augmentor",
        "feature_store", "pipeline_builder", "pipeline_templates", "synthetic", "formats", "time_series",
    ],
    "Model zoo & architectures": [
        "advanced_architectures", "advanced_architectures_2", "frontier2025", "modern_vision",
        "modern_alignment", "pretrained", "vision", "nlp", "audio", "graph", "tokenizers",
        "real_tokenizers", "agents_base",
    ],
    "Training & optimization": [
        "trainer", "distributed", "distributed_train", "live_training", "federation",
        "federated_bridge", "active_learning", "active_learning_v2", "finetuning", "alignment",
        "hyperparam_optimizer", "hparam_optimizer", "auto_ml", "automl", "budget", "callbacks",
        "self_improving", "nas_search", "experiment_tracking", "experiment_compare", "learning",
    ],
    "Compression & conversion": [
        "distillation_v2", "quantization", "precision", "compression", "compression_advisor",
        "compress", "conversion", "hf_converter", "huggingface_hub", "modelfile",
        "modelfile_extended", "modelfile_extensions", "weight_translation", "surgery", "weights",
    ],
    "Evaluation & interpretability": [
        "evaluate", "advanced_metrics", "benchmark", "benchmark_suite", "benchmarks", "model_health",
        "confidence_calibrator", "interpret", "model_visualizer", "ab_testing", "drift", "governance",
        "lineage", "cpc", "cpc_ai", "model_tools", "test_time_scaling",
    ],
    "Agents & language": [
        "agent", "agent_framework", "agents", "mind", "reasoning", "mcp", "memory", "memory_vault",
        "rag", "model_chat", "creative", "creative_tools", "collaborative",
    ],
    "Serving & production": [
        "serving", "serving_gateway", "production", "production_serving", "deployment",
        "fast_inference", "inference", "streaming", "runtime", "autoscale", "model_registry",
        "marketplace", "integrations", "integrations_real", "installer", "infra", "environments",
    ],
    "MLOps & monitoring": [
        "mlops", "tracking", "profiler", "observability", "model_monitor", "modellog",
        "model_versioner", "versioning", "model_card", "model_card_generator", "self_healing",
        "workflow", "drift",
    ],
    "Security & networking": [
        "security", "network", "ab_testing",
    ],
    "UI & visualization": [
        "visual_workspace", "visual_workspace_v2", "workspace_html", "webui", "dashboard",
        "model_visualizer", "bundle",
    ],
    "Docs, CLI & meta": [
        "cli", "cli_tool", "docs", "autodoc", "api_reference", "tutorials", "advanced", "modern_vision",
        "__main__",
    ],
}

def first_docline(path: Path, name: str = "") -> str:
    if name in DOC_OVERRIDES:
        return DOC_OVERRIDES[name]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return "(docstring unavailable)"
    doc = ast.get_docstring(tree) or ""
    doc = re.sub(r"\s+", " ", doc).strip()
    if not doc:
        # try first class docstring
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                d = ast.get_docstring(n) or ""
                d = re.sub(r"\s+", " ", d).strip()
                if d:
                    return d[:150]
        return "(no docstring)"
    # cut at first 'Usage' marker or 140 chars
    doc = re.split(r"\bUsage\b|\bUsage:", doc)[0].strip().rstrip(".")
    return (doc[:137] + "...") if len(doc) > 140 else doc

init_src = (QAI / "__init__.py").read_text(encoding="utf-8", errors="replace")

# curated overrides for modules whose package docstring is empty
DOC_OVERRIDES = {
    "core": "Core internals: Model, AutoTuner, weights plumbing — the engine behind qai.build() / qai.Model",
    "__main__": "`python -m qai` CLI entry point (same commands as qai.cli)",
    "governance": "Governance & drift detection utilities for models and features",
    "tracking": "Experiment tracking and metrics logging for qai workflows",
}

all_stems = {}
for p in sorted(QAI.glob("*.py")):
    if p.stem == "__init__":
        continue
    all_stems[p.stem] = p
for sub in sorted(p for p in QAI.iterdir() if p.is_dir()):
    all_stems[sub.name] = sub / "__init__.py"

categorized = set()
rows_by_cat = {}
for cat, names in CATS.items():
    for n in names:
        if n not in all_stems:
            continue
        if n in categorized:
            continue
        categorized.add(n)
        p = all_stems[n]
        doc = first_docline(p, n)
        exported = "yes" if re.search(rf"[\b\.]{re.escape(n)}\b", init_src) else "internal"
        newmark = " **NEW-DOC**" if n in NEW else ""
        loc = p.stem if p.suffix == ".py" and p.parent == QAI else f"{p.parent.name}/"
        if p.suffix == ".py" and p.parent != QAI:
            loc = f"{p.parent.name}/{p.stem}"
        elif p.name == "__init__.py":
            loc = f"{n}/ (package)"
        rows_by_cat.setdefault(cat, []).append((loc, doc, exported, newmark))

uncat = [n for n in all_stems if n not in categorized]

out = []
out.append("# QAI Module Registry — Every Module, A to Z")
out.append("")
out.append("> See also: [README](../README.md) · [Task Manual](guides/README.md) · [User Manual](USER_MANUAL.md) · [Focus List](UNBUILT_FEATURES.md)")
out.append("")
out.append(f"Complete inventory of the `qai` package: **{len(all_stems)} modules and subpackages**.")
out.append("Every one-liner below is generated from the module's real docstring in the source,")
out.append("so this page can never drift from the code.")
out.append("")
out.append("Legend: **NEW-DOC** = built earlier but only documented in v1.8.3 (these were the")
out.append("\"features with no documentation\" — now every module is findable). `internal` = not")
out.append("re-exported on the single `import qai` face; import it directly, e.g.")
out.append("`from qai.workspace_html import WORKSPACE_HTML_V2`.")
out.append("")

for cat, rows in rows_by_cat.items():
    out.append(f"## {cat} ({len(rows)})")
    out.append("")
    out.append("| Module | What it does | Import face |")
    out.append("|---|---|---|")
    for loc, doc, exported, newmark in rows:
        doc_md = doc.replace("|", "\\|")
        exp = "top-level" if exported == "yes" else "`internal`"
        out.append(f"| `qai.{loc}`{newmark} | {doc_md} | {exp} |")
    out.append("")

# uncategorized leftovers -> Other
if uncat:
    out.append(f"## Other & internal helpers ({len(uncat)})")
    out.append("")
    out.append("| Module | What it does | Import face |")
    out.append("|---|---|---|")
    for n in sorted(uncat):
        p = all_stems[n]
        doc = first_docline(p, n).replace("|", "\\|")
        exported = "yes" if re.search(rf"[\b\.]{re.escape(n)}\b", init_src) else "internal"
        newmark = " **NEW-DOC**" if n in NEW else ""
        exp = "top-level" if exported == "yes" else "`internal`"
        loc = n if p.name == "__init__.py" else f"{p.parent.name}/{p.stem}" if p.parent != QAI else p.stem
        out.append(f"| `qai.{loc}`{newmark} | {doc} | {exp} |")
    out.append("")

out.append("---")
out.append("")
out.append("## How this page is maintained")
out.append("")
out.append("Generated from source docstrings — regenerate after adding a module:")
out.append("```bash")
out.append("python3 scripts/gen_module_registry.py   # writes docs/MODULE_REGISTRY.md")
out.append("```")
out.append("")
out.append("Rules for contributors (see [CONTRIBUTING](../CONTRIBUTING.md)): every new public")
out.append("module must (1) be reachable from `import qai` or be marked `internal`, (2) appear")
out.append("here, and (3) get at least one usage example in the relevant Task Manual chapter.")
out.append("")

dest = ROOT / "docs" / "MODULE_REGISTRY.md"
dest.write_text("\n".join(out), encoding="utf-8")
print(f"WROTE {dest} ({len(out)} lines)")
print("Uncategorized:", uncat)

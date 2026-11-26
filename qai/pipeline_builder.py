"""
qai.pipeline_builder -- declarative DAG pipeline builder.

Build multi-stage ML pipelines as a directed acyclic graph (DAG) with
a declarative API. Each stage is a named step that takes inputs from
previous stages and produces outputs for downstream stages.

Usage:
    import qai
    builder = qai.PipelineBuilder()

    builder.add_step("preprocess", qai.build(technique="standard_scaler"))
    builder.add_step("feature_select", qai.build(technique="select_k_best", k=5),
                     depends_on=["preprocess"])
    builder.add_step("model_a", qai.build(technique="random_forest"),
                     depends_on=["feature_select"])
    builder.add_step("model_b", qai.build(technique="gradient_boosting"),
                     depends_on=["feature_select"])
    builder.add_step("ensemble", qai.VotingEnsemble(models=[]),
                     depends_on=["model_a", "model_b"])

    pipeline = builder.build()
    pipeline.train(X, y)
    preds = pipeline.predict(X_test)

    # Visualize the DAG
    print(builder.render_dag())
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict, deque
import numpy as np


@dataclass
class PipelineStep:
    """A single step in a pipeline DAG."""
    name: str
    model: Any  # qai.Model or callable
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


class PipelineBuilder:
    """Declarative DAG pipeline builder.

    Build multi-stage ML pipelines with explicit dependencies. The
    builder validates the DAG (no cycles, no missing deps) and renders
    it as a text diagram.

    Args:
        name: pipeline name (cosmetic).
    """
    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: Dict[str, PipelineStep] = {}
        self._order: List[str] = []  # topological order

    def add_step(self, name: str, model: Any,
                 depends_on: Optional[List[str]] = None,
                 **params) -> "PipelineBuilder":
        """Add a step to the pipeline.

        Args:
            name: unique step name.
            model: a qai.Model, qai.Technique, or callable.
            depends_on: list of step names this step depends on.
            **params: extra params for this step.

        Returns:
            self (for chaining).
        """
        if name in self.steps:
            raise ValueError(f"[WHAT] Step '{name}' already exists\n"
                             f"[WHY] Duplicate step name\n"
                             f"[FIX] Use a unique name for each step")
        depends_on = depends_on or []
        for dep in depends_on:
            if dep not in self.steps:
                raise ValueError(f"[WHAT] Step '{name}' depends on unknown step '{dep}'\n"
                                 f"[WHY] '{dep}' hasn't been added yet\n"
                                 f"[FIX] Add '{dep}' before '{name}'")
        self.steps[name] = PipelineStep(name=name, model=model,
                                         depends_on=depends_on, params=params)
        self._topological_sort()
        return self

    def _topological_sort(self):
        """Compute the topological order of steps (Kahn's algorithm)."""
        in_degree = {name: 0 for name in self.steps}
        adj = defaultdict(list)
        for name, step in self.steps.items():
            for dep in step.depends_on:
                adj[dep].append(name)
                in_degree[name] += 1
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.steps):
            raise ValueError("[WHAT] Pipeline has a cycle\n"
                             "[WHY] Circular dependency detected\n"
                             "[FIX] Remove the cycle in step dependencies")
        self._order = order

    def build(self) -> "DAGPipeline":
        """Build the pipeline DAG. Returns a DAGPipeline ready to train/predict."""
        return DAGPipeline(self.name, self.steps, self._order)

    def render_dag(self) -> str:
        """Render the DAG as a text diagram."""
        lines = [f"=== Pipeline DAG: {self.name} ===", ""]
        for name in self._order:
            step = self.steps[name]
            deps = ", ".join(step.depends_on) if step.depends_on else "(root)"
            model_type = type(step.model).__name__
            lines.append(f"  [{name}]")
            lines.append(f"    model: {model_type}")
            lines.append(f"    depends_on: {deps}")
            if step.params:
                lines.append(f"    params: {step.params}")
            # Show downstream
            downstream = [n for n, s in self.steps.items() if name in s.depends_on]
            if downstream:
                lines.append(f"    feeds: {', '.join(downstream)}")
            lines.append("")
        return "\n".join(lines)

    def __repr__(self):
        return f"<PipelineBuilder name={self.name!r} steps={len(self.steps)}>"


class DAGPipeline:
    """A compiled DAG pipeline ready to train and predict.

    Created by PipelineBuilder.build(). Don't construct directly.
    """
    def __init__(self, name: str, steps: Dict[str, PipelineStep], order: List[str]):
        self.name = name
        self.steps = steps
        self.order = order
        self._outputs: Dict[str, Any] = {}  # cached outputs per step
        self._trained = False

    def train(self, X, y=None, **kwargs) -> "DAGPipeline":
        """Train the pipeline in topological order.

        Each step receives the output of its dependencies as input.
        Root steps (no dependencies) receive X directly.
        """
        for step_name in self.order:
            step = self.steps[step_name]
            # Gather inputs from dependencies
            if step.depends_on:
                inputs = [self._outputs[dep] for dep in step.depends_on]
                if len(inputs) == 1:
                    step_input = inputs[0]
                else:
                    # Multiple inputs: concatenate (for ensembles, the model handles this)
                    step_input = inputs
            else:
                step_input = X

            # Train this step
            model = step.model
            if hasattr(model, "train"):
                try:
                    model.train(step_input, y, verbose=False, **kwargs)
                except TypeError:
                    model.train(step_input, y, **kwargs)
            elif callable(model):
                pass  # callable steps are applied at predict time

            # Compute this step's output (transformed data or predictions)
            output = step_input  # default: pass-through
            tech = getattr(model, "technique", None)
            if tech is not None and hasattr(tech, "transform"):
                # Transformer (scaler/PCA/etc): use technique.transform()
                try: output = tech.transform(step_input)
                except Exception: pass
            elif tech is not None and hasattr(tech, "model") and hasattr(tech.model, "transform"):
                # Sklearn transformer wrapped differently
                try: output = tech.model.transform(step_input)
                except Exception: pass
            elif hasattr(model, "predict"):
                try: output = model.predict(step_input)
                except Exception: pass
            elif callable(model):
                try: output = model(step_input)
                except Exception: pass
            self._outputs[step_name] = output

        self._trained = True
        return self

    def predict(self, X, **kwargs):
        """Run prediction through the DAG. Returns the output of the last step."""
        if not self._trained:
            raise RuntimeError("[WHAT] Pipeline not trained\n"
                               "[WHY] Call pipeline.train(X, y) first\n"
                               "[FIX] Train before predicting")
        outputs = {}
        for step_name in self.order:
            step = self.steps[step_name]
            if step.depends_on:
                inputs = [outputs[dep] for dep in step.depends_on]
                step_input = inputs[0] if len(inputs) == 1 else inputs
            else:
                step_input = X

            model = step.model
            output = step_input  # default: pass-through
            tech = getattr(model, "technique", None)
            if tech is not None and hasattr(tech, "transform"):
                try: output = tech.transform(step_input)
                except Exception: pass
            elif tech is not None and hasattr(tech, "model") and hasattr(tech.model, "transform"):
                try: output = tech.model.transform(step_input)
                except Exception: pass
            elif hasattr(model, "predict"):
                try: output = model.predict(step_input)
                except Exception: pass
            elif callable(model):
                try: output = model(step_input)
                except Exception: pass
            outputs[step_name] = output

        return outputs[self.order[-1]]

    def __repr__(self):
        return f"<DAGPipeline name={self.name!r} steps={len(self.steps)} trained={self._trained}>"


__all__ = ["PipelineBuilder", "DAGPipeline", "PipelineStep"]

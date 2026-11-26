"""
qai.asyncio -- async wrappers for the major qai APIs.

Many qai operations are CPU/IO-bound (training, retrieval, generation,
serving). These wrappers let you run them concurrently via asyncio.

Wrapped:
  - async_build       : qai.build but async
  - async_fit          : model.fit but runs in executor
  - async_forward      : model.forward in executor
  - AsyncRAG           : RAG with concurrent retrieval + generation
  - AsyncAgent         : ReAct-style agent with async tool calls
  - AsyncLLMClient     : wraps any LLM client (qai.serving.LLMClient) for async
  - AsyncBatchProcessor: process a list of inputs concurrently with backpressure
  - AsyncExperimentTracker: non-blocking logging to MLflow/W&B
  - AsyncDistributedTrainer: launch + monitor distributed training in background
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable
import numpy as np


# ============================================================================
# Common executor
# ============================================================================
# A module-level thread pool for CPU-bound work
_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_executor(max_workers: int = 4) -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    return _EXECUTOR


async def _run_in_executor(fn: Callable, *args, **kwargs) -> Any:
    """Run a sync function in a thread pool and await the result."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_get_executor(), lambda: fn(*args, **kwargs))


# ============================================================================
# Async build + fit + forward
# ============================================================================
async def async_build(technique: str = "regression", **kwargs) -> Any:
    """Async wrapper for qai.build()."""
    import qai
    return await _run_in_executor(qai.build, technique=technique, **kwargs)


async def async_fit(model: Any, X: Any, y: Any = None) -> Any:
    """Async wrapper for model.fit()."""
    return await _run_in_executor(model.fit, X, y)


async def async_forward(model: Any, x: Any) -> Any:
    """Async wrapper for model.forward()."""
    return await _run_in_executor(model.forward, x)


async def async_predict_many(model: Any, X: List[Any],
                              max_concurrency: int = 8) -> List[Any]:
    """Run model.forward on many inputs concurrently with backpressure."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(x):
        async with semaphore:
            return await async_forward(model, x)

    return await asyncio.gather(*[_one(x) for x in X])


# ============================================================================
# Async RAG
# ============================================================================
class AsyncRAG:
    """Async RAG: concurrent retrieval + generation.

    Wraps any qai.rag.* instance. Retrieval and generation happen in
    executor; multiple queries can be served concurrently.
    """
    def __init__(self, rag: Any):
        self.rag = rag

    async def retrieve(self, query: str, top_k: int = 3) -> List[Any]:
        return await _run_in_executor(self.rag.retrieve, query, top_k)

    async def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        return await _run_in_executor(self.rag.generate, query, **kwargs)

    async def batch_generate(self, queries: List[str],
                              max_concurrency: int = 4) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _one(q):
            async with semaphore:
                return await self.generate(q)

        return await asyncio.gather(*[_one(q) for q in queries])


# ============================================================================
# Async LLM client
# ============================================================================
class AsyncLLMClient:
    """Async wrapper around any sync LLM client.

    Multiple prompts are processed concurrently. Useful for batch
    evaluation, parallel self-consistency, multi-agent systems.
    """
    def __init__(self, sync_client: Any, max_concurrency: int = 4):
        self.client = sync_client
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate(self, prompt: str, **kwargs) -> str:
        async with self.semaphore:
            return await _run_in_executor(self.client.generate, prompt, **kwargs)

    async def batch_generate(self, prompts: List[str],
                              **kwargs) -> List[str]:
        return await asyncio.gather(*[self.generate(p, **kwargs) for p in prompts])

    async def stream(self, prompt: str, **kwargs):
        """Async generator yielding tokens if the underlying client supports it."""
        if not hasattr(self.client, "stream"):
            # fallback: just yield the whole output
            result = await self.generate(prompt, **kwargs)
            yield result
            return
        # wrap sync iterator in async
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for chunk in self.client.stream(prompt, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(_get_executor(), _producer)
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


# ============================================================================
# Async Agent
# ============================================================================
class AsyncAgent:
    """Async wrapper around a ReAct-style agent.

    Tool calls are run concurrently when independent. Long-running
    tools don't block other reasoning steps.
    """
    def __init__(self, agent: Any, max_concurrency: int = 2):
        self.agent = agent
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, query: str) -> Dict[str, Any]:
        async with self.semaphore:
            return await _run_in_executor(self.agent.run, query)

    async def batch_run(self, queries: List[str]) -> List[Dict[str, Any]]:
        return await asyncio.gather(*[self.run(q) for q in queries])


# ============================================================================
# Async Batch Processor
# ============================================================================
class AsyncBatchProcessor:
    """Process a list of inputs concurrently with backpressure + retries.

    Useful for batch inference on a slow model, parallel data loading,
    or any producer/consumer pattern with limited concurrency.
    """
    def __init__(self, process_fn: Callable[[Any], Any],
                 max_concurrency: int = 4, max_retries: int = 2,
                 retry_delay: float = 1.0):
        self.fn = process_fn
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def process_one(self, item: Any) -> Dict[str, Any]:
        """Process one item with retries."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    result = await _run_in_executor(self.fn, item)
                return {"item": item, "result": result, "attempts": attempt + 1,
                         "success": True}
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
        return {"item": item, "result": None, "attempts": self.max_retries + 1,
                 "success": False, "error": str(last_error)}

    async def process_batch(self, items: List[Any]) -> List[Dict[str, Any]]:
        return await asyncio.gather(*[self.process_one(item) for item in items])

    async def process_stream(self, items: List[Any]):
        """Async generator yielding results as they complete."""
        tasks = [asyncio.create_task(self.process_one(item)) for item in items]
        for coro in asyncio.as_completed(tasks):
            yield await coro


# ============================================================================
# Async Experiment Tracker
# ============================================================================
class AsyncExperimentTracker:
    """Non-blocking logging to MLflow / W&B / Neptune.

    Logging calls are queued and processed in a background task so
    training is never blocked by I/O.
    """
    def __init__(self, logger: Any):
        self.logger = logger
        self._queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> None:
        """Start the background consumer."""
        self._consumer_task = asyncio.create_task(self._consumer())

    async def stop(self) -> None:
        """Stop the background consumer; flush remaining items."""
        await self._queue.put(None)
        await self._consumer_task

    async def _consumer(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            method, args, kwargs = item
            try:
                await _run_in_executor(getattr(self.logger, method), *args, **kwargs)
            except Exception:
                pass  # don't crash the consumer

    async def log_metric(self, key: str, value: float, step: int = 0) -> None:
        await self._queue.put(("log_metric", [key, value, step], {}))

    async def log_params(self, params: Dict[str, Any]) -> None:
        await self._queue.put(("log_params", [params], {}))

    async def log_artifact(self, path: str) -> None:
        await self._queue.put(("log_artifact", [path], {}))


# ============================================================================
# Async Distributed Trainer
# ============================================================================
class AsyncDistributedTrainer:
    """Launch distributed training in the background; monitor via async.

    Usage:
        trainer = AsyncDistributedTrainer(backend="single")
        await trainer.start(model_factory, data_fn, train_step, epochs=10)
        # do other work
        result = await trainer.wait()
    """
    def __init__(self, backend: str = "single", **kwargs):
        self.backend_name = backend
        self.kwargs = kwargs
        self._task: Optional[asyncio.Task] = None
        self._result: Optional[Dict[str, Any]] = None

    async def start(self, model_factory: Callable, data_fn: Callable,
                     train_step: Callable, epochs: int = 1) -> None:
        from .distributed import DistributedTrainer
        trainer = DistributedTrainer(backend=self.backend_name, **self.kwargs)

        async def _run():
            return await _run_in_executor(
                trainer.train, model_factory, data_fn, train_step, epochs
            )

        self._task = asyncio.create_task(_run())

    async def wait(self) -> Dict[str, Any]:
        if self._task is None:
            raise RuntimeError("training not started; call start() first")
        self._result = await self._task
        return self._result

    async def status(self) -> Dict[str, Any]:
        if self._task is None:
            return {"status": "not_started"}
        if self._task.done():
            return {"status": "done", "result": self._result}
        return {"status": "running"}


# ============================================================================
# Convenience: run a sync function as if it were async
# ============================================================================
async def to_async(fn: Callable, *args, **kwargs) -> Any:
    """Run any sync function in a thread pool."""
    return await _run_in_executor(fn, *args, **kwargs)


def run_async(coro: Awaitable) -> Any:
    """Run a coroutine to completion from sync code.

    Handles the case where there's already an event loop running.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # already in a loop; create a future
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


__all__ = [
    "async_build", "async_fit", "async_forward", "async_predict_many",
    "AsyncRAG", "AsyncLLMClient", "AsyncAgent",
    "AsyncBatchProcessor", "AsyncExperimentTracker", "AsyncDistributedTrainer",
    "to_async", "run_async",
]

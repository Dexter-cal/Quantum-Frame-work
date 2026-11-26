"""
qai.distributed_train — multi-GPU and distributed training utilities.

Provides:
  - DistributedTrainer — wraps qai.Trainer with DDP (DistributedDataParallel)
  - init_distributed() — initialize the distributed process group
  - cleanup_distributed() — tear down the process group
  - get_world_size(), get_rank(), is_main_process() — helpers
  - FSDPTrainer — Fully Sharded Data Parallel for very large models

Usage (single-machine multi-GPU):
    # Launch with: torchrun --nproc_per_node=4 script.py
    import qai
    qai.init_distributed()
    m = qai.build(technique="transformer", ...)
    trainer = qai.DistributedTrainer(model=m, train_data=X, train_targets=y, ...)
    trainer.train(epochs=10)
    qai.cleanup_distributed()

Usage (FSDP for 70B+ models):
    trainer = qai.FSDPTrainer(model=m, train_data=X, ...)
    trainer.train(epochs=3)
"""
from __future__ import annotations
import os, sys, math, time, json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch():
    if not _TORCH:
        raise ImportError("[WHAT] Distributed training requires PyTorch\n[FIX] pip install torch")


# =====================================================================
# Process group management
# =====================================================================
def init_distributed(backend: str = "nccl"):
    """Initialize the distributed process group.

    Call this at the start of your training script. Use `torchrun` or
    `torch.distributed.launch` to start the process group.

    Args:
        backend: "nccl" (GPU, default), "gloo" (CPU), or "mpi"
    """
    _require_torch()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if dist.is_initialized():
        return  # already initialized
    # Detect from environment variables (set by torchrun)
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Single-process mode — no-op
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Set the current device (for nccl)
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Tear down the distributed process group."""
    _require_torch()
    if dist.is_initialized():
        dist.destroy_process_group()


def get_world_size() -> int:
    """Get the total number of processes."""
    if not _TORCH or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process (0 = main)."""
    if not _TORCH or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """True if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if _TORCH and dist.is_initialized():
        dist.barrier()


def all_reduce(tensor, op="sum"):
    """All-reduce a tensor across processes."""
    if not _TORCH or not dist.is_initialized():
        return tensor
    if op == "sum":
        op_enum = dist.ReduceOp.SUM
    elif op == "mean":
        op_enum = dist.ReduceOp.SUM
        # For mean, divide by world_size after
    elif op == "max":
        op_enum = dist.ReduceOp.MAX
    else:
        raise ValueError(f"Unknown reduce op: {op}")
    dist.all_reduce(tensor, op=op_enum)
    if op == "mean":
        tensor /= get_world_size()
    return tensor


def all_gather(tensor) -> torch.Tensor:
    """All-gather a tensor across processes. Returns a concatenated tensor."""
    if not _TORCH or not dist.is_initialized():
        return tensor
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)


# =====================================================================
# Distributed Sampler
# =====================================================================
class DistributedSampler:
    """Simple distributed sampler — splits data across processes.

    For torch.utils.data.Dataset, use torch.utils.data.distributed.
    DistributedSampler instead. This is a simplified version for plain
    arrays.
    """
    def __init__(self, n_samples: int, shuffle: bool = True, seed: int = 0):
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.world_size = get_world_size()
        self.rank = get_rank()
        # Pad to be divisible by world_size
        self.num_samples = math.ceil(n_samples / self.world_size)
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        indices = list(range(self.n_samples))
        # Add extra samples to make divisible
        padding = self.total_size - len(indices)
        if padding > 0:
            indices += indices[:padding]
        # Shuffle
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# =====================================================================
# Distributed Trainer (DDP)
# =====================================================================
class DistributedTrainer:
    """Distributed Data Parallel Trainer.

    Wraps a model with torch.nn.parallel.DistributedDataParallel and trains
    across multiple GPUs. Each process gets a subset of the data; gradients
    are synchronized automatically by DDP.

    Launch with:
        torchrun --nproc_per_node=4 your_script.py

    Or programmatically with torch.multiprocessing.spawn.

    Args:
        model: a qai.Model (will be moved to current CUDA device)
        train_data: training inputs (will be split across processes)
        train_targets: training targets
        eval_data: optional eval inputs
        eval_targets: optional eval targets
        optimizer, lr, lr_schedule, ...: same as qai.Trainer
        sync_grads: if True (default), gradients are synchronized (DDP default)
        find_unused_params: if True, DDP allows unused parameters (slower)

    Usage:
        import qai
        qai.init_distributed()
        m = qai.build(technique="transformer", ...)
        trainer = qai.DistributedTrainer(
            model=m, train_data=X, train_targets=y,
            optimizer="adamw", lr=3e-4, batch_size=32,
            output_dir="./checkpoints",
        )
        trainer.train(epochs=10)
        qai.cleanup_distributed()
    """
    def __init__(self, model, train_data, train_targets=None,
                 eval_data=None, eval_targets=None,
                 optimizer: str = "adamw", lr: float = 3e-4,
                 lr_schedule: str = "cosine", warmup_steps: int = 0,
                 weight_decay: float = 0.01, max_grad_norm: float = 1.0,
                 batch_size: int = 32, grad_accum_steps: int = 1,
                 fp16: bool = False, bf16: bool = False,
                 output_dir: str = "./checkpoints",
                 save_steps: int = 0, eval_steps: int = 0,
                 log_every: int = 50, seed: int = 42,
                 verbose: bool = True, find_unused_params: bool = False):
        _require_torch()
        self.model = model
        self.train_data = train_data
        self.train_targets = train_targets
        self.eval_data = eval_data
        self.eval_targets = eval_targets
        self.optimizer_name = optimizer
        self.lr = lr
        self.lr_schedule_name = lr_schedule
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.fp16 = fp16
        self.bf16 = bf16
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.log_every = log_every
        self.seed = seed
        self.verbose = verbose and is_main_process()
        self.world_size = get_world_size()
        self.rank = get_rank()

        # Move model to current device
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self._move_to_device()

        # Wrap with DDP if distributed
        self._ddp_model = None
        if dist.is_initialized() and self.world_size > 1:
            tech = model.technique
            # Find the inner nn.Module to wrap
            inner_module = self._find_inner_module(tech)
            if inner_module is not None:
                self._ddp_model = DDP(inner_module, device_ids=[self.rank] if torch.cuda.is_available() else None,
                                       find_unused_parameters=find_unused_params)

        # Create sampler
        n_samples = len(train_data) if hasattr(train_data, "__len__") else 1
        self.sampler = DistributedSampler(n_samples, shuffle=True, seed=seed)

        # Create optimizer
        from .trainer import make_optimizer, SCHEDULES
        self.optimizer = make_optimizer(model, optimizer, lr, weight_decay)
        total_steps = (n_samples // (batch_size * self.world_size)) * 10  # rough
        sched_cls = SCHEDULES.get(lr_schedule, SCHEDULES["cosine"])
        self.scheduler = sched_cls(self.optimizer, lr, max(total_steps, 1), warmup_steps)

        # Mixed precision
        self.use_amp = fp16 or bf16
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        self.scaler = torch.amp.GradScaler('cuda') if (fp16 and torch.cuda.is_available()) else None

        # State
        self.global_step = 0
        self.history = []
        os.makedirs(output_dir, exist_ok=True)

    def _move_to_device(self):
        """Move the model's torch submodules to the current device."""
        tech = self.model.technique
        for attr in vars(tech).values():
            if isinstance(attr, nn.Module):
                attr.to(self.device)
            elif isinstance(attr, (list, tuple)):
                for x in attr:
                    if isinstance(x, nn.Module):
                        x.to(self.device)

    def _find_inner_module(self, tech):
        """Find the main torch.nn.Module to wrap with DDP."""
        for attr_name in vars(tech):
            attr = getattr(tech, attr_name)
            if isinstance(attr, nn.Module) and not isinstance(attr, nn.Sequential):
                return attr
            if isinstance(attr, nn.Module):
                return attr
        return None

    def train(self, epochs: int = 1):
        """Run distributed training."""
        from .trainer import _nullcontext
        n_samples = len(self.train_data) if hasattr(self.train_data, "__len__") else 1
        steps_per_epoch = max(1, len(self.sampler) // self.batch_size)
        if self.verbose:
            print(f"[qai.DistributedTrainer] world_size={self.world_size}, "
                  f"steps_per_epoch={steps_per_epoch}")

        for epoch in range(epochs):
            self.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            n_batches = 0
            t_epoch = time.time()

            for batch_idx, sample_idx in enumerate(self.sampler):
                # Get batch
                X_batch = self.train_data[sample_idx]
                y_batch = self.train_targets[sample_idx] if self.train_targets is not None else None

                # Forward + backward
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype) if self.use_amp else _nullcontext():
                    loss = self._compute_loss(X_batch, y_batch)
                    loss = loss / self.grad_accum_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        # Get params from DDP model
                        params = list(self._ddp_model.parameters()) if self._ddp_model else self._get_params()
                        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1
                    epoch_loss += float(loss.item()) * self.grad_accum_steps
                    n_batches += 1

                    if self.verbose and self.global_step % self.log_every == 0:
                        cur_lr = self.optimizer.param_groups[0]["lr"]
                        print(f"  epoch {epoch+1}/{epochs} step {self.global_step} "
                              f"loss={float(loss.item()):.4f} lr={cur_lr:.2e}")

            # All-reduce loss across processes
            avg_loss = epoch_loss / max(n_batches, 1)
            if dist.is_initialized():
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                all_reduce(loss_tensor, op="mean")
                avg_loss = float(loss_tensor.item())

            epoch_time = time.time() - t_epoch
            log = {"epoch": epoch + 1, "step": self.global_step,
                   "train_loss": avg_loss, "epoch_time_s": epoch_time}
            self.history.append(log)
            if self.verbose:
                print(f"[qai.DistributedTrainer] epoch {epoch+1} done — "
                      f"loss={avg_loss:.4f} time={epoch_time:.1f}s")

            # Save checkpoint (only on main process)
            if is_main_process():
                self._save_checkpoint(f"epoch_{epoch+1}")

        # Final save
        if is_main_process():
            self._save_log()
        return self.model

    def _compute_loss(self, X_batch, y_batch):
        """Compute loss for a single sample."""
        from .trainer import _nullcontext
        tech = self.model.technique
        if hasattr(tech, "compute_loss"):
            return tech.compute_loss(X_batch, y_batch)
        # Generic
        X_t = torch.as_tensor(np.asarray(X_batch), dtype=torch.float32, device=self.device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(0)
        # Try forward through inner module
        if self._ddp_model is not None:
            try:
                if y_batch is not None:
                    y_t = torch.as_tensor(np.asarray(y_batch), dtype=torch.long, device=self.device)
                    logits = self._ddp_model(X_t)
                    if logits.dim() > 1 and logits.shape[0] != y_t.shape[0]:
                        logits = logits.view(y_t.shape[0], -1)
                    return F.cross_entropy(logits, y_t)
                else:
                    return torch.tensor(0.0, requires_grad=True, device=self.device)
            except Exception:
                return torch.tensor(0.0, requires_grad=True, device=self.device)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def _get_params(self):
        tech = self.model.technique
        params = []
        if hasattr(tech, "parameters"):
            try:
                params = list(tech.parameters())
            except Exception:
                pass
        if not params:
            for attr in vars(tech).values():
                if isinstance(attr, nn.Module):
                    params += list(attr.parameters())
        return [p for p in params if p.requires_grad]

    def _save_checkpoint(self, name: str):
        if not is_main_process():
            return
        try:
            path = os.path.join(self.output_dir, f"{name}.pt")
            import qai
            qai.save_model(self.model, path)
        except Exception as e:
            if self.verbose:
                print(f"[qai.DistributedTrainer] checkpoint save failed: {e}")

    def _save_log(self):
        try:
            path = os.path.join(self.output_dir, "training_log.json")
            with open(path, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception:
            pass


# =====================================================================
# FSDP Trainer — for very large models (70B+)
# =====================================================================
class FSDPTrainer(DistributedTrainer):
    """Fully Sharded Data Parallel Trainer.

    Like DDP but shards model parameters, gradients, and optimizer states
    across processes. Enables training models that don't fit on a single GPU.

    Requires: PyTorch 2.0+ (for stable FSDP API)

    Usage:
        import qai
        qai.init_distributed()
        m = qai.build(technique="llama_decoder", dim=4096, ...)
        trainer = qai.FSDPTrainer(model=m, train_data=X, ...)
        trainer.train(epochs=3)
    """
    def __init__(self, *args, **kwargs):
        _require_torch()
        if not hasattr(torch.distributed, "fsdp"):
            raise ImportError(
                "[WHAT] FSDP requires PyTorch 2.0+\n"
                "[FIX] pip install --upgrade torch")
        super().__init__(*args, **kwargs)
        # Wrap with FSDP instead of DDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        tech = self.model.technique
        inner_module = self._find_inner_module(tech)
        if inner_module is not None and dist.is_initialized():
            self._ddp_model = FSDP(inner_module,
                                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                                    device_id=self.rank if torch.cuda.is_available() else None,
                                    use_orig_params=True)


__all__ = ["DistributedTrainer", "FSDPTrainer", "DistributedSampler",
           "init_distributed", "cleanup_distributed",
           "get_world_size", "get_rank", "is_main_process",
           "barrier", "all_reduce", "all_gather"]

# Guide: Distributed & Federated Training

*One machine isn't enough, or the data can't leave where it lives.*

> Back to [Guides index](README.md) · Prev: [COMPRESS.md](COMPRESS.md) · Related: [TRAINING.md](TRAINING.md)

---

## What you'll learn

1. Data-parallel training on multiple GPUs (DDP)
2. Fully-sharded training (FSDP) for models bigger than one GPU
3. Federated learning — train across clients without centralizing data

---

## 1. DDP — Distributed Data Parallel

```python
from qai.core.distributed import init_distributed_context, DistributedDataParallelWrapper

init_distributed_context(world_size=4, rank=int(os.environ["RANK"]))
wrapped = DistributedDataParallelWrapper(model)
wrapped.train(X, y)   # gradients all-reduced across ranks each step
```

Each GPU holds a full model copy and a different data shard. Standard choice
for models that fit on one GPU.

## 2. FSDP — Fully Sharded Data Parallel

```python
# For models that DON'T fit on one GPU:
# parameters, gradients, and optimizer states are sharded across GPUs
wrapped = DistributedDataParallelWrapper(model, strategy="fsdp")
```

## 3. Federated learning (`qai.federation`)

**What:** many clients (hospitals, phones, banks) each hold private data. The
server never sees the data — only weight updates. Added in v1.8.0.

### Quick run

```python
from qai.federation import quick, create_federation

# Auto-split data across N clients, run N rounds
result = quick(model_template, X, y, n_rounds=2, n_clients=4)
print(result.to_markdown())   # full round-by-round federation report
```

### Full control

```python
from qai.federation import (
    FederatedServer, Client, FederationConfig,
    FedAvgAggregator, FedProxAggregator, WeightedAggregator,
)

config = FederationConfig(
    n_rounds=10,
    clients_per_round=4,
    client_epochs=2,
    lr=1e-3,
    batch_size=32,
    strategy="fedavg",          # or "fedprox" / "weighted"
    # Differential privacy (DP-SGD):
    dp_epsilon=8.0,
    dp_delta=1e-5,
    dp_max_grad_norm=1.0,
    secure_aggregation=True,    # server sees aggregated update only
    client_selection="random",  # or "best" / "worst" / "weighted"
)

server = FederatedServer(model_template, config=config)
clients = create_federation(model_template, n_clients=10, X, y)  # auto-splits data

for stats in server.run(clients):
    print(stats.avg_loss, stats.avg_acc, stats.global_acc, stats.weight_delta)
    # weight_delta = L2 distance from previous global weights (convergence signal)
```

### The three aggregation strategies

| Strategy | How it aggregates | When |
|---|---|---|
| `FedAvgAggregator` | weighted mean of client updates, weighted by sample count | default, balanced data |
| `FedProxAggregator` | FedAvg + proximal term client-side (clients stay near global weights) | heterogeneous client data (non-IID) |
| `WeightedAggregator` | custom per-client weights | you know client reliability |

### Privacy controls

- **DP-SGD** — Gaussian noise added to client updates, budgeted by epsilon/delta, gradients clipped to `max_grad_norm`
- **Secure aggregation** — the server only sees the *sum* of client updates
- **Client selection** — random / best / worst / sample-count-weighted per round

### Round statistics

Every round reports: `avg_loss`, `avg_acc`, `global_acc`, `weight_delta`,
`wall_time`. `to_markdown()` renders the whole run as a table.

---

## Choosing a strategy

| Situation | Use |
|---|---|
| Model fits on 1 GPU, have 4 GPUs | DDP |
| Model bigger than 1 GPU | FSDP |
| Data can't leave the clients (privacy/legal) | Federation + DP-SGD + secure aggregation |
| Federation clients are very non-IID | FedProx strategy |
| Just want it to run | `qai.federation.quick(...)` |

---

## Cheat sheet

| Task | Call |
|---|---|
| Multi-GPU training | `init_distributed_context(...)` + `DistributedDataParallelWrapper` |
| Huge model | `DistributedDataParallelWrapper(model, strategy="fsdp")` |
| Federated quickstart | `qai.federation.quick(model, X, y, n_rounds=2, n_clients=4)` |
| Split data to clients | `create_federation(model, n_clients=10, X, y)` |
| Configure everything | `FederationConfig(...)` |
| DP-SGD privacy | `dp_epsilon=8.0, dp_max_grad_norm=1.0` |
| Non-IID clients | `strategy="fedprox"` |
| Federation report | `result.to_markdown()` |

---

## Common gotchas

- **FedAvg on non-IID data drifts.** That's what FedProx fixes — the proximal term penalizes clients wandering from the global weights.
- **DP noise has a cost.** epsilon 8.0 is a reasonable utility/privacy trade-off; epsilon 1.0 protects more, learns less.
- **`weight_delta` trending to zero means convergence** — stop early and save the rounds.
- **Rank and world_size must match your launcher** (torchrun sets `RANK`/`WORLD_SIZE` env vars automatically).

---

*Next: [AGENTS_MCP.md](AGENTS_MCP.md) or back to [Guides index](README.md).*

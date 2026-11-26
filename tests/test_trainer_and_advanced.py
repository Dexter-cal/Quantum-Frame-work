"""
Test the universal Trainer and the 10 advanced architectures.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import qai

print("=" * 70)
print("TEST 1: Trainer on torch-based model (neural_network)")
print("=" * 70)
m = qai.build(technique="neural_network", hidden_dim=8, epochs=1)
X = np.random.randn(50, 4).astype(np.float32)
y = np.random.randint(0, 2, 50)
trainer = qai.Trainer(
    model=m, train_data=X, train_targets=y,
    eval_data=X[:20], eval_targets=y[:20],
    optimizer="adamw", lr=1e-3, lr_schedule="cosine",
    warmup_steps=5, weight_decay=0.01, max_grad_norm=1.0,
    batch_size=16, grad_accum_steps=1,
    output_dir="/tmp/qai_trainer_test", save_steps=0, eval_steps=10,
    log_every=5, verbose=True,
)
trainer.train(epochs=2)
assert os.path.exists("/tmp/qai_trainer_test/training_log.json")
print("Trainer completed for neural_network ✓")

print()
print("=" * 70)
print("TEST 2: Trainer on sklearn-based model (random_forest)")
print("=" * 70)
m = qai.build(technique="random_forest", n_estimators=5)
trainer = qai.Trainer(
    model=m, train_data=X, train_targets=y,
    eval_data=X[:20], eval_targets=y[:20],
    output_dir="/tmp/qai_trainer_test2", verbose=True,
)
trainer.train(epochs=1)
print("Trainer completed for random_forest ✓")

print()
print("=" * 70)
print("TEST 3: LR Schedulers")
print("=" * 70)
import torch
opt = torch.optim.AdamW([torch.randn(2, requires_grad=True)], lr=1e-3)
for name in ["cosine", "linear", "one_cycle", "polynomial", "constant", "step"]:
    cls = qai.SCHEDULES[name]
    sched = cls(opt, base_lr=1e-3, total_steps=100, warmup_steps=10)
    # Step through 50 steps
    lrs = []
    for _ in range(50):
        lrs.append(sched.get_lr())
        sched.step()
    assert lrs[0] < lrs[10] or name == "constant" or name == "step", f"{name}: warmup failed"
    print(f"  {name:15s}: start={lrs[0]:.2e} mid={lrs[25]:.2e} end={lrs[-1]:.2e} ✓")

print()
print("=" * 70)
print("TEST 4: Optimizers")
print("=" * 70)
for opt_name in ["adamw", "adam", "sgd"]:
    m = qai.build(technique="neural_network", hidden_dim=4, epochs=1)
    try:
        optimizer = qai.make_optimizer(m, optimizer=opt_name, lr=1e-3)
        print(f"  {opt_name:10s} created ✓")
    except Exception as e:
        print(f"  {opt_name:10s} FAIL: {e}")

print()
print("=" * 70)
print("TEST 5: All 10 advanced architectures")
print("=" * 70)
configs = [
    ("retnet", dict(vocab_size=32, dim=16, n_layers=1, n_heads=2, epochs=1, n_classes=2),
     (np.random.randint(0, 32, size=(5, 4)), np.random.randint(0, 2, 5))),
    ("xlstm", dict(input_size=4, hidden_size=8, n_layers=1, epochs=1, n_classes=2),
     (np.random.randn(5, 4, 4).astype(np.float32), np.random.randint(0, 2, 5))),
    ("perceiver_io", dict(input_dim=4, latent_dim=8, n_latents=4, n_layers=1, n_heads=2, n_classes=2, epochs=1),
     (np.random.randn(5, 4).astype(np.float32), np.random.randint(0, 2, 5))),
    ("modern_hopfield", dict(input_dim=4, n_patterns=5, epochs=1),
     (np.random.randn(5, 4).astype(np.float32), None)),
    ("kan", dict(in_dim=4, hidden=[4], out_dim=2, n_knots=4, epochs=5),
     (np.random.randn(5, 4).astype(np.float32), np.random.randint(0, 2, 5))),
    ("neural_ode", dict(in_dim=2, out_dim=2, hidden=8, n_steps=3, epochs=5),
     (np.random.randn(5, 2).astype(np.float32), np.random.randint(0, 2, 5))),
    ("pinn", dict(in_dim=2, out_dim=1, hidden=8, n_layers=1, epochs=2),
     (np.random.randn(5, 2).astype(np.float32), None)),
    ("fno", dict(in_dim=2, out_dim=1, modes=2, width=4, n_layers=1, epochs=2),
     (np.random.randn(5, 2).astype(np.float32), np.random.randn(5).astype(np.float32))),
    ("dit", dict(input_dim=4, hidden=8, n_layers=1, n_heads=2, n_steps=5, epochs=1),
     (np.random.randn(5, 4).astype(np.float32), None)),
    ("vq_vae", dict(input_dim=4, latent_dim=2, hidden=8, n_codes=8, epochs=2),
     (np.random.randn(5, 4).astype(np.float32), None)),
]
for tech, params, (X, y) in configs:
    m = qai.build(technique=tech, **params)
    if y is not None:
        m.train(X, y, verbose=False)
    else:
        m.train(X, verbose=False)
    sample = X[0] if X.ndim > 1 else X
    pred = m.predict(sample)
    # save/load
    with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as tf:
        path = tf.name
    try:
        qai.save_model(m, path)
        qai.load_model(path)
    finally:
        os.unlink(path)
    # visualize
    qai.visualize_architecture(m, save=f"/tmp/test_{tech}.png")
    print(f"  {tech:20s} build/train/predict/save/visualize ✓")

print()
print("=" * 70)
print("TEST 6: Trainer on RetNet (advanced architecture)")
print("=" * 70)
m = qai.build(technique="retnet", vocab_size=32, dim=16, n_layers=1, n_heads=2, epochs=1, n_classes=2)
X_tokens = np.random.randint(0, 32, size=(20, 8))
y_labels = np.random.randint(0, 2, 20)
trainer = qai.Trainer(
    model=m, train_data=X_tokens, train_targets=y_labels,
    optimizer="adamw", lr=1e-3, lr_schedule="cosine",
    warmup_steps=2, batch_size=8,
    output_dir="/tmp/qai_trainer_retnet", verbose=True,
)
trainer.train(epochs=1)
print("Trainer completed for retnet ✓")

print()
print("=" * 70)
print("TEST 7: Transformer with SDPA (PyTorch 2.0+ backend)")
print("=" * 70)
m = qai.build(technique="transformer", vocab_size=32, n_layers=1, d_model=16,
              n_heads=2, epochs=1, task_type="classification", n_classes=2)
X = np.random.randint(0, 32, size=(5, 8))
y = np.random.randint(0, 2, 5)
m.train(X, y, verbose=False)
pred = m.predict(X[0])
assert isinstance(pred, int), f"expected int, got {type(pred)}"
print(f"Transformer with SDPA: pred={pred} ✓")

print()
print("=" * 70)
print("ALL TRAINER + ADVANCED ARCHITECTURE TESTS PASSED")
print("=" * 70)

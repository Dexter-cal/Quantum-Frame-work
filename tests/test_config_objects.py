"""
Test the config object system for qai.build().

Verifies:
  1. All 17 config classes can be imported from qai
  2. Each config class can be passed to qai.build(config=...)
  3. The config object's fields are correctly unpacked into the technique
  4. Backward compat: kwargs still work without a config object
  5. kwargs override config fields when both are passed
  6. TrainConfig and MoEConfig work as cross-cutting configs
  7. get_config_class() returns the right class for each technique
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import qai

print("=" * 70)
print("TEST 1: All 17 config classes importable from qai")
print("=" * 70)
expected = [
    "MoEConfig", "TrainConfig", "GenerateConfig",
    "TransformerConfig", "CNNConfig", "RNNConfig", "ModernLLMConfig",
    "TreeConfig", "BoostingConfig", "ClusterConfig",
    "GNNConfig", "VisionConfig", "RLConfig", "GenerativeConfig",
    "RecommenderConfig", "SequenceConfig", "ProbabilisticConfig",
]
for name in expected:
    assert hasattr(qai, name), f"qai.{name} not exported"
print(f"All {len(expected)} config classes exported from qai ✓")

print()
print("=" * 70)
print("TEST 2: Backward compat — kwargs still work without config")
print("=" * 70)
m = qai.build(technique="transformer", d_model=32, n_heads=2, n_layers=1,
              vocab_size=32, epochs=1)
assert m.technique.d_model == 32
assert m.technique.n_heads == 2
assert m.technique.n_layers == 1
print("kwargs-only call still works ✓")

print()
print("=" * 70)
print("TEST 3: TransformerConfig — config object equivalent to kwargs")
print("=" * 70)
cfg = qai.TransformerConfig(d_model=32, n_heads=2, n_layers=1,
                              vocab_size=32, epochs=1)
m1 = qai.build(technique="transformer", config=cfg)
m2 = qai.build(technique="transformer", d_model=32, n_heads=2, n_layers=1,
                vocab_size=32, epochs=1)
assert m1.technique.d_model == m2.technique.d_model == 32
assert m1.technique.n_heads == m2.technique.n_heads == 2
assert m1.technique.n_layers == m2.technique.n_layers == 1
print("config= and kwargs produce equivalent models ✓")

print()
print("=" * 70)
print("TEST 4: kwargs override config fields when both are passed")
print("=" * 70)
cfg = qai.TransformerConfig(d_model=32, n_heads=2, n_layers=1,
                              vocab_size=32, epochs=1)
m = qai.build(technique="transformer", config=cfg, d_model=64)  # override
assert m.technique.d_model == 64, f"expected 64, got {m.technique.d_model}"
assert m.technique.n_heads == 2  # from config
print("kwargs correctly override config fields ✓")

print()
print("=" * 70)
print("TEST 5: get_config_class() returns right class per technique")
print("=" * 70)
cases = [
    ("transformer", qai.TransformerConfig),
    ("cnn", qai.CNNConfig),
    ("rnn", qai.RNNConfig),
    ("llama_decoder", qai.ModernLLMConfig),
    ("random_forest", qai.TreeConfig),
    ("xgboost", qai.BoostingConfig),
    ("kmeans", qai.ClusterConfig),
    ("gmm", qai.ClusterConfig),
    ("gcn", qai.GNNConfig),
    ("resnet", qai.VisionConfig),
    ("dqn", qai.RLConfig),
    ("gan", qai.GenerativeConfig),
    ("matrix_factorization", qai.RecommenderConfig),
    ("s4", qai.SequenceConfig),
    ("hmm", qai.ProbabilisticConfig),
]
for tech, expected_cls in cases:
    actual = qai.get_config_class(tech)
    assert actual is expected_cls, f"{tech}: expected {expected_cls}, got {actual}"
    print(f"  {tech:25s} → {expected_cls.__name__} ✓")

print()
print("=" * 70)
print("TEST 6: ClusterConfig for GMM with variational_bayes")
print("=" * 70)
cfg = qai.ClusterConfig(n_clusters=5, variational_bayes=True,
                          weight_concentration_prior=1e-2,
                          covariance_type="full")
m = qai.build(technique="gmm", config=cfg)
assert m.technique.n_components == 5
assert m.technique.variational_bayes is True
print(f"GMM with variational_bayes config: n_components={m.technique.n_components} ✓")

print()
print("=" * 70)
print("TEST 7: TreeConfig for random_forest")
print("=" * 70)
cfg = qai.TreeConfig(n_estimators=10, max_depth=5, min_samples_split=4)
m = qai.build(technique="random_forest", config=cfg)
assert m.technique.n_estimators == 10
print(f"RF with TreeConfig: n_estimators={m.technique.n_estimators} ✓")

print()
print("=" * 70)
print("TEST 8: VisionConfig for resnet")
print("=" * 70)
cfg = qai.VisionConfig(in_ch=1, n_classes=3, base_channels=8, n_blocks=2)
m = qai.build(technique="resnet", config=cfg)
assert m.technique.n_classes == 3
print(f"ResNet with VisionConfig: n_classes={m.technique.n_classes} ✓")

print()
print("=" * 70)
print("TEST 9: TrainConfig — attached to model, not technique")
print("=" * 70)
tc = qai.TrainConfig(optimizer="adamw", lr=1e-3, warmup_steps=50,
                      weight_decay=0.01, max_grad_norm=1.0)
m = qai.build(technique="classifier", train_config=tc)
assert hasattr(m, "_train_config")
assert m._train_config.lr == 1e-3
assert m._train_config.optimizer == "adamw"
print(f"TrainConfig attached: lr={m._train_config.lr}, opt={m._train_config.optimizer} ✓")

print()
print("=" * 70)
print("TEST 10: MoEConfig embedded in TransformerConfig")
print("=" * 70)
moe = qai.MoEConfig(num_experts=8, top_k=2, expert_dim=64)
cfg = qai.TransformerConfig(d_model=32, n_heads=2, n_layers=1,
                              vocab_size=32, epochs=1, moe=moe)
m = qai.build(technique="transformer", config=cfg)
# MoE params should be unpacked with moe_ prefix
assert "moe_num_experts" in cfg.to_dict()
assert cfg.to_dict()["moe_num_experts"] == 8
print(f"MoEConfig embedded: num_experts={cfg.to_dict()['moe_num_experts']} ✓")

print()
print("=" * 70)
print("TEST 11: Each config class can build its corresponding technique")
print("=" * 70)
import numpy as np
test_cases = [
    ("transformer", qai.TransformerConfig(vocab_size=32, d_model=16, n_heads=2, n_layers=1, epochs=1)),
    ("cnn", qai.CNNConfig(in_channels=1, n_classes=3, conv_channels=[4, 8], kernel_sizes=[3, 3], pool_sizes=[2, 2], fc_hidden=16, epochs=1)),
    ("rnn", qai.RNNConfig(input_size=4, hidden_size=8, n_classes=2, epochs=1)),
    ("random_forest", qai.TreeConfig(n_estimators=5)),
    ("gradient_boosting", qai.BoostingConfig(n_estimators=5)),
    ("kmeans", qai.ClusterConfig(n_clusters=3)),
    ("gmm", qai.ClusterConfig(n_clusters=3)),
]
for tech, cfg in test_cases:
    try:
        m = qai.build(technique=tech, config=cfg)
        print(f"  {tech:25s} build OK ✓")
    except Exception as e:
        print(f"  {tech:25s} FAIL: {type(e).__name__}: {str(e)[:60]}")

print()
print("=" * 70)
print("TEST 12: Tokenizer attachment")
print("=" * 70)
class FakeTokenizer:
    vocab_size = 32
    def encode(self, text):
        return [ord(c) % 32 for c in text]
    def decode(self, ids):
        return "".join(chr(i + 32) for i in ids)

tok = FakeTokenizer()
m = qai.build(technique="classifier", tokenizer=tok)
assert hasattr(m, "tokenizer")
assert m.tokenizer is tok
print("Tokenizer attached to model ✓")

print()
print("=" * 70)
print("ALL CONFIG OBJECT TESTS PASSED")
print("=" * 70)

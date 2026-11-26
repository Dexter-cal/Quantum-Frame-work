"""
test_all_visualizations.py — create models from all 40 technique families,
visualize each one, save screenshots.

Run:
    cd showcase
    python test_all_visualizations.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import warnings; warnings.filterwarnings('ignore')
import qai
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "visualization_screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Build-and-visualize tests for representatives of every technique family.
# Each entry: (technique_name, build_params, train_kind)
# train_kind: 'class' | 'reg' | 'unsup' | None (don't train)
TECHNIQUES = [
    # --- deep_learning ---
    ("cnn", dict(in_channels=1, n_classes=3, conv_channels=[8,16],
                  kernel_sizes=[3,3], pool_sizes=[2,2], fc_hidden=32,
                  epochs=1, batch_size=4, augment=False), "class"),
    ("neural_network", dict(hidden_dim=16, epochs=2, lr=0.01), "class"),
    ("transformer", dict(vocab_size=64, n_layers=1, d_model=32, n_heads=4,
                          epochs=1), "class"),
    ("rnn", {}, None),
    ("lstm", {}, None),
    ("gru", {}, None),
    ("seq2seq", {}, None),
    ("autoencoder", dict(epochs=1), "unsup"),
    ("vae", dict(epochs=1), "unsup"),
    ("gan", dict(epochs=1), "unsup"),
    ("diffusion", dict(epochs=1), "unsup"),
    ("moe", {}, None),
    ("mamba", {}, None),
    ("gnn", {}, None),
    ("dqn", {}, None),
    ("ppo", {}, None),
    ("self_supervised", {}, None),
    # --- classical_ml (linear) ---
    ("regression", {}, "reg"),
    ("classifier", {}, "class"),
    ("svm", {}, "class"),
    ("perceptron", {}, "class"),
    ("ridge", {}, "reg"),
    ("lasso", {}, "reg"),
    ("elastic_net", {}, "reg"),
    ("sgd_classifier", {}, "class"),
    ("sgd_regressor", {}, "reg"),
    ("poisson_regressor", {}, "reg"),
    ("lda", {}, "class"),
    ("qda", {}, "class"),
    ("mlp_regressor", dict(epochs=1), "reg"),
    # --- classical_ml (tree-based) ---
    ("decision_tree", {}, "class"),
    ("random_forest", dict(n_estimators=10), "class"),
    ("extra_trees", dict(n_estimators=10), "class"),
    ("gradient_boosting", dict(n_estimators=10), "class"),
    ("adaboost", dict(n_estimators=10), "class"),
    ("hist_gradient_boosting", {}, "class"),
    ("xgboost", {}, "class"),
    ("lightgbm", {}, "class"),
    ("catboost", {}, "class"),
    # --- classical_ml (instance-based + nb) ---
    ("knn", dict(k=3), "class"),
    ("knn_regressor", {}, "reg"),
    ("naive_bayes", {}, "class"),
    ("multinomial_naive_bayes", {}, "class"),
    ("gaussian_nb", {}, "class"),
    ("bernoulli_naive_bayes", {}, "class"),
    # --- unsupervised (clustering) ---
    ("kmeans", dict(n_clusters=3), "unsup"),
    ("gmm", dict(n_components=3), "unsup"),
    ("dbscan", dict(eps=0.5, min_samples=3), "unsup"),
    ("hierarchical", {}, "unsup"),
    ("agglomerative_clustering", {}, "unsup"),
    ("birch", {}, "unsup"),
    ("mean_shift", {}, "unsup"),
    ("spectral_clustering", {}, "unsup"),
    ("affinity_propagation", {}, "unsup"),
    ("optics", {}, "unsup"),
    ("bisecting_kmeans", dict(n_clusters=3), "unsup"),
    ("mini_batch_kmeans", dict(n_clusters=3), "unsup"),
    ("art", {}, "unsup"),
    ("neural_gas", {}, "unsup"),
    ("som", {}, "unsup"),
    # --- dim red / manifold / decomposition ---
    ("pca", dict(n_components=2), "unsup"),
    ("truncated_svd", dict(n_components=2), "unsup"),
    ("tsne", dict(n_components=2), "unsup"),
    ("fast_ica", {}, "unsup"),
    ("isomap", {}, "unsup"),
    ("lle", {}, "unsup"),
    ("sparse_pca", {}, "unsup"),
    ("kernel_pca", {}, "unsup"),
    ("incremental_pca", {}, "unsup"),
    ("gaussian_random_projection", {}, "unsup"),
    ("factor_analysis", {}, "unsup"),
    ("nmf", {}, "unsup"),
    ("mds", {}, "unsup"),
    ("spectral_embedding", {}, "unsup"),
    ("covariance", {}, "unsup"),
    # --- preprocessing ---
    ("standard_scaler", {}, "unsup"),
    ("minmax_scaler", {}, "unsup"),
    ("robust_scaler", {}, "unsup"),
    ("normalizer", {}, "unsup"),
    ("maxabs_scaler", {}, "unsup"),
    ("binarizer", {}, "unsup"),
    ("polynomial_features", {}, "unsup"),
    ("kbins_discretizer", {}, "unsup"),
    ("simple_imputer", {}, "unsup"),
    ("knn_imputer", {}, "unsup"),
    ("label_encoder", {}, None),
    ("one_hot_encoder", {}, None),
    ("target_encoder", {}, None),
    ("tfidf_vectorizer", {}, None),
    ("count_vectorizer", {}, None),
    ("hashing_vectorizer", {}, None),
    # --- anomaly ---
    ("isolation_forest", {}, "unsup"),
    ("one_class_svm", {}, "unsup"),
    ("local_outlier_factor", {}, "unsup"),
    ("kde", {}, "unsup"),
    ("kernel_density", {}, "unsup"),
    # --- reinforcement learning ---
    ("a2c", {}, None),
    ("sac", {}, None),
    ("ddpg", {}, None),
    ("td3", {}, None),
    ("sarsa", {}, None),
    ("tabular_policy", {}, None),
    # --- model_based_rl ---
    ("world_model", {}, None),
    ("dreamer", {}, None),
    ("muzero", {}, None),
    # --- planning ---
    ("mcts", {}, None),
    ("policy_iteration", {}, None),
    ("value_iteration", {}, None),
    # --- vision ---
    ("resnet", {}, None),
    ("unet", {}, None),
    ("vit", {}, None),
    ("deit", {}, None),
    ("densenet", {}, None),
    ("mobilenet", {}, None),
    ("efficientnet", {}, None),
    ("capsule", {}, None),
    ("nerf", {}, None),
    ("inr", {}, None),
    # --- speech ---
    ("wav2vec2", {}, None),
    ("hubert", {}, None),
    ("whisper", {}, None),
    # --- multimodal ---
    ("clip", {}, None),
    # --- modern architectures ---
    ("llama_decoder", {}, None),
    ("t5", {}, None),
    ("rwkv", {}, None),
    ("s4", {}, None),
    ("mamba2", {}, None),
    ("deepseek_moe", {}, None),
    ("jamba", {}, None),
    # --- recommender ---
    ("deepfm", {}, None),
    ("wide_deep", {}, None),
    ("matrix_factorization", {}, None),
    ("collab_filtering", {}, None),
    ("item2vec", {}, None),
    # --- probabilistic ---
    ("hmm", {}, None),
    ("bayesian_nn", {}, None),
    ("mc_dropout", {}, None),
    ("mixture_density", {}, None),
    ("topic_model", {}, None),
    # --- self_supervised ---
    ("contrastive_learning", {}, None),
    # --- sequence ---
    ("pointer_net", {}, None),
    ("ctc", {}, None),
    ("ttt", {}, None),
    # --- set ---
    ("set_transformer", {}, None),
    # --- memory_augmented ---
    ("dnc", {}, None),
    ("ntm", {}, None),
    # --- meta_learning ---
    ("maml", {}, None),
    # --- few_shot ---
    ("prototypical", {}, None),
    ("matching", {}, None),
    ("relation", {}, None),
    # --- metric_learning ---
    ("siamese", {}, None),
    # --- filtering ---
    ("kalman_filter", {}, None),
    # --- neuromorphic ---
    ("spiking_nn", {}, None),
    # --- feedforward ---
    ("rbf_network", {}, None),
    ("elm", {}, None),
    # --- reservoir ---
    ("echo_state", {}, None),
    # --- time_series ---
    ("time_series_arima", {}, None),
    # --- feature_selection ---
    ("rfe", {}, None),
    ("select_k_best", {}, None),
    ("variance_threshold", {}, None),
    # --- supervised (other) ---
    ("gaussian_process", {}, None),
    ("isotonic_regression", {}, None),
    ("dummy_regressor", {}, None),
    ("cca", {}, None),
    ("pls_canonical", {}, None),
    ("probability_calibration", {}, None),
    ("stacking_classifier", {}, None),
    ("voting_classifier", {}, None),
    # --- generative (other) ---
    ("energy_based", {}, None),
    ("normalizing_flow", {}, None),
    ("flow_matching", {}, None),
    ("score_based", {}, None),
    ("consistency_model", {}, None),
    ("rbm", {}, None),
    ("bernoulli_rbm", {}, None),
    ("boltzmann", {}, None),
    ("dbn", {}, None),
    # --- graph ---
    ("gcn", {}, None),
    ("gat", {}, None),
    ("gin", {}, None),
    ("graphsage", {}, None),
    ("mpnn", {}, None),
    # --- attention ---
    ("sparse_attention", {}, None),
]

# Techniques to visualize WITHOUT building (generic)
GENERIC_TECHNIQUES = [
    "transformer", "rnn", "lstm", "extra_trees", "adaboost", "tsne",
    "autoencoder", "vae", "gan", "diffusion",
    "gcn", "gat", "mcts", "world_model", "clip",
    "llama_decoder", "rwkv", "mamba", "deepseek_moe", "jamba",
    "hmm", "kalman_filter", "spiking_nn", "siamese", "prototypical",
    "dnc", "maml", "echo_state", "time_series_arima",
]


def main():
    print("=" * 70)
    print(f"Visualization Test: ALL {len(TECHNIQUES)} technique types + "
          f"{len(GENERIC_TECHNIQUES)} generic")
    print("=" * 70)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    y_class = (X[:, 0] > 0).astype(int)
    y_reg = X[:, 0] * 2.0 + 0.1 * rng.standard_normal(50)

    passed = 0
    failed = 0
    for tech_name, params, kind in TECHNIQUES:
        try:
            m = qai.build(technique=tech_name, **params)
            if kind == "class":
                try: m.train(X, y_class, verbose=False)
                except Exception: pass
            elif kind == "reg":
                try: m.train(X, y_reg, verbose=False)
                except Exception: pass
            elif kind == "unsup":
                try: m.train(X, verbose=False)
                except Exception: pass
            out_path = os.path.join(OUTPUT_DIR, f"{tech_name}.png")
            qai.visualize_architecture(m, save=out_path)
            assert os.path.exists(out_path), "file not created"
            size = os.path.getsize(out_path)
            assert size > 3000, f"file too small ({size} bytes)"
            print(f"  OK   {tech_name:30s} ({size:6d} bytes)")
            passed += 1
        except Exception as e:
            print(f"  FAIL {tech_name:30s} {str(e)[:80]}")
            failed += 1

    # Generic visualizations (no model needed)
    print()
    print("--- Generic visualizations (no model) ---")
    for tech_name in GENERIC_TECHNIQUES:
        try:
            out_path = os.path.join(OUTPUT_DIR, f"{tech_name}_generic.png")
            qai.visualize_technique(tech_name, save=out_path)
            assert os.path.exists(out_path)
            size = os.path.getsize(out_path)
            assert size > 3000
            print(f"  OK   {tech_name:30s} generic ({size:6d} bytes)")
            passed += 1
        except Exception as e:
            print(f"  FAIL {tech_name:30s} generic {str(e)[:80]}")
            failed += 1

    # Pipeline visualization
    print()
    print("--- Pipelines / bridges ---")
    try:
        builder = qai.PipelineBuilder("test_pipe")
        builder.add_step("scaler", qai.build(technique="standard_scaler"))
        builder.add_step("pca", qai.build(technique="pca", n_components=2),
                          depends_on=["scaler"])
        builder.add_step("model", qai.build(technique="random_forest", n_estimators=5),
                          depends_on=["pca"])
        out_path = os.path.join(OUTPUT_DIR, "pipeline_dag.png")
        qai.visualize_pipeline(builder, save=out_path)
        assert os.path.exists(out_path)
        print(f"  OK   pipeline_dag")
        passed += 1
    except Exception as e:
        print(f"  FAIL pipeline_dag: {str(e)[:80]}")
        failed += 1

    try:
        bridge = qai.FederatedBridge("test_bridge")
        bridge.add_piece("scaler", qai.build(technique="standard_scaler"), framework="qai")
        bridge.add_piece("clf", qai.build(technique="classifier"), framework="qai", input_from="scaler")
        out_path = os.path.join(OUTPUT_DIR, "federated_bridge.png")
        qai.visualize_architecture(bridge, save=out_path)
        assert os.path.exists(out_path)
        print(f"  OK   federated_bridge")
        passed += 1
    except Exception as e:
        print(f"  FAIL federated_bridge: {str(e)[:80]}")
        failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"Screenshots saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

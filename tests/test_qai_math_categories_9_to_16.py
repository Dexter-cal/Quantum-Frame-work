import unittest
import numpy as np
import qai.math as qmath

class TestQAIMathCategories9To16(unittest.TestCase):

    def test_category_9_transformer_attention(self):
        Q = np.random.randn(2, 4, 8)
        K = np.random.randn(2, 4, 8)
        V = np.random.randn(2, 4, 8)

        out, weights = qmath.scaled_dot_product_attention(Q, K, V)
        self.assertEqual(out.shape, (2, 4, 8))
        self.assertAlmostEqual(np.sum(weights[0, 0]), 1.0)

        W_q, W_k, W_v = np.eye(8), np.eye(8), np.eye(8)
        q_p, k_p, v_p = qmath.query_key_value_projection(Q, W_q, W_k, W_v)
        self.assertEqual(q_p.shape, (2, 4, 8))

        m_out, m_weights = qmath.multi_head_attention(Q, K, V, np.eye(8))
        self.assertEqual(m_out.shape, (2, 4, 8))

        c_out, c_weights = qmath.masked_attention(Q, K, V)
        self.assertEqual(c_out.shape, (2, 4, 8))

        x_out, x_weights = qmath.cross_attention(Q, K, V)
        self.assertEqual(x_out.shape, (2, 4, 8))

        pe = qmath.sinusoidal_positional_encoding(10, 16)
        self.assertEqual(pe.shape, (10, 16))

        l_pe = qmath.learned_positional_embedding(10, 16)
        self.assertEqual(l_pe.shape, (10, 16))

        norm_x = qmath.layer_norm(Q)
        self.assertEqual(norm_x.shape, Q.shape)

        res = qmath.residual_connection(Q, Q)
        self.assertTrue(np.allclose(res, Q * 2))

        ff_out = qmath.feed_forward_block(Q, np.eye(8), np.zeros(8), np.eye(8), np.zeros(8))
        self.assertEqual(ff_out.shape, Q.shape)

        drop_out = qmath.attention_dropout(weights, p=0.1, training=True)
        self.assertEqual(drop_out.shape, weights.shape)

    def test_category_10_diffusion_model_math(self):
        x0 = np.ones((2, 2))
        noise = np.random.randn(2, 2)

        x_t = qmath.forward_noise_step(x0, noise, beta_t=0.01)
        self.assertEqual(x_t.shape, (2, 2))

        x_t_cf = qmath.forward_noise_closed_form(x0, noise, alpha_bar_t=0.9)
        self.assertEqual(x_t_cf.shape, (2, 2))

        betas = qmath.noise_schedule(n_steps=10, mode="linear")
        self.assertEqual(len(betas), 10)

        a_bar = qmath.alpha_bar(betas)
        self.assertEqual(len(a_bar), 10)

        rev = qmath.reverse_denoise_step(x_t, noise, beta_t=0.01, alpha_bar_t=0.9, alpha_bar_prev=0.95)
        self.assertEqual(rev.shape, (2, 2))

        score = qmath.score_function(noise, sigma_t=0.1)
        self.assertEqual(score.shape, (2, 2))

        d_loss = qmath.denoising_score_matching_loss(noise, noise)
        self.assertAlmostEqual(d_loss, 0.0)

        ddim_step = qmath.ddim_sample_step(x_t, noise, alpha_bar_t=0.9, alpha_bar_prev=0.95)
        self.assertEqual(ddim_step.shape, (2, 2))

        v_sched = qmath.variance_schedule(betas)
        self.assertEqual(len(v_sched), 10)

        lat = qmath.latent_encode(x0)
        self.assertTrue(np.allclose(qmath.latent_decode(lat), x0))

    def test_category_11_reinforcement_learning_math(self):
        self.assertEqual(qmath.bellman_equation(1.0, 0.99, 10.0), 10.9)

        disc_ret = qmath.discounted_return([1.0, 1.0, 1.0], gamma=0.9)
        self.assertEqual(len(disc_ret), 3)

        self.assertAlmostEqual(qmath.td_error(1.0, 0.99, 10.0, 5.0), 5.9)
        self.assertAlmostEqual(qmath.advantage_function(10.9, 5.0), 5.9)
        self.assertIsInstance(qmath.policy_gradient([-0.5], [2.0]), float)
        self.assertAlmostEqual(qmath.q_value_update(1.0, 1.0, 0.99, 10.0, alpha=0.1), 1.989)

    def test_category_12_optimizer_math(self):
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])

        p_sgd = qmath.sgd_update(param, grad, lr=0.1)
        self.assertTrue(np.allclose(p_sgd, [0.99, 1.98]))

        p_mom, v_mom = qmath.momentum_update(param, grad, np.zeros(2))
        self.assertEqual(p_mom.shape, (2,))

        p_rms, s_rms = qmath.rmsprop_update(param, grad, np.zeros(2))
        self.assertEqual(p_rms.shape, (2,))

        p_adam, m, v = qmath.adam_update(param, grad, np.zeros(2), np.zeros(2), t=1)
        self.assertEqual(p_adam.shape, (2,))

        self.assertAlmostEqual(qmath.learning_rate_decay(0.1, epoch=1, decay_rate=0.1), 0.09090909)
        self.assertEqual(qmath.gradient_clipping([10.0, 10.0], max_norm=1.0).shape, (2,))

    def test_category_13_generative_models_gan_vae(self):
        d_loss = qmath.discriminator_loss([0.9, 0.8], [0.1, 0.2])
        g_loss = qmath.generator_loss([0.9, 0.8])
        self.assertGreater(d_loss, 0)
        self.assertGreater(g_loss, 0)

        self.assertIsInstance(qmath.minimax_objective(0.9, 0.1), float)
        self.assertGreaterEqual(qmath.wasserstein_distance([1, 2], [2, 3]), 0)
        self.assertAlmostEqual(qmath.reconstruction_loss([1.0], [1.0]), 0.0)

        z_repar = qmath.reparameterization_trick([0.0], [0.0])
        self.assertEqual(z_repar.shape, (1,))

        elbo = qmath.evidence_lower_bound([1.0], [1.0], [0.0], [0.0])
        self.assertIsInstance(elbo, float)

    def test_category_14_regularization_and_normalization(self):
        w = [1.0, -2.0, 3.0]
        self.assertAlmostEqual(qmath.l1_regularization(w, l1_lambda=0.1), 0.06)
        self.assertAlmostEqual(qmath.l2_regularization(w, l2_lambda=0.1), 0.07)

        d_mask = qmath.dropout_mask((2, 2), drop_prob=0.5)
        self.assertEqual(d_mask.shape, (2, 2))

        b_norm = qmath.batch_norm(np.random.randn(10, 4))
        self.assertEqual(b_norm.shape, (10, 4))

        g_norm = qmath.group_norm(np.random.randn(2, 4, 8, 8), num_groups=2)
        self.assertEqual(g_norm.shape, (2, 4, 8, 8))

    def test_category_15_ensemble_boosting_math(self):
        res = qmath.gradient_boosting_update([1.0, 2.0], [0.8, 1.8])
        self.assertTrue(np.allclose(res, [0.2, 0.2]))

        ada_w = qmath.adaboost_weight_update([0.5, 0.5], 0.1, [1, 0], [1, 1])
        self.assertEqual(len(ada_w), 2)

        bag_w = qmath.bagging_sample_weight(10)
        self.assertEqual(len(bag_w), 10)

    def test_category_16_gnn_math(self):
        adj = np.array([[0, 1], [1, 0]])
        adj_norm = qmath.adjacency_matrix_ops(adj, mode="normalize")
        self.assertEqual(adj_norm.shape, (2, 2))

        X = np.array([[1.0], [2.0]])
        W = np.array([[1.0]])
        gcn_out = qmath.graph_convolution(X, adj_norm, W)
        self.assertEqual(gcn_out.shape, (2, 1))

        msg_pass = qmath.message_passing(X, adj, aggregate_fn="sum")
        self.assertEqual(msg_pass.shape, (2, 1))

if __name__ == "__main__":
    unittest.main()

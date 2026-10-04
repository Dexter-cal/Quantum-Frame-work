import unittest
import numpy as np
import qai.math as qmath

class TestQAIMathComplete(unittest.TestCase):
    def test_linear_algebra_operations(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])

        self.assertAlmostEqual(qmath.dot([1, 2], [3, 4]), 11.0)
        self.assertEqual(qmath.matmul(A, b).shape, (2,))
        self.assertEqual(qmath.transpose(A).shape, (2, 2))
        self.assertAlmostEqual(qmath.determinant(A), 11.0)
        self.assertAlmostEqual(qmath.trace(A), 7.0)
        self.assertEqual(qmath.rank(A), 2)
        self.assertGreater(qmath.norm([3, 4], ord="l2"), 0)
        self.assertAlmostEqual(qmath.norm(qmath.normalize([3, 4])), 1.0)

        vals, vecs = qmath.eigen(A)
        self.assertEqual(len(vals), 2)

        U, S, Vh = qmath.svd(A)
        self.assertEqual(len(S), 2)

        Q, R = qmath.qr_decompose(A)
        self.assertEqual(Q.shape, (2, 2))

        L = qmath.cholesky_decompose(A)
        self.assertEqual(L.shape, (2, 2))

        x_sol = qmath.solve_linear_system(A, b)
        self.assertAlmostEqual(np.dot(A, x_sol)[0], b[0])

        self.assertEqual(qmath.identity(3).shape, (3, 3))
        self.assertEqual(qmath.zeros((2, 2)).shape, (2, 2))
        self.assertEqual(qmath.ones((2, 2)).shape, (2, 2))
        self.assertEqual(qmath.outer([1, 2], [3, 4]).shape, (2, 2))

    def test_calculus_and_gradients(self):
        f = lambda x: x[0]**2 + 3*x[1]
        x0 = np.array([2.0, 1.0])

        grad = qmath.gradient(f, x0)
        self.assertAlmostEqual(grad[0], 4.0, places=3)
        self.assertAlmostEqual(grad[1], 3.0, places=3)

        p_deriv = qmath.partial_derivative(f, x0, var_index=0)
        self.assertAlmostEqual(p_deriv, 4.0, places=3)

        f_1d = lambda x: x**3
        self.assertAlmostEqual(qmath.second_derivative(f_1d, 2.0), 12.0, places=2)

        H = qmath.hessian(f, x0)
        self.assertEqual(H.shape, (2, 2))

        vf = lambda x: np.array([x[0]**2, x[0]*x[1]])
        J = qmath.jacobian(vf, x0)
        self.assertEqual(J.shape, (2, 2))

        self.assertEqual(qmath.chain_rule(2.0, 3.0), 6.0)

        mse_g = qmath.mse_gradient([1.0, 2.0], [1.5, 2.5])
        self.assertEqual(len(mse_g), 2)

    def test_activations_and_derivatives(self):
        x = np.array([-1.0, 0.0, 1.0])

        self.assertEqual(len(qmath.sigmoid(x)), 3)
        self.assertEqual(len(qmath.sigmoid_derivative(x)), 3)
        self.assertEqual(len(qmath.tanh(x)), 3)
        self.assertEqual(len(qmath.tanh_derivative(x)), 3)
        self.assertEqual(len(qmath.relu(x)), 3)
        self.assertEqual(len(qmath.relu_derivative(x)), 3)
        self.assertEqual(len(qmath.leaky_relu(x)), 3)
        self.assertEqual(len(qmath.elu(x)), 3)
        self.assertEqual(len(qmath.gelu(x)), 3)
        self.assertEqual(len(qmath.swish(x)), 3)
        self.assertAlmostEqual(np.sum(qmath.softmax(x)), 1.0)
        self.assertEqual(len(qmath.softplus(x)), 3)
        self.assertEqual(len(qmath.linear(x)), 3)

    def test_loss_functions_and_gradients(self):
        y_t = [1.0, 0.0]
        y_p = [0.8, 0.2]

        self.assertGreater(qmath.mse(y_t, y_p), 0)
        self.assertGreater(qmath.mae(y_t, y_p), 0)
        self.assertGreater(qmath.huber_loss(y_t, y_p), 0)
        self.assertGreaterEqual(qmath.hinge_loss([1, -1], [0.8, -0.2]), 0)
        self.assertGreater(qmath.focal_loss(y_t, y_p), 0)
        self.assertGreater(qmath.cross_entropy(y_t, y_p), 0)
        self.assertGreater(qmath.binary_cross_entropy(y_t, y_p), 0)
        self.assertGreaterEqual(qmath.kl_divergence_loss([0.5, 0.5], [0.4, 0.6]), 0)

    def test_probability_and_statistics(self):
        data = [1, 2, 3, 4, 5, 5, 6]

        self.assertAlmostEqual(qmath.mean(data), 3.7142857, places=4)
        self.assertEqual(qmath.median(data), 4.0)
        self.assertEqual(qmath.mode(data), 5)
        self.assertGreater(qmath.variance(data), 0)
        self.assertGreater(qmath.std(data), 0)
        self.assertIsInstance(qmath.covariance([1, 2, 3], [2, 4, 6]), float)
        self.assertEqual(qmath.covariance_matrix([[1, 2], [3, 4]]).shape, (2, 2))
        self.assertAlmostEqual(qmath.correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertEqual(qmath.percentile(data, 50), 4.0)
        self.assertEqual(len(qmath.z_score(data)), 7)
        self.assertIsInstance(qmath.skewness(data), float)
        self.assertIsInstance(qmath.kurtosis(data), float)

        self.assertGreater(qmath.gaussian_pdf(0.0), 0)
        self.assertGreaterEqual(qmath.bernoulli_pmf(1, 0.5), 0)
        self.assertGreaterEqual(qmath.binomial_pmf(2, 5, 0.5), 0)
        self.assertGreaterEqual(qmath.poisson_pmf(2, 3.0), 0)
        self.assertGreater(qmath.uniform_pdf(0.5), 0)
        self.assertGreater(qmath.exponential_pdf(1.0), 0)

        self.assertAlmostEqual(qmath.bayes_theorem(0.01, 0.9, 0.05), 0.18)
        self.assertAlmostEqual(qmath.joint_probability(0.5, 0.8), 0.4)
        self.assertAlmostEqual(qmath.marginal_probability([0.1, 0.2, 0.3]), 0.6)
        self.assertAlmostEqual(qmath.conditional_probability(0.2, 0.5), 0.4)
        self.assertAlmostEqual(qmath.expectation([1, 2], [0.5, 0.5]), 1.5)

        mle = qmath.maximum_likelihood_estimate([1, 2, 3, 4, 5])
        self.assertIn("mu", mle)

        priors = qmath.class_prior([0, 0, 1, 1, 1])
        self.assertAlmostEqual(priors[0], 0.4)

        ci = qmath.confidence_interval([10, 12, 11, 9, 13])
        self.assertTrue(ci[0] < ci[1])

        self.assertEqual(len(qmath.sample([1, 2, 3], size=2)), 2)
        self.assertEqual(len(qmath.shuffle([1, 2, 3])), 3)

    def test_information_theory(self):
        p = [0.5, 0.5]
        q = [0.2, 0.8]

        self.assertAlmostEqual(qmath.entropy(p), 1.0)
        self.assertGreater(qmath.info_cross_entropy(p, q), 0)
        self.assertGreaterEqual(qmath.kl_divergence(p, q), 0)
        self.assertGreaterEqual(qmath.mutual_information([0, 1, 0, 1], [0, 1, 0, 1]), 0)
        self.assertGreaterEqual(qmath.information_gain([0, 0, 1, 1], [[0, 0], [1, 1]]), 0)
        self.assertAlmostEqual(qmath.gini_impurity([0, 0, 1, 1]), 0.5)
        self.assertAlmostEqual(qmath.perplexity(p), 2.0)

    def test_distance_and_similarity(self):
        x = [1, 2]
        y = [4, 6]

        self.assertAlmostEqual(qmath.euclidean_distance(x, y), 5.0)
        self.assertAlmostEqual(qmath.manhattan_distance(x, y), 7.0)
        self.assertGreater(qmath.cosine_similarity(x, y), 0)
        self.assertGreaterEqual(qmath.mahalanobis_distance(x, y, np.eye(2)), 0)
        self.assertAlmostEqual(qmath.hamming_distance([1, 0, 1], [1, 1, 0]), 2/3)

    def test_signal_and_convolution(self):
        signal = [1, 2, 3, 4]
        kernel = [1, 0, -1]

        conv1d = qmath.convolve(signal, kernel, mode="same")
        self.assertEqual(len(conv1d), 4)

        m2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        pooled = qmath.pooling(m2d, pool_size=(2, 2), mode="max")
        self.assertEqual(pooled.shape, (2, 2))
        self.assertEqual(pooled[0, 0], 6.0)

        fft = qmath.fourier_transform(signal)
        ifft = qmath.inverse_fourier_transform(fft)
        self.assertAlmostEqual(ifft[0].real, 1.0)

    def test_reference_helpers(self):
        overview = qmath.help()
        self.assertIn("qai.math", overview)

        detail = qmath.help("gini_impurity")
        self.assertIn("gini_impurity", detail)

        funcs = qmath.list_by_category("linear_algebra")
        self.assertIn("dot", funcs)

        with self.assertRaises(ValueError):
            qmath.list_by_category("unknown_category")

    def test_edge_and_error_cases(self):
        with self.assertRaises(ValueError):
            qmath.bayes_theorem(0.5, 0.5, marginal_b=0.0)

        with self.assertRaises(ValueError):
            qmath.conditional_probability(0.5, p_given=0.0)

        with self.assertRaises(ValueError):
            qmath.norm([1, 2], ord="invalid")

if __name__ == "__main__":
    unittest.main()

import unittest
import os
import qai
import qai.runtime as qruntime

class TestQAIRuntimeComplete(unittest.TestCase):
    def setUp(self):
        self.base_model = qai.build(type="regression")
        self.base_model.train([[1, 2], [3, 4]], [1.0, 2.0])
        self.runtime_model = qruntime.RuntimeModelWrapper(self.base_model, name="sales_regressor")

    def test_versioning_and_states(self):
        self.assertEqual(self.runtime_model.version(), "1.0.0")
        bumped = self.runtime_model.bump_version("minor")
        self.assertEqual(bumped, "1.1.0")
        self.assertIn(self.runtime_model.state, qruntime.VALID_STATES)

    def test_staging_deploy_promote_history(self):
        deploy_rec = self.runtime_model.deploy(environment="staging", auto_rollback_if="mae > 0.5")
        self.assertEqual(self.runtime_model.environment, "staging")
        self.assertEqual(deploy_rec["environment"], "staging")

        prom_rec = self.runtime_model.promote(from_env="staging", to_env="production")
        self.assertEqual(self.runtime_model.environment, "production")
        self.assertEqual(len(self.runtime_model.deployment_history()), 2)

    def test_promote_invalid_environment_raises(self):
        with self.assertRaises(RuntimeError):
            self.runtime_model.promote(from_env="staging", to_env="production")

    def test_ab_testing_router(self):
        model_b = qai.build(type="regression")
        model_b.train([[1, 2], [3, 4]], [1.1, 2.1])
        router = qruntime.ab_test(self.base_model, model_b, traffic_split=0.5)
        pred = router([[1, 2]])
        self.assertIsNotNone(pred)

    def test_footprint_and_signing(self):
        foot = self.runtime_model.footprint()
        self.assertIn("disk_size_mb", foot)

        sig = self.runtime_model.sign_model("secret_key")
        self.assertTrue(self.runtime_model.verify_signature("secret_key"))
        self.assertFalse(self.runtime_model.verify_signature("wrong_key"))

    def test_export_and_batch_convert(self):
        export_path = "/tmp/runtime_model_export.json"
        bundle = self.runtime_model.export(export_path, include_runtime=True, runtime_scope="predict_only")
        self.assertTrue(os.path.exists(export_path))
        self.assertTrue(bundle["include_runtime"])
        os.remove(export_path)

        converted = qruntime.batch_convert([self.base_model], target_format="onnx")
        self.assertEqual(len(converted), 1)

    def test_cli_engine(self):
        ls_out = qruntime.qai_runtime_cli(["list"])
        self.assertIn("Local Models", ls_out)

        ps_out = qruntime.qai_runtime_cli(["ps"])
        self.assertIn("Loaded Models", ps_out)

        show_out = qruntime.qai_runtime_cli(["show", "sales_regressor"])
        self.assertIn("Metadata", show_out)

        chat_out = qruntime.qai_runtime_cli(["chat"])
        self.assertIn("REPL", chat_out)

        serve_out = qruntime.qai_runtime_cli(["serve"])
        self.assertIn("5050", serve_out)

if __name__ == "__main__":
    unittest.main()

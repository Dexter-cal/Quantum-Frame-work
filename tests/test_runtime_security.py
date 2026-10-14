import unittest
import qai.runtime as qruntime

class TestRuntimeSecurity(unittest.TestCase):
    def test_publisher_trust_and_verification(self):
        qruntime.trust_add("acme_corp", "12345abcdef")
        model_name = "fraud_detector"
        sig = qruntime.hashlib.sha256(f"{model_name}_12345abcdef".encode()).hexdigest()

        self.assertTrue(qruntime.verify_publisher_signature(model_name, sig, "acme_corp"))
        self.assertFalse(qruntime.verify_publisher_signature(model_name, sig, "untrusted_publisher"))

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import qai
from qai.core.schema import SchemaError

class TestOutputAndDictSchema(unittest.TestCase):
    def test_dict_input_schema_happy_path(self):
        in_schema = {"f1": float, "f2": float}
        model = qai.build(type="regression", input_schema=in_schema)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model.train(X, y)

        res = model.predict({"f1": 1.0, "f2": 2.0})
        self.assertIsInstance(res, (float, int, np.number))

    def test_output_schema_validation(self):
        out_schema = {"prediction": float}
        model = qai.build(type="regression", output_schema=out_schema)
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        model.train(X, y)

        # Output schema expecting string when model produces numeric -> SchemaError
        invalid_out_model = qai.build(type="regression", output_schema={"score": str})
        invalid_out_model.train(X, y)

        with self.assertRaises(SchemaError) as ctx:
            invalid_out_model.predict([1.0])
        self.assertIn("[WHAT]", str(ctx.exception))

    def test_dict_schema_edge_cases(self):
        schema = qai.core.schema.Schema({"age": int, "income": float})

        # List of dicts
        data_list = [
            {"age": 25, "income": 50000.0},
            {"age": 30, "income": 65000.0}
        ]
        schema.validate(data_list, context="test list")

        # Missing field error
        with self.assertRaises(SchemaError):
            schema.validate_row({"age": 25})

        # Wrong type error
        with self.assertRaises(SchemaError):
            schema.validate_row({"age": "twenty-five", "income": 50000.0})

    def test_security_misuse_garbage_input(self):
        schema = qai.core.schema.Schema({"a": float})
        with self.assertRaises(SchemaError):
            schema.validate(12345)

if __name__ == "__main__":
    unittest.main()

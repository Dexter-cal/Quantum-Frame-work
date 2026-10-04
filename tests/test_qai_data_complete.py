import unittest
import os
import qai.data as qdata

class TestQAIDataComplete(unittest.TestCase):
    def setUp(self):
        self.raw_data = [
            [1, "Alice", "alice@example.com", 25, 100.0],
            [2, "Bob", "bob@example.com", 30, 200.0],
            [3, "Charlie", "charlie@example.com", 35, None],
            [1, "Alice", "alice@example.com", 25, 100.0]  # duplicate
        ]
        self.columns = ["id", "name", "email", "age", "score"]
        self.ds = qdata.DataDataset(self.raw_data, self.columns)

    def test_describe_and_info(self):
        desc = self.ds.describe()
        self.assertIn("age", desc)
        self.assertIn("score", desc)
        self.assertEqual(desc["age"]["count"], 4)

        info = self.ds.info()
        self.assertEqual(info["row_count"], 4)
        self.assertEqual(info["duplicate_count"], 1)
        self.assertEqual(info["null_counts"]["score"], 1)

    def test_cleaning_and_filtering(self):
        self.ds.remove_duplicates()
        self.assertEqual(len(self.ds.data), 3)

        self.ds.handle_missing(strategy="drop")
        self.assertEqual(len(self.ds.data), 2)

        self.ds.filter(lambda row: row["age"] > 20)
        self.assertEqual(len(self.ds.data), 2)

    def test_cell_and_formula(self):
        self.ds.cell(0, "age").set(26)
        self.assertEqual(self.ds.data[0][3], 26)

        self.ds.formula("double_age", "= age * 2")
        self.assertIn("double_age", self.ds.columns)
        self.assertEqual(self.ds.data[0][-1], 52)

    def test_search_snapshot_rollback_diff(self):
        results = self.ds.search("alice")
        self.assertGreater(len(results), 0)

        snap = self.ds.snapshot("v1")
        self.assertEqual(snap, "v1")

        self.ds.remove_duplicates()
        diff_res = self.ds.diff("v1")
        self.assertEqual(diff_res["row_count_delta"], -1)

        self.ds.rollback("v1")
        self.assertEqual(len(self.ds.data), 4)

    def test_expectations_and_pii(self):
        self.ds.define_expectations([
            {"column": "id", "rule": "not_null"},
            {"column": "email", "rule": "unique"}
        ])
        val_res = self.ds.validate()
        self.assertFalse(val_res["passed"])

        pii = self.ds.detect_pii()
        self.assertIn("email", pii)

        self.ds.anonymize(["email"])
        self.assertNotIn("alice@example.com", str(self.ds.data))

        report = self.ds.quality_report()
        self.assertIn("completeness", report)
        self.assertIn("uniqueness", report)

    def test_schema_and_stats_and_mask(self):
        schema = self.ds.infer_schema()
        self.assertIn("age", schema)

        stats = self.ds.column_stats("age")
        self.assertEqual(stats["name"], "age")

        self.ds.mask(["name"])
        self.assertEqual(self.ds.data[0][1], "MASKED_0")

    def test_export_and_ingestion(self):
        json_path = "/tmp/test_qai_data.json"
        parquet_path = "/tmp/test_qai_data.parquet"
        excel_path = "/tmp/test_qai_data.xlsx"

        self.ds.to_json(json_path)
        self.ds.to_parquet(parquet_path)
        self.ds.to_excel(excel_path)

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(parquet_path))
        self.assertTrue(os.path.exists(excel_path))

        self.assertEqual(qdata.detect_format(json_path), "json")
        self.assertEqual(qdata.detect_format(parquet_path), "parquet")

        loaded_ds = qdata.load(json_path)
        self.assertEqual(len(loaded_ds.data), 4)

        os.remove(json_path)
        os.remove(parquet_path)
        os.remove(excel_path)

    def test_batches_and_from_document(self):
        batch_list = list(self.ds.batches(batch_size=2))
        self.assertEqual(len(batch_list), 2)

        doc_path = "/tmp/test_doc.txt"
        with open(doc_path, "w") as f:
            f.write("Hello QAI framework document ingestion test.")

        doc_ds = qdata.from_document(doc_path, chunk_size=10)
        self.assertGreater(len(doc_ds.data), 0)
        os.remove(doc_path)

if __name__ == "__main__":
    unittest.main()

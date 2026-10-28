"""
Unit tests for DatasetHandler: loading, deduplication, target verification, and stratified splitting.
"""

import unittest
import pandas as pd
from src.config import FEATURE_NAMES, DEFAULT_DATASET_PATH
from src.dataset import DatasetHandler


class TestDatasetHandler(unittest.TestCase):
    """Test suite for dataset ingestion and data integrity."""

    def setUp(self):
        self.handler = DatasetHandler(DEFAULT_DATASET_PATH)

    def test_raw_dataset_file_exists(self):
        """Verify the raw CSV file exists and is readable."""
        self.assertTrue(self.handler.data_path.exists())
        raw_df = pd.read_csv(self.handler.data_path)
        self.assertGreaterEqual(len(raw_df), 300)
        for col in FEATURE_NAMES + ["target"]:
            self.assertIn(col, raw_df.columns)

    def test_deduplication_removes_exact_duplicates(self):
        """Verify duplicate records are pruned to prevent CV leakage."""
        X, y, audit = self.handler.load_and_clean()
        self.assertLess(audit["deduplicated_row_count"], audit["raw_row_count"])
        self.assertEqual(len(X), audit["deduplicated_row_count"])
        self.assertEqual(len(y), audit["deduplicated_row_count"])
        # Verify no remaining duplicates in X
        self.assertEqual(X.duplicated().sum(), 0)

    def test_target_label_verification(self):
        """Verify target is standardized so 1 represents disease (positive ST correlation)."""
        X, y, _ = self.handler.load_and_clean()
        # In cardiology, exercise ST depression (oldpeak) correlates positively with disease
        corr = X["oldpeak"].corr(y)
        self.assertGreater(corr, 0.20, f"Expected positive correlation between oldpeak and target, got {corr}")
        self.assertEqual(set(y.unique()), {0, 1})

    def test_stratified_train_test_split(self):
        """Verify stratified split preserves class prevalence across subsets."""
        X_train, X_test, y_train, y_test, meta = self.handler.get_stratified_split(test_size=0.20)
        self.assertEqual(len(X_train), meta["train_samples"])
        self.assertEqual(len(X_test), meta["test_samples"])
        
        # Prevalence difference between train and test must be less than 2%
        diff = abs(meta["train_prevalence"] - meta["test_prevalence"])
        self.assertLess(diff, 0.02, f"Prevalence discrepancy too large: {diff}")


if __name__ == "__main__":
    unittest.main()

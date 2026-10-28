"""
Dataset Loading, Deduplication, Target Verification, and Stratified Partitioning.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_DATASET_PATH,
    FEATURE_NAMES,
    RANDOM_STATE,
    TEST_SPLIT_SIZE
)

logger = logging.getLogger(__name__)


class DatasetHandler:
    """Manages dataset loading, target standardization, and stratified partitioning."""

    def __init__(self, data_path: Path = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_DATASET_PATH
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at path: {self.data_path}")

    def load_and_clean(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Loads the raw CSV, audits duplicate records, standardizes target polarity,
        and returns cleaned feature matrix X and target vector y.
        """
        raw_df = pd.read_csv(self.data_path)
        raw_count = len(raw_df)
        
        # Verify required columns exist
        missing_cols = [c for c in FEATURE_NAMES + ["target"] if c not in raw_df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Audit and drop duplicates to prevent cross-validation data leakage
        dup_count = int(raw_df.duplicated().sum())
        clean_df = raw_df.drop_duplicates().reset_index(drop=True)
        dedup_count = len(clean_df)

        logger.info(
            f"Dataset Audit: Loaded {raw_count} raw rows. "
            f"Removed {dup_count} duplicate records ({dup_count/raw_count*100:.1f}%). "
            f"Retained {dedup_count} unique patient records."
        )

        # Target label verification:
        # In physiological cardiology, exercise ST depression (oldpeak) and major vessels (ca)
        # correlate positively with ischemic coronary artery disease.
        # In this raw CSV, oldpeak has a negative correlation with raw target (-0.43),
        # confirming raw target 0 represents diseased subjects and 1 represents healthy subjects.
        # We standardize: 1 = Heart Disease Present, 0 = Healthy / No Disease.
        raw_target = clean_df["target"]
        if clean_df["oldpeak"].corr(raw_target) < 0:
            target = (1 - raw_target).astype(int)
            logger.info("Target Label Verification: Standardized raw target (inverted) so 1=Disease, 0=Healthy.")
        else:
            target = raw_target.astype(int)
            logger.info("Target Label Verification: Raw target retained (1=Disease, 0=Healthy).")

        X = clean_df[FEATURE_NAMES].copy()
        y = target.copy()

        # Compile audit metadata
        audit_meta = {
            "raw_row_count": raw_count,
            "duplicate_count": dup_count,
            "deduplicated_row_count": dedup_count,
            "disease_positive_count": int((y == 1).sum()),
            "disease_negative_count": int((y == 0).sum()),
            "prevalence_rate": float((y == 1).mean()),
            "feature_count": len(FEATURE_NAMES)
        }

        return X, y, audit_meta

    def get_stratified_split(
        self,
        test_size: float = TEST_SPLIT_SIZE,
        random_state: int = RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Partition dataset into strictly stratified train and test splits.
        Stratification preserves exact disease prevalence across training and evaluation subsets.
        """
        X, y, audit = self.load_and_clean()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(
            f"Stratified Split: {len(X_train)} train samples, {len(X_test)} test samples. "
            f"Train prevalence: {y_train.mean():.3f}, Test prevalence: {y_test.mean():.3f}."
        )

        split_info = {
            **audit,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_prevalence": float(y_train.mean()),
            "test_prevalence": float(y_test.mean())
        }

        return X_train, X_test, y_train, y_test, split_info

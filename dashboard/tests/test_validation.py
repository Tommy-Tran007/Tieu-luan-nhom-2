"""
Unit & Mutation Tests cho dashboard/validation.py (Giai đoạn 1 — Data
Validation).
Gồm 2 nhóm kiểm thử:
1. Kiểm thử dương tính trên dữ liệu thật của dự án (toàn bộ 17/17 hạng mục phải
PASS).
2. Kiểm thử âm tính (Mutate & Break): cố tình tiêm lỗi thiếu cột, trùng lặp id,
   lỗi CRIT-01 (gán is_raining=1 khi weather_matched==0), lỗi CRIT-02 (điền
   datetime khi
   is_timestamp_original==0), lỗi giá âm -> xác minh validation.py bắt đúng
   100% lỗi!
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.data_loader import load_all_result_tables, load_master_dataset
from dashboard.validation import (
    DataValidationError,
    run_full_data_validation,
    validate_master_dataset,
    validate_result_tables,
)


@pytest.fixture(scope="module")
def master_df() -> pd.DataFrame:
    return load_master_dataset()


@pytest.fixture(scope="module")
def sample_clean_df(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trích 2,000 dòng đại diện (gồm cả dòng có TS gốc, thiếu TS, khớp & không
    khớp thời tiết) để chạy mutation test nhanh.
    """
    part_wm1 = master_df.loc[master_df["weather_matched"] == 1].head(1_000)
    part_ts0 = master_df.loc[master_df["is_timestamp_original"] == 0].head(500)
    part_wm0_ts1 = master_df.loc[
        (master_df["is_timestamp_original"] == 1)
        & (master_df["weather_matched"] == 0)
    ].head(500)
    return pd.concat([part_wm1, part_ts0, part_wm0_ts1], ignore_index=True)


def test_full_real_data_validation_passes() -> None:
    """
    Toàn bộ 17 hạng mục kiểm toán trên dữ liệu thật của dự án phải PASS 100%.
    """
    report = run_full_data_validation()
    failed = [(c.check_id, c.message) for c in report.failed_checks]
    assert report.passed, f"Phát hiện hạng mục kiểm toán thất bại: {failed}"
    assert len(report.critical_failures) == 0
    assert report.metrics["n_rows"] == 270_898
    assert report.metrics["n_ts_original"] == 208_034
    assert report.metrics["n_weather_matched"] == 179_280


def test_validation_catches_missing_required_column(
    sample_clean_df: pd.DataFrame,
) -> None:
    """
    Mutation test: xóa cột 'price' phải bị bắt lỗi CRITICAL VAL-01_SCHEMA.
    """
    broken = sample_clean_df.drop(columns=["price"])
    report = validate_master_dataset(broken)
    assert not report.passed
    assert any(
        c.check_id == "VAL-01_SCHEMA" and c.status == "FAIL"
        for c in report.checks
    )
    with pytest.raises(DataValidationError):
        report.raise_if_critical()


def test_validation_catches_crit01_fake_rain_label(
    sample_clean_df: pd.DataFrame,
) -> None:
    """
    Mutation test CRIT-01: cố tình gán is_raining=1 trên dòng
    weather_matched==0 phải bị chặn CRITICAL.
    """
    broken = sample_clean_df.copy()
    idx_unmatched = broken.index[broken["weather_matched"] == 0][0]
    broken.loc[idx_unmatched, "is_raining"] = 1.0

    report = validate_master_dataset(broken)
    crit01 = [
        c for c in report.checks if c.check_id == "VAL-11_CRIT01_WEATHER_MATCH"
    ][0]
    assert crit01.status == "FAIL"
    assert crit01.severity == "CRITICAL"


def test_validation_catches_crit02_ffill_timestamp_leak(
    sample_clean_df: pd.DataFrame,
) -> None:
    """
    Mutation test CRIT-02: cố tình điền datetime/hour trên dòng
    is_timestamp_original==0 phải bị chặn CRITICAL.
    """
    broken = sample_clean_df.copy()
    idx_ts0 = broken.index[broken["is_timestamp_original"] == 0][0]
    broken.loc[idx_ts0, "hour"] = 8.0

    report = validate_master_dataset(broken)
    crit02 = [
        c for c in report.checks if c.check_id == "VAL-10_CRIT02_TIMESTAMP_EST"
    ][0]
    assert crit02.status == "FAIL"
    assert crit02.severity == "CRITICAL"


def test_validation_catches_duplicate_primary_key(
    sample_clean_df: pd.DataFrame,
) -> None:
    """
    Mutation test: nhân bản 1 dòng trùng id phải bị bắt lỗi CRITICAL
    VAL-05_PK_DUPLICATES.
    """
    broken = pd.concat(
        [sample_clean_df, sample_clean_df.iloc[[0]]], ignore_index=True
    )
    report = validate_master_dataset(broken)
    pk_chk = [
        c for c in report.checks if c.check_id == "VAL-05_PK_DUPLICATES"
    ][0]
    assert pk_chk.status == "FAIL"


def test_validation_catches_invalid_price_bound(
    sample_clean_df: pd.DataFrame,
) -> None:
    """Mutation test: tiêm giá âm phải bị bắt lỗi VAL-06_BOUNDS_OUTLIERS."""
    broken = sample_clean_df.copy()
    broken.loc[0, "price"] = np.float32(-10.0)
    report = validate_master_dataset(broken)
    bnd_chk = [
        c for c in report.checks if c.check_id == "VAL-06_BOUNDS_OUTLIERS"
    ][0]
    assert bnd_chk.status == "FAIL"


def test_result_tables_validation_catches_corrupted_permutation_table() -> (
    None
):
    """
    Mutation test HIGH-01: làm trùng cột trong bảng Permutation Importance phải
    bị phát hiện.
    """
    tables = load_all_result_tables()
    bad_perm = tables["permutation_importance"].copy()
    bad_perm.loc[0, "permuted_columns"] = (
        "name, distance"  # cố tình đếm trùng distance
    )
    tables["permutation_importance"] = bad_perm

    report = validate_result_tables(tables)
    perm_chk = [
        c for c in report.checks if c.check_id == "TBL-03_HIGH01_PERMUTATION"
    ][0]
    assert perm_chk.status == "FAIL"

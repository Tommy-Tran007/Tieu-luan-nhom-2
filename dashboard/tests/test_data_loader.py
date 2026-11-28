"""
Unit Tests cho dashboard/data_loader.py (Giai đoạn 1 — Data Layer).
Kiểm tra:
1. Đường dẫn tương đối pathlib hoạt động độc lập với thư mục hiện hành.
2. Nạp thành công master_rideshare.parquet, 14 bảng CSV kết quả và 4 tệp JSON
tổng hợp.
3. Ném DataLoadError rõ ràng khi tệp không tồn tại, tệp rỗng 0 byte hoặc JSON
hỏng.
4. Bộ đo RAM hoạt động chính xác và chứng minh hiệu quả downcasting của tệp
Parquet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.config import BASE_DIR, MASTER_PARQUET_PATH, TABLE_PATHS
from dashboard.data_loader import (
    DataLoadError,
    ensure_file_exists,
    estimate_unoptimized_ram_mb,
    load_all_result_tables,
    load_all_summaries,
    load_data_bundle,
    load_json_artifact,
    load_master_dataset,
    load_result_table,
    measure_ram_mb,
)


def test_relative_paths_resolve_inside_workspace() -> None:
    """
    Đảm bảo mọi đường dẫn cấu hình dùng đường dẫn tương đối và tồn tại thật.
    """
    assert BASE_DIR.exists()
    assert MASTER_PARQUET_PATH.exists()
    assert len(TABLE_PATHS) == 14
    for key, path in TABLE_PATHS.items():
        assert path.exists(), f"Thiếu tệp bảng CSV '{key}': {path}"


def test_load_master_dataset_shape_and_memory() -> None:
    """
    Kiểm tra nạp master_rideshare.parquet đúng kích thước 270,898 x 32 và tối
    ưu RAM.
    """
    df = load_master_dataset()
    assert df.shape == (270_898, 32)
    ram_mb = measure_ram_mb(df)
    est_raw_mb = estimate_unoptimized_ram_mb(df)
    assert 10.0 < ram_mb < 65.0
    assert est_raw_mb > ram_mb, (
        "Dữ liệu downcast phải tiết kiệm RAM hơn so với float64/object."
    )


def test_load_all_result_tables_and_summaries() -> None:
    """Kiểm tra nạp toàn bộ 14 bảng CSV và 4 tệp JSON tổng hợp."""
    tables = load_all_result_tables()
    assert set(tables.keys()) == set(TABLE_PATHS.keys())
    for key, tdf in tables.items():
        assert not tdf.empty, f"Bảng '{key}' bị rỗng."

    summaries = load_all_summaries()
    assert set(summaries.keys()) == {"stage3", "stage4", "stage5", "stage6"}
    assert "part6_surge_cap_policy_simulation" in summaries["stage6"]


def test_load_data_bundle_returns_complete_structure() -> None:
    """
    Kiểm tra hàm load_data_bundle trả về đủ master, tables, summaries và thông
    số RAM.
    """
    bundle = load_data_bundle()
    assert bundle["master"].shape == (270_898, 32)
    assert len(bundle["tables"]) == 14
    assert len(bundle["summaries"]) == 4
    assert "v5" in bundle
    assert bundle["ram_mb"] < bundle["est_unoptimized_ram_mb"]


def test_missing_file_raises_data_load_error(tmp_path: Path) -> None:
    """Kiểm thử âm tính: tệp không tồn tại phải ném DataLoadError rõ ràng."""
    fake_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(DataLoadError, match="Không tìm thấy tệp"):
        load_master_dataset(fake_path)


def test_empty_file_raises_data_load_error(tmp_path: Path) -> None:
    """Kiểm thử âm tính: tệp rỗng 0 byte phải bị chặn ngay lập tức."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(DataLoadError, match="đang rỗng"):
        ensure_file_exists(empty_file, "file kiểm thử rỗng")


def test_corrupted_json_raises_data_load_error(tmp_path: Path) -> None:
    """Kiểm thử âm tính: tệp JSON lỗi cú pháp phải ném DataLoadError."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DataLoadError, match="Lỗi giải mã JSON"):
        load_json_artifact(bad_json, "bad.json")


def test_unknown_table_key_raises_key_error() -> None:
    """Kiểm thử âm tính: khóa bảng không tồn tại phải báo KeyError."""
    with pytest.raises(KeyError):
        load_result_table("non_existent_table_key")

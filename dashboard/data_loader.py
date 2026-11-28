"""
Module: dashboard/data_loader.py
Mục đích: Tầng nạp dữ liệu thuần túy (Pure Data Layer) cho Dashboard
PRJ-EXT-01.
- Đọc an toàn các tệp Parquet, CSV và JSON đã qua xử lý bằng đường dẫn tương
  đối.
- Kiểm tra tồn tại tệp trước khi nạp và ném lỗi tường minh (DataLoadError) nếu
  thiếu/hỏng.
- Không sửa đổi, không xóa dòng, không điền khuyết (imputation) dữ liệu nguồn.
- Tách bạch hoàn toàn khỏi Business Logic, Model Inference và Presentation
  Layer (Streamlit).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.config import (
    AUDIT_FLAGS_PARQUET_PATH,
    CAB_CLEAN_PARQUET_PATH,
    MASTER_PARQUET_PATH,
    STAGE3_AUDIT_JSON_PATH,
    STAGE4_SUMMARY_JSON_PATH,
    STAGE5_SUMMARY_JSON_PATH,
    STAGE6_SUMMARY_JSON_PATH,
    TABLE_PATHS,
    THESIS_V5_JSON_PATH,
    WEATHER_CLEAN_CSV_PATH,
)


class DataLoadError(RuntimeError):
    """
    Ngoại lệ chuyên biệt khi thiếu tệp nguồn hoặc không thể giải mã tệp dữ
    liệu.
    """


def ensure_file_exists(path: Path, label: str) -> Path:
    """Xác minh tệp tồn tại và không rỗng (0 byte) trước khi nạp."""
    if not path.exists():
        raise DataLoadError(
            f"Không tìm thấy tệp {label} tại đường dẫn tương đối: {path}. "
            "Vui lòng chạy lại pipeline tiền xử lý/đánh giá (python "
            "main.py --stage all) trước khi mở Dashboard."
        )
    if not path.is_file() or path.stat().st_size == 0:
        raise DataLoadError(
            f"Tệp {label} tại {path} không hợp lệ hoặc đang rỗng (0 byte)."
        )
    return path


def measure_ram_mb(df: pd.DataFrame) -> float:
    """Đo dung lượng RAM thực tế (MB, deep=True) của một DataFrame."""
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def estimate_unoptimized_ram_mb(df: pd.DataFrame) -> float:
    """
    Ước lượng dung lượng RAM (MB) nếu cùng DataFrame đó lưu ở kiểu mặc định
    chưa tối ưu
    (object cho cột category và float64 cho cột float32/int8). Không làm thay
    đổi df gốc.
    """
    total_bytes = 0
    n_rows = len(df)
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            # Mỗi ô chuỗi Python object trung bình ~64 bytes (con trỏ 8B +
            # PyUnicodeObject ~56B)
            cats = df[col].cat.categories
            avg_str_len = (
                (sum(len(str(v)) for v in cats) / len(cats))
                if len(cats) > 0
                else 8.0
            )
            total_bytes += int(n_rows * (56.0 + avg_str_len))
        elif pd.api.types.is_float_dtype(
            dtype
        ) or pd.api.types.is_integer_dtype(dtype):
            total_bytes += n_rows * 8  # float64 / int64
        else:
            total_bytes += int(df[col].memory_usage(deep=True))
    return float(total_bytes / (1024**2))


def load_master_dataset(path: Path = MASTER_PARQUET_PATH) -> pd.DataFrame:
    """
    Nạp bảng dữ liệu sạch chính thức master_rideshare.parquet (270,898 x 32).
    Bảo toàn nguyên vẹn toàn bộ kiểu dữ liệu tối ưu (float32, int8, Int64,
    category) và các giá trị NaN có chủ đích.
    """
    verified_path = ensure_file_exists(
        path, "dữ liệu chính master_rideshare.parquet"
    )
    try:
        df = pd.read_parquet(verified_path)
    except Exception as exc:
        raise DataLoadError(
            f"Lỗi giải mã tệp Parquet tại {verified_path}: {exc}"
        ) from exc

    if df.empty:
        raise DataLoadError(f"Tệp {verified_path} có 0 dòng dữ liệu.")
    return df


def load_cab_cleaned(path: Path = CAB_CLEAN_PARQUET_PATH) -> pd.DataFrame:
    """
    Nạp bảng cab_rides_cleaned.parquet (trước Left Join) để kiểm toán chống
    fan-out.
    """
    verified_path = ensure_file_exists(path, "cab_rides_cleaned.parquet")
    try:
        return pd.read_parquet(verified_path)
    except Exception as exc:
        raise DataLoadError(f"Lỗi đọc {verified_path}: {exc}") from exc


def load_weather_cleaned(path: Path = WEATHER_CLEAN_CSV_PATH) -> pd.DataFrame:
    """
    Nạp bảng weather_cleaned.csv để kiểm toán trạm khí tượng và ngưỡng lượng
    mưa.
    """
    verified_path = ensure_file_exists(path, "weather_cleaned.csv")
    try:
        return pd.read_csv(verified_path)
    except Exception as exc:
        raise DataLoadError(f"Lỗi đọc {verified_path}: {exc}") from exc


def load_audit_flags(path: Path = AUDIT_FLAGS_PARQUET_PATH) -> pd.DataFrame:
    """
    Nạp bảng cờ kiểm toán R3 (stage4_audit_flags.parquet) đánh dấu tỷ lệ bản
    ghi nguyên bản.
    """
    verified_path = ensure_file_exists(path, "stage4_audit_flags.parquet")
    try:
        return pd.read_parquet(verified_path)
    except Exception as exc:
        raise DataLoadError(f"Lỗi đọc {verified_path}: {exc}") from exc


def load_result_table(
    table_key: str, tables_map: dict[str, Path] | None = None
) -> pd.DataFrame:
    """
    Nạp một bảng kết quả CSV chuẩn từ reports/tables/ theo khóa định danh.
    """
    mapping = tables_map if tables_map is not None else TABLE_PATHS
    if table_key not in mapping:
        raise KeyError(
            f"Khóa bảng '{table_key}' không tồn tại trong cấu hình "
            "TABLE_PATHS."
        )
    verified_path = ensure_file_exists(
        mapping[table_key], f"bảng kết quả '{table_key}'"
    )
    try:
        df = pd.read_csv(verified_path)
    except Exception as exc:
        raise DataLoadError(
            f"Lỗi đọc CSV '{table_key}' tại {verified_path}: {exc}"
        ) from exc
    if df.empty:
        raise DataLoadError(
            f"Bảng kết quả '{table_key}' tại {verified_path} bị rỗng."
        )
    return df


def load_all_result_tables(
    tables_map: dict[str, Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """Nạp toàn bộ 14 bảng CSV chuẩn đã kiểm toán tại reports/tables/."""
    mapping = tables_map if tables_map is not None else TABLE_PATHS
    return {key: load_result_table(key, mapping) for key in mapping}


def load_json_artifact(path: Path, label: str) -> dict[str, Any]:
    """Nạp một tệp JSON tổng hợp kết quả/kiểm toán (UTF-8)."""
    verified_path = ensure_file_exists(path, label)
    try:
        data = json.loads(verified_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataLoadError(
            f"Lỗi giải mã JSON '{label}' tại {verified_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not data:
        raise DataLoadError(
            f"Tệp JSON '{label}' tại {verified_path} không có cấu trúc đối "
            "tượng hợp lệ."
        )
    return data


def load_all_summaries() -> dict[str, dict[str, Any]]:
    """
    Nạp toàn bộ 4 tệp JSON tổng hợp kiểm toán từ Giai đoạn 3 đến Giai đoạn 6.
    """
    return {
        "stage3": load_json_artifact(
            STAGE3_AUDIT_JSON_PATH, "stage3_audit_summary.json"
        ),
        "stage4": load_json_artifact(
            STAGE4_SUMMARY_JSON_PATH, "stage4_summary.json"
        ),
        "stage5": load_json_artifact(
            STAGE5_SUMMARY_JSON_PATH, "stage5_summary.json"
        ),
        "stage6": load_json_artifact(
            STAGE6_SUMMARY_JSON_PATH, "stage6_summary.json"
        ),
    }


def load_v5_verification(path: Path = THESIS_V5_JSON_PATH) -> dict[str, Any]:
    """
    Nạp nguồn sự thật kiểm chứng hợp nhất thesis_v5_verification.json (nếu có).
    """
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return load_json_artifact(path, "thesis_v5_verification.json")


def load_data_bundle() -> dict[str, Any]:
    """
    Nạp trọn gói dữ liệu chuẩn cho Dashboard (master Parquet + 14 bảng CSV + 4
    JSON tổng hợp + v5 verification).
    Không thực hiện bất kỳ biến đổi làm sạch hay lọc phá hủy nào.
    """
    master = load_master_dataset()
    tables = load_all_result_tables()
    summaries = load_all_summaries()
    v5_data = load_v5_verification()
    return {
        "master": master,
        "tables": tables,
        "summaries": summaries,
        "v5": v5_data,
        "ram_mb": round(measure_ram_mb(master), 2),
        "est_unoptimized_ram_mb": round(
            estimate_unoptimized_ram_mb(master), 2
        ),
    }

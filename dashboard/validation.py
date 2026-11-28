"""
Module: dashboard/validation.py
Mục đích: Bộ kiểm toán và xác thực chất lượng dữ liệu tự động (Automated Data
Validation Gate)
cho Giai đoạn 1 và Runtime của Dashboard PRJ-EXT-01.

Thực hiện đầy đủ kiểm định 13 hạng mục:
1. Schema & Gatekeeper cỡ mẫu >= 100,000 dòng.
2. Kiểu dữ liệu (dtypes) & hiệu quả tối ưu hóa bộ nhớ RAM.
3. Missing values (phân biệt missing lỗi bắt buộc vs missing có chủ đích
CRIT-01/CRIT-02).
4. Duplicate rows & Khóa chính (Primary Key id).
5. Outliers & Miền giá trị vật lý (không xóa dữ liệu, chỉ kiểm toán minh bạch).
6. Khóa ngoại & Toàn vẹn quan hệ (product_id 1-1, 12 quận Boston, 72 tuyến,
location == source).
7. Kết quả phép ghép nối Left Join (0 fan-out so với cab_rides_cleaned.parquet
& khớp R3 flags).
8. CRIT-02: Timestamp gốc vs thiếu, 0 ffill, kiểm chứng trực tiếp múi giờ
America/New_York (Boston EST).
9. CRIT-01: Weather matching thực, 0 nhãn is_raining giả trên dòng không khớp,
ngưỡng rain <= 0.85 in/h.
10. Kiểm định toàn bộ 14 bảng CSV đầu vào Dashboard (bao gồm HIGH-01: 14 nhóm =
27/27 cột X).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dashboard.config import (
    BOSTON_TIMEZONE,
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    EMPIRICAL_PRICE_MAX,
    EMPIRICAL_PRICE_MIN,
    EXPECTED_ROUTE_COUNT,
    GATEKEEPER_MIN_ROWS,
    MASTER_NON_NULL_COLUMNS,
    MASTER_REQUIRED_COLUMNS,
    MAX_VALID_RAIN_INCH_PER_HOUR,
    TABLE_REQUIRED_COLUMNS,
    TIMESTAMP_DEPENDENT_COLS,
    ULTRA_SHORT_GPS_THRESHOLD,
    VALID_BOSTON_DISTRICTS,
    VALID_CAB_TYPES,
    VALID_SURGE_LEVELS,
    WEATHER_DEPENDENT_COLS,
)
from dashboard.data_loader import (
    estimate_unoptimized_ram_mb,
    load_all_result_tables,
    load_audit_flags,
    load_cab_cleaned,
    load_master_dataset,
    load_weather_cleaned,
    measure_ram_mb,
)


class DataValidationError(ValueError):
    """
    Ngoại lệ nghiêm trọng khi dữ liệu đầu vào vi phạm hợp đồng kiểm toán
    CRITICAL.
    """


@dataclass
class ValidationCheck:
    """Kết quả một hạng mục kiểm toán độc lập."""

    check_id: str
    category: str
    status: str  # "PASS" | "WARN" | "FAIL"
    severity: str  # "CRITICAL" | "HIGH" | "MEDIUM" | "INFO"
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """
    Báo cáo tổng hợp kiểm toán dữ liệu có thể xuất bảng hoặc kiểm tra tự động.
    """

    checks: list[ValidationCheck] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def critical_failures(self) -> list[ValidationCheck]:
        return [
            c
            for c in self.checks
            if c.status == "FAIL" and c.severity == "CRITICAL"
        ]

    @property
    def warning_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "WARN"]

    @property
    def passed(self) -> bool:
        return len(self.failed_checks) == 0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check_id": c.check_id,
                    "category": c.category,
                    "severity": c.severity,
                    "status": c.status,
                    "message": c.message,
                    "evidence": str(c.evidence),
                }
                for c in self.checks
            ]
        )

    def raise_if_critical(self) -> None:
        if self.critical_failures:
            details = "; ".join(
                f"[{c.check_id}] {c.message}" for c in self.critical_failures
            )
            raise DataValidationError(
                f"Kiểm toán dữ liệu thất bại mức CRITICAL: {details}"
            )


def _iqr_outlier_count(series: pd.Series) -> dict[str, Any]:
    """
    Đếm ngoại lai theo quy tắc Tukey 1.5*IQR mà không làm thay đổi dữ liệu gốc.
    """
    s = series.dropna().astype(np.float64)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_low = int((s < lower).sum())
    n_high = int((s > upper).sum())
    return {
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_fence": round(lower, 4),
        "upper_fence": round(upper, 4),
        "n_outliers_low": n_low,
        "n_outliers_high": n_high,
        "n_outliers_total": n_low + n_high,
        "pct_outliers": round((n_low + n_high) / max(len(s), 1) * 100.0, 2),
    }


def validate_master_dataset(
    df: pd.DataFrame,
    cab_clean_df: pd.DataFrame | None = None,
    weather_clean_df: pd.DataFrame | None = None,
    audit_flags_df: pd.DataFrame | None = None,
    tz_sample_size: int = 1_000,
) -> ValidationReport:
    """
    Thực hiện toàn bộ kiểm toán dữ liệu trên bảng chính
    master_rideshare.parquet
    (Schema, Dtypes, Missing, Duplicate, Outliers, Keys, Join, CRIT-02
    Timestamp EST, CRIT-01 Weather, RAM).
    """
    report = ValidationReport()
    n_rows = len(df)
    n_cols = df.shape[1]

    # 1. Schema & Gatekeeper cỡ mẫu
    missing_cols = [c for c in MASTER_REQUIRED_COLUMNS if c not in df.columns]
    report.checks.append(
        ValidationCheck(
            check_id="VAL-01_SCHEMA",
            category="Schema",
            status="PASS" if not missing_cols else "FAIL",
            severity="CRITICAL",
            message=(
                f"Đủ toàn bộ {len(MASTER_REQUIRED_COLUMNS)} cột bắt buộc "
                f"(hiện có {n_cols} cột)."
                if not missing_cols
                else f"Thiếu {len(missing_cols)} cột bắt buộc: {missing_cols}"
            ),
            evidence={
                "n_rows": n_rows,
                "n_cols": n_cols,
                "missing_cols": missing_cols,
            },
        )
    )
    if missing_cols:
        return report  # Không thể kiểm tra sâu hơn nếu thiếu cột bắt buộc

    report.checks.append(
        ValidationCheck(
            check_id="VAL-02_GATEKEEPER_SIZE",
            category="Sample Size",
            status="PASS" if n_rows >= GATEKEEPER_MIN_ROWS else "FAIL",
            severity="CRITICAL",
            message=(
                f"Cỡ mẫu đạt {n_rows:,} quan sát (ngưỡng tiên quyết Vòng 1 >"
                f"= {GATEKEEPER_MIN_ROWS:,})."
            ),
            evidence={"n_rows": n_rows, "min_required": GATEKEEPER_MIN_ROWS},
        )
    )

    # 2. Kiểu dữ liệu & Tối ưu bộ nhớ RAM
    ram_mb = measure_ram_mb(df)
    est_raw_mb = estimate_unoptimized_ram_mb(df)
    ram_saved_pct = (1.0 - ram_mb / max(est_raw_mb, 1e-6)) * 100.0
    cat_cols = [
        c
        for c in [
            "cab_type",
            "name",
            "product_id",
            "source",
            "destination",
            "location",
            "day_name",
            "time_of_day",
        ]
        if isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    dtype_ok = (
        pd.api.types.is_numeric_dtype(df["price"])
        and pd.api.types.is_numeric_dtype(df["distance"])
        and pd.api.types.is_numeric_dtype(df["surge_multiplier"])
        and pd.api.types.is_datetime64_any_dtype(df["datetime"])
        and pd.api.types.is_datetime64_any_dtype(df["date_hour"])
        and len(cat_cols) == 8
    )
    report.checks.append(
        ValidationCheck(
            check_id="VAL-03_DTYPES_RAM",
            category="Dtypes & Memory",
            status="PASS" if dtype_ok else "FAIL",
            severity="HIGH",
            message=(
                f"Kiểu dữ liệu chuẩn xác; RAM thực tế = {ram_mb:.2f} MB "
                f"(tiết kiệm ~{ram_saved_pct:.1f}% so với ước lượng chưa "
                f"tối ưu ~{est_raw_mb:.1f} MB; {len(cat_cols)}/8 cột "
                "category)."
            ),
            evidence={
                "ram_mb": round(ram_mb, 2),
                "est_unoptimized_ram_mb": round(est_raw_mb, 2),
                "ram_saved_pct": round(ram_saved_pct, 1),
                "category_cols_count": len(cat_cols),
            },
        )
    )

    # 3. Missing values ở các cột lõi KHÔNG ĐƯỢC THIẾU
    core_missing = {
        c: int(df[c].isna().sum())
        for c in MASTER_NON_NULL_COLUMNS
        if int(df[c].isna().sum()) > 0
    }
    report.checks.append(
        ValidationCheck(
            check_id="VAL-04_CORE_MISSING",
            category="Missing Values",
            status="PASS" if not core_missing else "FAIL",
            severity="CRITICAL",
            message=(
                "0 giá trị thiếu trên toàn bộ "
                f"{len(MASTER_NON_NULL_COLUMNS)} cột lõi bắt buộc."
                if not core_missing
                else (
                    "Phát hiện giá trị thiếu bất hợp lệ trên cột lõi: "
                    f"{core_missing}"
                )
            ),
            evidence={"core_missing_counts": core_missing},
        )
    )

    # 4. Duplicate & Primary Key (id)
    n_dup_rows = int(df.duplicated().sum())
    n_unique_id = int(df["id"].nunique(dropna=False))
    pk_ok = (n_dup_rows == 0) and (n_unique_id == n_rows)
    report.checks.append(
        ValidationCheck(
            check_id="VAL-05_PK_DUPLICATES",
            category="Primary Key & Duplicates",
            status="PASS" if pk_ok else "FAIL",
            severity="CRITICAL",
            message=(
                f"Khóa chính id duy nhất {n_unique_id:,}/{n_rows:,}; số "
                f"dòng trùng lặp = {n_dup_rows}."
            ),
            evidence={
                "n_unique_id": n_unique_id,
                "n_rows": n_rows,
                "n_dup_rows": n_dup_rows,
            },
        )
    )

    # 5. Outliers & Miền giá trị vật lý (chỉ kiểm toán, không xóa)
    p_min, p_max = float(df["price"].min()), float(df["price"].max())
    d_min, d_max = float(df["distance"].min()), float(df["distance"].max())
    observed_surges = {
        round(float(x), 2) for x in df["surge_multiplier"].dropna().unique()
    }
    invalid_surges = sorted(observed_surges - set(VALID_SURGE_LEVELS))
    uber_surge_gt1 = int(
        (
            (df["cab_type"].astype(str) == "Uber")
            & (df["surge_multiplier"] > 1.0)
        ).sum()
    )
    lyft_surge_gt1 = int(
        (
            (df["cab_type"].astype(str) == "Lyft")
            & (df["surge_multiplier"] > 1.0)
        ).sum()
    )
    n_ultra_short = int((df["distance"] <= ULTRA_SHORT_GPS_THRESHOLD).sum())
    n_ppm_na = int(df["price_per_mile"].isna().sum())

    price_iqr = _iqr_outlier_count(df["price"])
    dist_iqr = _iqr_outlier_count(df["distance"])

    bounds_ok = (
        (p_min >= EMPIRICAL_PRICE_MIN - 1e-6)
        and (p_max <= EMPIRICAL_PRICE_MAX + 1e-6)
        and (d_min >= EMPIRICAL_DISTANCE_MIN - 1e-4)
        and (d_max <= EMPIRICAL_DISTANCE_MAX + 1e-4)
        and (not invalid_surges)
        and (uber_surge_gt1 == 0)
    )
    report.checks.append(
        ValidationCheck(
            check_id="VAL-06_BOUNDS_OUTLIERS",
            category="Domain Bounds & Outliers",
            status="PASS" if bounds_ok else "FAIL",
            severity="HIGH",
            message=(
                f"Miền hợp lệ: price=[${p_min:.2f}, ${p_max:.2f}], "
                f"distance=[{d_min:.2f}, {d_max:.2f}] dặm, "
                f"Uber surge>1 = {uber_surge_gt1}, Lyft surge>1 = "
                f"{lyft_surge_gt1:,}. "
                f"Ghi nhận minh bạch {price_iqr['n_outliers_total']:,} "
                "price IQR outliers, "
                f"{dist_iqr['n_outliers_total']:,} distance IQR outliers và "
                f"{n_ultra_short} cuốc GPS <= 0.05 dặm (giữ nguyên không "
                "xóa)."
            ),
            evidence={
                "price_min": round(p_min, 2),
                "price_max": round(p_max, 2),
                "distance_min": round(d_min, 4),
                "distance_max": round(d_max, 4),
                "invalid_surges": invalid_surges,
                "uber_surge_gt1": uber_surge_gt1,
                "lyft_surge_gt1": lyft_surge_gt1,
                "ultra_short_gps_count": n_ultra_short,
                "price_per_mile_na_count": n_ppm_na,
                "price_iqr": price_iqr,
                "distance_iqr": dist_iqr,
            },
        )
    )

    # 6. Khóa ngoại & Toàn vẹn quan hệ
    invalid_cabs = sorted(
        set(df["cab_type"].astype(str).unique()) - set(VALID_CAB_TYPES)
    )
    invalid_sources = sorted(
        set(df["source"].astype(str).unique()) - set(VALID_BOSTON_DISTRICTS)
    )
    invalid_dests = sorted(
        set(df["destination"].astype(str).unique())
        - set(VALID_BOSTON_DISTRICTS)
    )
    self_loop_count = int(
        (df["source"].astype(str) == df["destination"].astype(str)).sum()
    )
    n_routes = int(
        df.groupby(["source", "destination"], observed=True).ngroups
    )

    # Kiểm tra ánh xạ 1-1 product_id <-> (cab_type, name)
    pid_to_pair_max = int(
        df.groupby("product_id", observed=True)["name"].nunique().max()
    )
    pair_to_pid_max = int(
        df.groupby(["cab_type", "name"], observed=True)["product_id"]
        .nunique()
        .max()
    )

    # Kiểm tra location == source trên 100% bản ghi weather_matched == 1
    wm1_mask = df["weather_matched"] == 1
    loc_mismatch = int(
        (
            df.loc[wm1_mask, "location"].astype(str)
            != df.loc[wm1_mask, "source"].astype(str)
        ).sum()
    )

    fk_ok = (
        not invalid_cabs
        and not invalid_sources
        and not invalid_dests
        and self_loop_count == 0
        and n_routes == EXPECTED_ROUTE_COUNT
        and pid_to_pair_max == 1
        and pair_to_pid_max == 1
        and loc_mismatch == 0
    )
    report.checks.append(
        ValidationCheck(
            check_id="VAL-07_RELATIONAL_KEYS",
            category="Foreign Keys & Relations",
            status="PASS" if fk_ok else "FAIL",
            severity="HIGH",
            message=(
                "Toàn vẹn quan hệ đạt chuẩn: 12 quận Boston, đúng "
                f"{n_routes} tuyến thực tế (0 self-loop), "
                "product_id <-> (cab_type, name) ánh xạ 1-1, 0 sai lệch "
                "location vs source."
            ),
            evidence={
                "invalid_cabs": invalid_cabs,
                "invalid_sources": invalid_sources,
                "invalid_dests": invalid_dests,
                "self_loop_count": self_loop_count,
                "n_routes": n_routes,
                "pid_to_pair_max": pid_to_pair_max,
                "pair_to_pid_max": pair_to_pid_max,
                "location_source_mismatch": loc_mismatch,
            },
        )
    )

    # 7. Kiểm tra kết quả Join chống Fan-out (nếu cung cấp cab_clean_df &
    # audit_flags_df)
    if cab_clean_df is not None:
        join_fanout = len(df) - len(cab_clean_df)
        same_ids = set(df["id"]) == set(cab_clean_df["id"])
        report.checks.append(
            ValidationCheck(
                check_id="VAL-08_JOIN_NO_FANOUT",
                category="Join Integrity",
                status="PASS" if (join_fanout == 0 and same_ids) else "FAIL",
                severity="CRITICAL",
                message=(
                    f"Left Join bảo toàn đúng {len(df):,} = "
                    f"{len(cab_clean_df):,} dòng (fan-out = {join_fanout}, "
                    "tập khóa id trùng khớp 100%)."
                ),
                evidence={
                    "master_rows": len(df),
                    "cab_clean_rows": len(cab_clean_df),
                    "fanout": join_fanout,
                    "same_id_set": same_ids,
                },
            )
        )

    if audit_flags_df is not None:
        flags_aligned = (len(audit_flags_df) == n_rows) and (
            set(audit_flags_df["id"]) == set(df["id"])
        )
        r3_pct = (
            float(audit_flags_df["is_strictly_unmodified_raw"].mean() * 100.0)
            if flags_aligned
            else 0.0
        )
        report.checks.append(
            ValidationCheck(
                check_id="VAL-09_R3_AUDIT_FLAGS",
                category="Provenance (R3)",
                status="PASS" if flags_aligned else "FAIL",
                severity="MEDIUM",
                message=(
                    "Khớp 100% cờ R3 (stage4_audit_flags.parquet): "
                    f"{r3_pct:.2f}% bản ghi gốc nguyên bản không qua scale-"
                    "fix."
                ),
                evidence={
                    "aligned": flags_aligned,
                    "strictly_unmodified_raw_pct": round(r3_pct, 2),
                },
            )
        )

    # 8. CRIT-02: Kiểm toán Timestamp, 0 ffill & Múi giờ Boston
    # America/New_York
    ts_flags = set(df["is_timestamp_original"].unique().tolist())
    ts1_mask = df["is_timestamp_original"] == 1
    ts0_mask = df["is_timestamp_original"] == 0
    n_ts1 = int(ts1_mask.sum())
    n_ts0 = int(ts0_mask.sum())

    # Trên dòng thiếu TS gốc (ts0), toàn bộ biến thời gian số & datetime phải
    # là NaN và weather_matched phải = 0
    ts0_leaks = {
        c: int(df.loc[ts0_mask, c].notna().sum())
        for c in TIMESTAMP_DEPENDENT_COLS
    }
    ts0_weather_leak = int((df.loc[ts0_mask, "weather_matched"] == 1).sum())
    ts0_tod_non_unknown = int(
        (df.loc[ts0_mask, "time_of_day"].astype(str) != "Unknown").sum()
    )

    # Trên dòng có TS gốc (ts1), kiểm tra 0 NaN và kiểm chứng chuyển đổi
    # America/New_York
    ts1_missing = {
        c: int(df.loc[ts1_mask, c].isna().sum())
        for c in TIMESTAMP_DEPENDENT_COLS
    }
    tz_mismatch_count = 0
    ts_min_str, ts_max_str = "", ""
    if (
        n_ts1 > 0
        and ts1_missing["time_stamp"] == 0
        and ts1_missing["datetime"] == 0
    ):
        sample_df = df.loc[ts1_mask, ["time_stamp", "datetime"]].iloc[
            np.linspace(0, n_ts1 - 1, min(tz_sample_size, n_ts1), dtype=int)
        ]
        expected_est = (
            pd.to_datetime(
                sample_df["time_stamp"].astype("float64"), unit="ms", utc=True
            )
            .dt.tz_convert(BOSTON_TIMEZONE)
            .dt.tz_localize(None)
        )
        tz_mismatch_count = int(
            (sample_df["datetime"].values != expected_est.values).sum()
        )
        ts_min_str = str(df.loc[ts1_mask, "datetime"].min())
        ts_max_str = str(df.loc[ts1_mask, "datetime"].max())

    crit02_ok = (
        ts_flags.issubset({0, 1})
        and all(v == 0 for v in ts0_leaks.values())
        and ts0_weather_leak == 0
        and ts0_tod_non_unknown == 0
        and all(v == 0 for v in ts1_missing.values())
        and tz_mismatch_count == 0
    )
    report.checks.append(
        ValidationCheck(
            check_id="VAL-10_CRIT02_TIMESTAMP_EST",
            category="CRIT-02 Timestamp & EST",
            status="PASS" if crit02_ok else "FAIL",
            severity="CRITICAL",
            message=(
                f"CRIT-02 ĐẠT: TS gốc = {n_ts1:,} "
                f"({n_ts1 / n_rows * 100:.2f}%), thiếu TS = {n_ts0:,} "
                f"({n_ts0 / n_rows * 100:.2f}%), "
                "0 dòng bị ffill/suy đoán, 0 sai lệch múi giờ "
                f"{BOSTON_TIMEZONE} (kiểm chứng "
                f"{min(tz_sample_size, n_ts1):,} mốc: {ts_min_str} -> "
                f"{ts_max_str})."
                if crit02_ok
                else (
                    f"VI PHẠM CRIT-02: ts0_leaks={ts0_leaks}, "
                    f"ts0_weather={ts0_weather_leak}, "
                    f"ts0_tod={ts0_tod_non_unknown}, "
                    f"ts1_missing={ts1_missing}, "
                    f"tz_mismatch={tz_mismatch_count}"
                )
            ),
            evidence={
                "n_ts_original": n_ts1,
                "n_ts_missing": n_ts0,
                "pct_ts_original": round(n_ts1 / max(n_rows, 1) * 100.0, 2),
                "ts0_leaks": ts0_leaks,
                "ts0_weather_leak": ts0_weather_leak,
                "tz_mismatch_count": tz_mismatch_count,
                "est_start": ts_min_str,
                "est_end": ts_max_str,
            },
        )
    )

    # 9. CRIT-01 & H4: Kiểm toán Weather Matching, tách bạch is_rain_sentinel &
    # 0 nhãn is_raining giả
    wm_flags = set(df["weather_matched"].unique().tolist())
    wm0_mask = df["weather_matched"] == 0
    n_wm1 = int(wm1_mask.sum())
    n_wm0 = int(wm0_mask.sum())
    wm0_leaks = {
        c: int(df.loc[wm0_mask, c].notna().sum())
        for c in WEATHER_DEPENDENT_COLS
    }

    # Trên dòng khớp thời tiết (weather_matched == 1): 6 biến thời tiết liên
    # tục & location phải đủ 100%;
    # riêng rain & is_raining chỉ không khuyết khi is_rain_sentinel == 0, và
    # bắt buộc NaN khi is_rain_sentinel == 1.
    non_rain_wea_cols = [
        c for c in WEATHER_DEPENDENT_COLS if c not in ("rain", "is_raining")
    ]
    wm1_non_rain_missing = {
        c: int(df.loc[wm1_mask, c].isna().sum()) for c in non_rain_wea_cols
    }

    rs_flags = set(df["is_rain_sentinel"].unique().tolist())
    rs1_mask = df["is_rain_sentinel"] == 1
    n_rs1 = int(rs1_mask.sum())
    wm1_valid_rain_mask = wm1_mask & (~rs1_mask)
    n_wm1_valid_rain = int(wm1_valid_rain_mask.sum())

    rs1_rain_leaks = int(
        df.loc[rs1_mask, "rain"].notna().sum()
        + df.loc[rs1_mask, "is_raining"].notna().sum()
    )
    rs1_unmatched_leaks = int((rs1_mask & wm0_mask).sum())
    valid_rain_missing = int(
        df.loc[wm1_valid_rain_mask, "rain"].isna().sum()
        + df.loc[wm1_valid_rain_mask, "is_raining"].isna().sum()
    )

    rain_logic_mismatch = 0
    max_rain_master = 0.0
    n_rain_true = 0
    if n_wm1_valid_rain > 0 and valid_rain_missing == 0:
        expected_rain_flag = (
            df.loc[wm1_valid_rain_mask, "rain"] > 0.0
        ).astype(np.float32)
        rain_logic_mismatch = int(
            (
                df.loc[wm1_valid_rain_mask, "is_raining"].astype(np.float32)
                != expected_rain_flag
            ).sum()
        )
        max_rain_master = float(df.loc[wm1_valid_rain_mask, "rain"].max())
        n_rain_true = int(
            (df.loc[wm1_valid_rain_mask, "is_raining"] == 1.0).sum()
        )

    wm_rate_all = n_wm1 / max(n_rows, 1) * 100.0
    wm_rate_ts = n_wm1 / max(n_ts1, 1) * 100.0

    crit01_ok = (
        wm_flags.issubset({0, 1})
        and rs_flags.issubset({0, 1})
        and all(v == 0 for v in wm0_leaks.values())
        and all(v == 0 for v in wm1_non_rain_missing.values())
        and rs1_rain_leaks == 0
        and rs1_unmatched_leaks == 0
        and valid_rain_missing == 0
        and rain_logic_mismatch == 0
        and max_rain_master <= MAX_VALID_RAIN_INCH_PER_HOUR
    )
    report.checks.append(
        ValidationCheck(
            check_id="VAL-11_CRIT01_WEATHER_MATCH",
            category="CRIT-01 & H4 Weather Matching",
            status="PASS" if crit01_ok else "FAIL",
            severity="CRITICAL",
            message=(
                f"CRIT-01 & H4 ĐẠT: Khớp thời tiết thực = {n_wm1:,} "
                f"({wm_rate_all:.2f}% toàn tập; {wm_rate_ts:.2f}% trên TS "
                "gốc), "
                f"tách bạch {n_rs1:,} dòng ô sentinel mưa (giữ rain/"
                f"is_raining = NaN), {n_wm1_valid_rain:,} dòng mưa/khô hợp "
                "lệ "
                f"(Mưa thực = {n_rain_true:,}, max(rain) = "
                f"{max_rain_master:.4f} in/h), 0 nhãn is_raining giả."
                if crit01_ok
                else (
                    f"VI PHẠM CRIT-01/H4: wm0_leaks={wm0_leaks}, "
                    f"non_rain_miss={wm1_non_rain_missing}, "
                    f"rs1_leaks={rs1_rain_leaks}, "
                    f"valid_rain_miss={valid_rain_missing}, "
                    f"logic_mismatch={rain_logic_mismatch}"
                )
            ),
            evidence={
                "n_weather_matched": n_wm1,
                "n_weather_unmatched": n_wm0,
                "n_rain_sentinel_rows": n_rs1,
                "n_valid_rain_obs": n_wm1_valid_rain,
                "match_rate_all_pct": round(wm_rate_all, 2),
                "match_rate_valid_ts_pct": round(wm_rate_ts, 2),
                "n_rain_true": n_rain_true,
                "n_dry_true": n_wm1_valid_rain - n_rain_true,
                "wm0_leaks": wm0_leaks,
                "rain_logic_mismatch": rain_logic_mismatch,
                "max_rain_in_per_hour": round(max_rain_master, 4),
            },
        )
    )

    if weather_clean_df is not None:
        w_rain_valid = weather_clean_df["rain"].dropna()
        w_max_rain = (
            float(w_rain_valid.max()) if len(w_rain_valid) > 0 else 0.0
        )
        w_neg_rain = int((w_rain_valid < 0).sum())
        w_sent_cnt = (
            int((weather_clean_df["is_rain_sentinel"] == 1).sum())
            if "is_rain_sentinel" in weather_clean_df.columns
            else 0
        )
        w_ok = (w_max_rain <= MAX_VALID_RAIN_INCH_PER_HOUR) and (
            w_neg_rain == 0
        )
        report.checks.append(
            ValidationCheck(
                check_id="VAL-12_WEATHER_CLEAN_CSV",
                category="Weather CSV Integrity",
                status="PASS" if w_ok else "FAIL",
                severity="MEDIUM",
                message=(
                    f"weather_cleaned.csv ({len(weather_clean_df):,} dòng): "
                    f"max(rain) = {w_max_rain:.4f} in/h "
                    f"(<= {MAX_VALID_RAIN_INCH_PER_HOUR}), giá trị âm = "
                    f"{w_neg_rain}, sentinel mưa giữ NaN = {w_sent_cnt:,}."
                ),
                evidence={
                    "rows": len(weather_clean_df),
                    "max_rain": round(w_max_rain, 4),
                    "neg_rain": w_neg_rain,
                    "rain_sentinels_nan": w_sent_cnt,
                },
            )
        )

    # 10. CRIT-03: Kiểm toán nguồn gốc surge_multiplier (không điền mù 1.0 trên
    # Lyft khuyết surge)
    so_flags = set(df["is_surge_original"].unique().tolist())
    so1_mask = df["is_surge_original"] == 1
    so0_mask = df["is_surge_original"] == 0
    n_so1 = int(so1_mask.sum())
    n_so0 = int(so0_mask.sum())
    so0_surge_leaks = int(
        df.loc[so0_mask, "surge_multiplier"].notna().sum()
        + df.loc[so0_mask, "distance_x_surge"].notna().sum()
    )
    so1_surge_missing = int(
        df.loc[so1_mask, "surge_multiplier"].isna().sum()
        + df.loc[so1_mask & df["distance"].notna(), "distance_x_surge"]
        .isna()
        .sum()
    )
    so0_uber_leaks = int(
        (so0_mask & (df["cab_type"].astype(str) == "Uber")).sum()
    )
    lyft_obs_mask = (df["cab_type"].astype(str) == "Lyft") & so1_mask
    lyft_obs_rate = (
        float((df.loc[lyft_obs_mask, "surge_multiplier"] > 1.0).mean() * 100.0)
        if lyft_obs_mask.any()
        else 0.0
    )

    crit03_ok = (
        so_flags.issubset({0, 1})
        and so0_surge_leaks == 0
        and so1_surge_missing == 0
        and so0_uber_leaks == 0
        and n_so0 > 0
    )
    report.checks.append(
        ValidationCheck(
            check_id="VAL-13_CRIT03_SURGE_PROVENANCE",
            category="CRIT-03 Surge Provenance",
            status="PASS" if crit03_ok else "FAIL",
            severity="CRITICAL",
            message=(
                f"CRIT-03 ĐẠT: Quan sát surge hợp lệ = {n_so1:,} (Uber API "
                f"1.0x + {int(lyft_obs_mask.sum()):,} Lyft thực, tỷ lệ "
                f"Surge Lyft = {lyft_obs_rate:.2f}%), "
                f"Lyft khuyết surge giữ NaN = {n_so0:,} (0 dòng bị "
                "fillna(1.0) mù)."
                if crit03_ok
                else (
                    f"VI PHẠM CRIT-03: so0_leaks={so0_surge_leaks}, "
                    f"so1_miss={so1_surge_missing}, "
                    f"so0_uber={so0_uber_leaks}"
                )
            ),
            evidence={
                "n_surge_original": n_so1,
                "n_lyft_surge_missing_nan": n_so0,
                "n_lyft_surge_observed": int(lyft_obs_mask.sum()),
                "lyft_observed_surge_rate_pct": round(lyft_obs_rate, 2),
                "so0_surge_leaks": so0_surge_leaks,
                "so1_surge_missing": so1_surge_missing,
            },
        )
    )

    # Lưu bảng thống kê nhanh vào report.metrics
    report.metrics = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "ram_mb": round(ram_mb, 2),
        "est_unoptimized_ram_mb": round(est_raw_mb, 2),
        "n_ts_original": n_ts1,
        "n_ts_missing": n_ts0,
        "n_weather_matched": n_wm1,
        "n_weather_unmatched": n_wm0,
        "n_routes": n_routes,
        "n_ultra_short_gps": n_ultra_short,
    }
    return report


def validate_result_tables(
    tables: dict[str, pd.DataFrame],
) -> ValidationReport:
    """
    Kiểm định schema và tính nhất quán nghiệp vụ trên toàn bộ 14 bảng CSV kết
    quả.
    """
    report = ValidationReport()

    # 1. Kiểm tra schema từng bảng CSV
    missing_table_keys = [k for k in TABLE_REQUIRED_COLUMNS if k not in tables]
    schema_errors: dict[str, list[str]] = {}
    for key, req_cols in TABLE_REQUIRED_COLUMNS.items():
        if key not in tables:
            continue
        tdf = tables[key]
        miss = [c for c in req_cols if c not in tdf.columns]
        if miss or tdf.empty:
            schema_errors[key] = miss if miss else ["EMPTY_TABLE"]

    report.checks.append(
        ValidationCheck(
            check_id="TBL-01_CSV_SCHEMAS",
            category="Result Tables Schema",
            status="PASS"
            if (not missing_table_keys and not schema_errors)
            else "FAIL",
            severity="CRITICAL",
            message=(
                f"Toàn bộ {len(TABLE_REQUIRED_COLUMNS)} bảng CSV kết quả "
                "đạt chuẩn schema & không rỗng."
                if (not missing_table_keys and not schema_errors)
                else (
                    f"Lỗi bảng CSV: thiếu bảng={missing_table_keys}, lỗi "
                    f"cột={schema_errors}"
                )
            ),
            evidence={
                "table_shapes": {k: list(v.shape) for k, v in tables.items()}
            },
        )
    )
    if missing_table_keys or schema_errors:
        return report

    # 2. Kiểm tra Data Dictionary đủ 41 cột và 0 đơn vị đo bị thiếu (MED-03 +
    # CRIT-03 + H4)
    dd = tables["data_dictionary"]
    missing_units = int(
        dd["unit"].isna().sum()
        + (dd["unit"].astype(str).str.strip() == "").sum()
    )
    dd_ok = (len(dd) == 41) and (missing_units == 0)
    report.checks.append(
        ValidationCheck(
            check_id="TBL-02_DATA_DICTIONARY",
            category="Data Dictionary",
            status="PASS" if dd_ok else "FAIL",
            severity="MEDIUM",
            message=(
                f"data_dictionary_complete.csv có {len(dd)} biến, "
                f"{missing_units} ô thiếu đơn vị đo."
            ),
            evidence={"n_vars": len(dd), "missing_units": missing_units},
        )
    )

    # 3. Kiểm tra HIGH-01: Permutation Importance đúng 14 nhóm rời nhau = 27
    # cột duy nhất & tổng ~100%
    perm = tables["permutation_importance"]
    all_listed = [
        c.strip()
        for s in perm["permuted_columns"]
        for c in str(s).split(",")
        if c.strip()
    ]
    share_sum = float(perm["m7_importance_share_pct"].sum())
    perm_ok = (
        (len(perm) == 14)
        and (len(all_listed) == 27)
        and (len(set(all_listed)) == 27)
        and (99.5 <= share_sum <= 100.5)
    )
    report.checks.append(
        ValidationCheck(
            check_id="TBL-03_HIGH01_PERMUTATION",
            category="HIGH-01 Permutation Importance",
            status="PASS" if perm_ok else "FAIL",
            severity="HIGH",
            message=(
                f"Permutation Importance hợp lệ: {len(perm)} nhóm rời nhau "
                f"= {len(set(all_listed))}/{len(all_listed)} cột duy nhất, "
                f"tổng tỷ trọng = {share_sum:.2f}%."
            ),
            evidence={
                "n_groups": len(perm),
                "n_listed_cols": len(all_listed),
                "n_unique_cols": len(set(all_listed)),
                "share_sum_pct": round(share_sum, 2),
            },
        )
    )

    # 4. Kiểm tra HIGH-03: Bảng mô phỏng Surge Cap có đủ 6 kịch bản & cột đối
    # chứng M2 OLS
    sim = tables["surge_cap_simulation"]
    sim_ok = (len(sim) == 6) and (
        sim["m2_ols_total_savings_usd"].notna().all()
    )
    report.checks.append(
        ValidationCheck(
            check_id="TBL-04_HIGH03_SIMULATION",
            category="HIGH-03 Surge Cap Simulation",
            status="PASS" if sim_ok else "FAIL",
            severity="HIGH",
            message=(
                f"Bảng mô phỏng Surge Cap có {len(sim)} kịch bản và đầy đủ "
                "cột đối chứng M2 OLS."
            ),
            evidence={
                "n_scenarios": len(sim),
                "caps": sim["surge_cap"].tolist(),
            },
        )
    )

    # 5. Kiểm tra đủ 10 mô hình M0a–M7, 13 kiểm định H1–H4 và 12 gói biểu giá
    counts_ok = (
        len(tables["test_metrics"]) == 10
        and len(tables["hypothesis_tests"]) == 13
        and len(tables["tier_tariffs"]) == 12
        and len(tables["prediction_interval"]) == 12
    )
    report.checks.append(
        ValidationCheck(
            check_id="TBL-05_EXPECTED_COUNTS",
            category="Completeness",
            status="PASS" if counts_ok else "FAIL",
            severity="HIGH",
            message=(
                f"Đủ số lượng chuẩn: {len(tables['test_metrics'])} mô hình, "
                f"{len(tables['hypothesis_tests'])} kiểm định H1–H4, "
                f"{len(tables['tier_tariffs'])} gói biểu giá, "
                f"{len(tables['prediction_interval'])} gói khoảng dự báo "
                "P10–P90."
            ),
            evidence={
                "n_models": len(tables["test_metrics"]),
                "n_hypotheses": len(tables["hypothesis_tests"]),
                "n_tariffs": len(tables["tier_tariffs"]),
                "n_pi_tiers": len(tables["prediction_interval"]),
            },
        )
    )
    return report


def run_full_data_validation() -> ValidationReport:
    """
    Chạy kiểm toán đầy đủ cả lớp dữ liệu Parquet/CSV gốc đã làm sạch lẫn 14
    bảng CSV kết quả.
    """
    master = load_master_dataset()
    cab_clean = load_cab_cleaned()
    weather_clean = load_weather_cleaned()
    audit_flags = load_audit_flags()
    tables = load_all_result_tables()

    master_report = validate_master_dataset(
        master,
        cab_clean_df=cab_clean,
        weather_clean_df=weather_clean,
        audit_flags_df=audit_flags,
    )
    tables_report = validate_result_tables(tables)

    combined = ValidationReport(
        checks=master_report.checks + tables_report.checks,
        metrics=master_report.metrics,
    )
    return combined


if __name__ == "__main__":
    rep = run_full_data_validation()
    print(
        f"=== KET QUA KIEM TOAN DU LIEU GIAI DOAN 1 (PASSED={rep.passed}) ==="
    )
    for chk in rep.checks:
        print(
            f"[{chk.status:4s} | {chk.severity:8s}] {chk.check_id}: "
            f"{chk.message}"
        )
    rep.raise_if_critical()
    print(
        f"\nTONG KET: {len(rep.checks) - len(rep.failed_checks)}/"
        f"{len(rep.checks)} hang muc DAT (0 loi CRITICAL)."
    )

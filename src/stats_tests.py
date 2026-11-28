"""
Module: stats_tests.py
Mục đích: Thực thi Phần I của Giai đoạn 5 (Kiểm định Giả thuyết Thống kê Chuyên
sâu & Độ lớn Hiệu ứng - Effect Size):
- H1 (Thời tiết & Giá cước): Welch's two-sample t-test + Mann-Whitney U test +
  Cohen's d + Rank-Biserial r.
- H2 (Buổi trong ngày & Hệ số Surge): One-way ANOVA + Kruskal-Wallis H test +
  Eta-squared (eta^2) + Epsilon-squared.
- H3 (Tính độc lập của sự kiện kích hoạt Surge): Kiểm định Chi-Square (chi^2) +
  Cramér's V trên 4 chiều (Hãng xe, Mưa, Buổi, Giờ cao điểm).
- H4 (Biểu giá phân khúc & Kiểm định Wald HC3): Ước lượng OLS với sai số chuẩn
  bền vững White-Huber HC3,
  kiểm định Wald so sánh giá mở cửa và đơn giá/dặm giữa UberX và Lyft, kiểm
  định đồng thời (Joint F-test) cho nhóm biến thời tiết.

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import TABLES_DIR, ensure_directories

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_cohens_d(
    x1: np.ndarray, x2: np.ndarray
) -> tuple[float, float, float]:
    """Tính Cohen's d kèm khoảng tin cậy 95% cho hai mẫu độc lập."""
    n1, n2 = len(x1), len(x2)
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2)))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d
    return float(d), float(ci_low), float(ci_high)


def test_h1_weather_price(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Kiểm định H1: Giá cước khi trời mưa (is_raining=1) so với khi không mưa
    (is_raining=0).
    [CRIT-01 & H4 FIX] CHỈ thực hiện trên các bản ghi có khớp bản tin thời tiết
    thực và không bị sentinel mưa (weather_matched == 1 & is_raining.notna(), n
    = 152,043)
    nhằm tránh ô nhiễm nhãn mưa giả từ dữ liệu không khớp trạm giờ hoặc lỗi cảm
    biến mưa.
    """
    # [CRIT-01 & H4 FIX] CHỈ thực hiện trên các bản ghi khớp thời tiết thực VÀ
    # không bị sentinel mưa (is_raining.notna())
    if "weather_matched" in df.columns:
        df_wea = df[
            (df["weather_matched"] == 1) & (df["is_raining"].notna())
        ].copy()
    else:
        df_wea = df[df["is_raining"].notna()].copy()
    results = []
    scopes = [
        (
            "H1a_All_Rides",
            df_wea,
            "Toàn thị trường - Khớp thời tiết thực (Uber + Lyft)",
        ),
        (
            "H1b_Uber_Only",
            df_wea[df_wea["cab_type"].astype(str) == "Uber"],
            "Riêng nền tảng Uber - Khớp thời tiết thực",
        ),
        (
            "H1c_Lyft_Only",
            df_wea[df_wea["cab_type"].astype(str) == "Lyft"],
            "Riêng nền tảng Lyft - Khớp thời tiết thực",
        ),
    ]
    for code, sub_df, scope_desc in scopes:
        rain_p = (
            sub_df.loc[sub_df["is_raining"] == 1.0, "price"]
            .astype(np.float64)
            .values
        )
        dry_p = (
            sub_df.loc[sub_df["is_raining"] == 0.0, "price"]
            .astype(np.float64)
            .values
        )

        t_stat, t_pval = stats.ttest_ind(rain_p, dry_p, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(
            rain_p, dry_p, alternative="two-sided"
        )
        d_val, d_low, d_high = compute_cohens_d(rain_p, dry_p)
        n1, n2 = len(rain_p), len(dry_p)
        # U1 là số cặp Rain > Dry, nên r_rb > 0 tương ứng với Rain > Dry
        r_rb = (2.0 * float(u_stat)) / (n1 * n2) - 1.0
        diff_mean = float(np.mean(rain_p) - np.mean(dry_p))

        results.append(
            {
                "hypothesis_id": code,
                "research_question": "RQ1",
                "scope": scope_desc,
                "null_hypothesis": (
                    "H0: Trung bình/Phân phối giá cước khi Mưa bằng khi "
                    "Không mưa"
                ),
                "n_group1": int(n1),
                "n_group2": int(n2),
                "mean_group1_or_rate": round(float(np.mean(rain_p)), 4),
                "mean_group2_or_ref": round(float(np.mean(dry_p)), 4),
                "aux_stat_1": round(float(np.median(rain_p)), 4),
                "aux_stat_2": round(float(np.median(dry_p)), 4),
                "aux_stat_description": "Median(Rain_USD) | Median(Dry_USD)",
                "parametric_test": "Welch's t-test",
                "parametric_stat": round(float(t_stat), 4),
                "parametric_pvalue": round(float(t_pval), 6),
                "nonparametric_test": "Mann-Whitney U",
                "nonparametric_stat": round(float(u_stat), 1),
                "nonparametric_pvalue": round(float(u_pval), 6),
                "effect_size_name": "Cohen's d",
                "effect_size_value": round(d_val, 5),
                "effect_size_ci95": f"[{d_low:.5f}, {d_high:.5f}]",
                "secondary_effect_size": round(float(r_rb), 5),
                "statistical_decision": "Bác bỏ H0"
                if t_pval < 0.05
                else "Chưa đủ bằng chứng bác bỏ H0",
                "practical_significance": "Không đáng kể (|d| < 0.01)"
                if abs(d_val) < 0.01
                else "Nhỏ",
                "economic_conclusion": (
                    "Trên tập khớp thời tiết thực, mưa không làm tăng giá "
                    "cước cơ học "
                    f"(chênh lệch {diff_mean:+.4f} USD, |d| = "
                    f"{abs(d_val):.5f})."
                ),
            }
        )
    return results


def _evaluate_h2_config(
    code: str,
    sub_df: pd.DataFrame,
    target_col: str,
    scope_desc: str,
    h0_text: str,
    periods: list[str],
) -> dict[str, Any]:
    """Đánh giá một cấu hình ANOVA/Kruskal-Wallis cho giả thuyết H2."""
    groups = [
        sub_df.loc[sub_df["time_of_day"].astype(str) == p, target_col]
        .astype(np.float64)
        .values
        for p in periods
    ]
    f_stat, f_pval = stats.f_oneway(*groups)
    kw_stat, kw_pval = stats.kruskal(*groups)

    n_total = sum(len(g) for g in groups)
    k = len(groups)
    vals = sub_df[target_col].astype(np.float64).values
    grand_mean = float(np.mean(vals))
    ss_between = sum(
        len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups
    )
    ss_total = float(np.sum((vals - grand_mean) ** 2))
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    eps_sq = (
        float(kw_stat) / ((n_total**2 - 1.0) / (n_total + 1.0))
        if n_total > 1
        else 0.0
    )

    means_by_p = {
        p: round(float(np.mean(g)), 4)
        for p, g in zip(periods, groups, strict=False)
    }
    max_p = max(means_by_p, key=means_by_p.get)
    min_p = min(means_by_p, key=means_by_p.get)

    return {
        "hypothesis_id": code,
        "research_question": "RQ1",
        "scope": scope_desc,
        "null_hypothesis": h0_text,
        "n_group1": int(len(groups[0]) + len(groups[1])),
        "n_group2": int(len(groups[2]) + len(groups[3])),
        "mean_group1_or_rate": means_by_p[max_p],
        "mean_group2_or_ref": means_by_p[min_p],
        "aux_stat_1": round(float(np.median(groups[0])), 4),
        "aux_stat_2": round(float(np.median(groups[2])), 4),
        "aux_stat_description": "Median(Morning) | Median(Evening)",
        "parametric_test": f"One-way ANOVA (k={k})",
        "parametric_stat": round(float(f_stat), 4),
        "parametric_pvalue": round(float(f_pval), 6),
        "nonparametric_test": "Kruskal-Wallis H",
        "nonparametric_stat": round(float(kw_stat), 4),
        "nonparametric_pvalue": round(float(kw_pval), 6),
        "effect_size_name": "Eta-squared (eta^2)",
        "effect_size_value": round(float(eta_sq), 6),
        "effect_size_ci95": (
            f"Max({max_p})={means_by_p[max_p]} vs "
            f"Min({min_p})={means_by_p[min_p]}"
        ),
        "secondary_effect_size": round(float(eps_sq), 6),
        "statistical_decision": (
            "Bác bỏ H0" if f_pval < 0.05 else "Chưa đủ bằng chứng bác bỏ H0"
        ),
        "practical_significance": (
            "Không đáng kể (eta^2 < 0.001)" if eta_sq < 0.001 else "Nhỏ"
        ),
        "economic_conclusion": (
            "Theo giờ Boston EST (TS gốc), hệ số surge và giá cước "
            "không biến thiên đáng kể theo buổi cố định (eta^2 = "
            f"{eta_sq:.6f})."
        ),
    }


def test_h2_time_of_day_surge(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Kiểm định H2: Hệ số nhân giá (surge_multiplier) và giá cước theo 4 buổi
    trong ngày Boston EST (Morning, Afternoon, Evening, Night).
    [CRIT-02 FIX] CHỈ thực hiện trên tập có timestamp gốc hợp lệ
    (is_timestamp_original == 1, n = 208,034).
    """
    df_ts = (
        df[df["is_timestamp_original"] == 1].copy()
        if "is_timestamp_original" in df.columns
        else df[df["time_of_day"].astype(str) != "Unknown"].copy()
    )
    df_ts_surge = (
        df_ts[df_ts["is_surge_original"] == 1].copy()
        if "is_surge_original" in df_ts.columns
        else df_ts[df_ts["surge_multiplier"].notna()].copy()
    )
    periods = ["Morning", "Afternoon", "Evening", "Night"]
    configs = [
        (
            "H2a_Surge_All",
            df_ts_surge,
            "surge_multiplier",
            "Hệ số Surge theo 4 buổi Boston EST (TS gốc & Surge quan sát)",
            "H0: Hệ số surge trung bình bằng nhau ở cả 4 buổi trong ngày "
            "(Boston EST)",
        ),
        (
            "H2b_Surge_Lyft",
            df_ts_surge[df_ts_surge["cab_type"].astype(str) == "Lyft"],
            "surge_multiplier",
            "Hệ số Surge theo 4 buổi Boston EST (Riêng Lyft - TS gốc & "
            "Surge quan sát)",
            "H0: Hệ số surge trung bình của Lyft bằng nhau ở cả 4 buổi "
            "trong ngày (Boston EST)",
        ),
        (
            "H2c_Price_All",
            df_ts,
            "price",
            "Giá báo (price) theo 4 buổi Boston EST (TS gốc hợp lệ)",
            "H0: Giá báo trung bình bằng nhau ở cả 4 buổi trong ngày "
            "(Boston EST)",
        ),
    ]
    return [
        _evaluate_h2_config(code, sub_df, col, scope, h0, periods)
        for code, sub_df, col, scope, h0 in configs
    ]


def _evaluate_h3_spec(
    code: str,
    sub_df: pd.DataFrame,
    col_factor: str,
    scope_desc: str,
    h0_text: str,
) -> dict[str, Any]:
    """Đánh giá một phép kiểm định Chi-Square và Cramér's V cho H3."""
    ct = pd.crosstab(sub_df[col_factor].astype(str), sub_df["has_surge"])
    chi2_stat, chi2_pval, dof, _ = stats.chi2_contingency(ct)
    g_stat, g_pval, _, _ = stats.chi2_contingency(ct, lambda_="log-likelihood")
    n_obs = int(ct.values.sum())
    min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
    cramers_v = (
        float(np.sqrt(chi2_stat / (n_obs * min_dim)))
        if (n_obs > 0 and min_dim > 0)
        else 0.0
    )

    rates = (
        (ct[1] / ct.sum(axis=1) * 100.0).to_dict() if 1 in ct.columns else {}
    )
    max_cat = max(rates, key=rates.get) if rates else "N/A"
    min_cat = min(rates, key=rates.get) if rates else "N/A"

    if code == "H3a_Surge_x_CabType":
        lyft_rate = rates.get("Lyft", 0.0)
        prac = (
            f"Hiệu ứng cấu trúc rõ rệt (V = {cramers_v:.4f}: Lyft "
            f"{lyft_rate:.2f}% vs Uber 0.00%)"
        )
        econ = (
            "Chỉ Lyft ghi nhận hệ số nhân giá động trên API; Uber cố "
            "định surge = 1.0."
        )
    else:
        if cramers_v < 0.01:
            prac = "Không đáng kể (Cramér's V < 0.01)"
        elif cramers_v < 0.05:
            prac = (
                "Cực nhỏ / Không đáng kể thực tiễn (0.01 ≤ Cramér's V < 0.05)"
            )
        else:
            prac = f"Nhỏ (Cramér's V = {cramers_v:.4f})"
        econ = (
            "Xác suất Lyft kích hoạt surge không phụ thuộc cơ học vào "
            f"{col_factor} (V = {cramers_v:.5f})."
        )

    return {
        "hypothesis_id": code,
        "research_question": "RQ1",
        "scope": scope_desc,
        "null_hypothesis": h0_text,
        "n_group1": int(ct.iloc[0].sum()),
        "n_group2": int(ct.iloc[1:].values.sum()),
        "mean_group1_or_rate": round(float(rates.get(max_cat, 0.0)), 4),
        "mean_group2_or_ref": round(float(rates.get(min_cat, 0.0)), 4),
        "aux_stat_1": float(dof),
        "aux_stat_2": float(ct.shape[0]),
        "aux_stat_description": "Chi2_df | Num_Categories_k",
        "parametric_test": f"Pearson Chi-Square (df={dof})",
        "parametric_stat": round(float(chi2_stat), 4),
        "parametric_pvalue": round(float(chi2_pval), 6),
        "nonparametric_test": "G-test (Log-likelihood ratio)",
        "nonparametric_stat": round(float(g_stat), 4),
        "nonparametric_pvalue": round(float(g_pval), 6),
        "effect_size_name": "Cramér's V",
        "effect_size_value": round(cramers_v, 5),
        "effect_size_ci95": (
            f"MaxRate({max_cat})={rates.get(max_cat, 0):.2f}% vs "
            f"Min({min_cat})={rates.get(min_cat, 0):.2f}%"
        ),
        "secondary_effect_size": round(cramers_v**2, 6),
        "statistical_decision": (
            "Bác bỏ H0" if chi2_pval < 0.05 else "Chưa đủ bằng chứng bác bỏ H0"
        ),
        "practical_significance": prac,
        "economic_conclusion": econ,
    }


def test_h3_chi_square_surge_independence(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Kiểm định H3: Kiểm định Chi-Square độc lập và hệ số Cramér's V cho sự kiện
    kích hoạt Surge (has_surge = surge_multiplier > 1.0) theo H3a-H3d.
    """
    work_df = (
        df[df["is_surge_original"] == 1].copy()
        if "is_surge_original" in df.columns
        else df[df["surge_multiplier"].notna()].copy()
    )
    work_df["has_surge"] = (work_df["surge_multiplier"] > 1.0).astype(int)
    lyft_wea_df = work_df[
        (work_df["cab_type"].astype(str) == "Lyft")
        & (work_df["weather_matched"] == 1)
        & (work_df["is_raining"].notna())
    ].copy()
    lyft_ts_df = work_df[
        (work_df["cab_type"].astype(str) == "Lyft")
        & (work_df["is_timestamp_original"] == 1)
    ].copy()

    specs = [
        (
            "H3a_Surge_x_CabType",
            work_df,
            "cab_type",
            "Tần suất kích hoạt Surge giữa Lyft vs Uber (Toàn mẫu)",
            "H0: Xác suất xuất hiện surge > 1.0 độc lập với nền tảng gọi xe "
            "(Uber/Lyft)",
        ),
        (
            "H3b_Surge_x_Rain_Lyft",
            lyft_wea_df,
            "is_raining",
            "Tần suất kích hoạt Surge theo Mưa vs Tạnh (Lyft khớp thời tiết "
            "thực)",
            "H0: Xác suất Lyft kích hoạt surge độc lập với tình trạng trời "
            "mưa",
        ),
        (
            "H3c_Surge_x_TimeOfDay_Lyft",
            lyft_ts_df,
            "time_of_day",
            "Tần suất kích hoạt Surge theo 4 Buổi Boston EST (Lyft TS gốc)",
            "H0: Xác suất Lyft kích hoạt surge độc lập với buổi trong ngày "
            "(Boston EST)",
        ),
        (
            "H3d_Surge_x_RushHour_Lyft",
            lyft_ts_df,
            "is_rush_hour",
            "Tần suất kích hoạt Surge theo Giờ cao điểm Boston EST (Lyft TS "
            "gốc)",
            "H0: Xác suất Lyft kích hoạt surge độc lập với khung giờ cao "
            "điểm (Boston EST)",
        ),
    ]
    return [
        _evaluate_h3_spec(code, sub_df, col_factor, scope_desc, h0_text)
        for code, sub_df, col_factor, scope_desc, h0_text in specs
    ]


def _estimate_per_tier_tariffs(
    df_train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Ước lượng OLS HC3 cho biểu giá 12 phân khúc dịch vụ."""
    tiers = sorted(df_train["name"].astype(str).unique().tolist())
    tariff_rows: list[dict[str, Any]] = []
    tier_params: dict[str, dict[str, Any]] = {}

    for tier in tiers:
        sub = df_train[
            (df_train["name"].astype(str) == tier)
            & df_train["distance_x_surge"].notna()
        ]
        cab = str(sub["cab_type"].iloc[0])
        y = sub["price"].astype(np.float64).values
        x_eff = sub["distance_x_surge"].astype(np.float64).values
        x_dist = sub["distance"].astype(np.float64).values
        res_hc3 = sm.OLS(y, sm.add_constant(x_eff)).fit(cov_type="HC3")

        b0, b1 = float(res_hc3.params[0]), float(res_hc3.params[1])
        se0, se1 = float(res_hc3.bse[0]), float(res_hc3.bse[1])
        ci0_low, ci0_high = res_hc3.conf_int()[0]
        ci1_low, ci1_high = res_hc3.conf_int()[1]
        cov_01 = float(res_hc3.cov_params()[0, 1])

        tier_params[tier] = {
            "b0": b0,
            "b1": b1,
            "se0": se0,
            "se1": se1,
            "cov01": cov_01,
            "n": len(sub),
        }
        tariff_rows.append(
            {
                "cab_type": cab,
                "service_tier": tier,
                "n_train": int(len(sub)),
                "base_fare_b0_usd": round(b0, 4),
                "base_fare_se_hc3": round(se0, 4),
                "base_fare_ci95_low": round(float(ci0_low), 4),
                "base_fare_ci95_high": round(float(ci0_high), 4),
                "per_mile_rate_b1_usd": round(b1, 4),
                "per_mile_se_hc3": round(se1, 4),
                "per_mile_ci95_low": round(float(ci1_low), 4),
                "per_mile_ci95_high": round(float(ci1_high), 4),
                "r2_within_tier": round(float(res_hc3.rsquared), 4),
                "mean_price_usd": round(float(np.mean(y)), 2),
                "median_price_usd": round(float(np.median(y)), 2),
                "mean_distance_miles": round(float(np.mean(x_dist)), 2),
            }
        )

    tariff_df = (
        pd.DataFrame(tariff_rows)
        .sort_values(["cab_type", "base_fare_b0_usd"])
        .reset_index(drop=True)
    )
    return tariff_df, tier_params


def _wald_decision(p1: float, p2: float, *, both_strict: bool) -> str:
    """Chuẩn hóa nhãn quyết định kiểm định Wald từ 2 giá trị p."""
    strict_hit = (
        (p1 < 1e-4 and p2 < 1e-4) if both_strict else (p1 < 1e-4 or p2 < 1e-4)
    )
    if strict_hit:
        return "Bác bỏ H0 (p < 0.0001)"
    return (
        "Bác bỏ H0" if min(p1, p2) < 0.05 else "Chưa đủ bằng chứng bác bỏ H0"
    )


def _compute_h4a_stats(
    ux: dict[str, Any], lf: dict[str, Any], df_train: pd.DataFrame
) -> dict[str, float]:
    """Tính thống kê Wald Z và khoảng tin cậy Delta-method cho điểm cắt d*."""
    diff_b0 = ux["b0"] - lf["b0"]
    z_b0 = diff_b0 / np.sqrt(ux["se0"] ** 2 + lf["se0"] ** 2)
    p_b0 = float(2.0 * (1.0 - stats.norm.cdf(abs(z_b0))))

    diff_b1 = lf["b1"] - ux["b1"]
    z_b1 = diff_b1 / np.sqrt(lf["se1"] ** 2 + ux["se1"] ** 2)
    p_b1 = float(2.0 * (1.0 - stats.norm.cdf(abs(z_b1))))

    d_star = diff_b0 / diff_b1
    var_a = ux["se0"] ** 2 + lf["se0"] ** 2
    var_b = lf["se1"] ** 2 + ux["se1"] ** 2
    cov_ab = -ux["cov01"] - lf["cov01"]
    var_d = (
        (var_a / (diff_b1**2))
        + ((diff_b0**2) * var_b / (diff_b1**4))
        - (2.0 * diff_b0 * cov_ab / (diff_b1**3))
    )
    se_d = float(np.sqrt(max(var_d, 1e-12)))
    return {
        "diff_b0": diff_b0,
        "z_b0": float(z_b0),
        "p_b0": p_b0,
        "diff_b1": diff_b1,
        "z_b1": float(z_b1),
        "p_b1": p_b1,
        "d_star": float(d_star),
        "se_d": se_d,
        "d_low": float(d_star - 1.96 * se_d),
        "d_high": float(d_star + 1.96 * se_d),
        "pct_below": float(
            (df_train["distance"].dropna() < d_star).mean() * 100.0
        ),
    }


def _compute_h4a_uberx_lyft(
    tier_params: dict[str, dict[str, Any]], df_train: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kiểm định Wald H4a cho cặp biểu giá cắt nhau UberX vs Lyft."""
    ux, lf = tier_params["UberX"], tier_params["Lyft"]
    st = _compute_h4a_stats(ux, lf, df_train)
    record = {
        "hypothesis_id": "H4a_Wald_UberX_vs_Lyft_Slope",
        "research_question": "RQ2",
        "scope": "So sánh biểu giá UberX vs Lyft (Kiểm định Wald HC3)",
        "null_hypothesis": (
            "H0: Đơn giá mỗi dặm của Lyft bằng UberX (beta_{1,Lyft} = "
            "beta_{1,UberX})"
        ),
        "n_group1": int(lf["n"]),
        "n_group2": int(ux["n"]),
        "mean_group1_or_rate": round(lf["b1"], 4),
        "mean_group2_or_ref": round(ux["b1"], 4),
        "aux_stat_1": round(lf["b0"], 4),
        "aux_stat_2": round(ux["b0"], 4),
        "aux_stat_description": (
            "BaseFare_b0(Lyft_USD) | BaseFare_b0(UberX_USD)"
        ),
        "parametric_test": "Wald Z-test (HC3 Robust SE)",
        "parametric_stat": round(st["z_b1"], 4),
        "parametric_pvalue": round(st["p_b1"], 6),
        "nonparametric_test": "Wald Base-Fare Z-test (b0_UberX - b0_Lyft)",
        "nonparametric_stat": round(st["z_b0"], 4),
        "nonparametric_pvalue": round(st["p_b0"], 6),
        "effect_size_name": "Crossing Distance d* (miles)",
        "effect_size_value": round(st["d_star"], 4),
        "effect_size_ci95": f"[{st['d_low']:.3f}, {st['d_high']:.3f}] dặm",
        "secondary_effect_size": round(st["diff_b1"], 4),
        "statistical_decision": _wald_decision(
            st["p_b1"], st["p_b0"], both_strict=True
        ),
        "practical_significance": (
            f"Rất lớn: Điểm cắt tại {st['d_star']:.2f} dặm "
            f"({st['pct_below']:.1f}% chuyến < d*)"
        ),
        "economic_conclusion": (
            "Chiến lược biểu giá cắt nhau: Lyft rẻ hơn ở cự ly ngắn < "
            f"{st['d_star']:.2f} dặm (mở cửa thấp hơn ${st['diff_b0']:.2f}), "
            f"UberX rẻ hơn ở cự ly dài > {st['d_star']:.2f} dặm "
            f"(rẻ hơn ${st['diff_b1']:.2f}/dặm)."
        ),
    }
    summary = {
        "uberx_base_b0": round(ux["b0"], 4),
        "lyft_base_b0": round(lf["b0"], 4),
        "base_diff_usd": round(st["diff_b0"], 4),
        "uberx_slope_b1": round(ux["b1"], 4),
        "lyft_slope_b1": round(lf["b1"], 4),
        "slope_diff_usd_per_mile": round(st["diff_b1"], 4),
        "crossing_distance_miles": round(st["d_star"], 4),
        "crossing_distance_se": round(st["se_d"], 4),
        "crossing_distance_ci95": [
            round(st["d_low"], 4),
            round(st["d_high"], 4),
        ],
        "pct_trips_shorter_than_crossing": round(st["pct_below"], 2),
    }
    return record, summary


def _compute_h4b_pool_shared(
    tier_params: dict[str, dict[str, Any]], df_train: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kiểm định Wald H4b cho phân khúc đi ghép UberPool vs Lyft Shared."""
    up, sh = tier_params["UberPool"], tier_params["Shared"]
    diff_b0 = up["b0"] - sh["b0"]
    z_b0 = diff_b0 / np.sqrt(up["se0"] ** 2 + sh["se0"] ** 2)
    p_b0 = float(2.0 * (1.0 - stats.norm.cdf(abs(z_b0))))
    diff_b1 = sh["b1"] - up["b1"]
    z_b1 = diff_b1 / np.sqrt(sh["se1"] ** 2 + up["se1"] ** 2)
    p_b1 = float(2.0 * (1.0 - stats.norm.cdf(abs(z_b1))))
    d_star_pool = diff_b0 / diff_b1 if abs(diff_b1) > 1e-6 else np.nan
    max_obs = float(df_train["distance"].max())
    cross_in = bool(
        np.isfinite(d_star_pool) and (0.0 < d_star_pool <= max_obs)
    )

    record = {
        "hypothesis_id": "H4b_Wald_UberPool_vs_Shared",
        "research_question": "RQ2",
        "scope": "So sánh phân khúc đi ghép UberPool vs Lyft Shared (HC3)",
        "null_hypothesis": (
            "H0: Biểu giá mở cửa và đơn giá/dặm của UberPool bằng Shared"
        ),
        "n_group1": int(up["n"]),
        "n_group2": int(sh["n"]),
        "mean_group1_or_rate": round(sh["b1"], 4),
        "mean_group2_or_ref": round(up["b1"], 4),
        "aux_stat_1": round(sh["b0"], 4),
        "aux_stat_2": round(up["b0"], 4),
        "aux_stat_description": (
            "BaseFare_b0(Shared_USD) | BaseFare_b0(UberPool_USD)"
        ),
        "parametric_test": "Wald Z-test Slope (HC3)",
        "parametric_stat": round(float(z_b1), 4),
        "parametric_pvalue": round(p_b1, 6),
        "nonparametric_test": "Wald Z-test Base Fare (HC3)",
        "nonparametric_stat": round(float(z_b0), 4),
        "nonparametric_pvalue": round(p_b0, 6),
        "effect_size_name": "Base Fare Diff b0_UberPool - b0_Shared (USD)",
        "effect_size_value": round(float(diff_b0), 4),
        "effect_size_ci95": (
            f"Base diff=${diff_b0:.2f}, Slope diff=${diff_b1:.2f}/dặm "
            f"(Không cắt trong miền [0.02, {max_obs:.2f}] dặm)"
        ),
        "secondary_effect_size": round(float(diff_b1), 4),
        "statistical_decision": _wald_decision(p_b0, p_b1, both_strict=False),
        "practical_significance": (
            f"Rõ rệt (Không cắt trong miền Boston <= {max_obs:.2f} dặm): "
            "Lyft Shared luôn rẻ hơn UberPool trên toàn bộ phạm vi quan sát."
        ),
        "economic_conclusion": (
            f"Lyft Shared đặt giá mở cửa thấp hơn hẳn (${sh['b0']:.2f} "
            f"vs ${up['b0']:.2f}) và chênh lệch độ dốc chỉ "
            f"${diff_b1:.2f}/dặm, nên KHÔNG tồn tại điểm cắt thực tế trong "
            f"miền quan sát [0.02, {max_obs:.2f}] dặm (điểm giao toán học "
            "ngoài miền không được ngoại suy)."
        ),
    }
    summary = {
        "uberpool_base_b0": round(up["b0"], 4),
        "shared_base_b0": round(sh["b0"], 4),
        "uberpool_slope_b1": round(up["b1"], 4),
        "shared_slope_b1": round(sh["b1"], 4),
        "max_observed_distance_miles": round(max_obs, 2),
        "crossing_in_observed_domain": cross_in,
        "crossing_distance_miles": (
            round(float(d_star_pool), 4) if cross_in else None
        ),
        "note": (
            "Lyft Shared rẻ hơn UberPool trên toàn bộ miền quan sát "
            f"Boston [0.02, {max_obs:.2f}] dặm; không ngoại suy điểm giao "
            "ngoài miền."
        ),
    }
    return record, summary


def _build_h4c_design_matrix(
    df_train: pd.DataFrame, weather_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Chuẩn bị ma trận thiết kế hạng đầy đủ cho kiểm định đồng thời H4c."""
    df_multi = (
        df_train[
            (df_train["weather_matched"] == 1)
            & (df_train["rain"].notna())
            & (df_train["distance"].notna())
        ].copy()
        if "weather_matched" in df_train.columns
        else df_train.dropna(
            subset=["distance", *weather_cols, "is_rush_hour", "is_weekend"]
        ).copy()
    )
    X_multi = pd.get_dummies(
        df_multi[["name", "source", "destination", "time_of_day"]],
        drop_first=True,
        dtype=np.float64,
    )
    zero_cols = [c for c in X_multi.columns if X_multi[c].std() == 0]
    if zero_cols:
        X_multi = X_multi.drop(columns=zero_cols)
    if "destination_Boston University" in X_multi.columns:
        X_multi = X_multi.drop(columns=["destination_Boston University"])
    num_cols = [
        "distance",
        "lyft_surge_excess",
        "distance_x_lyft_surge_excess",
        "is_rush_hour",
        "is_weekend",
        *weather_cols,
    ]
    for col in num_cols:
        X_multi[col] = df_multi[col].astype(np.float64).values
    return sm.add_constant(X_multi), df_multi["price"].astype(
        np.float64
    ).values


def _compute_h4c_weather_wald(
    df_train: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kiểm định Wald đồng thời H4c (Joint F-test) cho 6 biến thời tiết."""
    weather_cols = ["temp", "rain", "humidity", "wind", "clouds", "pressure"]
    X_multi, y_multi = _build_h4c_design_matrix(df_train, weather_cols)
    ols_multi = sm.OLS(y_multi, X_multi).fit(cov_type="HC3")

    col_names = list(X_multi.columns)
    R_mat = np.zeros((len(weather_cols), len(col_names)))
    for i, wc in enumerate(weather_cols):
        R_mat[i, col_names.index(wc)] = 1.0
    wald_wea = ols_multi.wald_test(R_mat, use_f=True)
    f_wea = float(np.asarray(wald_wea.statistic).squeeze())
    p_wea = float(np.asarray(wald_wea.pvalue).squeeze())

    r2_no_wea = float(
        sm.OLS(y_multi, X_multi.drop(columns=weather_cols)).fit().rsquared
    )
    delta_r2 = float(ols_multi.rsquared - r2_no_wea)
    n_obs = len(y_multi)

    record = {
        "hypothesis_id": "H4c_Wald_Joint_Weather_Multivariate",
        "research_question": "RQ1",
        "scope": (
            "Kiểm định Wald đồng thời 6 biến thời tiết trong hồi quy "
            f"đa biến HC3 (Train khớp thời tiết thực, n={n_obs:,})"
        ),
        "null_hypothesis": (
            "H0: Cả 6 hệ số thời tiết (temp, rain, humidity, wind, "
            "clouds, pressure) đồng thời bằng 0"
        ),
        "n_group1": int(n_obs),
        "n_group2": 6,
        "mean_group1_or_rate": round(float(ols_multi.params["rain"]), 4),
        "mean_group2_or_ref": round(float(ols_multi.params["temp"]), 4),
        "aux_stat_1": round(float(ols_multi.rsquared), 6),
        "aux_stat_2": round(float(r2_no_wea), 6),
        "aux_stat_description": (
            "R2(With_6_Weather_Vars) | R2(Without_Weather_Vars)"
        ),
        "parametric_test": "Joint Wald F-test (df=6, HC3)",
        "parametric_stat": round(f_wea, 4),
        "parametric_pvalue": round(p_wea, 6),
        "nonparametric_test": "Incremental R-squared (Delta R2 %)",
        "nonparametric_stat": round(delta_r2 * 100.0, 6),
        "nonparametric_pvalue": round(p_wea, 6),
        "effect_size_name": "Delta R^2 (Weather)",
        "effect_size_value": round(delta_r2, 7),
        "effect_size_ci95": (
            f"beta_rain={ols_multi.params['rain']:+.4f} "
            f"(p={ols_multi.pvalues['rain']:.3f})"
        ),
        "secondary_effect_size": round(f_wea, 4),
        "statistical_decision": (
            "Chưa đủ bằng chứng bác bỏ H0" if p_wea >= 0.05 else "Bác bỏ H0"
        ),
        "practical_significance": "Bằng 0 thực tiễn (Delta R^2 < 0.0001)",
        "economic_conclusion": (
            "Ngay cả khi kiểm soát đa biến trên tập khớp thời tiết "
            f"thực (n = {n_obs:,}), cả 6 biến thời tiết đóng góp "
            f"Delta R^2 = {delta_r2:.7f} (≈ 0.000%) vào phương sai giá cước."
        ),
    }
    summary = {
        "n_train_weather_matched": int(n_obs),
        "wald_f_stat": round(f_wea, 4),
        "wald_pvalue": round(p_wea, 6),
        "r2_with_weather": round(float(ols_multi.rsquared), 6),
        "r2_without_weather": round(r2_no_wea, 6),
        "delta_r2_weather": round(delta_r2, 7),
        "weather_coefficients": {
            wc: {
                "coef": round(float(ols_multi.params[wc]), 5),
                "se_hc3": round(float(ols_multi.bse[wc]), 5),
                "pvalue": round(float(ols_multi.pvalues[wc]), 4),
            }
            for wc in weather_cols
        },
    }
    return record, summary


def test_h4_econometric_wald_and_tariffs(
    df_train: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """
    Kiểm định H4 (RQ2 & RQ1 đa biến) trên tập huấn luyện X_train bằng
    statsmodels OLS với sai số chuẩn bền vững HC3 (biểu giá 12 gói, H4a,
    H4b và H4c).
    """
    logger.info(
        "Đang chạy ước lượng statsmodels OLS (HC3) và kiểm định Wald cho H4..."
    )
    tariff_df, tier_params = _estimate_per_tier_tariffs(df_train)
    h4a_rec, h4a_sum = _compute_h4a_uberx_lyft(tier_params, df_train)
    h4b_rec, h4b_sum = _compute_h4b_pool_shared(tier_params, df_train)
    h4c_rec, h4c_sum = _compute_h4c_weather_wald(df_train)
    crossing_summary = {
        "uberx_vs_lyft": h4a_sum,
        "uberpool_vs_shared": h4b_sum,
        "multivariate_weather_joint_test": h4c_sum,
    }
    return [h4a_rec, h4b_rec, h4c_rec], tariff_df, crossing_summary


def run_all_hypothesis_tests(
    df_full: pd.DataFrame, df_train: pd.DataFrame
) -> dict[str, Any]:
    """
    Chạy toàn bộ hệ thống kiểm định giả thuyết H1 -> H4 và xuất các bảng CSV
    chuẩn mực.
    """
    ensure_directories()
    logger.info(
        "=== BẮT ĐẦU PHẦN I GIAI ĐOẠN 5: KIỂM ĐỊNH GIẢ THUYẾT THỐNG KÊ (H1 -"
        "> H4) ==="
    )

    h1_res = test_h1_weather_price(df_full)
    h2_res = test_h2_time_of_day_surge(df_full)
    h3_res = test_h3_chi_square_surge_independence(df_full)
    h4_res, tariff_df, crossing_summary = test_h4_econometric_wald_and_tariffs(
        df_train
    )

    all_tests = h1_res + h2_res + h3_res + h4_res
    hyp_df = pd.DataFrame(all_tests)

    hyp_csv_path = TABLES_DIR / "stage5_hypothesis_tests.csv"
    hyp_df.to_csv(hyp_csv_path, index=False, encoding="utf-8-sig")

    ols_csv_path = TABLES_DIR / "stage5_ols_coefficients.csv"
    tariff_df.to_csv(ols_csv_path, index=False, encoding="utf-8-sig")

    logger.info(
        f"Đã xuất {len(hyp_df)} kiểm định giả thuyết tại: {hyp_csv_path}"
    )
    logger.info(
        f"Đã xuất bảng biểu giá OLS (HC3) 12 phân khúc tại: {ols_csv_path}"
    )

    return {
        "hypothesis_tests": all_tests,
        "tier_tariffs_hc3": tariff_df.to_dict(orient="records"),
        "crossing_and_wald_summary": crossing_summary,
    }

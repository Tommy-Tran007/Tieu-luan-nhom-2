"""
Unit & Metric Reconciliation Tests cho dashboard/business_logic.py (Giai đoạn
2).
Đối chiếu trực tiếp từng KPI động với kết quả độc lập và báo cáo GĐ3–GĐ6,
đồng thời kiểm thử cưỡng chế CRIT-01, CRIT-02, HIGH-02, HIGH-03 và trường hợp
lọc rỗng (0 dòng).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.business_logic import (
    FilterParams,
    apply_global_filters,
    compute_all_pair_comparisons,
    compute_binned_empirical_tariff,
    compute_data_quality_summary,
    compute_dayofweek_est_profile,
    compute_executive_kpis,
    compute_hourly_est_profile,
    compute_price_histogram_bins,
    compute_route_catalog,
    compute_surge_distribution,
    compute_tier_boxplot_stats,
    compute_time_of_day_est_profile,
    compute_weather_rain_comparison,
    compute_weather_reference_medians,
    extract_surge_cap_scenario,
    prepare_model_leaderboard,
    prepare_permutation_family_summary,
    prepare_segment_errors,
    prepare_surge_cap_display_table,
)
from dashboard.data_loader import load_data_bundle


@pytest.fixture(scope="module")
def bundle() -> dict:
    return load_data_bundle()


def test_executive_kpis_match_independent_ground_truth(bundle: dict) -> None:
    """
    Đối chiếu KPI Tab 1 với số liệu thực chứng v5 (bảo toàn nguồn gốc surge
    6.80%).
    """
    master = bundle["master"]
    tables = bundle["tables"]
    kpis = compute_executive_kpis(
        master, tables["test_metrics"], full_dataset_rows=len(master)
    )

    assert not kpis["is_empty"]
    assert kpis["total_obs"] == 270_898
    assert kpis["uber_count"] == 140_477
    assert kpis["lyft_count"] == 130_421
    assert kpis["lyft_obs_surge_count"] == 92_384
    assert kpis["lyft_surge_missing_count"] == 38_037
    assert kpis["uber_surge_count"] == 0
    assert kpis["uber_surge_rate_pct"] == 0.0
    assert kpis["lyft_surge_count"] == 6_280
    assert kpis["lyft_surge_rate_pct"] == 6.80
    assert np.isclose(
        kpis["mean_price_usd"],
        float(master["price"].astype(np.float64).mean()),
        atol=0.01,
    )
    assert np.isclose(
        kpis["median_price_usd"],
        float(master["price"].astype(np.float64).median()),
        atol=0.01,
    )
    assert kpis["m7_test_r2"] == 0.9626
    assert kpis["m7_test_rmse_usd"] == 1.7830
    assert kpis["m7_test_mae_usd"] == 1.1338
    assert kpis["m7_test_mape_pct"] == 8.19
    assert kpis["m2_test_r2"] == 0.9539
    assert kpis["m2_test_rmse_usd"] == 1.9802


def test_default_global_filter_retains_all_rows_including_nan_surge(
    bundle: dict,
) -> None:
    """
    CRIT-03: Bộ lọc mặc định [1.0, 3.0] không được đánh rơi 38,037 dòng Lyft
    khuyết surge (NaN).
    """
    master = bundle["master"]
    res_default = apply_global_filters(master, FilterParams())
    assert res_default.total_after == 270_898
    assert int(res_default.df["surge_multiplier"].isna().sum()) == 38_037

    res_narrow = apply_global_filters(
        master, FilterParams(surge_min=1.25, surge_max=3.0)
    )
    assert res_narrow.total_after == 6_280
    assert int(res_narrow.df["surge_multiplier"].isna().sum()) == 0


def test_empty_filter_uber_plus_surge_handled_gracefully(bundle: dict) -> None:
    """
    Chọn Uber + surge_only=True phải trả về 0 dòng kèm giải thích kinh tế rõ
    ràng, không crash KPI.
    """
    master = bundle["master"]
    tables = bundle["tables"]
    res = apply_global_filters(
        master, FilterParams(cab_types=("Uber",), surge_only=True)
    )

    assert res.is_empty
    assert res.total_after == 0
    assert res.empty_reason is not None and "Uber" in res.empty_reason

    kpis = compute_executive_kpis(res.df, tables["test_metrics"])
    assert kpis["is_empty"]
    assert kpis["mean_price_usd"] is None
    assert compute_tier_boxplot_stats(res.df).empty
    assert compute_price_histogram_bins(res.df).empty
    assert compute_route_catalog(res.df).empty
    w_df, w_meta = compute_weather_rain_comparison(res.df)
    assert w_df.empty and w_meta["n_matched"] == 0
    h_df, h_meta = compute_hourly_est_profile(res.df)
    assert h_df.empty and h_meta["n_valid_ts"] == 0


def test_crit01_weather_comparison_matches_h1a_exactly(bundle: dict) -> None:
    """
    CRIT-01 & H4: So sánh Mưa vs Tạnh khóa weather_matched==1 & sạch sentinel
    mưa (n = 152,043) khớp H1a.
    """
    master = bundle["master"]
    hyp = bundle["tables"]["hypothesis_tests"]
    h1a = hyp.loc[hyp["hypothesis_id"] == "H1a_All_Rides"].iloc[0]

    w_df, meta = compute_weather_rain_comparison(master)
    assert meta["n_matched"] == 179_280
    assert meta["n_valid_rain_obs"] == 152_043
    assert meta["n_rain_sentinel"] == 27_237
    assert meta["n_rain"] == int(h1a["n_group1"]) == 24_495
    assert meta["n_dry"] == int(h1a["n_group2"]) == 127_548
    assert np.isclose(
        meta["all_rain_mean_usd"], float(h1a["mean_group1_or_rate"]), atol=1e-4
    )
    assert np.isclose(
        meta["all_dry_mean_usd"], float(h1a["mean_group2_or_ref"]), atol=1e-4
    )
    assert not w_df.empty


def test_crit02_temporal_profiles_lock_original_est_timestamp(
    bundle: dict,
) -> None:
    """
    CRIT-02: Hồ sơ theo giờ, thứ và 4 buổi bắt buộc khóa
    is_timestamp_original==1 (n = 208,034, Boston EST).
    """
    master = bundle["master"]
    hourly, meta = compute_hourly_est_profile(master)
    assert meta["n_valid_ts"] == 208_034
    assert int(hourly["trips"].sum()) == 208_034
    assert meta["trough_hour_est"] == 3
    assert meta["peak_hour_est"] == 19

    dow = compute_dayofweek_est_profile(master)
    assert int(dow["trips"].sum()) == 208_034
    assert len(dow["day_of_week"].unique()) == 7

    tod = compute_time_of_day_est_profile(master)
    assert int(tod["trips"].sum()) == 208_034
    assert "Unknown" not in set(tod["time_of_day"].astype(str))


def test_high02_crossing_points_match_stage6_and_no_extrapolation(
    bundle: dict,
) -> None:
    """
    HIGH-02: Đúng 5 cặp đối xứng; UberPool vs Shared không cắt trong [0.02,
    7.86] dặm; UberX vs Lyft = 3.21 dặm (Spec A).
    """
    master = bundle["master"]
    tariffs = bundle["tables"]["tier_tariffs"]
    pairs = compute_all_pair_comparisons(tariffs, master)

    assert len(pairs) == 5
    pool = pairs.loc[pairs["pair_id"] == "pool_shared"].iloc[0]
    assert not bool(pool["crossing_in_domain"])
    assert pd.isna(pool["crossing_distance_miles"])

    std = pairs.loc[pairs["pair_id"] == "standard"].iloc[0]
    assert bool(std["crossing_in_domain"])
    assert std["crossing_distance_miles"] == 3.21
    assert 84.0 <= float(std["pct_trips_shorter_than_dstar"]) <= 88.5

    xl = pairs.loc[pairs["pair_id"] == "xl"].iloc[0]
    assert xl["crossing_distance_miles"] == 3.75

    binned = compute_binned_empirical_tariff(master, "UberX", "Lyft")
    assert not binned.empty


def test_high03_surge_cap_simulation_proxy_labels_and_values(
    bundle: dict,
) -> None:
    """
    HIGH-03 & C2: Bảng hiển thị chuẩn Quote Proxy (0 nhãn 'doanh thu'); kịch
    bản 2.00x & 1.75x khớp CSV v5.
    """
    sim = bundle["tables"]["surge_cap_simulation"]
    disp = prepare_surge_cap_display_table(sim)
    all_headers = " ".join(disp.columns).lower()
    assert "quote proxy" in all_headers
    assert "ràng buộc cung" in all_headers
    assert "doanh thu" not in all_headers

    cap175 = extract_surge_cap_scenario(
        sim, 1.75, "eps_050", supply_drop_pct=0
    )
    assert cap175["is_simulation"] is True
    assert cap175["affected_quotes_count"] == 704
    assert cap175["mean_saving_m7_usd"] == 4.61
    assert cap175["mean_saving_m2_usd"] == 5.14
    assert cap175["quote_proxy_change_pct"] == -0.116
    assert cap175["tree_saturation_warning"] is None

    cap200 = extract_surge_cap_scenario(
        sim, 2.00, "static_000", supply_drop_pct=0
    )
    assert cap200["affected_quotes_count"] == 53
    assert cap200["mean_saving_m7_usd"] == 5.53
    assert cap200["mean_saving_m2_usd"] == 5.88
    assert cap200["tree_saturation_warning"] is not None


def test_c2_surge_cap_supply_drop_monotonicity(bundle: dict) -> None:
    """
    C2 Short-Side Rationing: Mất cung càng lớn (-10% < -5% < tĩnh 0% <= cận
    trên) trên mọi kịch bản áp trần.
    """
    sim = bundle["tables"]["surge_cap_simulation"]
    for cap in (2.00, 1.75, 1.50, 1.25, 1.00):
        s0 = extract_surge_cap_scenario(
            sim, cap, "static_000", supply_drop_pct=0
        )
        s5 = extract_surge_cap_scenario(sim, cap, "eps_050", supply_drop_pct=5)
        s10 = extract_surge_cap_scenario(
            sim, cap, "eps_050", supply_drop_pct=10
        )
        up50 = extract_surge_cap_scenario(
            sim, cap, "eps_050", supply_drop_pct=0
        )

        assert (
            s10["quote_proxy_change_pct"]
            < s5["quote_proxy_change_pct"]
            < s0["quote_proxy_change_pct"]
        )
        assert s0["quote_proxy_change_pct"] <= up50["quote_proxy_change_pct"]
        assert (
            s10["unfulfilled_quotes_est"] > s5["unfulfilled_quotes_est"] > 0.0
        )
        # Đảm bảo khi supply_drop > 0, chọn elasticity khác nhau KHÔNG làm hỏng
        # đơn điệu
        s5_e80 = extract_surge_cap_scenario(
            sim, cap, "eps_080", supply_drop_pct=5
        )
        assert s5_e80["quote_proxy_change_pct"] == s5["quote_proxy_change_pct"]

    # Riêng 2.00x đạt khẩu vị bảo thủ tau <= 0.10% trên mọi chế độ cung
    assert (
        abs(
            extract_surge_cap_scenario(
                sim, 2.00, "static_000", supply_drop_pct=10
            )["quote_proxy_change_pct"]
        )
        <= 0.10
    )
    # Trần 1.75x vượt tau = 0.10% ngay từ cầu tĩnh
    assert (
        abs(
            extract_surge_cap_scenario(
                sim, 1.75, "static_000", supply_drop_pct=0
            )["quote_proxy_change_pct"]
        )
        > 0.10
    )


def test_supporting_aggregations_and_data_quality_summary(
    bundle: dict,
) -> None:
    """
    Kiểm tra bảng hộp 12 gói, 72 tuyến, phân bố surge Lyft thực (92,384), tổng
    hợp XAI và Data Quality Summary.
    """
    master = bundle["master"]
    tables = bundle["tables"]
    summaries = bundle["summaries"]

    box = compute_tier_boxplot_stats(master)
    assert len(box) == 12
    assert int(box["count"].sum()) == 270_898

    routes = compute_route_catalog(master)
    assert len(routes) == 72

    surge_dist = compute_surge_distribution(master)
    assert len(surge_dist) == 7
    assert int(surge_dist["trips"].sum()) == 92_384

    lb = prepare_model_leaderboard(tables["test_metrics"])
    assert lb.iloc[0]["model_code"] == "M7_HistGradientBoosting"

    fam = prepare_permutation_family_summary(tables["permutation_importance"])
    assert np.isclose(float(fam["m7_share_pct"].sum()), 100.0, atol=0.1)

    seg = prepare_segment_errors(tables["segment_errors"], "service_tier")
    assert len(seg) == 12

    w_ref = compute_weather_reference_medians(master)
    assert 35.0 <= w_ref["temp"] <= 45.0

    dq = compute_data_quality_summary(
        master,
        summaries["stage3"],
        summaries["stage4"],
        bundle["ram_mb"],
        bundle["est_unoptimized_ram_mb"],
    )
    assert dq["false_rain_labels_count"] == 0
    assert dq["ts_original_rows"] == 208_034
    assert dq["weather_matched_rows"] == 179_280
    assert dq["valid_rain_obs_rows"] == 152_043
    assert dq["rain_sentinel_rows"] == 27_237
    assert dq["surge_original_rows"] == 232_861
    assert dq["lyft_surge_missing_rows"] == 38_037

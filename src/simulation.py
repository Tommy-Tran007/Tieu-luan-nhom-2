"""
Module: simulation.py
Mục đích: Mô phỏng Chính sách Kinh tế Vi mô về Trần Giá Linh hoạt
(Counterfactual Surge Cap Policy Simulation) trả lời Câu hỏi Nghiên cứu RQ4,
có tích hợp Độ co giãn của cầu theo giá theo Cohen et al. (2016, NBER 22627).

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import BASE_CAT_FEATURES, TABLES_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_CAP_SCENARIOS: list[tuple[str, float, str]] = [
    (
        "Baseline_No_Cap_3.00x",
        3.00,
        "Hiện trạng (Không áp trần, tối đa 3.00x)",
    ),
    (
        "Cap_2.00x_Mild",
        2.00,
        "Kịch bản A: Trần 2.00x (Cắt đỉnh cực đoan — Đạt chuẩn bảo thủ "
        "tau <= 0.10%)",
    ),
    (
        "Cap_1.75x_Balanced",
        1.75,
        "Kịch bản B: Trần 1.75x (Mở rộng có điều kiện — Ngưỡng tau = "
        "0.30%-0.50%)",
    ),
    (
        "Cap_1.50x_Moderate",
        1.50,
        "Kịch bản C: Trần 1.50x (Bảo vệ khách hàng giờ cao điểm)",
    ),
    (
        "Cap_1.25x_Strict",
        1.25,
        "Kịch bản D: Trần 1.25x (Trần siết chặt mạnh)",
    ),
    (
        "Cap_1.00x_No_Surge",
        1.00,
        "Kịch bản E: Trần 1.00x (Xóa bỏ hoàn toàn Surge - Đối chứng)",
    ),
]


def _predict_capped_prices(
    X_lyft_base: pd.DataFrame,
    orig_surge: np.ndarray,
    actual_prices: np.ndarray,
    pred_base_m7: np.ndarray,
    pred_base_m2: np.ndarray,
    cap_val: float,
    affected_mask: np.ndarray,
    model_m7: Any,
    model_m2: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Ước lượng giá báo phản thực tế (M7 & M2) khi áp trần surge cap_val."""
    if int(np.sum(affected_mask)) == 0:
        return actual_prices.copy(), actual_prices.copy()
    X_cap = X_lyft_base.copy()
    capped_s = np.minimum(orig_surge, float(cap_val))
    X_cap["surge_multiplier"] = capped_s
    X_cap["lyft_surge_excess"] = (capped_s - 1.0) * X_cap["is_lyft"].astype(
        np.float64
    )
    X_cap["distance_x_surge"] = X_cap["distance"].astype(np.float64) * capped_s
    X_cap["distance_x_lyft_surge_excess"] = X_cap["distance"].astype(
        np.float64
    ) * X_cap["lyft_surge_excess"].astype(np.float64)

    pred_cap_m7 = np.clip(model_m7.predict(X_cap), 2.5, None)
    pred_cap_m2 = np.clip(model_m2.predict(X_cap), 2.5, None)
    ratio_m7 = np.clip(pred_cap_m7 / pred_base_m7, 0.25, 1.0)
    ratio_m2 = np.clip(pred_cap_m2 / pred_base_m2, 0.25, 1.0)
    new_m7 = np.where(affected_mask, actual_prices * ratio_m7, actual_prices)
    new_m2 = np.where(affected_mask, actual_prices * ratio_m2, actual_prices)
    return new_m7, new_m2


def _compute_demand_and_supply_metrics(
    actual_prices: np.ndarray,
    new_prices_m7: np.ndarray,
    affected_mask: np.ndarray,
    total_lyft_rev_base: float,
    cap_val: float,
) -> dict[str, float]:
    """Tính doanh thu proxy tĩnh, cận trên cầu và ràng buộc cung bên ngắn."""
    n_affected = int(np.sum(affected_mask))
    static_rev_m7 = float(np.sum(new_prices_m7))
    static_pct = (
        (static_rev_m7 - total_lyft_rev_base) / total_lyft_rev_base * 100.0
    )
    rel_p = (new_prices_m7 - actual_prices) / actual_prices

    q_demand_mid = 1.0 + (-0.50) * rel_p
    ret_mid = float(np.sum(q_demand_mid - 1.0))
    el_mid_pct = (
        (float(np.sum(new_prices_m7 * q_demand_mid)) - total_lyft_rev_base)
        / total_lyft_rev_base
        * 100.0
    )
    q_demand_hi = 1.0 + (-0.80) * rel_p
    ret_hi = float(np.sum(q_demand_hi - 1.0))
    el_hi_pct = (
        (float(np.sum(new_prices_m7 * q_demand_hi)) - total_lyft_rev_base)
        / total_lyft_rev_base
        * 100.0
    )

    # Ràng buộc cung bên ngắn phản ứng theo mức cắt giá tương đối rel_p <= 0:
    # q_S = clip(1 + eps_S * rel_p, floor, 1.0) với eps_S in {0.30, 0.60}.
    q_supply_mod = np.where(
        affected_mask, np.clip(1.0 + 0.30 * rel_p, 0.50, 1.0), 1.0
    )
    q_supply_sev = np.where(
        affected_mask, np.clip(1.0 + 0.60 * rel_p, 0.40, 1.0), 1.0
    )
    q_realized_drop5 = np.where(
        affected_mask, np.minimum(q_demand_mid, q_supply_mod), 1.0
    )
    q_realized_drop10 = np.where(
        affected_mask, np.minimum(q_demand_mid, q_supply_sev), 1.0
    )
    drop5_pct = (
        (float(np.sum(new_prices_m7 * q_realized_drop5)) - total_lyft_rev_base)
        / total_lyft_rev_base
        * 100.0
    )
    drop10_pct = (
        (
            float(np.sum(new_prices_m7 * q_realized_drop10))
            - total_lyft_rev_base
        )
        / total_lyft_rev_base
        * 100.0
    )
    unfulfilled_5 = float(
        np.sum(
            np.where(
                affected_mask,
                np.maximum(0.0, q_demand_mid - q_realized_drop5),
                0.0,
            )
        )
    )
    unfulfilled_10 = float(
        np.sum(
            np.where(
                affected_mask,
                np.maximum(0.0, q_demand_mid - q_realized_drop10),
                0.0,
            )
        )
    )
    if n_affected > 0:
        assert drop10_pct < drop5_pct < static_pct, (
            f"Vi phạm đơn điệu kinh tế vi mô tại cap={cap_val}: "
            f"drop10={drop10_pct:.4f}, drop5={drop5_pct:.4f}, "
            f"static={static_pct:.4f}"
        )
    return {
        "static_rev_m7": static_rev_m7,
        "static_pct": static_pct,
        "ret_mid": ret_mid,
        "el_mid_pct": el_mid_pct,
        "ret_hi": ret_hi,
        "el_hi_pct": el_hi_pct,
        "drop5_pct": drop5_pct,
        "drop10_pct": drop10_pct,
        "unfulfilled_drop5": unfulfilled_5,
        "unfulfilled_drop10": unfulfilled_10,
    }


def _evaluate_cap_scenario(
    scen_code: str,
    cap_val: float,
    scen_desc: str,
    base: dict[str, Any],
    model_m7: Any,
    model_m2: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    """Đánh giá đầy đủ 1 kịch bản trần Surge Cap trên mẫu Lyft quan sát."""
    actual_p = base["actual_prices"]
    orig_s = base["orig_surge"]
    affected = orig_s > cap_val + 1e-6
    n_aff = int(np.sum(affected))
    new_m7, new_m2 = _predict_capped_prices(
        base["X_lyft_base"],
        orig_s,
        actual_p,
        base["pred_base_m7"],
        base["pred_base_m2"],
        cap_val,
        affected,
        model_m7,
        model_m2,
    )
    sav_m7 = actual_p - new_m7
    tot_sav = float(np.sum(sav_m7))
    mean_sav = float(np.mean(sav_m7[affected])) if n_aff > 0 else 0.0
    mean_pct = (
        float(np.mean((sav_m7[affected] / actual_p[affected]) * 100.0))
        if n_aff > 0
        else 0.0
    )
    ds = _compute_demand_and_supply_metrics(
        actual_p, new_m7, affected, base["total_lyft_rev_base"], cap_val
    )
    sav_m2 = actual_p - new_m2
    m2_rev = float(np.sum(new_m2))
    m2_pct = (
        (m2_rev - base["total_lyft_rev_base"])
        / base["total_lyft_rev_base"]
        * 100.0
    )
    row = {
        "scenario_code": scen_code,
        "surge_cap": cap_val,
        "scenario_description": scen_desc,
        "affected_trips_count": n_aff,
        "affected_lyft_trips_pct": round(n_aff / base["n_lyft"] * 100.0, 2),
        "affected_surge_trips_pct": round(
            n_aff / base["n_surge_trips"] * 100.0, 2
        ),
        "mean_orig_price_affected_usd": (
            round(float(np.mean(actual_p[affected])), 2) if n_aff > 0 else 0.0
        ),
        "mean_capped_price_affected_usd": (
            round(float(np.mean(new_m7[affected])), 2) if n_aff > 0 else 0.0
        ),
        "mean_rider_saving_per_affected_trip_usd": round(mean_sav, 2),
        "mean_rider_saving_pct_affected": round(mean_pct, 2),
        "total_consumer_savings_usd": round(tot_sav, 2),
        "static_lyft_revenue_usd": round(ds["static_rev_m7"], 2),
        "static_lyft_rev_change_pct": round(ds["static_pct"], 3),
        "static_market_rev_change_pct": round(
            -tot_sav / base["total_market_rev_base"] * 100.0, 3
        ),
        "retained_trips_eps_050": round(ds["ret_mid"], 1),
        "net_lyft_rev_change_eps_050_pct": round(ds["el_mid_pct"], 3),
        "retained_trips_eps_080": round(ds["ret_hi"], 1),
        "net_lyft_rev_change_eps_080_pct": round(ds["el_hi_pct"], 3),
        "supply_constrained_rev_change_drop5_pct": round(ds["drop5_pct"], 3),
        "supply_constrained_rev_change_drop10_pct": round(ds["drop10_pct"], 3),
        "unfulfilled_quotes_drop5": round(ds["unfulfilled_drop5"], 1),
        "unfulfilled_quotes_drop10": round(ds["unfulfilled_drop10"], 1),
        "m2_ols_mean_saving_affected_usd": round(
            float(np.mean(sav_m2[affected])) if n_aff > 0 else 0.0, 2
        ),
        "m2_ols_total_savings_usd": round(float(np.sum(sav_m2)), 2),
        "m2_ols_static_lyft_rev_change_pct": round(m2_pct, 3),
    }
    return row, new_m7


def _build_simulation_extra_info(
    base: dict[str, Any], distributions: dict[str, list[float]]
) -> dict[str, Any]:
    """Đóng gói metadata & 3 giả định kinh tế học bắt buộc cho RQ4."""
    n_lyft = base["n_lyft"]
    rev_base = base["total_lyft_rev_base"]
    mkt_base = base["total_market_rev_base"]
    return {
        "n_total_lyft_quotes": int(n_lyft),
        "n_total_lyft_trips": int(n_lyft),
        "n_total_surge_quotes": int(base["n_surge_trips"]),
        "n_total_surge_trips": int(base["n_surge_trips"]),
        "total_lyft_baseline_quote_proxy_usd": round(rev_base, 2),
        "total_lyft_baseline_revenue_usd": round(rev_base, 2),
        "total_market_baseline_quote_proxy_usd": round(mkt_base, 2),
        "total_market_baseline_revenue_usd": round(mkt_base, 2),
        "economic_assumptions_and_limitations": [
            "1. Bản chất API Quote Proxy (CRIT-03): Tổng "
            f"${rev_base:,.2f} là tổng giá trị trên {n_lyft:,} "
            "lượt báo giá Lyft có surge quan sát thực (is_surge_original "
            "== 1, tách biệt khỏi 38,037 lượt khuyết surge) — KHÔNG đồng "
            "nhất với doanh thu kế toán của các chuyến hoàn thành.",
            "2. Nguyên tắc phân phối bên ngắn (Short-Side Rationing, C2): "
            "Sản lượng thực hiện Q_realized = min(Q_D, Q_S). Hai cột co "
            "giãn cầu (Cohen et al., 2016: epsilon_d in {-0.50, -0.80}) chỉ "
            "là cận trên lạc quan khi dư cung tài xế; khi cung tài xế nhạy "
            "giá theo mức cắt giảm (epsilon_S in {0.30, 0.60}; Q_S = "
            "clip(1 + epsilon_S * dp/p, sàn, 1.0)), mọi cầu vượt Q_S trở "
            "thành dư cầu không được phục vụ sum(max(0, Q_D - Q_realized)) "
            "và Proxy giảm đơn điệu theo độ nhạy cung.",
            "3. Giới hạn Ngoại suy Cây ở vùng hiếm (Tree Saturation): Tại "
            "kịch bản trần 2.00x (53 lượt có surge 2.50x-3.00x), mô hình "
            "cây M7 bão hòa nhánh lá nên báo tiết kiệm thấp hơn nhẹ; cần "
            "đối chứng song song với cột M2 OLS tương tác tuyến tính.",
        ],
        "counterfactual_surge_trip_distributions": distributions,
    }


def run_surge_cap_policy_simulation(
    df_feat: pd.DataFrame,
    valid_cols: list[str],
    model_m7: Any,
    model_m2: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Thực hiện mô phỏng phản thực tế (Counterfactual Simulation) khi áp dụng các
    mức trần hệ số nhân giá động C in {3.00, 2.00, 1.75, 1.50, 1.25, 1.00} trên
    nền tảng Lyft (92,384 lượt báo giá có surge quan sát thực).
    """
    logger.info(
        "Đang chạy mô phỏng chính sách trần giá linh hoạt (Surge Cap Policy "
        "Simulation)..."
    )
    obs_mask = (
        df_feat["is_surge_original"] == 1
        if "is_surge_original" in df_feat.columns
        else df_feat["surge_multiplier"].notna()
    )
    lyft_mask = (df_feat["cab_type"].astype(str) == "Lyft") & obs_mask
    lyft_df = df_feat.loc[lyft_mask].copy().reset_index(drop=True)
    actual_prices = lyft_df["price"].astype(np.float64).values
    orig_surge = lyft_df["surge_multiplier"].astype(np.float64).values

    X_lyft_base = lyft_df[valid_cols].copy()
    for c in BASE_CAT_FEATURES:
        X_lyft_base[c] = X_lyft_base[c].astype("category")

    surge_mask = orig_surge > 1.0
    base = {
        "n_lyft": len(lyft_df),
        "n_surge_trips": int(np.sum(surge_mask)),
        "actual_prices": actual_prices,
        "orig_surge": orig_surge,
        "X_lyft_base": X_lyft_base,
        "pred_base_m7": np.clip(model_m7.predict(X_lyft_base), 2.5, None),
        "pred_base_m2": np.clip(model_m2.predict(X_lyft_base), 2.5, None),
        "total_lyft_rev_base": float(np.sum(actual_prices, dtype=np.float64)),
        "total_market_rev_base": float(
            df_feat.loc[obs_mask, "price"].astype(np.float64).sum()
        ),
    }
    distributions: dict[str, list[float]] = {
        "Actual_Baseline": actual_prices[surge_mask].round(2).tolist()
    }
    sim_rows: list[dict[str, Any]] = []
    for code, cap_val, desc in _CAP_SCENARIOS:
        row, new_m7 = _evaluate_cap_scenario(
            code, cap_val, desc, base, model_m7, model_m2
        )
        sim_rows.append(row)
        if cap_val in (2.00, 1.75, 1.50, 1.25):
            distributions[f"Cap_{cap_val:.2f}x"] = (
                new_m7[surge_mask].round(2).tolist()
            )

    sim_df = pd.DataFrame(sim_rows)
    out_path = TABLES_DIR / "stage6_surge_cap_simulation.csv"
    sim_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu bảng mô phỏng chính sách trần Surge tại: {out_path}")
    return sim_df, _build_simulation_extra_info(base, distributions)

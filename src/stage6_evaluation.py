"""
Module: stage6_evaluation.py
Mục đích: Điều phối thực thi trọn vẹn Giai đoạn 6 (Chẩn đoán Phần dư Nâng cao,
Phân rã Sai số Đa chiều,
Khoảng Dự báo Bất định Quantile P10-P90, Diễn giải Mô hình Đa phương pháp & Mô
phỏng Chính sách Surge Cap),
xuất 5 bảng CSV, 7 biểu đồ 300 DPI (fig15 -> fig21) và tệp tổng hợp
reports/results/stage6_summary.json.

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import (
    BASE_CAT_FEATURES,
    BASE_DIR,
    FIG_STAGE6_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TABLES_DIR,
    TARGET_COL,
    TEST_SIZE,
    ensure_directories,
)
from evaluation import (
    diagnose_residuals,
    run_multidimensional_segment_errors,
    run_quantile_prediction_intervals,
)
from features import (
    create_data_splits,
    engineer_features_stage4,
    get_feature_columns_for_modeling,
    load_master_dataset,
)
from interpretation import (
    build_tier_tariffs_and_pairs_table,
    compute_pdp_and_ice_data,
    run_permutation_importance_analysis,
)
from simulation import run_surge_cap_policy_simulation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _plot_fig15_scatter_and_qq(
    ax1: Any,
    ax2: Any,
    y_pred_m7: np.ndarray,
    residuals_m7: np.ndarray,
    sample_idx: np.ndarray,
    diag_m7: dict[str, Any],
) -> None:
    """Vẽ Panel A (Residual vs Pred) và Panel B (Normal Q-Q) cho Fig 15."""
    ax1.scatter(
        y_pred_m7[sample_idx],
        residuals_m7[sample_idx],
        alpha=0.22,
        s=12,
        color="#0d9488",
        edgecolors="none",
        label="Phần dư mẫu Test (n=7,000)",
    )
    ax1.axhline(
        0.0, color="#dc2626", ls="--", lw=1.8, label="Đường cân bằng e = 0"
    )
    bins_pred = np.linspace(
        np.percentile(y_pred_m7, 1), np.percentile(y_pred_m7, 99), 18
    )
    bc, bm, bs = [], [], []
    for b0, b1 in zip(bins_pred[:-1], bins_pred[1:], strict=False):
        m_b = (y_pred_m7 >= b0) & (y_pred_m7 < b1)
        if np.sum(m_b) > 20:
            bc.append(0.5 * (b0 + b1))
            bm.append(float(np.mean(residuals_m7[m_b])))
            bs.append(float(np.std(residuals_m7[m_b])))
    ax1.plot(
        bc,
        bm,
        "o-",
        color="#1e3a8a",
        lw=2.2,
        markersize=5,
        label="Trung bình phần dư theo dải giá",
    )
    ax1.fill_between(
        bc,
        np.array(bm) - np.array(bs),
        np.array(bm) + np.array(bs),
        color="#93c5fd",
        alpha=0.25,
        label="Dải ±1 SD (Phương sai tăng theo giá -> Dùng HC3)",
    )
    ax1.set_xlim(2, 62)
    ax1.set_ylim(-14, 14)
    ax1.set_title(
        "A. Phần dư theo Giá dự báo (Breusch-Pagan "
        f"LM={diag_m7['breusch_pagan_lm_stat']:.1f}, p<0.0001)\n[Kỳ vọng "
        "phần dư bám sát 0; Phương sai mở rộng nhẹ ở phân khúc xe hạng sang]",
        fontsize=10.8,
        fontweight="bold",
    )
    ax1.set_xlabel("Giá báo API dự báo — Predicted Price ŷ (USD)", fontsize=10)
    ax1.set_ylabel("Phần dư — Residual e = y - ŷ (USD)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    std_res = (residuals_m7 - np.mean(residuals_m7)) / np.std(
        residuals_m7, ddof=1
    )
    osm, osr = stats.probplot(std_res[sample_idx], dist="norm", fit=False)
    ax2.scatter(
        osm,
        osr,
        alpha=0.35,
        s=12,
        color="#2563eb",
        label="Quantile thực nghiệm (M7)",
    )
    ax2.plot(
        [-4, 4],
        [-4, 4],
        color="#dc2626",
        ls="--",
        lw=1.8,
        label="Đường chuẩn lý thuyết N(0,1)",
    )
    ax2.set_xlim(-4.0, 4.0)
    ax2.set_ylim(-6.5, 9.5)
    ax2.set_title(
        "B. Biểu đồ Normal Q-Q Plot của Phần dư Chuẩn hóa\n[Bám sát chuẩn "
        "miền trung tâm [-1.5, +1.5]; Đuôi phải do surge ẩn Uber, "
        f"Kurtosis={diag_m7['kurtosis']:.2f}]",
        fontsize=10.3,
        fontweight="bold",
    )
    ax2.set_xlabel(
        "Phân vị lý thuyết N(0, 1) — Theoretical Quantiles", fontsize=10
    )
    ax2.set_ylabel(
        "Phân vị phần dư chuẩn hóa (cắt hiển thị tại [-6.5, +9.5])",
        fontsize=9.8,
    )
    ax2.legend(loc="upper left", fontsize=8.8, frameon=True)


def _plot_fig15_hist_and_box(
    ax3: Any,
    ax4: Any,
    df_test_meta: pd.DataFrame,
    residuals_m7: np.ndarray,
    diag_m7: dict[str, Any],
    tariffs_df: pd.DataFrame,
) -> None:
    """Vẽ Panel C (Histogram phần dư) và Panel D (Boxplot 12 gói) Fig 15."""
    in_range = residuals_m7[(residuals_m7 >= -8.0) & (residuals_m7 <= 8.0)]
    pct_in = len(in_range) / len(residuals_m7) * 100.0
    ax3.hist(
        in_range,
        bins=70,
        density=True,
        color="#0d9488",
        alpha=0.65,
        edgecolor="#0f172a",
        label=f"Phần dư trong [-8, +8] USD ({pct_in:.1f}% mẫu)",
    )
    ax3.axvline(
        -2.0,
        color="#e11d48",
        ls="--",
        lw=1.8,
        label=(
            f"Biên độ ±2.00 USD ({diag_m7['pct_within_2usd']:.1f}% lượt đạt)"
        ),
    )
    ax3.axvline(2.0, color="#e11d48", ls="--", lw=1.8)
    ax3.axvline(0.0, color="#1e3a8a", ls="-", lw=1.5)
    ax3.set_xlim(-8.0, 8.0)
    ax3.set_title(
        f"C. Phân phối Xác suất Phần dư (Mean={diag_m7['residual_mean']:+.3f} "
        f"USD, SD={diag_m7['residual_std']:.3f} USD)\n"
        f"[{diag_m7['pct_within_1usd']:.1f}% <= ±1 USD | "
        f"{diag_m7['pct_within_2usd']:.1f}% <= ±2 USD | "
        f"{diag_m7['pct_within_3usd']:.1f}% <= ±3 USD]",
        fontsize=10.4,
        fontweight="bold",
    )
    ax3.set_xlabel(
        "Phần dư e = y - ŷ (USD, hiển thị miền [-8, +8] không dồn biên)",
        fontsize=10,
    )
    ax3.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8.5, frameon=True)

    tier_order = tariffs_df.sort_values("mean_price_usd")[
        "service_tier"
    ].tolist()
    box_df = pd.DataFrame(
        {
            "service_tier": df_test_meta["name"].astype(str).values,
            "residual": residuals_m7,
        }
    )
    sns.boxplot(
        data=box_df,
        x="residual",
        y="service_tier",
        order=tier_order,
        ax=ax4,
        palette="viridis",
        hue="service_tier",
        legend=False,
        showfliers=False,
        width=0.6,
    )
    ax4.axvline(0.0, color="#dc2626", ls="--", lw=1.6)
    ax4.set_xlim(-6.5, 6.5)
    ax4.set_title(
        "D. Phân phối Phần dư (IQR Boxplot) theo 12 Phân khúc Dịch vụ\n"
        f"[Trung vị bám sát 0 (Median toàn tập = "
        f"{diag_m7['residual_median']:+.3f} USD); lệch hệ thống rất nhỏ]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax4.set_xlabel(
        "Phần dư e = y - ŷ (USD, ẩn ngoại lai cực đoan để rõ IQR)", fontsize=10
    )
    ax4.set_ylabel(
        "Phân khúc dịch vụ (Sắp xếp từ Giá rẻ -> Hạng sang)", fontsize=10
    )


def _save_fig15_residual_diagnostics(
    df_test_meta: pd.DataFrame,
    y_pred_m7: np.ndarray,
    residuals_m7: np.ndarray,
    diag_m7: dict[str, Any],
    tariffs_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> str:
    """Xuất Fig 15: Chẩn đoán phần dư 4 ô của M7."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5), dpi=300)
    (ax1, ax2), (ax3, ax4) = axes
    sample_idx = rng.choice(
        len(residuals_m7), size=min(7000, len(residuals_m7)), replace=False
    )
    _plot_fig15_scatter_and_qq(
        ax1, ax2, y_pred_m7, residuals_m7, sample_idx, diag_m7
    )
    _plot_fig15_hist_and_box(
        ax3, ax4, df_test_meta, residuals_m7, diag_m7, tariffs_df
    )
    fig.suptitle(
        "HÌNH 6.1 (FIG 15): CHẨN ĐOÁN PHẦN DƯ TOÀN DIỆN CỦA MÔ HÌNH DỰ BÁO "
        "CHÍNH M7 (HISTGRADIENTBOOSTING)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    p = FIG_STAGE6_DIR / "fig15_residual_diagnostics.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig16_tier_panel(ax_l: Any, seg_err_df: pd.DataFrame) -> None:
    """Vẽ Panel A của Fig 16: MAE ($) vs MAPE (%) theo 12 gói xe."""
    tier_err = seg_err_df.loc[
        seg_err_df["dimension"] == "service_tier"
    ].sort_values("mean_actual_usd", ascending=True)
    y_pos = np.arange(len(tier_err)) * 1.28
    w_bar, bar_offset = 0.42, 0.25

    ax_top = ax_l.twiny()
    ax_l.barh(
        y_pos - bar_offset,
        tier_err["m7_mae_usd"],
        height=w_bar,
        color="#0d9488",
        edgecolor="#0f172a",
        label="MAE (USD - Trục dưới)",
    )
    ax_top.barh(
        y_pos + bar_offset,
        tier_err["m7_mape_pct"],
        height=w_bar,
        color="#f59e0b",
        edgecolor="#0f172a",
        label="MAPE (% - Trục trên)",
    )
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(
        [
            f"{r['segment_value']} (${r['mean_actual_usd']:.1f})"
            for _, r in tier_err.iterrows()
        ],
        fontsize=9.2,
        fontweight="bold",
    )
    ax_l.set_xlim(0, 3.6)
    ax_top.set_xlim(0, 36.0)
    ax_l.set_ylim(-0.85, y_pos[-1] + 0.95)
    ax_l.grid(True, axis="x", alpha=0.35)
    ax_top.grid(False)
    mae_sh = float(
        tier_err.loc[tier_err["segment_value"] == "Shared", "m7_mae_usd"].iloc[
            0
        ]
    )
    mae_up = float(
        tier_err.loc[
            tier_err["segment_value"] == "UberPool", "m7_mae_usd"
        ].iloc[0]
    )
    ax_l.set_title(
        "A. Đối sánh MAE (USD) và MAPE (%) theo 12 Gói xe (2 Trục Độc lập)\n"
        f"[Shared & UberPool có MAE rất thấp "
        f"({min(mae_sh, mae_up):.2f}–{max(mae_sh, mae_up):.2f} USD) nhưng "
        "MAPE cao vì mẫu số giá nhỏ]",
        fontsize=10.2,
        fontweight="bold",
        pad=26,
    )
    ax_l.set_xlabel(
        "Sai số tuyệt đối trung bình — MAE (USD)",
        fontsize=10,
        color="#0f766e",
        fontweight="bold",
    )
    ax_top.set_xlabel(
        "Sai số phần trăm tuyệt đối trung bình — MAPE (%)",
        fontsize=10,
        color="#b45309",
        fontweight="bold",
    )
    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_top.get_legend_handles_labels()
    ax_l.legend(
        h1 + h2, l1 + l2, loc="upper right", fontsize=8.6, frameon=True
    )
    for (_, r), yi in zip(tier_err.iterrows(), y_pos, strict=False):
        ax_l.text(
            r["m7_mae_usd"] + 0.05,
            yi - bar_offset,
            f"${r['m7_mae_usd']:.2f}",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="#0f766e",
        )
        ax_top.text(
            r["m7_mape_pct"] + 0.5,
            yi + bar_offset,
            f"{r['m7_mape_pct']:.1f}%",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="#b45309",
        )


def _plot_fig16_heatmap_panel(
    ax_r: Any,
    df_test_meta: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_m7: np.ndarray,
) -> None:
    """Vẽ Panel B của Fig 16: Heatmap MAPE theo Quãng đường × Chế độ Surge."""
    work_hm = df_test_meta.copy()
    work_hm["ape_pct"] = np.abs(y_test - y_pred_m7) / y_test * 100.0
    work_hm["dist_bin"] = pd.cut(
        work_hm["distance"],
        bins=[0.0, 1.5, 3.0, 4.5, 100.0],
        labels=[
            "Ngắn (<1.5mi)",
            "Vừa (1.5-3.0mi)",
            "Khá dài (3.0-4.5mi)",
            "Dài (>4.5mi)",
        ],
    )
    work_hm["surge_bin"] = pd.cut(
        work_hm["surge_multiplier"],
        bins=[0.9, 1.05, 1.55, 2.05, 10.0],
        labels=[
            "1.0x (Bình thường)",
            "1.25x-1.50x (Surge nhẹ)",
            "1.75x-2.00x (Surge cao)",
            ">=2.50x (Cực đại)",
        ],
    )
    piv_mape = work_hm.pivot_table(
        index="surge_bin",
        columns="dist_bin",
        values="ape_pct",
        aggfunc="mean",
        observed=False,
    )
    sns.heatmap(
        piv_mape,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu_r",
        cbar_kws={"label": "MAPE trung bình (%)"},
        ax=ax_r,
        linewidths=1.0,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    vals = piv_mape.to_numpy(dtype=float)
    ax_r.set_title(
        "B. Ma trận Sai số MAPE (%) theo Quãng đường × Chế độ Surge\n[MAPE "
        f"toàn lưới từ {float(np.nanmin(vals)):.2f}% đến "
        f"{float(np.nanmax(vals)):.2f}%; thấp nhất khi có surge quan sát ở "
        "cự ly vừa–dài]",
        fontsize=10.3,
        fontweight="bold",
    )
    ax_r.set_xlabel(
        "Phân nhóm Quãng đường Lượt báo giá (Distance Bucket)", fontsize=10
    )
    ax_r.set_ylabel("Chế độ Hệ số Nhân giá Động (Surge Regime)", fontsize=10)


def _save_fig16_error_by_segment(
    df_test_meta: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_m7: np.ndarray,
    seg_err_df: pd.DataFrame,
) -> str:
    """Xuất Fig 16: Phân rã sai số đa chiều theo gói xe, cự ly và surge."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17.2, 7.8), dpi=300)
    _plot_fig16_tier_panel(ax_l, seg_err_df)
    _plot_fig16_heatmap_panel(ax_r, df_test_meta, y_test, y_pred_m7)
    fig.suptitle(
        "HÌNH 6.2 (FIG 16): PHÂN RÃ SAI SỐ ĐA CHIỀU THEO PHÂN KHÚC DỊCH VỤ, "
        "CỰ LY VÀ CHẾ ĐỘ SURGE",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE6_DIR / "fig16_error_by_segment.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig17_all_tiers_panel(
    ax_all: Any, tariffs_df: pd.DataFrame, d_vec: np.ndarray
) -> None:
    """Vẽ Panel A của Fig 17: 12 đường biểu giá tuyến tính OLS HC3."""
    palette_12 = sns.color_palette("tab20", 12)
    for idx_r, (_, r) in enumerate(
        tariffs_df.sort_values("mean_price_usd", ascending=False).iterrows()
    ):
        y_line = r["base_fare_b0_usd"] + r["per_mile_rate_b1_usd"] * d_vec
        ls = "--" if r["cab_type"] == "Lyft" else "-"
        ax_all.plot(
            d_vec,
            y_line,
            ls=ls,
            lw=2.0,
            color=palette_12[idx_r],
            label=(
                f"{r['cab_type']} - {r['service_tier']}: "
                f"${r['base_fare_b0_usd']:.2f} + "
                f"${r['per_mile_rate_b1_usd']:.2f}×d"
            ),
        )
    ax_all.set_title(
        "A. Cấu trúc Hàm Chi phí Tuyến tính 12 Phân khúc (Statsmodels OLS "
        "HC3)\n[Nét liền = Uber, Nét đứt = Lyft; Tách lớp rõ rệt qua 7 phân "
        "khúc từ Đi ghép -> SUV Hạng sang]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_all.set_xlabel(
        "Quãng đường hiệu dụng d = Distance × Surge (Dặm)", fontsize=10
    )
    ax_all.set_ylabel("Giá báo API kỳ vọng E[Price | d] (USD)", fontsize=10)
    ax_all.set_xlim(0.2, 6.0)
    ax_all.set_ylim(2, 54)
    ax_all.legend(loc="upper left", fontsize=7.8, ncol=2, frameon=True)


def _plot_fig17_pairs_panel(
    ax_pair: Any, tariffs_df: pd.DataFrame, d_vec: np.ndarray
) -> None:
    """Vẽ Panel B của Fig 17: 3 cặp đối đầu có điểm cắt trong miền."""
    t_map = {r["service_tier"]: r for _, r in tariffs_df.iterrows()}
    focus_pairs = [
        ("UberX", "Lyft", "#1d4ed8", "#e11d48", (3.35, 6.2)),
        ("UberXL", "Lyft XL", "#0284c7", "#be123c", (4.05, 16.5)),
        ("Black", "Lux Black", "#0f172a", "#9f1239", (0.75, 22.8)),
    ]
    for u_t, l_t, col_u, col_l, txt_xy in focus_pairs:
        ru, rl = t_map[u_t], t_map[l_t]
        ax_pair.plot(
            d_vec,
            ru["base_fare_b0_usd"] + ru["per_mile_rate_b1_usd"] * d_vec,
            "-",
            color=col_u,
            lw=2.3,
            label=(
                f"Uber {u_t} (${ru['base_fare_b0_usd']:.2f} + "
                f"${ru['per_mile_rate_b1_usd']:.2f}×d)"
            ),
        )
        ax_pair.plot(
            d_vec,
            rl["base_fare_b0_usd"] + rl["per_mile_rate_b1_usd"] * d_vec,
            "--",
            color=col_l,
            lw=2.3,
            label=(
                f"Lyft {l_t} (${rl['base_fare_b0_usd']:.2f} + "
                f"${rl['per_mile_rate_b1_usd']:.2f}×d)"
            ),
        )
        d_c = (ru["base_fare_b0_usd"] - rl["base_fare_b0_usd"]) / (
            rl["per_mile_rate_b1_usd"] - ru["per_mile_rate_b1_usd"]
        )
        if 0.2 <= d_c <= 5.8:
            p_c = ru["base_fare_b0_usd"] + ru["per_mile_rate_b1_usd"] * d_c
            ax_pair.scatter(
                [d_c],
                [p_c],
                color="#f59e0b",
                edgecolor="#0f172a",
                s=85,
                zorder=6,
            )
            ax_pair.annotate(
                f"Điểm cắt {u_t}/{l_t}\nd* = {d_c:.2f} dặm (${p_c:.2f})",
                xy=(d_c, p_c),
                xytext=txt_xy,
                arrowprops={"arrowstyle": "->", "color": "#0f172a", "lw": 1.3},
                fontsize=8.3,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#fef3c7",
                    "edgecolor": "#d97706",
                },
            )
    ax_pair.set_title(
        "B. Đối sánh Cạnh tranh Cặp Đôi & Điểm cắt Biểu giá (Crossing "
        "Tariffs)\n[Lyft luôn đặt giá mở cửa thấp hơn nhưng đơn giá/dặm cao "
        "hơn ở cả phân khúc 4 chỗ và 6 chỗ XL]",
        fontsize=10.6,
        fontweight="bold",
    )
    ax_pair.set_xlabel(
        "Quãng đường hiệu dụng d = Distance × Surge (Dặm)", fontsize=10
    )
    ax_pair.set_ylabel("Giá báo API kỳ vọng E[Price | d] (USD)", fontsize=10)
    ax_pair.set_xlim(0.2, 6.0)
    ax_pair.set_ylim(4, 42)
    ax_pair.legend(loc="upper left", fontsize=8.4, frameon=True)


def _save_fig17_tier_tariffs(tariffs_df: pd.DataFrame) -> str:
    """Xuất Fig 17: Đường biểu giá kinh tế lượng 12 gói & điểm cắt."""
    fig, (ax_all, ax_pair) = plt.subplots(1, 2, figsize=(17.0, 6.6), dpi=300)
    d_vec = np.linspace(0.3, 6.0, 150)
    _plot_fig17_all_tiers_panel(ax_all, tariffs_df, d_vec)
    _plot_fig17_pairs_panel(ax_pair, tariffs_df, d_vec)
    fig.suptitle(
        "HÌNH 6.3 (FIG 17): DIỄN GIẢI BIỂU GIÁ KINH TẾ LƯỢNG 12 GÓI XE & CƠ "
        "CHẾ CẠNH TRANH ĐỐI XỨNG UBER VS LYFT",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE6_DIR / "fig17_tier_tariff_lines.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig18_family_panel(
    ax_fam: Any, family_summary: list[dict[str, Any]], wt_share: float
) -> None:
    """Vẽ Panel B của Fig 18: Tỷ trọng 4 họ đặc trưng kinh tế."""
    fam_df = pd.DataFrame(family_summary).sort_values(
        "m7_share_pct", ascending=True
    )
    fam_colors = ["#94a3b8", "#f59e0b", "#2563eb", "#0d9488"][-len(fam_df) :]
    bars_f = ax_fam.barh(
        fam_df["domain_family"],
        fam_df["m7_share_pct"],
        color=fam_colors,
        edgecolor="#0f172a",
        height=0.55,
    )
    ax_fam.set_xlim(0, max(125, float(fam_df["m7_share_pct"].max()) * 1.72))
    ax_fam.set_title(
        "B. Tỷ trọng Quyền lực Dự báo theo 4 Họ Kinh tế (M7 — 27/27 cột)\n"
        f"[Nhóm Thời tiết & Khung giờ ≈ {wt_share:.2f}% -> Khớp tuyệt đối "
        "với Kiểm định H1–H4]",
        fontsize=9.9,
        fontweight="bold",
    )
    ax_fam.set_xlabel(
        "Tỷ trọng đóng góp vào Tổng mức tăng RMSE (%)", fontsize=10
    )
    for bar, (_, r) in zip(bars_f, fam_df.iterrows(), strict=False):
        w = bar.get_width()
        ax_fam.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}% (Tổng ΔRMSE = +${r['m7_sum_delta_rmse_usd']:.3f})",
            va="center",
            fontsize=9.2,
            fontweight="bold",
        )


def _save_fig18_permutation_importance(
    perm_df: pd.DataFrame, family_summary: list[dict[str, Any]]
) -> str:
    """Xuất Fig 18: Permutation Importance theo 14 nhóm và 4 họ đặc trưng."""
    fig, (ax_pi, ax_fam) = plt.subplots(1, 2, figsize=(17.0, 6.6), dpi=300)
    p_sorted = perm_df.sort_values("m7_delta_rmse_mean_usd", ascending=True)
    y_pi = np.arange(len(p_sorted))
    h_pi = 0.38
    ax_pi.barh(
        y_pi - h_pi / 2,
        p_sorted["m7_delta_rmse_mean_usd"],
        xerr=p_sorted["m7_delta_rmse_std_usd"],
        height=h_pi,
        color="#0d9488",
        edgecolor="#0f172a",
        label="M7: HistGradientBoosting (Dự báo chính)",
    )
    ax_pi.barh(
        y_pi + h_pi / 2,
        p_sorted["m2_delta_rmse_mean_usd"],
        xerr=p_sorted["m2_delta_rmse_std_usd"],
        height=h_pi,
        color="#2563eb",
        edgecolor="#0f172a",
        label="M2: OLS Tương tác Kinh tế (Diễn giải chính)",
    )
    ax_pi.set_yticks(y_pi)
    ax_pi.set_yticklabels(
        p_sorted["feature_group"], fontsize=8.8, fontweight="bold"
    )
    max_x = max(
        p_sorted["m7_delta_rmse_mean_usd"].max(),
        p_sorted["m2_delta_rmse_mean_usd"].max(),
    )
    ax_pi.set_xlim(-0.2, max_x * 1.52)
    wt_share = float(
        perm_df.loc[
            perm_df["domain_family"] == "4. Thời tiết & Khung giờ",
            "m7_importance_share_pct",
        ].sum()
    )
    ax_pi.set_title(
        "A. Mức tăng Sai số RMSE (ΔRMSE, USD) khi Hoán vị 14 Nhóm Biến Rời "
        f"nhau (27/27 cột)\n[Gói xe, Quãng đường & Surge chi phối "
        f"~{100.0 - wt_share:.2f}% quyền lực dự báo; Không đếm trùng]",
        fontsize=9.9,
        fontweight="bold",
    )
    ax_pi.set_xlabel(
        "Mức tăng RMSE trên Tập Test khi làm nhiễu biến - ΔRMSE (USD)",
        fontsize=10,
    )
    ax_pi.legend(loc="lower right", fontsize=8.6, frameon=True)
    for i, (_, r) in enumerate(p_sorted.iterrows()):
        v7 = r["m7_delta_rmse_mean_usd"]
        lbl_pi = (
            f"+${v7:.3f} ({r['m7_importance_share_pct']:.2f}%)"
            if v7 >= 0.01
            else (f"+${v7:.4f} (~0%)" if v7 >= 0 else f"-${abs(v7):.4f} (~0%)")
        )
        ax_pi.text(
            max(v7, r["m2_delta_rmse_mean_usd"], 0.0) + 0.18,
            i,
            lbl_pi,
            va="center",
            fontsize=8.1,
            fontweight="bold",
        )
    _plot_fig18_family_panel(ax_fam, family_summary, wt_share)
    fig.suptitle(
        "HÌNH 6.4 (FIG 18): PHÂN TÍCH MỨC ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG QUA "
        "HOÁN VỊ (PERMUTATION IMPORTANCE)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE6_DIR / "fig18_permutation_importance.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig19_distance_panel(ax_p1: Any, pdp_data: dict[str, Any]) -> None:
    """Vẽ Panel A của Fig 19: PDP & ICE theo quãng đường."""
    d_grid = np.array(pdp_data["distance_grid"])
    for row_ice in np.array(pdp_data["ice_distance_m7"])[:25]:
        ax_p1.plot(d_grid, row_ice, color="#cbd5e1", alpha=0.45, lw=0.9)
    ax_p1.plot(
        d_grid,
        pdp_data["pdp_distance_m7"],
        color="#0d9488",
        lw=3.0,
        label="PDP Toàn mẫu - M7 (HistGB)",
    )
    ax_p1.plot(
        d_grid,
        pdp_data["pdp_distance_m2"],
        color="#1d4ed8",
        ls="--",
        lw=2.4,
        label="PDP Toàn mẫu - M2 (OLS Tương tác)",
    )
    for t_name, t_col in [
        ("Black SUV", "#7c3aed"),
        ("UberX", "#ea580c"),
        ("Shared", "#64748b"),
    ]:
        ax_p1.plot(
            d_grid,
            pdp_data["tier_pdp_distance_curves"][t_name],
            ls=":",
            lw=2.0,
            color=t_col,
            label=f"PDP {t_name} (M7)",
        )
    ax_p1.axvspan(
        0.3, 1.0, color="#fef3c7", alpha=0.45, label="Vùng giá sàn (<1.0 dặm)"
    )
    ax_p1.set_title(
        "A. PDP & ICE theo Quãng đường (distance)\n[M7 tự học vùng phẳng "
        "giá sàn tối thiểu ở <1.0 dặm]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_p1.set_xlabel("Quãng đường hành trình — Distance (Dặm)", fontsize=10)
    ax_p1.set_ylabel(
        "Giá báo dự báo kỳ vọng E[ŷ | distance] (USD)", fontsize=10
    )
    ax_p1.legend(loc="upper left", fontsize=8.0, frameon=True)


def _save_fig19_partial_dependence(pdp_data: dict[str, Any]) -> str:
    """Xuất Fig 19: PDP, ICE theo distance/surge và bề mặt 2D."""
    fig, (ax_p1, ax_p2, ax_p3) = plt.subplots(
        1, 3, figsize=(18.0, 5.8), dpi=300
    )
    _plot_fig19_distance_panel(ax_p1, pdp_data)

    s_grid = np.array(pdp_data["surge_grid"])
    for row_ice in np.array(pdp_data["ice_surge_lyft_m7"])[:25]:
        ax_p2.plot(s_grid, row_ice, color="#fecdd3", alpha=0.45, lw=0.9)
    ax_p2.plot(
        s_grid,
        pdp_data["pdp_surge_lyft_m7"],
        "o-",
        color="#e11d48",
        lw=2.8,
        label="PDP Lyft Surge - M7 (HistGB)",
    )
    ax_p2.plot(
        s_grid,
        pdp_data["pdp_surge_lyft_m2"],
        "s--",
        color="#1e3a8a",
        lw=2.2,
        label="PDP Lyft Surge - M2 (OLS Tương tác)",
    )
    ax_p2.set_title(
        "B. PDP & ICE theo Hệ số nhân Surge (Nội bộ Lyft)\n[Độ dốc ICE phân "
        "kỳ mạnh theo cự ly và hạng xe]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_p2.set_xlabel(
        "Hệ số nhân giá động — Surge Multiplier (1.0x → 3.0x)", fontsize=10
    )
    ax_p2.set_ylabel("Giá báo dự báo kỳ vọng trên Lyft (USD)", fontsize=10)
    ax_p2.legend(loc="upper left", fontsize=8.5, frameon=True)

    d_2d = np.array(pdp_data["grid_2d_distance"])
    s_2d = np.array(pdp_data["grid_2d_surge"])
    z_2d = np.array(pdp_data["grid_2d_lyft_fare"])
    cp = ax_p3.contourf(d_2d, s_2d, z_2d, levels=14, cmap="Spectral_r")
    cs = ax_p3.contour(
        d_2d, s_2d, z_2d, levels=7, colors="#0f172a", linewidths=0.8
    )
    ax_p3.clabel(cs, inline=True, fontsize=8.5, fmt="$%.1f")
    fig.colorbar(cp, ax=ax_p3, label="Giá báo dự báo gói Lyft 4 chỗ (USD)")
    ax_p3.set_title(
        "C. Bề mặt Phụ thuộc Đồng thời 2D (Distance × Surge)\n[Đường đồng "
        "mức cong khẳng định tương tác nhân tính d × s]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_p3.set_xlabel("Quãng đường - Distance (Dặm)", fontsize=10)
    ax_p3.set_ylabel("Hệ số nhân giá - Surge Multiplier", fontsize=10)

    fig.suptitle(
        "HÌNH 6.5 (FIG 19): ĐƯỜNG CONG PHỤ THUỘC RIÊNG PHẦN (PDP), ĐƯỜNG CÁ "
        "THỂ (ICE) & BỀ MẶT TƯƠNG TÁC 2 CHIỀU",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    p = FIG_STAGE6_DIR / "fig19_partial_dependence.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig20_tradeoff_panel(ax_s1: Any, sim_df: pd.DataFrame) -> None:
    """Vẽ Panel A của Fig 20: Đánh đổi Quote Proxy % theo trần Surge Cap."""
    active = sim_df.loc[sim_df["surge_cap"].isin([2.00, 1.75, 1.50, 1.25])]
    x_c = np.arange(len(active))
    w_c = 0.26
    bars_spec = [
        (
            -w_c,
            "static_lyft_rev_change_pct",
            "#94a3b8",
            "API Quote Proxy Tĩnh (Không xét co giãn cầu: ε = 0.0)",
        ),
        (
            0.0,
            "net_lyft_rev_change_eps_050_pct",
            "#2563eb",
            "API Quote Proxy Cận trên Cầu (Cohen et al., ε = -0.50)",
        ),
        (
            w_c,
            "net_lyft_rev_change_eps_080_pct",
            "#0d9488",
            "API Quote Proxy Cận trên Cầu (Cạnh tranh cao, ε = -0.80)",
        ),
    ]
    bar_groups = [
        ax_s1.bar(
            x_c + off,
            active[col],
            width=w_c,
            color=c_hex,
            edgecolor="#0f172a",
            label=lbl,
        )
        for off, col, c_hex, lbl in bars_spec
    ]
    ax_s1.axhline(0.0, color="#0f172a", lw=1.2)
    ax_s1.set_xticks(x_c)
    ax_s1.set_xticklabels(
        [
            f"Trần {r['surge_cap']:.2f}x\n(Tiết kiệm "
            f"${r['mean_rider_saving_per_affected_trip_usd']:.2f}/cuốc)"
            for _, r in active.iterrows()
        ],
        fontsize=9.3,
        fontweight="bold",
    )
    c175 = active.loc[active["surge_cap"] == 1.75].iloc[0]
    c150 = active.loc[active["surge_cap"] == 1.50].iloc[0]
    min_rev = float(active[[s[1] for s in bars_spec]].min().min())
    ax_s1.set_ylim(min(-1.65, min_rev * 1.30), 0.35)
    ax_s1.set_title(
        "A. Đánh đổi Giá trị Báo giá Lyft (Quote Proxy %) và Tiết kiệm theo "
        "Trần Surge\n[Trần 1.50x–1.75x: giảm "
        f"${c175['mean_rider_saving_per_affected_trip_usd']:.2f}–$"
        f"{c150['mean_rider_saving_per_affected_trip_usd']:.2f}/lượt; "
        f"báo giá ròng -{abs(c175['net_lyft_rev_change_eps_050_pct']):.2f}% "
        f"đến -{abs(c150['net_lyft_rev_change_eps_050_pct']):.2f}% (ε=-0.5)]",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_s1.set_ylabel(
        "Thay đổi Tổng Giá trị Báo giá Lyft - Quote Proxy (%)", fontsize=9.8
    )
    ax_s1.legend(loc="lower left", fontsize=8.5, frameon=True)
    for grp in bar_groups:
        for bar in grp:
            v = bar.get_height()
            ax_s1.text(
                bar.get_x() + bar.get_width() / 2,
                v - 0.08 if v < 0 else v + 0.03,
                f"{v:.2f}%",
                ha="center",
                va="top" if v < 0 else "bottom",
                fontsize=8.2,
                fontweight="bold",
            )


def _save_fig20_surge_cap(
    sim_df: pd.DataFrame, sim_extra: dict[str, Any]
) -> str:
    """Xuất Fig 20: Mô phỏng chính sách kinh tế trần giá Surge Cap."""
    fig, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(17.0, 6.4), dpi=300)
    _plot_fig20_tradeoff_panel(ax_s1, sim_df)

    dist_dict = sim_extra["counterfactual_surge_trip_distributions"]
    lbl_map = [
        ("Actual_Baseline", "1. Hiện trạng\n(Trần 3.00x)"),
        ("Cap_2.00x", "2. Áp trần 2.00x\n(Kịch bản A)"),
        ("Cap_1.75x", "3. Áp trần 1.75x\n(Kịch bản B - Tối ưu)"),
        ("Cap_1.50x", "4. Áp trần 1.50x\n(Kịch bản C)"),
        ("Cap_1.25x", "5. Áp trần 1.25x\n(Kịch bản D)"),
    ]
    rows_d = [
        {"Scenario": lbl, "Price": val}
        for k_d, lbl in lbl_map
        for val in dist_dict[k_d]
    ]
    sns.boxplot(
        data=pd.DataFrame(rows_d),
        x="Scenario",
        y="Price",
        hue="Scenario",
        palette=["#e11d48", "#f59e0b", "#0d9488", "#2563eb", "#64748b"],
        legend=False,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.25},
        ax=ax_s2,
        width=0.52,
    )
    ax_s2.set_title(
        "B. Phân phối Giá trên 6,280 Lượt Lyft có Surge Trước & Sau Áp Trần\n"
        "[Thu hẹp đỉnh sốc >=2.0x; ngoại lai >$55 còn lại là Lux Black XL "
        "đường dài]",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_s2.set_xlabel(
        "Kịch bản Chính sách Trần Giá Động (Surge Cap Scenario)", fontsize=10
    )
    ax_s2.set_ylabel("Giá báo API trong Khung Surge (USD)", fontsize=10)
    fig.suptitle(
        "HÌNH 6.6 (FIG 20): MÔ PHỎNG CHÍNH SÁCH KINH TẾ TRẦN GIÁ LINH HOẠT "
        "(SURGE CAP POLICY SIMULATION - RQ4)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE6_DIR / "fig20_surge_cap_simulation.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def _plot_fig21_bands_panel(
    ax_q1: Any,
    df_test_meta: pd.DataFrame,
    y_test: np.ndarray,
    pi_summary: dict[str, Any],
    pi_preds: dict[str, np.ndarray],
    rng: np.random.RandomState,
) -> None:
    """Vẽ Panel A của Fig 21: Dải báo giá Quantile [P10, P50, P90]."""
    for tier_q, col_line, col_fill in [
        ("UberX", "#1d4ed8", "#93c5fd"),
        ("Black SUV", "#0d9488", "#5eead4"),
    ]:
        mask_q = (df_test_meta["name"].astype(str) == tier_q) & (
            df_test_meta["surge_multiplier"] == 1.0
        )
        d_q = df_test_meta.loc[mask_q, "distance"].values
        y_q = y_test[mask_q]
        p10_q, p50_q, p90_q = (
            pi_preds["p10"][mask_q],
            pi_preds["p50"][mask_q],
            pi_preds["p90"][mask_q],
        )
        bins_q = np.linspace(0.4, 5.4, 16)
        bc_q, m10, m50, m90 = [], [], [], []
        for b0, b1 in zip(bins_q[:-1], bins_q[1:], strict=False):
            in_b = (d_q >= b0) & (d_q < b1)
            if np.sum(in_b) > 10:
                bc_q.append(0.5 * (b0 + b1))
                m10.append(float(np.mean(p10_q[in_b])))
                m50.append(float(np.mean(p50_q[in_b])))
                m90.append(float(np.mean(p90_q[in_b])))
        sub_pts = rng.choice(len(d_q), size=min(400, len(d_q)), replace=False)
        ax_q1.scatter(
            d_q[sub_pts], y_q[sub_pts], color=col_line, alpha=0.18, s=11
        )
        ax_q1.plot(
            bc_q,
            m50,
            "o-",
            color=col_line,
            lw=2.4,
            label=f"{tier_q}: Dự báo Trung vị P50",
        )
        ax_q1.fill_between(
            bc_q,
            m10,
            m90,
            color=col_fill,
            alpha=0.35,
            label=f"{tier_q}: Dải bất định 80% [P10, P90]",
        )
    ax_q1.set_xlim(0.3, 5.6)
    ax_q1.set_ylim(4, 52)
    ax_q1.set_title(
        "A. Dải Báo giá Bất định Quantile [P10, P50, P90] theo Quãng đường\n"
        f"[Tỷ lệ bao phủ thực tế PICP toàn tập Test = "
        f"{pi_summary['empirical_coverage_picp_pct']:.2f}% (Kỳ vọng lý "
        "thuyết = 80.0%)]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_q1.set_xlabel("Quãng đường chuyến đi - Distance (Dặm)", fontsize=10)
    ax_q1.set_ylabel(
        "Giá báo API thực tế & Dải dự báo [P10, P90] (USD)", fontsize=10
    )
    ax_q1.legend(loc="upper left", fontsize=8.5, frameon=True)


def _save_fig21_prediction_interval(
    df_test_meta: pd.DataFrame,
    y_test: np.ndarray,
    pi_summary: dict[str, Any],
    pi_df: pd.DataFrame,
    pi_preds: dict[str, np.ndarray],
    rng: np.random.RandomState,
) -> str:
    """Xuất Fig 21: Khoảng dự báo bất định Quantile P10–P90."""
    fig, (ax_q1, ax_q2) = plt.subplots(1, 2, figsize=(17.0, 6.4), dpi=300)
    _plot_fig21_bands_panel(
        ax_q1, df_test_meta, y_test, pi_summary, pi_preds, rng
    )
    pi_sorted = pi_df.sort_values("mean_actual_usd", ascending=True)
    y_q2 = np.arange(len(pi_sorted))
    bars_cov = ax_q2.barh(
        y_q2,
        pi_sorted["picp_coverage_pct"],
        color="#0d9488",
        edgecolor="#0f172a",
        height=0.6,
    )
    ax_q2.axvline(
        80.0,
        color="#e11d48",
        ls="--",
        lw=1.8,
        label="Mức bao phủ danh nghĩa mục tiêu (80.0%)",
    )
    ax_q2.set_yticks(y_q2)
    ax_q2.set_yticklabels(
        [
            f"{r['cab_type']} - {r['service_tier']}"
            for _, r in pi_sorted.iterrows()
        ],
        fontsize=9.2,
        fontweight="bold",
    )
    ax_q2.set_xlim(0, 126)
    ax_q2.set_ylim(-0.6, len(pi_sorted) + 0.9)
    mpiw = pi_summary["mean_interval_width_mpiw_usd"]
    ax_q2.set_title(
        "B. Tỷ lệ Bao phủ Thực nghiệm PICP (%) và Độ rộng Dải giá MPIW "
        f"(USD) theo 12 Gói xe\n[Độ rộng dải báo giá trung bình chỉ "
        f"{mpiw:.2f} USD (±{mpiw / 2:.2f} USD quanh P50)]",
        fontsize=10.3,
        fontweight="bold",
    )
    ax_q2.set_xlabel(
        "Tỷ lệ Chuyến đi Thực tế Nằm trong Khoảng Dự báo [P10, P90] (%)",
        fontsize=10,
    )
    ax_q2.legend(loc="upper right", fontsize=8.8, frameon=True)
    for bar, (_, r) in zip(bars_cov, pi_sorted.iterrows(), strict=False):
        w = bar.get_width()
        ax_q2.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"PICP={w:.1f}% | Rộng=${r['mpiw_width_usd']:.2f}",
            va="center",
            fontsize=8.4,
            fontweight="bold",
        )
    fig.suptitle(
        "HÌNH 6.7 (FIG 21): ƯỚC LƯỢNG KHOẢNG DỰ BÁO BẤT ĐỊNH P10–P90 BẰNG "
        "QUANTILE HISTGRADIENTBOOSTING",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = FIG_STAGE6_DIR / "fig21_prediction_interval.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.relative_to(BASE_DIR).as_posix()


def generate_stage6_figures(
    df_test_meta: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_m7: np.ndarray,
    y_pred_m2: np.ndarray,
    diag_m7: dict[str, Any],
    seg_err_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    pair_comparisons: list[dict[str, Any]],
    perm_df: pd.DataFrame,
    family_summary: list[dict[str, Any]],
    pdp_data: dict[str, Any],
    sim_df: pd.DataFrame,
    sim_extra: dict[str, Any],
    pi_summary: dict[str, Any],
    pi_df: pd.DataFrame,
    pi_preds: dict[str, np.ndarray],
) -> list[str]:
    """Vẽ và lưu 7 biểu đồ chuẩn xuất bản 300 DPI của Giai đoạn 6."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.8

    rng = np.random.RandomState(RANDOM_STATE)
    residuals_m7 = y_test - y_pred_m7
    return [
        _save_fig15_residual_diagnostics(
            df_test_meta, y_pred_m7, residuals_m7, diag_m7, tariffs_df, rng
        ),
        _save_fig16_error_by_segment(
            df_test_meta, y_test, y_pred_m7, seg_err_df
        ),
        _save_fig17_tier_tariffs(tariffs_df),
        _save_fig18_permutation_importance(perm_df, family_summary),
        _save_fig19_partial_dependence(pdp_data),
        _save_fig20_surge_cap(sim_df, sim_extra),
        _save_fig21_prediction_interval(
            df_test_meta, y_test, pi_summary, pi_df, pi_preds, rng
        ),
    ]


def _build_rq1_entry(perm_df: pd.DataFrame) -> dict[str, str]:
    """Tạo bản ghi tổng hợp bằng chứng RQ1 (thời tiết & khung giờ)."""
    wt_share = float(
        perm_df.loc[
            perm_df["domain_family"] == "4. Thời tiết & Khung giờ",
            "m7_importance_share_pct",
        ].sum()
    )
    hyp_df = pd.read_csv(TABLES_DIR / "stage5_hypothesis_tests.csv").set_index(
        "hypothesis_id"
    )
    h1a = hyp_df.loc["H1a_All_Rides"]
    h2a = hyp_df.loc["H2a_Surge_All"]
    h3b = hyp_df.loc["H3b_Surge_x_Rain_Lyft"]
    h4c = hyp_df.loc["H4c_Wald_Joint_Weather_Multivariate"]
    return {
        "rq_id": "RQ1",
        "question": (
            "Các yếu tố thời tiết (mưa, nhiệt độ, độ ẩm, gió) và khung giờ "
            "tác động như thế nào đến giá báo và xác suất kích hoạt Surge?"
        ),
        "empirical_evidence": (
            "Trên tập khớp thời tiết thực không sentinel mưa (n = "
            f"{int(h1a['n_group1'] + h1a['n_group2']):,}) và timestamp "
            f"gốc Boston EST: Welch's t-test mưa vs tạnh: t = "
            f"{h1a['parametric_stat']:.4f}, p = "
            f"{h1a['parametric_pvalue']:.4f}, Cohen's d = "
            f"{h1a['effect_size_value']:+.5f}; ANOVA 4 buổi EST (n = "
            f"{int(h2a['n_group1'] + h2a['n_group2']):,}): F = "
            f"{h2a['parametric_stat']:.4f}, p = "
            f"{h2a['parametric_pvalue']:.4f}, eta^2 = "
            f"{h2a['effect_size_value']:.6f}; Chi-Square Lyft Surge x Mưa "
            f"(n = {int(h3b['n_group1'] + h3b['n_group2']):,}): Cramér's V = "
            f"{h3b['effect_size_value']:.5f} (p = "
            f"{h3b['parametric_pvalue']:.4f}); Joint Wald HC3 6 biến thời "
            f"tiết (n_train = {int(h4c['n_group1']):,}): F = "
            f"{h4c['parametric_stat']:.4f}, p = "
            f"{h4c['parametric_pvalue']:.4f}, Delta R2 = "
            f"{h4c['effect_size_value']:.7f}; Permutation Importance nhóm "
            f"thời tiết & khung giờ (27/27 cột): tổng {wt_share:.2f}%."
        ),
        "conclusion": (
            "Trong 24 ngày lịch cuối thu ôn hòa tại Boston (19.6–55.4 °F, "
            "mưa giờ <= 0.78 inch/h, không có bão tuyết cực đoan) theo độ "
            "phân giải từng giờ, thời tiết và khung giờ cố định KHÔNG tạo "
            "phụ phí cơ học lên giá báo hay xác suất surge; biến quyết định "
            "thực sự là mất cân bằng cung–cầu vi mô tức thời không quan sát "
            "được trong bảng."
        ),
        "confidence_level": (
            "Rất cao (Đồng thuận 100% qua Kiểm định Tham số, Phi tham số, "
            "Đa biến HC3 trên tập khớp thực, Lasso L1 và Permutation "
            "Importance)."
        ),
    }


def _build_rq2_entry(
    s5_summary: dict[str, Any], pair_comparisons: list[dict[str, Any]]
) -> dict[str, str]:
    """Tạo bản ghi tổng hợp bằng chứng RQ2 (cấu trúc biểu giá & điểm cắt)."""
    wald = s5_summary["part1_hypothesis_testing"]["crossing_and_wald_summary"]
    cx_s5, cp_s5 = wald["uberx_vs_lyft"], wald["uberpool_vs_shared"]
    ux_pair = next(p for p in pair_comparisons if p["uber_tier"] == "UberX")
    xl_pair = next(p for p in pair_comparisons if p["uber_tier"] == "UberXL")
    xl_cross = xl_pair.get("crossing_distance_miles")
    xl_txt = (
        f"Lyft XL rẻ hơn UberXL dưới {xl_cross:.2f} dặm"
        if xl_cross is not None
        else (
            "UberXL–Lyft XL không cắt trong miền quan sát [0.02, "
            f"{cp_s5['max_observed_distance_miles']:.2f}] dặm"
        )
    )
    xl_rng = (
        f"{ux_pair['crossing_distance_miles']:.2f}–{xl_cross:.2f} dặm"
        if xl_cross is not None
        else f"> {ux_pair['crossing_distance_miles']:.2f} dặm"
    )
    return {
        "rq_id": "RQ2",
        "question": (
            "Cấu trúc biểu giá cơ bản và chiến lược cạnh tranh giữa 12 phân "
            "khúc dịch vụ của Uber và Lyft khác biệt ra sao?"
        ),
        "empirical_evidence": (
            "Kiểm định Wald HC3 xác nhận chiến lược Biểu giá Cắt nhau "
            "(Crossing Tariffs): Lyft có giá mở cửa thấp hơn UberX "
            f"${cx_s5['base_diff_usd']:.2f} (${cx_s5['lyft_base_b0']:.2f} vs "
            f"${cx_s5['uberx_base_b0']:.2f}) nhưng đơn giá mỗi dặm cao hơn "
            f"${cx_s5['slope_diff_usd_per_mile']:.2f}/dặm "
            f"(${cx_s5['lyft_slope_b1']:.2f} vs "
            f"${cx_s5['uberx_slope_b1']:.2f}), tạo điểm cắt tại d* = "
            f"{cx_s5['crossing_distance_miles']:.2f} dặm (95% CI: "
            f"[{cx_s5['crossing_distance_ci95'][0]:.2f}, "
            f"{cx_s5['crossing_distance_ci95'][1]:.2f}] dặm; "
            f"{cx_s5['pct_trips_shorter_than_crossing']:.1f}% lượt < d*). "
            f"Ở phân khúc 6 chỗ: {xl_txt}, còn ở phân khúc đi ghép Lyft "
            f"Shared (${cp_s5['shared_base_b0']:.2f}) luôn rẻ hơn UberPool "
            f"(${cp_s5['uberpool_base_b0']:.2f}) trên toàn bộ miền quan sát "
            f"[0.02, {cp_s5['max_observed_distance_miles']:.2f}] dặm."
        ),
        "conclusion": (
            "Lyft theo đuổi chiến lược thâm nhập chặng ngắn đô thị (< "
            f"{ux_pair['crossing_distance_miles']:.2f} dặm) bằng giá mở cửa "
            "thấp, trong khi UberX tối ưu lợi thế ở các chặng dài hơn điểm "
            f"cắt ({xl_rng}) và không công bố hệ số nhân Surge trực tiếp "
            "trên API."
        ),
        "confidence_level": (
            "Rất cao (Bác bỏ H0 ở mức p < 0.0001 với sai số chuẩn bền vững "
            "White-Huber HC3)."
        ),
    }


def _build_rq3_entry(
    diag_m7: dict[str, Any], pi_summary: dict[str, Any]
) -> dict[str, str]:
    """Tạo bản ghi tổng hợp bằng chứng RQ3 (hiệu năng dự báo & SMART)."""
    test_m = pd.read_csv(TABLES_DIR / "stage5_test_metrics.csv").set_index(
        "model_code"
    )
    cv_m = pd.read_csv(TABLES_DIR / "stage5_cv_results.csv").set_index(
        "model_code"
    )
    m7_t = test_m.loc["M7_HistGradientBoosting"]
    m7_cv = cv_m.loc["M7_HistGradientBoosting"]
    m2_t = test_m.loc["M2_Interaction_OLS"]
    mpiw = pi_summary["mean_interval_width_mpiw_usd"]
    return {
        "rq_id": "RQ3",
        "question": (
            "Mô hình học máy có thể dự báo giá báo API có điều kiện chính "
            "xác đến mức nào và vượt qua chỉ tiêu SMART (R2 >= 0.90, MAPE "
            "<= 10%) không?"
        ),
        "empirical_evidence": (
            "Trên tập quan sát surge thực (n = 232,861): M7 "
            "(HistGradientBoosting) đạt 5-Fold CV RMSE = "
            f"${m7_cv['cv_rmse_mean']:.4f} (±{m7_cv['cv_rmse_std']:.4f}), "
            f"Test R2 = {m7_t['test_r2']:.4f}, RMSE = "
            f"${m7_t['test_rmse']:.4f}, MAE = ${m7_t['test_mae']:.4f}, "
            f"MAPE = {m7_t['test_mape_pct']:.2f}%, với "
            f"{diag_m7['pct_within_2usd']:.2f}% lượt có sai số trong phạm "
            "vi ±$2.00. M2 (OLS Tương tác Kinh tế lượng) đạt Test R2 = "
            f"{m2_t['test_r2']:.4f}, RMSE = ${m2_t['test_rmse']:.4f}, "
            f"MAPE = {m2_t['test_mape_pct']:.2f}%. Khoảng dự báo bất định "
            "Quantile [P10, P90] đạt độ bao phủ thực nghiệm PICP = "
            f"{pi_summary['empirical_coverage_picp_pct']:.2f}% với độ rộng "
            f"trung bình ${mpiw:.2f}."
        ),
        "conclusion": (
            "Cả hai nhánh mô hình có điều kiện (Học máy phi tuyến M5–M7 và "
            "Kinh tế lượng tương tác M2–M4) đều vượt mục tiêu SMART (R2 >= "
            "0.90, MAPE <= 10%) trên Random Split, Temporal Split (EST gốc) "
            "và tập nguyên bản R3."
        ),
        "confidence_level": (
            "Rất cao (Kiểm chứng chéo qua 5-Fold CV, 3 chiến lược chia dữ "
            "liệu và tập nguyên bản R3)."
        ),
    }


def _build_rq4_entry(
    sim_df: pd.DataFrame, sim_extra: dict[str, Any]
) -> dict[str, str]:
    """Tạo bản ghi tổng hợp bằng chứng RQ4 (mô phỏng trần Surge Cap)."""
    c175 = sim_df.loc[sim_df["surge_cap"] == 1.75].iloc[0]
    c150 = sim_df.loc[sim_df["surge_cap"] == 1.50].iloc[0]
    c200 = sim_df.loc[sim_df["surge_cap"] == 2.00].iloc[0]
    d5_200 = c200["supply_constrained_rev_change_drop5_pct"]
    d10_200 = c200["supply_constrained_rev_change_drop10_pct"]
    d5_175 = c175["supply_constrained_rev_change_drop5_pct"]
    d10_175 = c175["supply_constrained_rev_change_drop10_pct"]
    d10_150 = c150["supply_constrained_rev_change_drop10_pct"]
    return {
        "rq_id": "RQ4",
        "question": (
            "Chính sách áp trần hệ số nhân giá động (Surge Cap) tác động như "
            "thế nào đến thặng dư người dùng và tổng giá trị báo giá nền "
            "tảng (Quote Proxy)?"
        ),
        "empirical_evidence": (
            "Mô phỏng phản thực tế trên "
            f"{sim_extra['n_total_lyft_quotes']:,} lượt báo giá Lyft quan "
            "sát surge thực (Quote-as-Trip Proxy, tổng giá trị báo giá $"
            f"{sim_extra['total_lyft_baseline_quote_proxy_usd']:,.2f}) áp "
            "dụng nguyên tắc bên ngắn Q_realized = min(Q_D, Q_S): (1) Trần "
            f"2.00x tác động {int(c200['affected_trips_count']):,} lượt "
            "(surge 2.5x–3.0x): M7 tiết kiệm "
            f"${c200['mean_rider_saving_per_affected_trip_usd']:.2f}/lượt "
            f"(M2 OLS: ${c200['m2_ols_mean_saving_affected_usd']:.2f}/lượt), "
            f"Proxy {c200['static_lyft_rev_change_pct']:+.3f}% (tĩnh), "
            f"{d5_200:+.3f}% (cung nhạy giá ε_S = 0.30), "
            f"{d10_200:+.3f}% (cung nhạy giá ε_S = 0.60) — "
            f"đạt ngưỡng bảo thủ tau <= 0.10%. (2) Trần 1.75x bảo vệ "
            f"{int(c175['affected_trips_count']):,} lượt, tiết kiệm "
            f"${c175['mean_rider_saving_per_affected_trip_usd']:.2f}/lượt "
            f"(+${c175['total_consumer_savings_usd']:,.2f}), Proxy "
            f"{c175['static_lyft_rev_change_pct']:+.3f}% (tĩnh), "
            f"{d5_175:+.3f}% (cung nhạy giá ε_S = 0.30), "
            f"{d10_175:+.3f}% (cung nhạy giá ε_S = 0.60) — "
            "chỉ đạt ngưỡng mở rộng tau = 0.30%–0.50%. (3) Trần 1.50x tiết "
            f"kiệm ${c150['mean_rider_saving_per_affected_trip_usd']:.2f}/"
            f"lượt trên {int(c150['affected_trips_count']):,} lượt, Proxy "
            f"{c150['static_lyft_rev_change_pct']:+.3f}% (tĩnh) đến "
            f"{d10_150:+.3f}% (cung nhạy giá ε_S = 0.60)."
        ),
        "conclusion": (
            "Các kết quả mô phỏng phản thực tế trên tập Lyft quan sát thực "
            "là cơ sở sàng lọc ứng viên thử nghiệm thực địa Switchback A/B; "
            "đánh giá đầy đủ dưới ràng buộc cung tài xế min(Q_D, Q_S) được "
            "trình bày chi tiết tại Mục 5.1."
        ),
        "confidence_level": (
            "Trung bình–Cao (Phản thực tế có đối chứng M7/M2 trên tập quan "
            "sát thực, có điều kiện giả định Quote-as-Trip Proxy)."
        ),
    }


def _assemble_stage6_summary(
    diag_m7: dict[str, Any],
    diag_m2: dict[str, Any],
    pi_summary: dict[str, Any],
    pi_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    pair_comparisons: list[dict[str, Any]],
    perm_df: pd.DataFrame,
    family_summary: list[dict[str, Any]],
    pdp_data: dict[str, Any],
    sim_df: pd.DataFrame,
    sim_extra: dict[str, Any],
    rq_synthesis: list[dict[str, str]],
    saved_figs: list[str],
) -> dict[str, Any]:
    """Đóng gói cấu trúc JSON kết quả toàn diện của Giai đoạn 6."""
    meta_keys = [
        "n_total_lyft_quotes",
        "n_total_lyft_trips",
        "n_total_surge_quotes",
        "n_total_surge_trips",
        "total_lyft_baseline_quote_proxy_usd",
        "total_lyft_baseline_revenue_usd",
        "total_market_baseline_quote_proxy_usd",
        "total_market_baseline_revenue_usd",
        "economic_assumptions_and_limitations",
    ]
    pdp_keys = [
        "distance_grid",
        "pdp_distance_m7",
        "pdp_distance_m2",
        "tier_pdp_distance_curves",
        "surge_grid",
        "pdp_surge_lyft_m7",
        "pdp_surge_lyft_m2",
    ]
    return {
        "stage": (
            "Stage 6 - Residual Diagnostics, Model Interpretation (XAI) & "
            "Surge Cap Policy Simulation"
        ),
        "part1_residual_diagnostics": {
            "M7_HistGradientBoosting": diag_m7,
            "M2_Interaction_OLS": diag_m2,
        },
        "part2_prediction_intervals_p10_p90": {
            "overall": pi_summary,
            "by_service_tier": pi_df.to_dict(orient="records"),
        },
        "part3_tier_tariffs_and_competition": {
            "tier_tariffs_12": tariffs_df.to_dict(orient="records"),
            "uber_vs_lyft_pair_comparisons": pair_comparisons,
        },
        "part4_permutation_importance": {
            "feature_groups": perm_df.to_dict(orient="records"),
            "domain_family_summary": family_summary,
        },
        "part5_pdp_and_ice_summary": {k: pdp_data[k] for k in pdp_keys},
        "part6_surge_cap_policy_simulation": {
            "metadata": {k: sim_extra[k] for k in meta_keys},
            "scenarios": sim_df.to_dict(orient="records"),
        },
        "part7_research_questions_synthesis": rq_synthesis,
        "figures": saved_figs,
    }


def _load_stage6_inputs(test_size: float, random_state: int) -> dict[str, Any]:
    """Nạp dữ liệu đã xử lý, chia tập Train/Test và nạp 2 mô hình GĐ5."""
    df_feat = engineer_features_stage4(load_master_dataset())
    splits, _ = create_data_splits(
        df_feat, test_size=test_size, random_state=random_state
    )
    rand_tr_idx, rand_te_idx = splits["random_80_20"]
    valid_cols = get_feature_columns_for_modeling()
    X_all = df_feat[valid_cols].copy()
    for c in BASE_CAT_FEATURES:
        X_all[c] = X_all[c].astype("category")
    y_all = df_feat[TARGET_COL].astype(np.float64).values

    s5_summary = json.loads(
        (RESULTS_DIR / "stage5_summary.json").read_text(encoding="utf-8")
    )
    return {
        "df_feat": df_feat,
        "valid_cols": valid_cols,
        "X_train": X_all.iloc[rand_tr_idx],
        "y_train": y_all[rand_tr_idx],
        "X_test": X_all.iloc[rand_te_idx],
        "y_test": y_all[rand_te_idx],
        "df_test_meta": df_feat.iloc[rand_te_idx]
        .copy()
        .reset_index(drop=True),
        "model_m7": joblib.load(MODELS_DIR / "best_model.joblib"),
        "model_m2": joblib.load(MODELS_DIR / "m2_interaction_ols.joblib"),
        "s5_summary": s5_summary,
        "best_hgb_params": s5_summary["part2_hyperparameter_tuning"][
            "M7_HistGradientBoosting"
        ]["best_params"],
    }


def run_stage6(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Thực thi toàn diện Giai đoạn 6: Chẩn đoán phần dư, Khoảng dự báo P10-P90,
    Diễn giải XAI & Mô phỏng Chính sách Surge Cap.
    """
    ensure_directories()
    logger.info(
        "=== BẮT ĐẦU GIAI ĐOẠN 6: RESIDUAL DIAGNOSTICS, XAI INTERPRETATION "
        "& SURGE CAP POLICY SIMULATION ==="
    )
    ctx = _load_stage6_inputs(test_size, random_state)
    y_pred_m7 = ctx["model_m7"].predict(ctx["X_test"])
    y_pred_m2 = ctx["model_m2"].predict(ctx["X_test"])

    diag_m7 = diagnose_residuals(
        ctx["y_test"], y_pred_m7, X_exog_bp=ctx["df_test_meta"]
    )
    diag_m2 = diagnose_residuals(
        ctx["y_test"], y_pred_m2, X_exog_bp=ctx["df_test_meta"]
    )
    seg_err_df = run_multidimensional_segment_errors(
        ctx["df_test_meta"], ctx["y_test"], y_pred_m7, y_pred_m2
    )
    seg_err_df.to_csv(
        TABLES_DIR / "stage6_segment_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pi_summary, pi_df, pi_preds = run_quantile_prediction_intervals(
        ctx["X_train"],
        ctx["y_train"],
        ctx["X_test"],
        ctx["y_test"],
        best_hgb_params=ctx["best_hgb_params"],
        random_state=random_state,
    )
    pi_df.to_csv(
        TABLES_DIR / "stage6_prediction_interval.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ols_coef_df = pd.read_csv(TABLES_DIR / "stage5_ols_coefficients.csv")
    tariffs_df, pair_comparisons = build_tier_tariffs_and_pairs_table(
        ols_coef_df
    )
    perm_df, family_summary = run_permutation_importance_analysis(
        ctx["model_m7"],
        ctx["model_m2"],
        ctx["X_test"],
        ctx["y_test"],
        n_repeats=8,
        sample_size=15000,
        random_state=random_state,
    )
    pdp_data = compute_pdp_and_ice_data(
        ctx["model_m7"],
        ctx["model_m2"],
        ctx["X_test"],
        random_state=random_state,
    )
    sim_df, sim_extra = run_surge_cap_policy_simulation(
        ctx["df_feat"], ctx["valid_cols"], ctx["model_m7"], ctx["model_m2"]
    )
    saved_figs = generate_stage6_figures(
        ctx["df_test_meta"],
        ctx["y_test"],
        y_pred_m7,
        y_pred_m2,
        diag_m7,
        seg_err_df,
        tariffs_df,
        pair_comparisons,
        perm_df,
        family_summary,
        pdp_data,
        sim_df,
        sim_extra,
        pi_summary,
        pi_df,
        pi_preds,
    )
    rq_synthesis = [
        _build_rq1_entry(perm_df),
        _build_rq2_entry(ctx["s5_summary"], pair_comparisons),
        _build_rq3_entry(diag_m7, pi_summary),
        _build_rq4_entry(sim_df, sim_extra),
    ]
    summary = _assemble_stage6_summary(
        diag_m7,
        diag_m2,
        pi_summary,
        pi_df,
        tariffs_df,
        pair_comparisons,
        perm_df,
        family_summary,
        pdp_data,
        sim_df,
        sim_extra,
        rq_synthesis,
        saved_figs,
    )
    summary_path = RESULTS_DIR / "stage6_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu trọn bộ kết quả Giai đoạn 6 tại: {summary_path}")
    return summary


if __name__ == "__main__":
    run_stage6()

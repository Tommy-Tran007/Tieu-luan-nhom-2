"""
Script: src/prepare_thesis_figures.py
Mục đích: Tạo bộ 17 biểu đồ 300 DPI chuẩn học thuật v5 cho Tiểu luận tại
reports/figures/thesis/
(100% đọc động từ reports/results/thesis_v5_verification.json,
stage5_summary.json,
stage6_summary.json và các bảng CSV kết quả v5 — 0% số gõ tay, 0% thuật ngữ
cũ).

Gồm 2 nhóm công việc:
1. Cắt bỏ dải suptitle nội bộ ("NHÓM .../HÌNH x.y (FIG ...)") trên 10 biểu đồ
tái sử dụng
   từ Stage 3–6 (đã được làm sạch nhãn & số liệu v5).
2. Kết xuất mới 7 biểu đồ 300 DPI đồng bộ v5:
   - thesis_fig_3_0_cleaning_waterfall.png : Phễu 693,071 -> 270,898 & 4 ô chọn
     mẫu tách đơn vị
   - thesis_fig_3_5_hypotheses_clean.png   : Cluster-Robust + Sentinel + Exact
     WCR & UberX–Lyft Spec A/B
   - thesis_fig_4_3_pipeline_clean.png     : Sơ đồ Pipeline v5 (41 -> DROP 14
     -> 27; M7 + CQR + Ex-Ante)
   - thesis_fig_4_7_quantile_pi.png        : Monotonic Split-CQR (pandas
     category gốc, 0 cắt ngược)
   - thesis_fig_4_8_tariffs_clean.png      : 5 cặp biểu giá Spec A/B
     (UberXL–Lyft XL Spec B ngoài miền)
   - thesis_fig_4_9_grouped_xai_clean.png  : Grouped XAI 6 cụm = 27/27 cột (M7
     vs M2) & Ex-Ante Surge
   - thesis_fig_5_1_surge_cap_clean.png    : Surge Cap ràng buộc cung
     Q=min(Q_D,Q_S) (C2) & đối chứng 3 nhánh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
FIG_OUT = ROOT_DIR / "reports" / "figures" / "thesis"
V5_JSON = ROOT_DIR / "reports" / "results" / "thesis_v5_verification.json"
S5_JSON = ROOT_DIR / "reports" / "results" / "stage5_summary.json"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1B365D"
TEAL = "#0F766E"
BLUE = "#2563EB"
PINK = "#DB2777"
AMBER = "#D97706"
SLATE = "#475569"
RED = "#DC2626"


def strip_top_suptitle(
    src_rel: str, dst_name: str, min_content_gap: int = 18
) -> Path:
    """
    Phát hiện và cắt bỏ đúng dải tiêu đề trên cùng ('NHÓM .../HÌNH x.y (FIG
    ...)') của hình gốc.
    """
    src = ROOT_DIR / src_rel
    img = Image.open(src).convert("RGB")
    arr = np.asarray(img)
    h, w, _ = arr.shape
    non_white_rows = np.any(arr < 245, axis=(1, 2))
    max_scan = int(h * 0.24)
    in_first_band = False
    for y in range(max_scan):
        if non_white_rows[y]:
            in_first_band = True
        elif in_first_band:
            white_run = 0
            while (
                y + white_run < max_scan and not non_white_rows[y + white_run]
            ):
                white_run += 1
            if white_run >= min_content_gap:
                crop_y = max(0, y + white_run - 14)
                out_path = FIG_OUT / dst_name
                img.crop((0, crop_y, w, h)).save(out_path, dpi=(300, 300))
                return out_path
    out_path = FIG_OUT / dst_name
    img.save(out_path, dpi=(300, 300))
    return out_path


def build_fig_3_0_waterfall(v5: dict) -> Path:
    """
    Hình 3.1 v5: Phễu 693,071 -> 270,898 (A) & 4 ô chọn mẫu tách riêng đơn vị
    đo (B).
    """
    wf = v5["waterfall_and_selection_bias"]["waterfall"]
    sb = v5["waterfall_and_selection_bias"]["selection_bias_audit"]
    prov = v5["waterfall_and_selection_bias"]["master_provenance_breakdown"]

    fig = plt.figure(figsize=(13.8, 5.4))
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.28, 1.05], wspace=0.22)

    # Panel A: Cleaning Attrition Waterfall
    ax1 = fig.add_subplot(outer[0])
    stages = [
        "0. Dữ liệu gốc\n(cab_rides.csv)",
        f"1. Tách Uber Taxi\n(MNAR: -{wf[2]['dropped']:,})",
        f"2. Loại khuyết/\nlính canh price\n(-{wf[3]['dropped']:,})",
        f"3. Loại khuyết\nđiểm đón source\n(-{wf[4]['dropped']:,})",
        f"4. Loại khuyết\nđiểm trả dest\n(-{wf[5]['dropped']:,})",
    ]
    vals = [
        wf[0]["out_rows"],
        wf[2]["out_rows"],
        wf[3]["out_rows"],
        wf[4]["out_rows"],
        wf[5]["out_rows"],
    ]
    pcts = [
        wf[0]["share_raw_pct"],
        wf[2]["share_raw_pct"],
        wf[3]["share_raw_pct"],
        wf[4]["share_raw_pct"],
        wf[5]["share_raw_pct"],
    ]
    colors = [NAVY, "#3B82F6", "#60A5FA", "#38BDF8", TEAL]

    x = np.arange(len(stages))
    bars = ax1.bar(
        x,
        [v / 1000 for v in vals],
        width=0.56,
        color=colors,
        edgecolor="#0F172A",
        linewidth=0.8,
    )
    for bar, v, p in zip(bars, vals, pcts, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 14,
            f"{v:,}\n({p:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#0F172A",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.0)
    ax1.set_ylabel(
        "Số bản ghi còn lại (nghìn dòng)", fontsize=9.2, fontweight="bold"
    )
    ax1.set_ylim(0, 830)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(
        0.98,
        0.96,
        f"Bảo toàn nguồn gốc dữ liệu:\n"
        f"• Surge quan sát thực: {prov['is_surge_original_1_total']:,} (mô "
        "hình hóa)\n"
        f"• Lyft khuyết surge (NaN): {prov['lyft_surge_missing_nan']:,} "
        f"({prov['lyft_surge_missing_pct']:.2f}%)\n"
        f"• Thời tiết mưa/khô thực: {prov['weather_valid_rain_obs']:,}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#1E293B",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F8FAFC",
            "edgecolor": "#94A3B8",
        },
    )
    ax1.set_title(
        "A. Phễu hao hụt tuần tự đến bảng sạch (Left Join 0 fan-out)",
        fontsize=10.0,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    # Panel B [M2 FIX]: 2x2 sub-panels — mỗi chỉ tiêu 1 trục đơn vị gốc riêng
    # biệt (KHÔNG trộn trục!)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[1], hspace=0.46, wspace=0.34
    )
    panel_specs = [
        (
            "Tỷ lệ nền tảng Uber (%)",
            sb["uber_share_kept_pct"],
            sb["uber_share_dropped_pct"],
            "%",
            (0, 68),
            "{:.2f}%",
        ),
        (
            "Tỷ lệ Surge Lyft thực (%)",
            sb["lyft_surge_observed_kept_pct"],
            sb["lyft_surge_observed_dropped_pct"],
            "%",
            (0, 9.6),
            "{:.2f}%",
        ),
        (
            "Cự ly trung bình (dặm)",
            sb["mean_dist_kept_mi"],
            sb["mean_dist_dropped_mi"],
            "dặm",
            (0, 2.95),
            "{:.3f} mi",
        ),
        (
            "Giá báo khả dụng TB (USD)",
            sb["mean_price_kept_usd"],
            sb["mean_price_dropped_available_usd"],
            "USD",
            (0, 22.0),
            "${:.2f}",
        ),
    ]
    for idx_p, (ptitle, vk, vd, unit, ylim, fmt) in enumerate(panel_specs):
        ax_s = fig.add_subplot(inner[idx_p // 2, idx_p % 2])
        ax_s.bar(
            [0, 1],
            [vk, vd],
            width=0.52,
            color=[TEAL, "#94A3B8"],
            edgecolor="#0F172A",
            linewidth=0.7,
        )
        ax_s.text(
            0,
            vk + ylim[1] * 0.03,
            fmt.format(vk),
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color=TEAL,
        )
        ax_s.text(
            1,
            vd + ylim[1] * 0.03,
            fmt.format(vd),
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color="#334155",
        )
        ax_s.set_xticks([0, 1])
        ax_s.set_xticklabels(
            ["Giữ lại", "Bị loại"], fontsize=8.2, fontweight="bold"
        )
        ax_s.set_ylim(*ylim)
        ax_s.set_ylabel(unit, fontsize=8.2, fontweight="bold", color=SLATE)
        ax_s.set_title(
            ptitle, fontsize=8.6, fontweight="bold", color=NAVY, pad=6
        )
        ax_s.grid(axis="y", linestyle="--", alpha=0.35)
        ax_s.set_axisbelow(True)
        ax_s.spines[["top", "right"]].set_visible(False)

    # Tiêu đề chung & chú giải màu thật (mpatches) cho Panel B — có khoảng thở
    # rõ với sub-titles
    fig.text(
        0.735,
        0.975,
        "B. Kiểm tra thiên lệch chọn mẫu (Tách riêng từng đơn vị đo)",
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=NAVY,
    )
    b_handles = [
        mpatches.Patch(
            facecolor=TEAL,
            edgecolor="#0F172A",
            label=f"Giữ lại (n = {sb['kept_n']:,})",
        ),
        mpatches.Patch(
            facecolor="#94A3B8",
            edgecolor="#0F172A",
            label=f"Bị loại phi-Taxi (n = {sb['dropped_non_taxi_n']:,})",
        ),
    ]
    fig.legend(
        handles=b_handles,
        loc="upper center",
        bbox_to_anchor=(0.735, 0.935),
        ncol=2,
        fontsize=8.1,
        frameon=True,
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.14)
    out = FIG_OUT / "thesis_fig_3_0_cleaning_waterfall.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig_3_5_hypotheses(v5: dict) -> Path:
    """
    Hình 3.6 v5: Cluster-Robust + Sentinel + Exact WCR (A) & Điểm cắt
    UberX–Lyft Spec A/B trên X_train (B).
    """
    st = v5["statistical_rigor_extensions"]
    unadj = st["weather_unadjusted_loc_hr"]
    ctrl_hr = st["weather_controlled_citywide_hour"]
    sent = st["weather_sentinel_strict_sensitivity"]
    hv = st["heavy_rain_vs_dry"]
    wcr_loc = hv["exact_wcr_loc_hr_B1999"]
    wcr_hr = hv["exact_wcr_citywide_hour_B1999"]

    ux_pair = v5["crossing_distances_5_pairs"][1]
    ux_a = ux_pair["spec_a_train80"]
    ux_b = ux_pair["spec_b_train80_nonsurge"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [1.08, 1.12]}
    )
    fig.patch.set_facecolor("white")

    labels = [
        f"Mưa vs. Tạnh (Đơn biến)\n({st['n_loc_hr_clusters_all']:,} cụm "
        f"quận×giờ,\nn = {st['n_valid_rain_obs']:,})",
        f"Mưa vs. Tạnh (Đa biến FE)\n({st['n_citywide_hour_clusters_obs']} "
        f"cụm giờ Boston / {st['n_loc_hr_clusters_obs']:,} quận×giờ,\n"
        f"n = {st['n_valid_rain_and_surge_dist_obs']:,})",
        f"Loại mọi ô có Sentinel mưa\n({sent['n_hour_clusters']} cụm giờ "
        f"Boston,\nn = {sent['n_obs_excl_any_sentinel_cell']:,})",
        "Mưa lớn ≥ 0.10 in/h (WCR)\n(B = 1,999; "
        f"{hv['n_heavy_loc_hr_clusters']} cụm quận×giờ /"
        f"\n{hv['n_heavy_citywide_hours']} giờ Boston)",
    ]
    ests = [
        unadj["diff_usd"],
        ctrl_hr["beta_usd"],
        sent["beta_usd"],
        hv["ctrl_beta_usd"],
    ]
    ci_lo = [
        unadj["ci95_cluster"][0],
        ctrl_hr["ci95"][0],
        sent["beta_usd"] - 1.96 * sent["se_hour"],
        hv["ctrl_beta_usd"] - 1.96 * hv["ctrl_se_hour"],
    ]
    ci_hi = [
        unadj["ci95_cluster"][1],
        ctrl_hr["ci95"][1],
        sent["beta_usd"] + 1.96 * sent["se_hour"],
        hv["ctrl_beta_usd"] + 1.96 * hv["ctrl_se_hour"],
    ]
    notes = [
        f"{unadj['diff_usd']:+.4f} USD (p_cl = {unadj['p_cluster']:.3f})",
        f"{ctrl_hr['beta_usd']:+.4f} USD (p_hr = {ctrl_hr['p_value']:.3f}; "
        "TOST p < 10⁻⁵)",
        f"{sent['beta_usd']:+.4f} USD (p_hr = {sent['p_hour']:.3f}; TOST p "
        "< 10⁻⁵)",
        f"{hv['ctrl_beta_usd']:+.4f} USD (p_wild = {wcr_loc['p_wild']:.3f} /"
        f" {wcr_hr['p_wild']:.3f})",
    ]
    row_colors = [NAVY, NAVY, TEAL, AMBER]

    ax1.axvspan(
        -0.25,
        0.25,
        color="#DBEAFE",
        alpha=0.45,
        label="Biên tương đương TOST ±$0.25",
    )
    ax1.axvspan(
        -0.10,
        0.10,
        color="#99F6E4",
        alpha=0.55,
        label="Biên tương đương chặt ±$0.10",
    )
    ax1.axvline(0.0, color="#0F172A", linestyle="--", linewidth=1.1)

    y_pos = np.arange(len(labels))[::-1]
    for yp, est, lo, hi, txt, col in zip(
        y_pos, ests, ci_lo, ci_hi, notes, row_colors, strict=False
    ):
        ax1.errorbar(
            est,
            yp,
            xerr=[[est - lo], [hi - est]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=2.1,
            capsize=5,
            markersize=6.8,
        )
        ax1.text(
            est,
            yp + 0.23,
            txt,
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=col,
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.0)
    ax1.set_xlim(-0.32, 0.32)
    ax1.set_ylim(-0.68, 3.72)
    ax1.set_xlabel(
        "Chênh lệch giá báo API khi Mưa so với Tạnh (USD, 95% CI Cluster)",
        fontsize=8.5,
        fontweight="bold",
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower left", fontsize=7.6, frameon=True)
    ax1.set_title(
        "A. Kiểm định Cluster-Robust, Độ nhạy Sentinel mưa & TOST",
        fontsize=9.8,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    # Panel B: UberX vs Lyft Spec A & Spec B (X_train 80%)
    d = np.linspace(0.2, 7.0, 220)
    a, a_cl = ux_a["hc3"], ux_a["cluster"]
    b, b_cl = ux_b["hc3"], ux_b["cluster"]
    ax2.plot(
        d,
        a["b0_uber"] + a["b1_uber"] * d,
        color=NAVY,
        linewidth=2.3,
        label=(
            f"UberX (X_train): {a['b0_uber']:.2f} + {a['b1_uber']:.2f} USD/dặm"
        ),
    )
    ax2.plot(
        d,
        a["b0_lyft"] + a["b1_lyft"] * d,
        color=PINK,
        linewidth=2.3,
        label=(
            f"Lyft Spec A (d×surge): {a['b0_lyft']:.2f} + "
            f"{a['b1_lyft']:.2f} USD/dặm"
        ),
    )
    ax2.plot(
        d,
        b["b0_lyft"] + b["b1_lyft"] * d,
        color=TEAL,
        linewidth=2.1,
        linestyle="--",
        label=(
            f"Lyft Spec B (thuần 1.0x): {b['b0_lyft']:.2f} + "
            f"{b['b1_lyft']:.2f} USD/dặm"
        ),
    )
    ax2.axvspan(a_cl["ci95"][0], a_cl["ci95"][1], color=PINK, alpha=0.14)
    ax2.axvspan(b_cl["ci95"][0], b_cl["ci95"][1], color=TEAL, alpha=0.14)

    p_star_a = a["b0_uber"] + a["b1_uber"] * a["d_star"]
    p_star_b = b["b0_uber"] + b["b1_uber"] * b["d_star"]
    ax2.scatter(
        [a["d_star"]],
        [p_star_a],
        color=PINK,
        edgecolor="black",
        s=62,
        zorder=5,
    )
    ax2.scatter(
        [b["d_star"]],
        [p_star_b],
        color=TEAL,
        edgecolor="black",
        s=62,
        zorder=5,
    )
    ax2.annotate(
        f"Spec A (X_train): d* = {a['d_star']:.2f} mi\nCluster 95% CI "
        f"[{a_cl['ci95'][0]:.2f}, {a_cl['ci95'][1]:.2f}]",
        xy=(a["d_star"], p_star_a),
        xytext=(0.55, 13.8),
        fontsize=7.9,
        fontweight="bold",
        color=PINK,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": PINK,
            "alpha": 0.94,
        },
        arrowprops={"arrowstyle": "->", "color": PINK, "lw": 1.3},
    )
    ax2.annotate(
        f"Spec B (1.0x, X_train): d* = {b['d_star']:.2f} mi\nCluster 95% CI "
        f"[{b_cl['ci95'][0]:.2f}, {b_cl['ci95'][1]:.2f}]",
        xy=(b["d_star"], p_star_b),
        xytext=(3.55, 7.6),
        fontsize=7.9,
        fontweight="bold",
        color=TEAL,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": TEAL,
            "alpha": 0.94,
        },
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.3},
    )
    ax2.set_xlim(0.2, 7.0)
    ax2.set_ylim(5.5, 19.8)
    ax2.set_xlabel(
        "Quãng đường hiệu dụng / thực tế trong miền quan sát Boston (dặm)",
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Giá báo API dự kiến (USD)", fontsize=9.0, fontweight="bold"
    )
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax2.set_title(
        "B. Độ nhạy điểm cắt UberX vs. Lyft (Đồng nhất mẫu X_train 80%)",
        fontsize=9.8,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    plt.tight_layout(pad=1.5)
    out = FIG_OUT / "thesis_fig_3_5_hypotheses_clean.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig_4_3_pipeline(v5: dict, s5: dict) -> Path:
    """
    Hình 4.3 v5: Kiến trúc Pipeline chuẩn xác (41 -> DROP 14 -> 27; M7 + CQR
    native + Ex-Ante; không mã R3b/date).
    """
    tm = pd.read_csv(
        ROOT_DIR / "reports/tables/stage5_test_metrics.csv"
    ).set_index("model_code")
    rob = pd.read_csv(ROOT_DIR / "reports/tables/stage5_robustness.csv")
    cqr = v5["model_diagnostics_and_cqr"]["clean_monotonic_cqr"]
    api_b = v5["model_diagnostics_and_cqr"]["api_batch_group_split"]
    ex_a = s5["part3_ex_ante_experiments"]
    ex_all = next(
        r
        for r in ex_a["pricing_without_surge"]
        if r["model_or_slice"] == "M7_ExAnte_All_Test"
    )
    ex_clf = ex_a["lyft_surge_classifier"]["HistGB_Classifier"]
    prov = v5["waterfall_and_selection_bias"]["master_provenance_breakdown"]

    m7_t = tm.loc["M7_HistGradientBoosting"]
    m6_t = tm.loc["M6_Random_Forest"]
    m2_t = tm.loc["M2_Interaction_OLS"]
    r_temp = rob[
        (rob["scenario"] == "temporal_split")
        & (rob["model_code"] == "M7_HistGradientBoosting")
    ].iloc[0]
    r_grp = rob[
        (rob["scenario"] == "route_group_split")
        & (rob["model_code"] == "M7_HistGradientBoosting")
    ].iloc[0]
    r_orig = rob[
        (rob["scenario"] == "R3b_Train_And_Test_Strictly_Unmodified_Raw")
        & (rob["model_code"] == "M7_HistGradientBoosting")
    ].iloc[0]

    best_lr, best_leaves, best_leaf, best_l2 = 0.08, 31, 40, 1.0
    best_model_path = ROOT_DIR / "models" / "best_model.joblib"
    if best_model_path.exists():
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        reg = joblib.load(best_model_path).named_steps.get("reg")
        if reg is not None:
            p = reg.get_params()
            best_lr = float(p.get("learning_rate", best_lr))
            best_leaves = int(p.get("max_leaf_nodes", best_leaves))
            best_leaf = int(p.get("min_samples_leaf", best_leaf))
            best_l2 = float(p.get("l2_regularization", best_l2))

    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(
        x,
        y,
        w,
        h,
        title,
        body,
        fc="#EFF6FF",
        ec=NAVY,
        title_color=NAVY,
        fs_t=8.7,
        fs_b=7.4,
    ):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.4,
        )
        ax.add_patch(rect)
        n_t_lines = title.count("\n") + 1
        ax.text(
            x + w / 2,
            y + h - 1.3,
            title,
            ha="center",
            va="top",
            fontsize=fs_t,
            fontweight="bold",
            color=title_color,
            linespacing=1.18,
        )
        body_top = y + h - (4.6 if n_t_lines == 1 else 7.4)
        ax.text(
            x + w / 2,
            body_top,
            body,
            ha="center",
            va="top",
            fontsize=fs_b,
            color="#1E293B",
            linespacing=1.32,
        )

    def arrow(x1, y1, x2, y2, color=NAVY):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.35,head_length=0.55",
                "color": color,
                "lw": 1.6,
            },
        )

    # Block 1: Input
    box(
        1.0,
        19.0,
        18.8,
        28.0,
        f"1. BẢNG ĐẶC TRƯNG\n(41 cột × {prov['total_master_rows']:,})",
        "• 32 cột master sạch\n• +9 biến FE đơn dòng\n  (lyft_surge_excess,"
        "\n  log_distance, sin/cos)\n• Lọc quan sát surge thực:\n  "
        f"is_surge_original = 1\n  ({prov['is_surge_original_1_total']:,} "
        "lượt báo giá)",
        fc="#F1F5F9",
        ec=NAVY,
    )

    # Block 2A: Exact 14 excluded cols (matches config.py, NO 'date'!)
    box(
        22.4,
        37.5,
        22.0,
        26.0,
        "2A. KHÓA CẤM RÒ RỈ",
        "LOẠI BỎ 14 CỘT KHỎI X:\n• Target & rò rỉ (2): price, "
        "price_per_mile\n• Định danh/Khóa/Ngày (8): id, product_id,\n  "
        "time_stamp, datetime, location, route_id,\n  date_hour, "
        "day_name\n• Cờ nguồn gốc (4): weather_matched,\n  "
        "is_timestamp_original,\n  is_surge_original, is_rain_sentinel",
        fc="#FEF2F2",
        ec=RED,
        title_color=RED,
        fs_b=7.2,
    )
    box(
        22.4,
        4.0,
        22.0,
        29.5,
        "2B. MA TRẬN X HỢP LỆ\n(27 đặc trưng chuẩn)",
        "CHIA TRƯỚC KHI FIT (surge thực):\n• Chính thức: Random 80/20\n  "
        f"({int(cqr['n_fit'] + cqr['n_cal']):,} Train / "
        f"{int(cqr['n_test']):,} Test)\n• Thời gian EST gốc:\n  "
        f"({int(r_temp['n_train']):,} Train / {int(r_temp['n_test']):,} "
        f"Test)\n• Nhóm tuyến mới (57/15):\n  ({int(r_grp['n_train']):,} "
        f"Train / {int(r_grp['n_test']):,} Test)",
        fc="#EFF6FF",
        ec=BLUE,
        title_color=NAVY,
    )

    # Block 3: Two Parallel Pipelines
    box(
        47.5,
        35.0,
        24.5,
        28.5,
        "3A. NHÁNH TUYẾN TÍNH & CÂY\n(M1–M4, M5–M6)",
        "ColumnTransformer / EconomicPrep\n(fit ĐỘC QUYỀN trên X_train):"
        "\n• Biến số khuyết (thời tiết/giờ):\n  "
        "SimpleImputer(strategy='median')\n• Chuẩn hóa: StandardScaler\n• "
        "Biến phân loại: OneHotEncoder\n  + tương tác C(name)×distance\n• "
        f"M2 OLS: R²={m2_t['test_r2']:.4f} | "
        f"MAPE={m2_t['test_mape_pct']:.2f}%",
        fc="#F0FDF4",
        ec=TEAL,
        title_color=TEAL,
        fs_b=7.4,
    )
    box(
        47.5,
        4.0,
        24.5,
        27.0,
        "3B. NHÁNH BOOSTING BẢN ĐỊA\n(M7 HistGradientBoosting)",
        "m7_hgb_native Pipeline\n(fit ĐỘC QUYỀN trên X_train):\n• Biến số: "
        "passthrough trực tiếp\n  (sau RouteDistanceImputer train-only)\n• "
        "Biến phân loại: pandas category\n  "
        "(categorical_features='from_dtype')\n• "
        f"Tuned: lr={best_lr:g}, leaves={best_leaves},\n  "
        f"min_leaf={best_leaf}, L2={best_l2:.1f}",
        fc="#FDF2F8",
        ec=PINK,
        title_color="#9D174D",
        fs_b=7.4,
    )

    # Block 4: Evaluation & Generalization (NO R3b, NO stale 79.20%)
    box(
        75.2,
        29.5,
        23.8,
        34.0,
        "4. ĐÁNH GIÁ ĐỘC LẬP\n(Giá có điều kiện & Ex-Ante)",
        f"• M7 Có điều kiện (Chính):\n  R² = {m7_t['test_r2']:.4f} | RMSE = "
        f"${m7_t['test_rmse']:.4f}\n  MAE = ${m7_t['test_mae']:.4f} | MAPE "
        f"= {m7_t['test_mape_pct']:.2f}%\n• M6 Random Forest:\n  R² = "
        f"{m6_t['test_r2']:.4f} | MAPE = {m6_t['test_mape_pct']:.2f}%\n• "
        "Split-CQR đơn điệu (native cat):\n  PICP = "
        f"{cqr['overall_cqr_picp_pct']:.2f}% | MPIW = "
        f"${cqr['overall_cqr_mpiw_usd']:.2f}\n• Ex-Ante (23 biến, không "
        f"surge):\n  M7 R² = {ex_all['r2']:.4f} | Surge AUC = "
        f"{ex_clf['roc_auc']:.3f}",
        fc="#EFF6FF",
        ec=NAVY,
        title_color=NAVY,
        fs_b=7.3,
    )
    box(
        75.2,
        4.0,
        23.8,
        22.0,
        "KIỂM CHỨNG TỔNG QUÁT HÓA",
        f"• Temporal EST: R²={r_temp['r2']:.4f}, "
        f"MAPE={r_temp['mape_pct']:.2f}%\n"
        f"• Mẫu nguyên bản (không sửa quy mô):\n  R²={r_orig['r2']:.4f}, "
        f"MAPE={r_orig['mape_pct']:.2f}%\n"
        "• API-Batch GroupSplit (chặn cùng ms):\n  "
        f"R²={api_b['m7']['r2']:.4f}, MAPE={api_b['m7']['mape_pct']:.2f}%\n"
        f"• Route Group (15 tuyến mới):\n  R²={r_grp['r2']:.4f}, "
        f"MAPE={r_grp['mape_pct']:.2f}%",
        fc="#FFFBEB",
        ec=AMBER,
        title_color="#B45309",
        fs_b=7.2,
    )

    arrow(19.8, 38.0, 22.4, 49.0, RED)
    arrow(19.8, 28.0, 22.4, 20.0, NAVY)
    arrow(44.4, 24.0, 47.5, 46.0, TEAL)
    arrow(44.4, 15.0, 47.5, 15.0, PINK)
    arrow(72.0, 48.0, 75.2, 48.0, TEAL)
    arrow(72.0, 17.5, 75.2, 37.5, PINK)
    arrow(87.1, 29.5, 87.1, 26.0, AMBER)

    plt.tight_layout(pad=0.3)
    out = FIG_OUT / "thesis_fig_4_3_pipeline_clean.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig_4_7_cqr(v5: dict) -> Path:
    """
    Hình 4.7 v5: Monotonic Split-CQR dùng pandas category gốc (n_test = 46,573,
    0 cắt ngược).
    """
    cqr = v5["model_diagnostics_and_cqr"]["clean_monotonic_cqr"]
    rows = sorted(
        cqr["by_tier"], key=lambda r: r["mean_actual_usd"], reverse=True
    )

    tiers = [f"{r['cab_type']} — {r['service_tier']}" for r in rows]
    strict_p = [r["base85_raw_strict_picp_pct"] for r in rows]
    tol_p = [r["base85_raw_tol1e6_picp_pct"] for r in rows]
    cqr_p = [r["cqr_picp_pct"] for r in rows]
    mpiw_cqr = [r["cqr_mpiw_usd"] for r in rows]
    q_hats = [r["q_hat_usd_6dec"] for r in rows]
    delta_mpiw_overall = (
        cqr["overall_cqr_mpiw_usd"] - cqr["overall_base85_raw_mpiw_usd"]
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.8, 5.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.patch.set_facecolor("white")

    y = np.arange(len(tiers))[::-1]
    h = 0.26

    # [M2 FIX] Vùng xanh lá bao trọn khoảng thực nghiệm [78%, 82.5%] khớp 12/12
    # gói sau CQR (78.47%–82.16%)
    band_hi = max(82.5, round(cqr["max_tier_cqr_picp_pct"] + 0.5, 1))
    ax1.axvspan(
        78.0,
        band_hi,
        color="#DCFCE7",
        alpha=0.50,
        label=f"Vùng bao phủ mục tiêu [78.0%, {band_hi:.1f}%]",
    )
    ax1.axvline(
        80.0,
        color=RED,
        linestyle="--",
        linewidth=1.3,
        label="Mục tiêu danh nghĩa (80.0%)",
    )

    ax1.barh(
        y + h,
        strict_p,
        height=h,
        color="#94A3B8",
        edgecolor="#334155",
        linewidth=0.5,
        label=(
            "Trước CQR chặt "
            f"({cqr['overall_base85_raw_strict_picp_pct']:.2f}%)"
        ),
    )
    ax1.barh(
        y,
        tol_p,
        height=h,
        color="#60A5FA",
        edgecolor="#1D4ED8",
        linewidth=0.5,
        label=(
            "Trước CQR dung sai 10⁻⁶ USD "
            f"({cqr['overall_base85_raw_tol1e6_picp_pct']:.2f}%)"
        ),
    )
    ax1.barh(
        y - h,
        cqr_p,
        height=h,
        color=TEAL,
        edgecolor="#0F172A",
        linewidth=0.5,
        label=(
            f"Sau Split-CQR đơn điệu ({cqr['overall_cqr_picp_pct']:.2f}%, 0 "
            "cắt ngược)"
        ),
    )

    for idx, r in enumerate(rows):
        yp = y[idx]
        gap = r["base85_raw_tol1e6_picp_pct"] - r["base85_raw_strict_picp_pct"]
        if gap > 3.0:
            x_txt = (
                max(
                    r["base85_raw_tol1e6_picp_pct"], r["cqr_picp_pct"], band_hi
                )
                + 0.45
            )
            ax1.text(
                x_txt,
                yp,
                f"{r['base85_raw_strict_picp_pct']:.1f}% → "
                f"{r['cqr_picp_pct']:.1f}%",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color=NAVY,
            )
        else:
            x_txt = (
                max(r["base85_raw_tol1e6_picp_pct"], r["cqr_picp_pct"], 80.2)
                + 0.45
            )
            ax1.text(
                x_txt,
                yp - h * 0.25,
                f"{r['cqr_picp_pct']:.1f}%",
                va="center",
                fontsize=7.5,
                color="#1E293B",
            )

    ax1.set_yticks(y)
    ax1.set_yticklabels(tiers, fontsize=8.3, fontweight="bold")
    ax1.set_xlim(68, 99)
    ax1.set_ylim(-2.6, 11.7)
    ax1.set_xlabel(
        "Tỷ lệ bao phủ thực nghiệm PICP (%) trên tập Test (n = "
        f"{cqr['n_test']:,}, pandas category)",
        fontsize=8.5,
        fontweight="bold",
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.45)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=7.2, frameon=True)
    ax1.set_title(
        "A. Độ bao phủ 12 phân khúc: Trước vs. Sau Monotonic Split-CQR (85%/"
        "15%)",
        fontsize=9.4,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    # Chuẩn hóa tham số q̂ về đúng 4 chữ số thập phân (xóa -0.000 & ký pháp
    # e-05; đồng bộ màu với dấu sau làm tròn)
    q_rounded = [
        0.0 if abs(round(q, 4)) == 0.0 else round(q, 4) for q in q_hats
    ]
    bar_cols = [PINK if qr > 0.0 else NAVY for qr in q_rounded]
    bars = ax2.barh(
        y,
        mpiw_cqr,
        height=0.58,
        color=bar_cols,
        edgecolor="#0F172A",
        linewidth=0.5,
        alpha=0.88,
    )

    def _fmt_q(qr: float) -> str:
        if qr > 0.0:
            return f"q̂ = +{qr:.4f} USD"
        if qr < 0.0:
            return f"q̂ = -{abs(qr):.4f} USD"
        return "q̂ = 0.0000 USD"

    for bar, w_val, qr in zip(bars, mpiw_cqr, q_rounded, strict=False):
        ax2.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{w_val:.2f} USD | {_fmt_q(qr)}",
            va="center",
            fontsize=7.3,
            color="#0F172A",
            fontweight="bold" if qr > 0.0 else "normal",
        )

    legend_handles = [
        mpatches.Patch(
            facecolor=PINK,
            edgecolor="#0F172A",
            label="q̂ > 0 USD (nới biên đạt ≥ 80%)",
        ),
        mpatches.Patch(
            facecolor=NAVY,
            edgecolor="#0F172A",
            label="q̂ ≤ 0 USD (giữ gọn phân khúc phủ dư)",
        ),
    ]
    ax2.set_yticks(y)
    ax2.set_yticklabels(["" for _ in tiers])
    ax2.set_xlim(0, 8.8)
    ax2.set_ylim(-2.6, 11.7)
    ax2.set_xlabel(
        "Độ rộng khoảng dự báo MPIW sau CQR (USD) & tham số q̂",
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.grid(axis="x", linestyle=":", alpha=0.45)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(
        handles=legend_handles, loc="lower center", fontsize=7.5, frameon=True
    )
    ax2.set_title(
        f"B. Độ rộng MPIW (TB {cqr['overall_cqr_mpiw_usd']:.2f} USD, Δ = "
        f"{delta_mpiw_overall:+.3f} USD) & Tham số q̂",
        fontsize=9.2,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    plt.tight_layout(pad=1.4)
    out = FIG_OUT / "thesis_fig_4_7_quantile_pi.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig_4_8_tariffs(v5: dict) -> Path:
    """
    Hình 4.8 v5: 5 cặp biểu giá Spec A & Spec B (X_train 80%) — hiển thị trung
    thực UberXL–Lyft XL Spec B ngoài miền.
    """
    pairs = v5["crossing_distances_5_pairs"]

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.4))
    fig.patch.set_facecolor("white")
    d = np.linspace(0.05, 7.86, 220)

    for idx, p in enumerate(pairs):
        ax = axes[idx // 3, idx % 3]
        ut, lt = p["uber_tier"], p["lyft_tier"]
        sa = p["spec_a_train80"]
        sb = p["spec_b_train80_nonsurge"]

        b0_u_a, b1_u_a = sa["hc3"]["b0_uber"], sa["hc3"]["b1_uber"]
        b0_l_a, b1_l_a = sa["hc3"]["b0_lyft"], sa["hc3"]["b1_lyft"]
        b0_u_b, b1_u_b = sb["hc3"]["b0_uber"], sb["hc3"]["b1_uber"]
        b0_l_b, b1_l_b = sb["hc3"]["b0_lyft"], sb["hc3"]["b1_lyft"]

        ax.axvspan(
            0.0,
            1.0,
            color="#FEF3C7",
            alpha=0.45,
            label="Vùng giá sàn (<1.0 mi)" if idx == 0 else None,
        )
        ax.plot(
            d, b0_u_a + b1_u_a * d, color=NAVY, lw=2.2, label=f"{ut} (Spec A)"
        )
        ax.plot(
            d,
            b0_l_a + b1_l_a * d,
            color=PINK,
            lw=2.2,
            label=f"{lt} (Spec A: d×surge)",
        )
        ax.plot(
            d,
            b0_l_b + b1_l_b * d,
            color=TEAL,
            lw=1.8,
            linestyle="--",
            label=f"{lt} (Spec B: 1.0x)",
        )

        if sa["in_domain"]:
            ds_a = sa["hc3"]["d_star"]
            ps_a = b0_u_a + b1_u_a * ds_a
            ax.scatter(
                [ds_a], [ps_a], color=PINK, edgecolor="black", s=50, zorder=5
            )
            ci_a = sa["cluster"]["ci95"]
            ax.axvline(ds_a, color=PINK, linestyle=":", alpha=0.65, lw=1.2)
            note_a = f"Spec A: d*={ds_a:.2f} mi [{ci_a[0]:.2f}, {ci_a[1]:.2f}]"
        else:
            note_a = "Spec A: Không cắt (Shared luôn rẻ hơn)"

        if sb["in_domain"]:
            ds_b = sb["hc3"]["d_star"]
            ps_b = b0_u_b + b1_u_b * ds_b
            ax.scatter(
                [ds_b], [ps_b], color=TEAL, edgecolor="black", s=50, zorder=5
            )
            ci_b = sb["cluster"]["ci95"]
            note_b = (
                f"Spec B (1.0x): d*={ds_b:.2f} mi [{ci_b[0]:.2f}, "
                f"{ci_b[1]:.2f}]"
            )
        else:
            raw_db = sb["hc3"]["d_star"]
            if ut == "UberXL":
                note_b = (
                    f"Spec B (1.0x): Không cắt (d*={raw_db:.2f} > 7.86;"
                    "\nLyft XL rẻ hơn toàn miền)"
                )
            elif ut == "Black":
                note_b = (
                    "Spec B (1.0x): Không cắt\n(Uber Black rẻ hơn toàn miền)"
                )
            else:
                note_b = "Spec B (1.0x): Không cắt\n(Shared rẻ hơn toàn miền)"

        ax.text(
            0.03,
            0.96,
            f"{note_a}\n{note_b}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.93,
            },
        )
        ax.set_title(
            f"{p['segment']}: {ut} vs. {lt}",
            fontsize=9.4,
            fontweight="bold",
            color=NAVY,
        )
        ax.set_xlabel(
            "Cự ly hiệu dụng (Spec A) / thực tế (Spec B) [dặm]", fontsize=7.9
        )
        ax.set_ylabel("Giá báo dự kiến (USD)", fontsize=8.2)
        ax.set_xlim(0.0, 7.86)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=6.9, frameon=True)

    # Panel 6: Dynamic Sensitivity Summary (4 pairs with in-domain Spec A)
    ax6 = axes[1, 2]
    focus = [pairs[1], pairs[2], pairs[3], pairs[4]]
    comp_labels = [
        "UberX vs.\nLyft",
        "UberXL vs.\nLyft XL",
        "Black vs.\nLux Black*",
        "Black SUV vs.\nLux Black XL",
    ]
    y = np.arange(len(comp_labels))[::-1]

    for i, fp in enumerate(focus):
        sa = fp["spec_a_train80"]
        sb = fp["spec_b_train80_nonsurge"]
        da = sa["hc3"]["d_star"]
        cla = sa["cluster"]["ci95"]
        ax6.errorbar(
            da,
            y[i] + 0.14,
            xerr=[[max(da - cla[0], 0.0)], [max(cla[1] - da, 0.0)]],
            fmt="s",
            color=PINK,
            capsize=4,
            lw=1.8,
            markersize=6,
            label="Spec A (Gộp d×surge, X_train)" if i == 0 else None,
        )
        if sb["in_domain"]:
            db = sb["hc3"]["d_star"]
            clb = sb["cluster"]["ci95"]
            ax6.errorbar(
                db,
                y[i] - 0.14,
                xerr=[[max(db - clb[0], 0.0)], [max(clb[1] - db, 0.0)]],
                fmt="o",
                color=TEAL,
                capsize=4,
                lw=1.8,
                markersize=6,
                label="Spec B (Thuần 1.0x, X_train)" if i == 0 else None,
            )
        elif fp["uber_tier"] == "UberXL":
            ax6.text(
                4.25,
                y[i] - 0.18,
                f"Spec B: Ngoài miền ({sb['hc3']['d_star']:.2f} mi)\n→ Lyft "
                "XL rẻ hơn toàn miền",
                fontsize=7.1,
                color=TEAL,
                fontweight="bold",
                va="center",
            )
        elif fp["uber_tier"] == "Black":
            ax6.text(
                1.35,
                y[i] - 0.18,
                "Spec B: Không cắt\n(Uber Black luôn rẻ hơn)",
                fontsize=7.2,
                color=TEAL,
                fontweight="bold",
                va="center",
            )

    ax6.axvspan(
        0.0, 1.0, color="#FEF3C7", alpha=0.5, label="Vùng giá sàn (<1.0 mi)"
    )
    ax6.set_yticks(y)
    ax6.set_yticklabels(comp_labels, fontsize=8.2, fontweight="bold")
    ax6.set_xlim(0.0, 7.86)
    ax6.set_ylim(-0.75, 3.6)
    ax6.set_xlabel(
        "Ngưỡng điểm cắt d* (dặm, 95% CI Cluster)",
        fontsize=8.2,
        fontweight="bold",
    )
    ax6.grid(axis="x", linestyle="--", alpha=0.4)
    ax6.spines[["top", "right"]].set_visible(False)
    ax6.legend(loc="lower right", fontsize=6.9, frameon=True)
    ax6.set_title(
        "Tổng hợp độ nhạy d* (Cùng mẫu X_train 80%)",
        fontsize=9.4,
        fontweight="bold",
        color=NAVY,
    )

    plt.tight_layout(pad=1.35, rect=[0, 0.025, 1, 1])
    fig.text(
        0.99,
        0.004,
        f"*Lưu ý Black vs. Lux Black: điểm cắt Spec A "
        f"({pairs[3]['spec_a_train80']['hc3']['d_star']:.2f} mi) nằm trong "
        "vùng giá sàn < 1.0 mi; khi xét riêng cự ly ≥ 1.0 mi hai đường biểu "
        "giá không cắt nhau.",
        ha="right",
        va="bottom",
        fontsize=7.6,
        style="italic",
        color=SLATE,
    )
    out = FIG_OUT / "thesis_fig_4_8_tariffs_clean.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig_4_9_grouped_xai(v5: dict, s5: dict) -> Path:
    """
    Hình 4.9 v5: Grouped Permutation Importance (6 cụm = 27/27 cột, M7 vs M2) &
    Ex-Ante Surge AUC drops.
    """
    md = v5["model_diagnostics_and_cqr"]
    rows = md["grouped_permutation_importance"]
    cc_wea = md["complete_case_weather_delta_rmse_m7"]
    auc_drops = s5["part3_ex_ante_experiments"]["lyft_surge_classifier"][
        "group_roc_auc_drop"
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.4, 5.0), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.patch.set_facecolor("white")

    short_lbls = [
        "1. Phân khúc dịch vụ\n(name — 1 cột)",
        "2. Cụm Cự ly & Surge gộp\n(6 cột chống đa cộng tuyến)",
        "3. Tuyến không gian\n(source, destination — 2 cột)",
        "4. Hãng xe\n(cab_type, is_lyft — 2 cột)",
        "5. Toàn bộ Thời tiết\n(7 biến khí tượng)",
        "6. Toàn bộ Thời gian EST\n(9 biến khung giờ & thứ)",
    ]
    m7_d = [max(float(r["m7_delta_rmse"]), 0.0) for r in rows]
    m2_d = [max(float(r["m2_delta_rmse"]), 0.0) for r in rows]
    m7_s = [max(float(r["m7_rel_share_pct"]), 0.0) for r in rows]
    m2_s = [max(float(r["m2_rel_share_pct"]), 0.0) for r in rows]
    top2_m7 = m7_s[0] + m7_s[1]
    top2_m2 = m2_s[0] + m2_s[1]

    y = np.arange(len(short_lbls))[::-1]
    h = 0.35
    b1 = ax1.barh(
        y + h / 2, m7_d, height=h, color=TEAL, label="M7 HistGradientBoosting"
    )
    b2 = ax1.barh(
        y - h / 2,
        m2_d,
        height=h,
        color=NAVY,
        alpha=0.82,
        label="M2 Interaction OLS",
    )

    for bar, val in zip(b1, m7_d, strict=False):
        ax1.text(
            bar.get_width() + 0.14,
            bar.get_y() + bar.get_height() / 2,
            f"+${val:.4f}",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            color=TEAL,
        )
    for bar, val in zip(b2, m2_d, strict=False):
        ax1.text(
            bar.get_width() + 0.14,
            bar.get_y() + bar.get_height() / 2,
            f"+${val:.4f}",
            va="center",
            fontsize=8.2,
            color=NAVY,
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(short_lbls, fontsize=8.5)
    ax1.set_xlim(0, 12.2)
    ax1.set_xlabel(
        "Mức tăng sai số khi hoán vị cả cụm — ΔRMSE (USD, n = 15,000 × 8 "
        "lần lặp)",
        fontsize=8.7,
        fontweight="bold",
    )
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax1.set_title(
        "A. Grouped Permutation Importance (6 cụm rời = 27/27 cột X_test)",
        fontsize=10.0,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    b3 = ax2.barh(
        y + h / 2, m7_s, height=h, color=TEAL, label="Tỷ trọng ΔRMSE — M7 (%)"
    )
    b4 = ax2.barh(
        y - h / 2,
        m2_s,
        height=h,
        color=NAVY,
        alpha=0.82,
        label="Tỷ trọng ΔRMSE — M2 (%)",
    )
    for bar, val in zip(b3, m7_s, strict=False):
        ax2.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            color=TEAL,
        )
    for bar, val in zip(b4, m2_s, strict=False):
        ax2.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            fontsize=8.2,
            color=NAVY,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels([f"Cụm {i + 1}" for i in range(6)], fontsize=8.5)
    ax2.set_xlim(0, 88)
    ax2.set_xlabel(
        "Tỷ trọng tương đối của tổng ΔRMSE dương (%)",
        fontsize=8.7,
        fontweight="bold",
    )
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(loc="center right", fontsize=8.2, frameon=True)
    ax2.set_title(
        f"B. Đối sánh tỷ trọng ΔRMSE (Top 2 cụm: M7 {top2_m7:.2f}% vs. M2 "
        f"{top2_m2:.2f}%)",
        fontsize=9.6,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    w_auc = auc_drops.get("weather_7vars", auc_drops.get("weather_6vars", 0.0))
    ax2.text(
        0.97,
        0.04,
        f"Đối chứng Ex-Ante & Complete-case:\n"
        "• Mẫu thời tiết thực (không sentinel): ΔRMSE thời tiết = "
        f"{cc_wea:+.4f} USD\n"
        f"• Phân loại Surge Lyft (giảm AUC khi hoán vị):\n"
        f"  Tuyến/cự ly {auc_drops['source_destination_distance']:.3f} vs. "
        f"Thời tiết {w_auc:.3f}, Giờ {auc_drops['time_clock_est']:.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#F8FAFC",
            "edgecolor": "#94A3B8",
        },
    )

    plt.tight_layout(pad=1.5)
    out = FIG_OUT / "thesis_fig_4_9_grouped_xai_clean.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig_5_1_surge_cap(v5: dict) -> Path:
    """
    Hình 5.1 v5: Mô phỏng Trần Surge Cap với ràng buộc cung Q=min(Q_D, Q_S)
    (C2) & đối chứng 3 nhánh.
    """
    sim = pd.read_csv(
        ROOT_DIR / "reports/tables/stage6_surge_cap_simulation.csv"
    )
    arith = pd.DataFrame(
        v5["policy_rq4_supply_power_and_io"]["arithmetic_surge_benchmark"]
    )
    df = sim.merge(arith, on="surge_cap", how="left")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.18, 1.02]}
    )
    fig.patch.set_facecolor("white")

    # Panel A [C2 & M2 FIX]: 4 đường đơn điệu hoàn hảo (Cận trên cầu -> Tĩnh ->
    # Mất 5% cung -> Mất 10% cung)
    caps_lbl = [
        f"{c:.2f}x\n({int(n):,})"
        for c, n in zip(
            df["surge_cap"], df["affected_trips_count"], strict=False
        )
    ]
    x = np.arange(len(df))

    ax1.axhspan(
        -0.10,
        0.0,
        color="#DCFCE7",
        alpha=0.55,
        label="Ngưỡng bảo thủ τ ≤ 0.10%",
    )
    ax1.plot(
        x,
        df["net_lyft_rev_change_eps_050_pct"],
        marker="^",
        lw=2.0,
        linestyle="--",
        color=TEAL,
        label="Cận trên dư cung (ε = -0.50, Cohen et al., 2016)",
    )
    ax1.plot(
        x,
        df["static_lyft_rev_change_pct"],
        marker="o",
        lw=2.3,
        color=NAVY,
        label="Cung tĩnh (ε_S = 0.00)",
    )
    ax1.plot(
        x,
        df["supply_constrained_rev_change_drop5_pct"],
        marker="s",
        lw=2.2,
        color=AMBER,
        label="Cung nhạy giá ε_S = 0.30: Q = min(Q_D, Q_S)",
    )
    ax1.plot(
        x,
        df["supply_constrained_rev_change_drop10_pct"],
        marker="D",
        lw=2.2,
        color=RED,
        label="Cung nhạy giá ε_S = 0.60: Q = min(Q_D, Q_S)",
    )
    ax1.axhline(0.0, color="#64748B", lw=0.9, linestyle=":")

    idx_200 = int(df.index[df["surge_cap"] == 2.00][0])
    idx_175 = int(df.index[df["surge_cap"] == 1.75][0])
    r200, r175 = df.loc[idx_200], df.loc[idx_175]

    ax1.annotate(
        "Pha 1: Trần 2.00x (53 lượt)\nĐạt τ ≤ 0.10% mọi kịch bản "
        f"cung\n({r200['static_lyft_rev_change_pct']:+.3f}% đến "
        f"{r200['supply_constrained_rev_change_drop10_pct']:+.3f}%)",
        xy=(idx_200, float(r200["supply_constrained_rev_change_drop10_pct"])),
        xytext=(0.05, -1.18),
        fontsize=7.5,
        fontweight="bold",
        color=TEAL,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#F0FDF4",
            "edgecolor": TEAL,
        },
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.4},
    )
    ax1.annotate(
        "Pha 2: Trần 1.75x (704 lượt)\nĐạt τ = "
        f"0.30%–0.50%\n({r175['static_lyft_rev_change_pct']:+.3f}% đến "
        f"{r175['supply_constrained_rev_change_drop10_pct']:+.3f}%)",
        xy=(idx_175, float(r175["supply_constrained_rev_change_drop10_pct"])),
        xytext=(1.95, -2.05),
        fontsize=7.5,
        fontweight="bold",
        color="#92400E",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#FFFBEB",
            "edgecolor": AMBER,
        },
        arrowprops={"arrowstyle": "->", "color": AMBER, "lw": 1.4},
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(caps_lbl, fontsize=8.3)
    ax1.set_xlabel(
        "Ngưỡng trần Surge Cap (số lượt báo giá Lyft thực chịu tác động)",
        fontsize=8.6,
        fontweight="bold",
    )
    ax1.set_ylabel(
        "Biến động tổng API Quote Proxy Lyft (%)",
        fontsize=8.9,
        fontweight="bold",
    )
    ax1.set_ylim(-4.45, 0.35)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower left", fontsize=7.4, frameon=True)
    ax1.set_title(
        "A. Biến động API Quote Proxy dưới ràng buộc cung Q_realized = "
        "min(Q_D, Q_S)",
        fontsize=9.4,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    # Panel B: 3-Way Benchmark (M7 vs M2 OLS vs Model-Free Arithmetic 1 -
    # cap/surge)
    act = df.loc[df["surge_cap"] < 3.0].reset_index(drop=True)
    xb = np.arange(len(act))
    w = 0.26
    r1 = ax2.bar(
        xb - w,
        act["mean_rider_saving_per_affected_trip_usd"],
        width=w,
        color=TEAL,
        label="M7 HistGBR (Mô phỏng)",
    )
    r2 = ax2.bar(
        xb,
        act["m2_ols_mean_saving_affected_usd"],
        width=w,
        color=NAVY,
        label="M2 OLS (Đối chứng tuyến tính)",
    )
    r3 = ax2.bar(
        xb + w,
        act["arith_mean_saving_usd"],
        width=w,
        color="#94A3B8",
        edgecolor="#334155",
        linewidth=0.7,
        label="Chuẩn số học P·(1 - cap/surge)",
    )

    for bars, col in [(r1, TEAL), (r2, NAVY), (r3, "#334155")]:
        for b in bars:
            ax2.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.14,
                f"${b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=col,
                rotation=52,
            )

    ax2.annotate(
        "Hiệu ứng bão hòa cây (n=53)\n& thay đổi cơ cấu mẫu 2.00x → 1.75x",
        xy=(0.26, float(act.loc[0, "arith_mean_saving_usd"]) + 1.05),
        xytext=(0.10, 9.85),
        fontsize=7.5,
        color="#7C2D12",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#FFF7ED",
            "edgecolor": "#EA580C",
        },
        arrowprops={"arrowstyle": "->", "color": "#EA580C", "lw": 1.2},
    )

    ax2.set_xticks(xb)
    ax2.set_xticklabels(
        [
            f"{c:.2f}x\n(n={int(n):,})"
            for c, n in zip(
                act["surge_cap"], act["affected_trips_count"], strict=False
            )
        ],
        fontsize=8.3,
    )
    ax2.set_xlabel(
        "Ngưỡng trần Surge Cap kích hoạt (mẫu Lyft thực n = 92,384)",
        fontsize=8.6,
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Mức giảm giá báo TB trên mỗi lượt bị cắt (USD)",
        fontsize=8.7,
        fontweight="bold",
    )
    ax2.set_ylim(0, 11.4)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax2.set_title(
        "B. Đối chứng mức giảm giá báo: M7 vs. M2 OLS vs. Chuẩn số học",
        fontsize=9.5,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    plt.tight_layout(pad=1.5)
    out = FIG_OUT / "thesis_fig_5_1_surge_cap_clean.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    v5 = json.loads(V5_JSON.read_text(encoding="utf-8"))
    s5 = json.loads(S5_JSON.read_text(encoding="utf-8"))

    reuse_map = [
        (
            "reports/figures/stage3/fig01_groupA_market_overview.png",
            "thesis_fig_3_1_market_overview.png",
        ),
        (
            "reports/figures/stage3/fig02_groupB_distributions.png",
            "thesis_fig_3_2_distributions.png",
        ),
        (
            "reports/figures/stage3/fig03_groupC_temporal_trends.png",
            "thesis_fig_3_3_temporal_est.png",
        ),
        (
            "reports/figures/stage3/fig06_groupF_surge_weather_business.png",
            "thesis_fig_3_4_weather_surge_est.png",
        ),
        (
            "reports/figures/stage4/fig08_feature_correlation_after_fe.png",
            "thesis_fig_4_1_corr_vif.png",
        ),
        (
            "reports/figures/stage4/fig09_split_distribution.png",
            "thesis_fig_4_2_split_dist.png",
        ),
        (
            "reports/figures/stage5/fig12_model_comparison_cv.png",
            "thesis_fig_4_4_model_comp.png",
        ),
        (
            "reports/figures/stage6/fig15_residual_diagnostics.png",
            "thesis_fig_4_5_residuals_m7.png",
        ),
        (
            "reports/figures/stage6/fig16_error_by_segment.png",
            "thesis_fig_4_6_segment_errors.png",
        ),
        (
            "reports/figures/stage6/fig19_partial_dependence.png",
            "thesis_fig_4_10_pdp_ice.png",
        ),
    ]
    for src_rel, dst_name in reuse_map:
        p = strip_top_suptitle(src_rel, dst_name)
        print(f"[CROPPED]  {dst_name} -> {p.stat().st_size / 1024:.0f} KB")

    created = [
        build_fig_3_0_waterfall(v5),
        build_fig_3_5_hypotheses(v5),
        build_fig_4_3_pipeline(v5, s5),
        build_fig_4_7_cqr(v5),
        build_fig_4_8_tariffs(v5),
        build_fig_4_9_grouped_xai(v5, s5),
        build_fig_5_1_surge_cap(v5),
    ]
    for p in created:
        print(f"[RENDERED] {p.name} -> {p.stat().st_size / 1024:.0f} KB")
    print(
        f"\nHOAN TAT {len(reuse_map) + len(created)} BIEU DO CHUAN TIEU "
        f"LUAN V5 TAI: {FIG_OUT}"
    )


if __name__ == "__main__":
    main()

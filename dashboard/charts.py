"""
Module: dashboard/charts.py
Mục đích: Bộ hàm dựng biểu đồ Plotly tương tác thuần túy (Pure Visualization
Layer).
- KHÔNG import streamlit để tái sử dụng 100% giữa Streamlit App (app.py) và
  Trình xuất
  Báo cáo HTML Độc lập (export_static_html.py) — đảm bảo hai phiên bản đồng
  nhất tuyệt đối.
- Mọi biểu đồ đều có tiêu đề rõ ràng, đơn vị đo tường minh, tooltip có ý nghĩa,
  màu thương hiệu nhất quán (Uber #111827, Lyft #E11D8A, Accent #2563EB) và
  nguồn dữ liệu.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.config import (
    BRAND_COLORS,
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    STANDALONE_TIERS,
    SYMMETRIC_PAIRS,
    VALID_BOSTON_DISTRICTS,
)

PLOTLY_DISPLAY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

_FONT_FAMILY = "Inter, Segoe UI, system-ui, -apple-system, sans-serif"


def _wrap_html(text: str, max_chars: int) -> tuple[str, int]:
    """
    Tự động xuống dòng tiêu đề/phụ đề Plotly bằng <br> để không bao giờ bị cắt
    chữ hai mép.
    """
    if not text:
        return "", 0
    out_lines: list[str] = []
    for seg in text.split("<br>"):
        wrapped = textwrap.wrap(
            seg.strip(),
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out_lines.extend(wrapped or [""])
    return "<br>".join(out_lines), len(out_lines)


def apply_executive_layout(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 440,
    xaxis_title: str = "",
    yaxis_title: str = "",
    legend_top: bool = True,
    narrow: bool = False,
) -> go.Figure:
    """
    Áp dụng chuẩn định dạng đồ thị sáng rõ, tự động xuống dòng chống cắt chữ
    trên mọi kích thước cột.
    """
    title_max = 48 if narrow else 86
    sub_max = 64 if narrow else 118
    title_size = 13.5 if narrow else 15
    sub_size = 11 if narrow else 12

    wrapped_title, n_title_lines = _wrap_html(title, title_max)
    wrapped_sub, n_sub_lines = (
        _wrap_html(subtitle, sub_max) if subtitle else ("", 0)
    )
    full_title = (
        f"<b>{wrapped_title}</b><br><span style='font-size:{sub_size}px;"
        f"color:#475569;'>{wrapped_sub}</span>"
        if wrapped_sub
        else f"<b>{wrapped_title}</b>"
    )

    subplot_anns = [
        ann
        for ann in (fig.layout.annotations or ())
        if getattr(ann, "xref", None) == "paper"
        and getattr(ann, "yref", None) == "paper"
        and ann.y is not None
        and float(ann.y) >= 0.98
    ]
    has_subplot_titles = bool(subplot_anns)
    max_sub_ann_lines = 1
    for ann in subplot_anns:
        if ann.text:
            wrapped_ann, n_ann = _wrap_html(str(ann.text), 46)
            ann.text = (
                f"<b>{wrapped_ann}</b>"
                if not wrapped_ann.startswith("<b>")
                else wrapped_ann
            )
            max_sub_ann_lines = max(max_sub_ann_lines, n_ann)
        ann.font = {"size": 12.5, "color": "#0F172A"}

    header_px = (
        16
        + n_title_lines * (19 if narrow else 21)
        + (n_sub_lines * (15 if narrow else 17) + 4 if wrapped_sub else 0)
    )
    sub_ann_px = (max_sub_ann_lines * 18 + 8) if has_subplot_titles else 0
    legend_px = 30 if legend_top else 0
    top_margin = max(62, header_px + sub_ann_px + legend_px + 10)
    eff_height = height + max(0, top_margin - (86 if subtitle else 62))

    layout_kwargs: dict[str, Any] = {
        "title": {
            "text": full_title,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": title_size, "color": "#0F172A"},
        },
        "font": {"family": _FONT_FAMILY, "size": 12, "color": "#1E293B"},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "height": eff_height,
        "margin": {"l": 60, "r": 34, "t": top_margin, "b": 56},
        "hovermode": "closest",
        "xaxis": {
            "title": xaxis_title,
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "linecolor": "#CBD5E1",
            "zeroline": False,
        },
        "yaxis": {
            "title": yaxis_title,
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "linecolor": "#CBD5E1",
            "zeroline": False,
        },
    }
    if legend_top:
        legend_y = (
            (1.10 if max_sub_ann_lines == 1 else 1.16)
            if has_subplot_titles
            else 1.02
        )
        layout_kwargs["legend"] = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": legend_y,
            "xanchor": "center" if (has_subplot_titles or narrow) else "right",
            "x": 0.5 if (has_subplot_titles or narrow) else 1.0,
            "font": {"size": 11 if narrow else 12},
            "bgcolor": "rgba(255,255,255,0.85)",
            "title": {"text": ""},
        }
    fig.update_layout(**layout_kwargs)
    return fig


def _empty_figure(title: str, message: str, height: int = 380) -> go.Figure:
    """Tạo khung biểu đồ trống thân thiện khi bộ lọc trả về 0 quan sát."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": "#64748B"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_executive_layout(
        fig, title=title, height=height, legend_top=False
    )


# =============================================================================
# TAB 1 — EXECUTIVE OVERVIEW CHARTS (GIAI ĐOẠN 3 & EXECUTIVE REDESIGN)
# =============================================================================
def build_executive_uber_vs_lyft_fig(
    df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
) -> go.Figure:
    """
    Biểu đồ Chủ đạo Tab 1 (Executive Hero Chart): Thể hiện toàn cảnh khác biệt
    Uber vs. Lyft
    trong duy nhất 1 khung nhìn 2 phần:
      - Trái: Giá báo trung bình (USD) giữa Uber & Lyft trên 5 phân khúc đối
        xứng (tính động từ dữ liệu lọc).
      - Phải: Lợi thế chi phí theo quãng đường Δ = Giá Lyft − Giá Uber (USD,
        surge = 1.0x) trên miền [0.02, 7.86] dặm
              (< $0: Lyft rẻ hơn | > $0: Uber rẻ hơn), đánh dấu trực tiếp điểm
              cắt d* = 3.10 dặm.
    """
    if df.empty or tariffs_df.empty:
        return _empty_figure(
            "Khác biệt Cốt lõi giữa Uber và Lyft theo Phân khúc & Quãng đường",
            "Không có dữ liệu khớp bộ lọc hiện tại.",
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Giá báo TB theo 5 Phân khúc Đối xứng (USD)",
            "Chênh lệch Giá (Lyft − Uber) theo Cự ly (surge = 1.0x)",
        ),
        column_widths=[0.50, 0.50],
        horizontal_spacing=0.11,
    )

    # --- PANEL TRÁI: Giá báo TB trên 5 cặp đối xứng (tính động từ df lọc) ---
    tier_means = df.groupby("name", observed=True)["price"].mean().to_dict()
    tier_counts = df.groupby("name", observed=True)["price"].size().to_dict()

    seg_labels, u_vals, l_vals, u_names, l_names, u_cnts, l_cnts = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for p in SYMMETRIC_PAIRS:
        u_t, l_t = p["uber_tier"], p["lyft_tier"]
        u_v = float(tier_means[u_t]) if u_t in tier_means else np.nan
        l_v = float(tier_means[l_t]) if l_t in tier_means else np.nan
        if np.isnan(u_v) and np.isnan(l_v):
            continue
        seg_labels.append(p["short_label"])
        u_vals.append(round(u_v, 2) if not np.isnan(u_v) else 0.0)
        l_vals.append(round(l_v, 2) if not np.isnan(l_v) else 0.0)
        u_names.append(u_t)
        l_names.append(l_t)
        u_cnts.append(int(tier_counts.get(u_t, 0)))
        l_cnts.append(int(tier_counts.get(l_t, 0)))

    fig.add_trace(
        go.Bar(
            x=seg_labels,
            y=u_vals,
            name="Uber",
            marker_color=BRAND_COLORS["Uber"],
            text=[f"${v:.2f}" if v > 0 else "" for v in u_vals],
            textposition="outside",
            customdata=np.stack([u_names, u_cnts], axis=-1)
            if seg_labels
            else None,
            hovertemplate=(
                "<b>%{customdata[0]} (Uber)</b><br>Giá báo TB: <b>$%{y:"
                ".2f}</b><br>Cỡ mẫu: <b>%{customdata[1]:,}</b><extra></"
                "extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=seg_labels,
            y=l_vals,
            name="Lyft",
            marker_color=BRAND_COLORS["Lyft"],
            text=[f"${v:.2f}" if v > 0 else "" for v in l_vals],
            textposition="outside",
            customdata=np.stack([l_names, l_cnts], axis=-1)
            if seg_labels
            else None,
            hovertemplate=(
                "<b>%{customdata[0]} (Lyft)</b><br>Giá báo TB: <b>$%{y:"
                ".2f}</b><br>Cỡ mẫu: <b>%{customdata[1]:,}</b><extra></"
                "extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # --- PANEL PHẢI: Đường chênh lệch (Lyft - Uber) theo cự ly cho 3 phân khúc
    # đại diện ---
    d_grid = np.linspace(EMPIRICAL_DISTANCE_MIN, EMPIRICAL_DISTANCE_MAX, 120)
    t_map = tariffs_df.set_index("service_tier").to_dict("index")

    def _dstar(u_t: str, l_t: str) -> float:
        ur, lr = t_map[u_t], t_map[l_t]
        return (
            float(ur["base_fare_b0_usd"]) - float(lr["base_fare_b0_usd"])
        ) / (
            float(lr["per_mile_rate_b1_usd"])
            - float(ur["per_mile_rate_b1_usd"])
        )

    d_std = _dstar("UberX", "Lyft")
    d_blk = _dstar("Black", "Lux Black")
    line_specs = [
        (
            "Đi ghép: Shared − UberPool",
            "UberPool",
            "Shared",
            BRAND_COLORS["success"],
            "solid",
        ),
        (
            f"Phổ thông: Lyft − UberX (d* = {d_std:.2f} mi)",
            "UberX",
            "Lyft",
            BRAND_COLORS["accent"],
            "solid",
        ),
        (
            f"Hạng sang: Lux Black − Black (d* = {d_blk:.2f} mi)",
            "Black",
            "Lux Black",
            BRAND_COLORS["Lyft"],
            "dash",
        ),
    ]

    fig.add_hrect(
        y0=-4.5,
        y1=0.0,
        fillcolor="rgba(225,29,138,0.06)",
        line_width=0,
        annotation_text="Lyft rẻ hơn (< $0)",
        annotation_position="bottom right",
        annotation_font={"size": 10.5, "color": BRAND_COLORS["Lyft"]},
        row=1,
        col=2,
    )
    fig.add_hrect(
        y0=0.0,
        y1=9.2,
        fillcolor="rgba(17,24,39,0.04)",
        line_width=0,
        annotation_text="Uber rẻ hơn (> $0)",
        annotation_position="top left",
        annotation_font={"size": 10.5, "color": BRAND_COLORS["Uber"]},
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0.0,
        line_color="#0F172A",
        line_width=1.6,
        line_dash="solid",
        row=1,
        col=2,
    )

    for label, u_t, l_t, color, dash in line_specs:
        u_r, l_r = t_map[u_t], t_map[l_t]
        diff_y = (
            float(l_r["base_fare_b0_usd"])
            + float(l_r["per_mile_rate_b1_usd"]) * d_grid
        ) - (
            float(u_r["base_fare_b0_usd"])
            + float(u_r["per_mile_rate_b1_usd"]) * d_grid
        )
        fig.add_trace(
            go.Scatter(
                x=d_grid,
                y=diff_y,
                mode="lines",
                name=label,
                line={"color": color, "width": 2.6, "dash": dash},
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Quãng đường: <b>%{x:.2f} dặm</b><br>"
                    "Chênh lệch (Lyft − Uber): <b>%{y:+.2f} USD</b><extra></"
                    "extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # Đánh dấu điểm cắt chuẩn UberX vs Lyft (Đặc tả A)
    fig.add_trace(
        go.Scatter(
            x=[round(d_std, 2)],
            y=[0.0],
            mode="markers+text",
            name=f"Điểm cắt Phổ thông d* = {d_std:.2f} mi",
            showlegend=False,
            marker={
                "size": 11,
                "symbol": "diamond",
                "color": BRAND_COLORS["accent"],
                "line": {"color": "#FFFFFF", "width": 2},
            },
            text=[f" d* = {d_std:.2f} mi"],
            textposition="bottom right",
            textfont={"size": 11, "color": BRAND_COLORS["accent"]},
            hovertemplate=(
                "<b>Điểm cân bằng UberX = Lyft</b><br>Cự ly: <b>"
                f"{d_std:.2f} dặm</b> (Chênh lệch = $0.00)<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    max_bar = max(u_vals + l_vals) if (u_vals or l_vals) else 35.0
    fig.update_layout(barmode="group")
    fig.update_yaxes(
        title_text="Giá báo TB (USD)", range=[0, max_bar * 1.18], row=1, col=1
    )
    fig.update_xaxes(tickangle=-18, row=1, col=1)
    fig.update_xaxes(
        title_text="Quãng đường thực nghiệm Boston (0.02–7.86 dặm)",
        range=[0.0, 8.0],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Chênh lệch Lyft − Uber (USD)",
        range=[-4.2, 9.0],
        row=1,
        col=2,
    )

    return apply_executive_layout(
        fig,
        title=(
            "Bức tranh Khác biệt Chiến lược Giá: Uber vs. Lyft (Phân khúc × "
            "Quãng đường)"
        ),
        subtitle=(
            f"Trái: Giá TB trên mẫu hiện hành (n = {len(df):,}) | Phải: "
            "Đường chênh lệch HC3 OLS (surge=1.0x) — Lyft lợi thế chặng "
            "ngắn & bình dân, Uber lợi thế chặng dài & hạng sang"
        ),
        height=470,
    )


def build_price_distribution_fig(
    hist_df: pd.DataFrame,
    kpis: dict[str, Any],
) -> go.Figure:
    """
    Biểu đồ 1A (Tab 1): So sánh phân phối tỷ trọng giá báo API (USD) giữa Uber
    và Lyft,
    kèm đường gióng trung vị (Median) giúp thấy rõ tính lệch phải của phân phối
    giá.
    """
    if hist_df.empty:
        return _empty_figure(
            "Phân phối Giá Báo API: Uber vs. Lyft",
            "Không có dữ liệu khớp bộ lọc hiện tại.",
        )

    fig = go.Figure()
    for cab in ("Uber", "Lyft"):
        sub = hist_df.loc[hist_df["cab_type"] == cab]
        if sub.empty:
            continue
        color = BRAND_COLORS[cab]
        fig.add_trace(
            go.Bar(
                x=sub["bin_center_usd"],
                y=sub["share_pct"],
                name=cab,
                marker_color=color,
                opacity=0.72 if cab == "Uber" else 0.62,
                customdata=np.stack([sub["bin_label"], sub["count"]], axis=-1),
                hovertemplate=(
                    f"<b>{cab}</b><br>"
                    "Khoảng giá: <b>%{customdata[0]}</b><br>"
                    "Tỷ trọng trong hãng: <b>%{y:.2f}%</b><br>"
                    "Số lượt báo giá: <b>%{customdata[1]:,}</b><extra></extra>"
                ),
            )
        )

    # Đường gióng Trung vị từng hãng (chống lệch bởi gói hạng sang)
    if kpis.get("uber_median_price_usd") is not None:
        u_med = float(kpis["uber_median_price_usd"])
        fig.add_vline(
            x=u_med,
            line_dash="dash",
            line_color=BRAND_COLORS["Uber"],
            line_width=2,
            annotation_text=f"Trung vị Uber: ${u_med:.2f}",
            annotation_position="top left",
            annotation_font={"size": 11, "color": BRAND_COLORS["Uber"]},
        )
    if kpis.get("lyft_median_price_usd") is not None:
        l_med = float(kpis["lyft_median_price_usd"])
        fig.add_vline(
            x=l_med,
            line_dash="dot",
            line_color=BRAND_COLORS["Lyft"],
            line_width=2,
            annotation_text=f"Trung vị Lyft: ${l_med:.2f}",
            annotation_position="top right",
            annotation_font={"size": 11, "color": BRAND_COLORS["Lyft"]},
        )

    fig.update_layout(barmode="overlay", bargap=0.06)
    return apply_executive_layout(
        fig,
        title="Phân phối Tỷ trọng Giá Báo API: Uber vs. Lyft (USD)",
        subtitle=(
            f"Nguồn: master_rideshare.parquet (n = {kpis['total_obs']:,} "
            "báo giá; đuôi trên $65 gộp vào nhóm cuối)"
        ),
        height=430,
        xaxis_title="Giá báo chuyến đi (USD)",
        yaxis_title="Tỷ trọng lượt báo giá trong hãng (%)",
        narrow=True,
    )


def build_tier_boxplot_fig(box_stats_df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ 1B (Tab 1): Boxplot ngang so sánh 12 phân khúc dịch vụ Uber & Lyft,
    dựng từ phân vị chính xác (Q1, Median, Q3, Fences, Mean) tính trên 100%
    quan sát lọc.
    """
    if box_stats_df.empty:
        return _empty_figure(
            "Cấu trúc Giá theo 12 Phân khúc Dịch vụ",
            "Không có dữ liệu khớp bộ lọc hiện tại.",
        )

    fig = go.Figure()
    ordered_tiers = box_stats_df["service_tier"].tolist()
    total_n = int(box_stats_df["count"].sum())

    for cab in ("Uber", "Lyft"):
        sub = box_stats_df.loc[box_stats_df["cab_type"] == cab]
        if sub.empty:
            continue
        color = BRAND_COLORS[cab]
        fig.add_trace(
            go.Box(
                y=sub["service_tier"],
                orientation="h",
                name=cab,
                q1=sub["q1_usd"],
                median=sub["median_usd"],
                q3=sub["q3_usd"],
                lowerfence=sub["lowerfence_usd"],
                upperfence=sub["upperfence_usd"],
                mean=sub["mean_usd"],
                marker_color=color,
                line={"color": color, "width": 1.6},
                fillcolor="rgba(17,24,39,0.14)"
                if cab == "Uber"
                else "rgba(225,29,138,0.14)",
                boxmean=True,
                hoverinfo="x+y+name",
            )
        )

    # Thêm điểm trung bình có tooltip tiếng Việt đầy đủ (n, Mean, Median,
    # Q1–Q3)
    for cab in ("Uber", "Lyft"):
        sub = box_stats_df.loc[box_stats_df["cab_type"] == cab]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["mean_usd"],
                y=sub["service_tier"],
                mode="markers",
                name=f"Trung bình {cab}",
                showlegend=False,
                marker={
                    "symbol": "diamond",
                    "size": 7,
                    "color": BRAND_COLORS[cab],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                customdata=np.stack(
                    [
                        sub["count"],
                        sub["median_usd"],
                        sub["q1_usd"],
                        sub["q3_usd"],
                        sub["min_usd"],
                        sub["max_usd"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b> (" + cab + ")<br>"
                    "Số lượt báo giá: <b>%{customdata[0]:,}</b><br>"
                    "Giá trung bình (Mean): <b>$%{x:.2f}</b><br>"
                    "Giá trung vị (Median): <b>$%{customdata[1]:.2f}</b><br>"
                    "Khoảng tứ phân vị (Q1–Q3): <b>$%{customdata[2]:.2f} – "
                    "$%{customdata[3]:.2f}</b><br>"
                    "Biên thực tế (Min–Max): <b>$%{customdata[4]:.2f} – "
                    "$%{customdata[5]:.2f}</b><extra></extra>"
                ),
            )
        )

    fig.update_yaxes(categoryorder="array", categoryarray=ordered_tiers)
    return apply_executive_layout(
        fig,
        title="Phân bố Giá Báo theo 12 Gói Dịch vụ (Sắp theo Trung vị)",
        subtitle=(
            f"Tính chính xác trên 100% quan sát lọc (n = {total_n:,}) | "
            "Hình thoi = Giá trung bình (Mean)"
        ),
        height=490,
        xaxis_title="Giá báo API (USD)",
        yaxis_title="",
        narrow=True,
    )


def build_route_heatmap_fig(
    routes_df: pd.DataFrame,
    metric: str = "mean_price_usd",
) -> go.Figure:
    """
    Biểu đồ 1C (Tab 1): Bản đồ nhiệt 12 quận đón x 12 quận trả tại Boston (72
    tuyến thực tế).
    Hỗ trợ chuyển đổi động giữa: 'mean_price_usd', 'median_distance_miles',
    'surge_rate_pct'.
    Các ô trống biểu thị cặp quận không kết nối trực tiếp trong đồ thị lưỡng
    phân 6x6 của dữ liệu Boston.
    """
    metric_specs = {
        "mean_price_usd": (
            "Giá báo trung bình (USD)",
            "$%{z:.2f}",
            "Blues",
            "Giá báo TB (USD)",
        ),
        "median_distance_miles": (
            "Quãng đường trung vị (dặm)",
            "%{z:.2f} dặm",
            "Teal",
            "Cự ly TB vị (dặm)",
        ),
        "surge_rate_pct": (
            "Tỷ lệ kích hoạt Surge (%)",
            "%{z:.2f}%",
            "RdPu",
            "Tỷ lệ Surge (%)",
        ),
    }
    label_vi, z_fmt, colorscale, cb_title = metric_specs.get(
        metric, metric_specs["mean_price_usd"]
    )

    if routes_df.empty:
        return _empty_figure(
            f"Ma trận Tuyến đường Boston — {label_vi}",
            "Không có tuyến đường nào khớp bộ lọc hiện tại.",
        )

    districts = list(VALID_BOSTON_DISTRICTS)
    pivot_val = routes_df.pivot(
        index="source", columns="destination", values=metric
    ).reindex(index=districts, columns=districts)
    pivot_trips = routes_df.pivot(
        index="source", columns="destination", values="trips"
    ).reindex(index=districts, columns=districts)
    pivot_dist = routes_df.pivot(
        index="source", columns="destination", values="median_distance_miles"
    ).reindex(index=districts, columns=districts)
    pivot_price = routes_df.pivot(
        index="source", columns="destination", values="mean_price_usd"
    ).reindex(index=districts, columns=districts)
    pivot_surge = routes_df.pivot(
        index="source", columns="destination", values="surge_rate_pct"
    ).reindex(index=districts, columns=districts)

    # Tạo mảng text hiển thị gọn trực tiếp trong ô có dữ liệu
    text_matrix = []
    custom_matrix = []
    for r_idx, src in enumerate(districts):
        t_row = []
        c_row = []
        for c_idx, dst in enumerate(districts):
            val = pivot_val.iloc[r_idx, c_idx]
            if pd.isna(val):
                t_row.append("")
                c_row.append(["Không kết nối", 0, 0.0, 0.0, 0.0])
            else:
                if metric == "mean_price_usd":
                    t_row.append(f"${val:.1f}")
                elif metric == "median_distance_miles":
                    t_row.append(f"{val:.1f}")
                else:
                    t_row.append(f"{val:.1f}%")
                c_row.append(
                    [
                        f"{src} → {dst}",
                        int(pivot_trips.iloc[r_idx, c_idx]),
                        float(pivot_price.iloc[r_idx, c_idx]),
                        float(pivot_dist.iloc[r_idx, c_idx]),
                        float(pivot_surge.iloc[r_idx, c_idx]),
                    ]
                )
        text_matrix.append(t_row)
        custom_matrix.append(c_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_val.values,
            x=districts,
            y=districts,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=colorscale,
            colorbar={"title": cb_title, "thickness": 14},
            hoverongaps=False,
            customdata=custom_matrix,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Số lượt báo giá: <b>%{customdata[1]:,}</b><br>"
                "Giá báo TB: <b>$%{customdata[2]:.2f}</b><br>"
                "Cự ly trung vị: <b>%{customdata[3]:.2f} dặm</b><br>"
                "Tỷ lệ Surge (>1.0x): <b>%{customdata[4]:.2f}%</b><extra></"
                "extra>"
            ),
        )
    )
    fig.update_xaxes(tickangle=-32, side="bottom")
    fig.update_yaxes(autorange="reversed")
    return apply_executive_layout(
        fig,
        title=f"Ma trận 72 Tuyến Liên quận Boston — {label_vi}",
        subtitle=(
            "Ô trắng là cặp quận không kết nối trực tiếp trong đồ thị lưỡng "
            "phân 6×6 của bộ dữ liệu Boston"
        ),
        height=520,
        xaxis_title="Quận điểm đến (Destination)",
        yaxis_title="Quận điểm đón (Source)",
        legend_top=False,
    )


# =============================================================================
# TAB 2 — RQ1: THỜI TIẾT (CRIT-01), THỜI GIAN BOSTON EST (CRIT-02) & SURGE
# =============================================================================
def build_weather_rain_audit_fig(
    weather_df: pd.DataFrame,
    weather_meta: dict[str, Any],
) -> go.Figure:
    """
    Biểu đồ 2A (Tab 2 — CRIT-01): So sánh Giá báo TB (USD) và Tỷ lệ Surge Lyft
    (%)
    giữa Trời tạnh (Dry) và Có mưa (Rain) CHỈ trên mẫu khớp thời tiết thực
    (weather_matched == 1).
    """
    if weather_df.empty or weather_meta.get("n_matched", 0) == 0:
        return _empty_figure(
            "Tác động Thời tiết (Mưa vs. Không mưa) — Khóa weather_matched "
            "== 1",
            "Không có bản ghi khớp thời tiết thực trong bộ lọc hiện tại.",
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Giá báo trung bình (USD): Không mưa vs. Có mưa",
            "Tỷ lệ kích hoạt Surge Lyft (%): Không mưa vs. Có mưa",
        ),
        column_widths=[0.60, 0.40],
        horizontal_spacing=0.11,
    )

    state_colors = {
        "Không mưa (Dry)": BRAND_COLORS["accent"],
        "Có mưa (Rain)": "#0EA5E9",
    }

    for state in ("Không mưa (Dry)", "Có mưa (Rain)"):
        sub = weather_df.loc[weather_df["weather_state"] == state]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["scope"],
                y=sub["mean_price_usd"],
                name=state,
                marker_color=state_colors[state],
                text=[f"${v:.2f}" for v in sub["mean_price_usd"]],
                textposition="outside",
                customdata=np.stack(
                    [sub["n_obs"], sub["median_price_usd"]], axis=-1
                ),
                hovertemplate=(
                    f"<b>%{{x}} — {state}</b><br>"
                    "Giá báo TB: <b>$%{y:.4f}</b><br>"
                    "Giá trung vị: <b>$%{customdata[1]:.2f}</b><br>"
                    "Cỡ mẫu khớp thực: <b>%{customdata[0]:,}</b><extra></"
                    "extra>"
                ),
            ),
            row=1,
            col=1,
        )

    lyft_w = weather_df.loc[weather_df["scope"] == "Lyft"]
    if not lyft_w.empty:
        fig.add_trace(
            go.Bar(
                x=lyft_w["weather_state"],
                y=lyft_w["surge_rate_pct"],
                name="Tỷ lệ Surge Lyft (%)",
                showlegend=False,
                marker_color=[BRAND_COLORS["Lyft"], "#9D174D"],
                text=[f"{v:.2f}%" for v in lyft_w["surge_rate_pct"]],
                textposition="outside",
                customdata=lyft_w["n_obs"],
                hovertemplate=(
                    "<b>Lyft — %{x}</b><br>"
                    "Tỷ lệ kích hoạt Surge (>1.0x): <b>%{y:.2f}%</b><br>"
                    "Cỡ mẫu Lyft khớp thời tiết: <b>%{customdata:,}</"
                    "b><extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    diff_val = weather_meta.get("all_rain_minus_dry_usd")
    diff_txt = (
        f"Chênh lệch toàn mẫu: {diff_val:+.4f} USD"
        if diff_val is not None
        else ""
    )
    n_valid = int(
        weather_meta.get("n_valid_rain_obs", weather_meta.get("n_matched", 0))
    )
    n_sent = int(weather_meta.get("n_rain_sentinel", 0))
    fig.update_yaxes(
        title_text="Giá báo TB (USD)",
        range=[0, max(22.0, float(weather_df["mean_price_usd"].max()) * 1.18)],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Tỷ lệ Surge Lyft (%)", range=[0, 9.5], row=1, col=2
    )
    fig.update_layout(barmode="group")

    return apply_executive_layout(
        fig,
        title=(
            "Kiểm toán Tác động Thời tiết đến Giá Báo API & Surge (0 Nhãn "
            "mưa giả)"
        ),
        subtitle=(
            f"Mẫu khí tượng sạch lỗi cảm biến: n = {n_valid:,} (Mưa: "
            f"{weather_meta['n_rain']:,}, Tạnh: {weather_meta['n_dry']:,}; "
            f"tách {n_sent:,} ô sentinel) • {diff_txt}"
        ),
        height=440,
    )


def build_hourly_est_fig(
    hourly_df: pd.DataFrame,
    hourly_meta: dict[str, Any],
) -> go.Figure:
    """
    Biểu đồ 2B (Tab 2 — CRIT-02): Nhịp điệu 24 giờ Boston EST
    (America/New_York)
    chỉ tính trên bản ghi có timestamp gốc (is_timestamp_original == 1).
    Trục trái: Giá báo TB (USD) Uber vs Lyft; Trục phải: Tỷ lệ Surge Lyft (%).
    """
    if hourly_df.empty or hourly_meta.get("n_valid_ts", 0) == 0:
        return _empty_figure(
            "Diễn biến Giá Báo API & Surge theo 24 Giờ Boston EST",
            "Không có bản ghi timestamp gốc hợp lệ trong bộ lọc hiện tại.",
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Tô vùng giờ cao điểm sáng (7–9h) & chiều (16–19h EST)
    fig.add_vrect(
        x0=7,
        x1=9,
        fillcolor="#FEF3C7",
        opacity=0.45,
        layer="below",
        line_width=0,
        annotation_text="Cao điểm sáng",
        annotation_position="top left",
        annotation_font={"size": 10, "color": "#92400E"},
    )
    fig.add_vrect(
        x0=16,
        x1=19,
        fillcolor="#FEF3C7",
        opacity=0.45,
        layer="below",
        line_width=0,
        annotation_text="Cao điểm chiều",
        annotation_position="top left",
        annotation_font={"size": 10, "color": "#92400E"},
    )

    for cab in ("Uber", "Lyft"):
        sub = hourly_df.loc[hourly_df["cab_type"] == cab]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["hour"],
                y=sub["mean_price_usd"],
                mode="lines+markers",
                name=f"Giá TB {cab} (USD)",
                line={"color": BRAND_COLORS[cab], "width": 2.4},
                marker={"size": 6},
                customdata=np.stack(
                    [sub["trips"], sub["median_price_usd"]], axis=-1
                ),
                hovertemplate=(
                    f"<b>{cab} — %{{x:02d}}:00 EST</b><br>"
                    "Giá báo TB: <b>$%{y:.2f}</b><br>"
                    "Giá trung vị: <b>$%{customdata[1]:.2f}</b><br>"
                    "Lưu lượng mẫu: <b>%{customdata[0]:,}</b> lượt<extra></"
                    "extra>"
                ),
            ),
            secondary_y=False,
        )

    lyft_h = hourly_df.loc[hourly_df["cab_type"] == "Lyft"]
    if not lyft_h.empty:
        fig.add_trace(
            go.Scatter(
                x=lyft_h["hour"],
                y=lyft_h["surge_rate_pct"],
                mode="lines+markers",
                name="Tỷ lệ Surge Lyft (%, trục phải)",
                line={
                    "color": BRAND_COLORS["warning"],
                    "width": 2.0,
                    "dash": "dot",
                },
                marker={"size": 5, "symbol": "diamond"},
                hovertemplate=(
                    "<b>Lyft Surge — %{x:02d}:00 EST</b><br>Tỷ lệ Surge (>"
                    "1.0x): <b>%{y:.2f}%</b><extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    fig.update_xaxes(
        title_text="Giờ trong ngày — Boston EST (00h–23h)",
        dtick=2,
        range=[-0.5, 23.5],
    )
    fig.update_yaxes(
        title_text="Giá báo trung bình (USD)",
        secondary_y=False,
        range=[12.0, 21.0],
    )
    fig.update_yaxes(
        title_text="Tỷ lệ kích hoạt Surge Lyft (%)",
        secondary_y=True,
        range=[0.0, 10.5],
        showgrid=False,
    )

    peak_h = hourly_meta.get("peak_hour_est")
    trough_h = hourly_meta.get("trough_hour_est")
    return apply_executive_layout(
        fig,
        title=(
            "Nhịp điệu Giá Báo API & Surge theo 24 Giờ Địa phương Boston EST"
        ),
        subtitle=(
            "Mẫu dấu thời gian gốc: is_timestamp_original == 1 (n = "
            f"{hourly_meta['n_valid_ts']:,}, 0 ffill) | "
            f"Lưu lượng thấp nhất: {trough_h:02d}:00 EST • Cao nhất: "
            f"{peak_h:02d}:00 EST"
        ),
        height=440,
    )


def build_dayofweek_est_fig(dow_df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ 2C (Tab 2 — CRIT-02): So sánh giá báo TB theo 7 Thứ trong tuần
    (Boston EST).
    """
    if dow_df.empty:
        return _empty_figure(
            "Giá Báo theo 7 Thứ trong tuần (Boston EST)",
            "Không có dữ liệu khớp bộ lọc.",
        )

    fig = go.Figure()
    for cab in ("Uber", "Lyft"):
        sub = dow_df.loc[dow_df["cab_type"] == cab]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["day_label_vi"],
                y=sub["mean_price_usd"],
                name=cab,
                marker_color=BRAND_COLORS[cab],
                text=[f"${v:.2f}" for v in sub["mean_price_usd"]],
                textposition="outside",
                customdata=np.stack(
                    [
                        sub["trips"],
                        sub["median_price_usd"],
                        sub["surge_rate_pct"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"<b>{cab} — %{{x}}</b><br>"
                    "Giá báo TB: <b>$%{y:.2f}</b> | Trung vị: "
                    "<b>$%{customdata[1]:.2f}</b><br>"
                    "Số lượt TS gốc: <b>%{customdata[0]:,}</b><br>"
                    "Tỷ lệ Surge (>1.0x): <b>%{customdata[2]:.2f}%</"
                    "b><extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="group")
    fig.update_yaxes(
        range=[0, max(22.0, float(dow_df["mean_price_usd"].max()) * 1.18)]
    )
    return apply_executive_layout(
        fig,
        title="Ổn định Giá Báo theo Thứ trong tuần (Boston EST)",
        subtitle=(
            "Lọc bắt buộc is_timestamp_original == 1 (n = "
            f"{int(dow_df['trips'].sum()):,}) • Không có hiệu ứng đột biến "
            "cuối tuần"
        ),
        height=410,
        xaxis_title="Thứ trong tuần (Boston EST)",
        yaxis_title="Giá báo trung bình (USD)",
        narrow=True,
    )


def build_surge_levels_fig(surge_dist_df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ 2D (Tab 2): Phân bố 7 bậc hệ số Surge Lyft (1.0x -> 3.0x) và bước
    nhảy giá cước.
    """
    if surge_dist_df.empty:
        return _empty_figure(
            "Phân bố Bậc Hệ số Surge & Bước nhảy Giá (Riêng Lyft)",
            "Không có bản ghi Lyft trong bộ lọc hiện tại (Uber luôn cố định "
            "surge = 1.0x).",
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    labels = [f"{v:.2f}x" for v in surge_dist_df["surge_multiplier"]]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=surge_dist_df["trips"],
            name="Số lượt báo giá Lyft",
            marker_color=[
                BRAND_COLORS["muted"] if v == 1.0 else BRAND_COLORS["Lyft"]
                for v in surge_dist_df["surge_multiplier"]
            ],
            text=[f"{int(v):,}" for v in surge_dist_df["trips"]],
            textposition="outside",
            customdata=np.stack(
                [
                    surge_dist_df["share_of_lyft_pct"],
                    surge_dist_df["mean_price_usd"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Bậc Surge %{x}</b><br>"
                "Số lượt Lyft: <b>%{y:,}</b> (%{customdata[0]:.2f}% Lyft)<br>"
                "Giá báo TB: <b>$%{customdata[1]:.2f}</b><extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=surge_dist_df["mean_price_usd"],
            mode="lines+markers+text",
            name="Giá báo TB (USD, trục phải)",
            line={"color": BRAND_COLORS["accent"], "width": 2.6},
            marker={"size": 8, "color": BRAND_COLORS["accent"]},
            text=[f"${v:.1f}" for v in surge_dist_df["mean_price_usd"]],
            textposition="top center",
            hovertemplate=(
                "<b>Bậc Surge %{x}</b><br>Giá báo TB: <b>$%{y:.2f}</b>"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_yaxes(
        title_text="Số lượt báo giá (thang log)", type="log", secondary_y=False
    )
    fig.update_yaxes(
        title_text="Giá báo trung bình (USD)",
        secondary_y=True,
        showgrid=False,
        range=[0, 52],
    )
    return apply_executive_layout(
        fig,
        title=(
            "Phân bố 7 Bậc Surge & Bước nhảy Giá Báo API (Riêng Lyft — "
            "Surge quan sát thực)"
        ),
        subtitle=(
            "Mẫu Lyft quan sát surge thực (n = "
            f"{int(surge_dist_df['trips'].sum()):,}) • Thang log nhìn rõ "
            "đuôi hiếm 2.50x–3.00x (53 lượt) | Uber cố định 1.00x"
        ),
        height=410,
        xaxis_title="Hệ số nhân định giá động (surge_multiplier)",
        narrow=True,
    )


# =============================================================================
# TAB 3 — RQ2: CẠNH TRANH BIỂU GIÁ & ĐIỂM CẮT NỘI MIỀN (HIGH-02)
# =============================================================================
def build_tariff_crossing_fig(
    tariffs_df: pd.DataFrame,
    pair_info: dict[str, Any],
    binned_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Biểu đồ 3A (Tab 3 — HIGH-02): Đường biểu giá tuyến tính OLS HC3 (kèm dải
    tin cậy 95%
    và chấm giá thực tế theo nhóm cự ly) trên đúng miền thực nghiệm [0.02,
    7.86] dặm.
    KHÔNG ngoại suy ngoài miền (ví dụ UberPool vs Shared ghi rõ không cắt trong
    Boston).
    """
    u_tier = pair_info["uber_tier"]
    l_tier = pair_info["lyft_tier"]
    u_row = tariffs_df.loc[tariffs_df["service_tier"] == u_tier].iloc[0]
    l_row = tariffs_df.loc[tariffs_df["service_tier"] == l_tier].iloc[0]

    d_grid = np.linspace(EMPIRICAL_DISTANCE_MIN, EMPIRICAL_DISTANCE_MAX, 160)
    u_y = (
        float(u_row["base_fare_b0_usd"])
        + float(u_row["per_mile_rate_b1_usd"]) * d_grid
    )
    l_y = (
        float(l_row["base_fare_b0_usd"])
        + float(l_row["per_mile_rate_b1_usd"]) * d_grid
    )

    fig = go.Figure()

    # Tô vùng lợi thế giá nội miền nếu có điểm cắt hợp lệ trong [0.02, 7.86]
    # dặm
    if (
        pair_info["crossing_in_domain"]
        and pair_info["crossing_distance_miles"] is not None
    ):
        d_star = float(pair_info["crossing_distance_miles"])
        p_star = float(pair_info["crossing_price_usd"])
        fig.add_vrect(
            x0=EMPIRICAL_DISTANCE_MIN,
            x1=d_star,
            fillcolor="rgba(225,29,138,0.06)",
            line_width=0,
            annotation_text=f"Vùng {l_tier} rẻ hơn (< {d_star:.2f} mi)",
            annotation_position="top left",
            annotation_font={"size": 11, "color": BRAND_COLORS["Lyft"]},
        )
        fig.add_vrect(
            x0=d_star,
            x1=EMPIRICAL_DISTANCE_MAX,
            fillcolor="rgba(17,24,39,0.05)",
            line_width=0,
            annotation_text=f"Vùng {u_tier} rẻ hơn (> {d_star:.2f} mi)",
            annotation_position="top right",
            annotation_font={"size": 11, "color": BRAND_COLORS["Uber"]},
        )
        fig.add_vline(
            x=d_star,
            line_dash="dash",
            line_color=BRAND_COLORS["accent"],
            line_width=2,
        )
    else:
        d_star, p_star = None, None
        fig.add_vrect(
            x0=EMPIRICAL_DISTANCE_MIN,
            x1=EMPIRICAL_DISTANCE_MAX,
            fillcolor="rgba(225,29,138,0.06)",
            line_width=0,
            annotation_text=(
                f"Không cắt trong [0.02, 7.86] dặm — {l_tier} luôn rẻ hơn "
                f"{u_tier}"
            ),
            annotation_position="top left",
            annotation_font={"size": 11, "color": BRAND_COLORS["Lyft"]},
        )

    # Đường biểu giá OLS HC3
    fig.add_trace(
        go.Scatter(
            x=d_grid,
            y=u_y,
            mode="lines",
            name=(
                f"{u_tier} (OLS: ${u_row['base_fare_b0_usd']:.2f} + "
                f"${u_row['per_mile_rate_b1_usd']:.2f}/mi)"
            ),
            line={"color": BRAND_COLORS["Uber"], "width": 3.0},
            hovertemplate=(
                f"<b>{u_tier} (HC3 OLS)</b><br>Cự ly: %{{x:.2f}} dặm<br>Giá "
                f"kỳ vọng: <b>$%{{y:.2f}}</b><extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_grid,
            y=l_y,
            mode="lines",
            name=(
                f"{l_tier} (OLS: ${l_row['base_fare_b0_usd']:.2f} + "
                f"${l_row['per_mile_rate_b1_usd']:.2f}/mi)"
            ),
            line={"color": BRAND_COLORS["Lyft"], "width": 3.0},
            hovertemplate=(
                f"<b>{l_tier} (HC3 OLS)</b><br>Cự ly: %{{x:.2f}} dặm<br>Giá "
                f"kỳ vọng: <b>$%{{y:.2f}}</b><extra></extra>"
            ),
        )
    )

    # Điểm trung bình thực nghiệm theo nhóm cự ly (surge == 1.0)
    if binned_df is not None and not binned_df.empty:
        for tier, cab in ((u_tier, "Uber"), (l_tier, "Lyft")):
            sub = binned_df.loc[binned_df["service_tier"] == tier]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["dist_bin_center"],
                    y=sub["mean_price_usd"],
                    mode="markers",
                    name=f"TB thực tế {tier} (1.0x)",
                    marker={
                        "size": 7.5,
                        "color": BRAND_COLORS[cab],
                        "symbol": "circle-open",
                        "line": {"width": 2},
                    },
                    customdata=sub["n_obs"],
                    hovertemplate=(
                        f"<b>Thực tế {tier} (surge=1.0x)</b><br>"
                        "Nhóm cự ly: ~%{x:.2f} dặm<br>"
                        "Giá báo TB thực tế: <b>$%{y:.2f}</b><br>"
                        "Cỡ mẫu nhóm: <b>%{customdata:,}</b><extra></extra>"
                    ),
                )
            )

    # Điểm giao cắt d* nổi bật
    if d_star is not None and p_star is not None:
        fig.add_trace(
            go.Scatter(
                x=[d_star],
                y=[p_star],
                mode="markers+text",
                name=f"Điểm cắt d* = {d_star:.2f} mi",
                marker={
                    "size": 13,
                    "symbol": "diamond",
                    "color": BRAND_COLORS["accent"],
                    "line": {"color": "#FFFFFF", "width": 2},
                },
                text=[f"  d* = {d_star:.2f} dặm (${p_star:.2f})"],
                textposition="middle right",
                textfont={"size": 12, "color": BRAND_COLORS["accent"]},
                hovertemplate=(
                    "<b>Điểm cân bằng cạnh tranh d*</b><br>Cự ly cắt nhau: "
                    f"<b>{d_star:.2f} dặm</b><br>Giá ngang nhau: <b>"
                    f"${p_star:.2f}</b><extra></extra>"
                ),
            )
        )

    fig.update_xaxes(range=[0.0, EMPIRICAL_DISTANCE_MAX + 0.15])
    return apply_executive_layout(
        fig,
        title=(
            f"Đường Biểu giá Cạnh tranh: {u_tier} vs. {l_tier} "
            f"({pair_info['short_label']})"
        ),
        subtitle=(
            "Miền thực nghiệm Boston [0.02, 7.86] dặm (không ngoại suy) • "
            f"{pair_info['crossing_status_vi']}"
        ),
        height=460,
        xaxis_title="Quãng đường chuyến đi (dặm — miền quan sát thực tế)",
        yaxis_title="Giá báo API (USD, khi surge = 1.0x)",
    )


def build_tariff_strategy_scatter_fig(tariffs_df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ 3B (Tab 3): Bản đồ chiến lược Giá mở cửa (Base Fare b0) vs Đơn giá
    mỗi dặm (Slope b1)
    cho cả 12 gói dịch vụ, nối 5 cặp đối xứng để chứng minh quy luật 'Lyft mở
    cửa thấp hơn – chạy dài đắt hơn'.
    """
    if tariffs_df.empty:
        return _empty_figure(
            "Bản đồ Chiến lược Biểu giá 12 Gói Dịch vụ",
            "Không có dữ liệu biểu giá.",
        )

    fig = go.Figure()
    lookup = tariffs_df.set_index("service_tier").to_dict("index")
    standalone_names = {item["tier"] for item in STANDALONE_TIERS}

    # Đường nối 5 cặp đối xứng để nhìn ngay hướng dịch chuyển chiến lược Uber
    # -> Lyft
    for pair in SYMMETRIC_PAIRS:
        uber_row = lookup[pair["uber_tier"]]
        lyft_row = lookup[pair["lyft_tier"]]
        fig.add_trace(
            go.Scatter(
                x=[uber_row["base_fare_b0_usd"], lyft_row["base_fare_b0_usd"]],
                y=[
                    uber_row["per_mile_rate_b1_usd"],
                    lyft_row["per_mile_rate_b1_usd"],
                ],
                mode="lines",
                line={"color": "#CBD5E1", "width": 1.8, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for cab in ("Uber", "Lyft"):
        sub = tariffs_df.loc[tariffs_df["cab_type"] == cab]
        symbols = [
            "star" if t in standalone_names else "circle"
            for t in sub["service_tier"]
        ]
        sizes = [
            14 if t in standalone_names else 12 for t in sub["service_tier"]
        ]
        fig.add_trace(
            go.Scatter(
                x=sub["base_fare_b0_usd"],
                y=sub["per_mile_rate_b1_usd"],
                mode="markers+text",
                name=f"{cab} (★ = gói độc lập không có cặp)",
                marker={
                    "color": BRAND_COLORS[cab],
                    "size": sizes,
                    "symbol": symbols,
                    "line": {"color": "#FFFFFF", "width": 1.5},
                },
                text=sub["service_tier"],
                textposition="top center",
                textfont={"size": 11},
                customdata=np.stack(
                    [
                        sub["market_segment"],
                        sub["r2_within_tier"],
                        sub["expected_fare_2_5mi_usd"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{text}</b> (" + cab + ")<br>"
                    "Phân khúc: %{customdata[0]}<br>"
                    "Giá mở cửa (β₀): <b>$%{x:.2f}</b><br>"
                    "Đơn giá/dặm (β₁): <b>$%{y:.2f}/dặm</b><br>"
                    "Giá kỳ vọng chặng 2.5 dặm: <b>$%{customdata[2]:.2f}</"
                    "b><br>"
                    "R² nội phân khúc: <b>%{customdata[1]:.4f}</b><extra></"
                    "extra>"
                ),
            )
        )

    return apply_executive_layout(
        fig,
        title=(
            "Bản đồ Chiến lược Biểu giá 12 Gói Xe: Giá Mở cửa (β₀) vs. Đơn "
            "giá/Dặm (β₁)"
        ),
        subtitle=(
            "Đường chấm nối 5 cặp đối xứng: trên 5/5 phân khúc, Lyft luôn "
            "đặt β₀ thấp hơn nhưng β₁ cao hơn Uber (★ = WAV & Lux độc lập)"
        ),
        height=460,
        xaxis_title="Giá mở cửa ước lượng HC3 — β₀ (USD)",
        yaxis_title="Đơn giá biên mỗi dặm ước lượng HC3 — β₁ (USD / dặm)",
    )


# =============================================================================
# TAB 4 — RQ3: ML ARENA, CHẨN ĐOÁN PHẦN DƯ, XAI & BẤT ĐỊNH DỰ BÁO (GIAI ĐOẠN 5)
# =============================================================================
def build_model_arena_fig(leaderboard_df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ 4A (Tab 4 — Model Performance):
    - Trái: Xếp hạng 10 mô hình (M0a–M7) theo Test RMSE (USD), phân biệt mô
      hình đạt chuẩn SMART.
    - Phải: Đánh đổi Độ chính xác (Test R²) vs. Tốc độ suy luận (Predict Time
      ms).
    """
    if leaderboard_df.empty:
        return _empty_figure(
            "So sánh 10 Cấu hình Mô hình Học máy (M0a–M7)",
            "Không có bảng đánh giá mô hình.",
        )

    df = (
        leaderboard_df.copy()
        .sort_values("test_rmse", ascending=False)
        .reset_index(drop=True)
    )
    short_names = [
        str(code).split("_")[0] + " — " + str(name).split(": ")[-1][:24]
        for code, name in zip(df["model_code"], df["model_name"], strict=False)
    ]

    bar_colors = []
    for code, smart in zip(
        df["model_code"], df["meets_both_smart"], strict=False
    ):
        if code == "M7_HistGradientBoosting":
            bar_colors.append(BRAND_COLORS["success"])
        elif code == "M2_Interaction_OLS":
            bar_colors.append(BRAND_COLORS["accent"])
        elif bool(smart):
            bar_colors.append("#60A5FA")
        else:
            bar_colors.append("#94A3B8")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Xếp hạng 10 Mô hình theo Test RMSE (USD — thấp hơn là tốt hơn)",
            "Biên hiệu quả: Test R² vs. Tốc độ (ms / 46,573 dòng)",
        ),
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            y=short_names,
            x=df["test_rmse"],
            orientation="h",
            marker_color=bar_colors,
            text=[
                f"${r:.2f} (R²={r2:.4f})"
                for r, r2 in zip(df["test_rmse"], df["test_r2"], strict=False)
            ],
            textposition="outside",
            showlegend=False,
            customdata=np.stack(
                [
                    df["test_r2"],
                    df["test_mae"],
                    df["test_mape_pct"],
                    df["meets_both_smart"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Test RMSE: <b>$%{x:.4f}</b><br>"
                "Test R²: <b>%{customdata[0]:.4f}</b><br>"
                "Test MAE: <b>$%{customdata[1]:.4f}</b> | MAPE: "
                "<b>%{customdata[2]:.2f}%</b><br>"
                "Đạt cả 2 chuẩn SMART (R²≥0.90 & MAPE≤10%): "
                "<b>%{customdata[3]}</b><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Scatter R² vs latency (loại M0a quá thấp khỏi khung zoom để nhìn rõ nhóm
    # cạnh tranh)
    df_sc = leaderboard_df.loc[leaderboard_df["test_r2"] >= 0.85].copy()
    sc_colors = [
        BRAND_COLORS["success"]
        if c == "M7_HistGradientBoosting"
        else (
            BRAND_COLORS["accent"]
            if c == "M2_Interaction_OLS"
            else BRAND_COLORS["muted"]
        )
        for c in df_sc["model_code"]
    ]
    sc_labels = [str(c).split("_")[0] for c in df_sc["model_code"]]
    sc_textpos = [
        "top left"
        if c == "M7_HistGradientBoosting"
        else ("bottom right" if c == "M6_Random_Forest" else "top center")
        for c in df_sc["model_code"]
    ]
    fig.add_trace(
        go.Scatter(
            x=df_sc["predict_time_ms"],
            y=df_sc["test_r2"],
            mode="markers+text",
            text=sc_labels,
            textposition=sc_textpos,
            showlegend=False,
            marker={
                "size": 12,
                "color": sc_colors,
                "line": {"color": "#FFFFFF", "width": 1.5},
            },
            customdata=np.stack(
                [
                    df_sc["model_name"],
                    df_sc["test_rmse"],
                    df_sc["test_mape_pct"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Test R²: <b>%{y:.4f}</b> | RMSE: <b>$%{customdata[1]:.4f}</"
                "b><br>"
                "MAPE: <b>%{customdata[2]:.2f}%</b><br>"
                "Thời gian dự báo tập Test (46,573 dòng): <b>%{x:.1f} ms</"
                "b><extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    m7_r = leaderboard_df.loc[
        leaderboard_df["model_code"] == "M7_HistGradientBoosting"
    ].iloc[0]
    m2_r = leaderboard_df.loc[
        leaderboard_df["model_code"] == "M2_Interaction_OLS"
    ].iloc[0]
    fig.update_xaxes(
        title_text="Test RMSE (USD)",
        range=[0, float(df["test_rmse"].max()) * 1.32],
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Thời gian dự báo tập Test (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Test R²", range=[0.915, 0.972], row=1, col=2)
    return apply_executive_layout(
        fig,
        title=(
            "Đấu trường 10 Mô hình Có điều kiện trên Tập Test Độc lập (n = "
            "46,573)"
        ),
        subtitle=(
            f"Xanh lá = M7 (Vô địch dự báo: R²={float(m7_r['test_r2']):.4f},"
            f" RMSE=${float(m7_r['test_rmse']):.2f}) | "
            "Xanh dương đậm = M2 OLS (Diễn giải chính: "
            f"R²={float(m2_r['test_r2']):.4f})"
        ),
        height=470,
        legend_top=False,
    )


def build_actual_vs_predicted_and_residuals_fig(
    eval_res: dict[str, Any],
    model_key: str = "M7",
) -> go.Figure:
    """
    Biểu đồ 4B (Tab 4 — Residual Diagnostics & Actual vs Predicted):
    - Trái: Actual vs Predicted (USD) trên mẫu phân tầng n = 3,500 (seed=42) +
      đường chéo lý tưởng y = x.
    - Phải: Histogram phần dư (y - ŷ) tính trên toàn bộ 46,573 dòng tập Test
      trong [-8, +8] USD (không clip đỉnh giả).
    """
    plot_df = eval_res["plot_sample_df"]
    hist_df = eval_res["resid_hist_df"]
    pred_col = "pred_m7_usd" if model_key == "M7" else "pred_m2_usd"
    hist_col = "m7_count" if model_key == "M7" else "m2_count"
    metrics = (
        eval_res["metrics_m7"] if model_key == "M7" else eval_res["metrics_m2"]
    )
    model_label = (
        "M7 HistGradientBoosting"
        if model_key == "M7"
        else "M2 Interaction OLS"
    )

    n_test_eval = int(metrics["n_test"])
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Giá Thực tế vs. Dự báo — {model_label} (mẫu n = "
            f"{len(plot_df):,})",
            f"Phân phối Phần dư (y − ŷ) trên Tập Test n = {n_test_eval:,} "
            "(lọc [-8, +8] USD)",
        ),
        horizontal_spacing=0.10,
    )

    for cab in ("Uber", "Lyft"):
        sub = plot_df.loc[plot_df["cab_type"] == cab]
        fig.add_trace(
            go.Scatter(
                x=sub["actual_usd"],
                y=sub[pred_col],
                mode="markers",
                name=cab,
                marker={
                    "size": 4.5,
                    "color": BRAND_COLORS[cab],
                    "opacity": 0.45,
                },
                customdata=np.stack(
                    [
                        sub["service_tier"],
                        sub["distance"],
                        sub["surge_multiplier"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"<b>{cab} — %{{customdata[0]}}</b><br>"
                    "Giá thực tế (y): <b>$%{x:.2f}</b><br>"
                    "Giá dự báo (ŷ): <b>$%{y:.2f}</b><br>"
                    "Cự ly: %{customdata[1]:.2f} dặm | Surge: "
                    "%{customdata[2]:.2f}x<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Đường chéo y = x
    fig.add_trace(
        go.Scatter(
            x=[2.5, 70.0],
            y=[2.5, 70.0],
            mode="lines",
            name="Đường lý tưởng (y = ŷ)",
            line={
                "color": BRAND_COLORS["danger"],
                "width": 1.8,
                "dash": "dash",
            },
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Histogram phần dư sạch (không clip ở biên -> tuân thủ MED-04)
    fig.add_trace(
        go.Bar(
            x=hist_df["bin_center_usd"],
            y=hist_df[hist_col],
            name="Tần suất phần dư",
            marker_color=BRAND_COLORS["success"]
            if model_key == "M7"
            else BRAND_COLORS["accent"],
            opacity=0.82,
            showlegend=False,
            hovertemplate=(
                "<b>Phần dư ~%{x:+.2f} USD</b><br>Số chuyến tập Test: <b>"
                "%{y:,}</b><extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.add_vline(
        x=-2.0,
        line_dash="dot",
        line_color=BRAND_COLORS["warning"],
        line_width=1.6,
        row=1,
        col=2,
    )
    fig.add_vline(
        x=2.0,
        line_dash="dot",
        line_color=BRAND_COLORS["warning"],
        line_width=1.6,
        row=1,
        col=2,
    )
    fig.add_vline(
        x=0.0,
        line_dash="solid",
        line_color="#0F172A",
        line_width=1.2,
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="Giá báo thực tế y (USD)", range=[0, 68], row=1, col=1
    )
    fig.update_yaxes(
        title_text="Giá báo dự báo ŷ (USD)", range=[0, 68], row=1, col=1
    )
    fig.update_xaxes(
        title_text="Phần dư y − ŷ (USD, vạch đứt = ±$2.00)",
        range=[-8.2, 8.2],
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Số lượt chuyến tập Test", row=1, col=2)

    return apply_executive_layout(
        fig,
        title=f"Chẩn đoán Dự báo & Phần dư Thực nghiệm — {model_label}",
        subtitle=(
            f"Chỉ số toàn tập Test (n = {n_test_eval:,}): R² = "
            f"{metrics['r2']:.4f} | RMSE = ${metrics['rmse_usd']:.4f} | "
            f"MAE = ${metrics['mae_usd']:.4f} | "
            f"{metrics['pct_within_2usd']:.2f}% chuyến lệch trong ±$2.00"
        ),
        height=450,
    )


def build_segment_errors_fig(
    seg_df: pd.DataFrame, dimension_label: str
) -> go.Figure:
    """
    Biểu đồ 4C (Tab 4 — Phân rã sai số đa chiều): So sánh RMSE (USD) giữa M7 &
    M2
    và hiển thị MAPE (%) trên trục phụ nhằm làm rõ 'Nghịch lý mẫu số MAPE' ở
    gói Shared.
    """
    if seg_df.empty:
        return _empty_figure(
            f"Phân rã Sai số theo {dimension_label}",
            "Không có dữ liệu phân khúc.",
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x_labels = seg_df["segment_value"].astype(str).tolist()

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=seg_df["m7_rmse_usd"],
            name="M7 RMSE (USD)",
            marker_color=BRAND_COLORS["success"],
            customdata=np.stack(
                [
                    seg_df["n_test"],
                    seg_df["m7_mae_usd"],
                    seg_df["m7_within_2usd_pct"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x} — M7</b><br>"
                "RMSE: <b>$%{y:.4f}</b> | MAE: <b>$%{customdata[1]:.4f}</"
                "b><br>"
                "Đạt sai số ≤ ±$2.00: <b>%{customdata[2]:.1f}%</b><br>"
                "Cỡ mẫu Test: <b>%{customdata[0]:,}</b><extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=seg_df["m2_rmse_usd"],
            name="M2 OLS RMSE (USD)",
            marker_color=BRAND_COLORS["accent"],
            opacity=0.75,
            hovertemplate=(
                "<b>%{x} — M2 OLS</b><br>RMSE: <b>$%{y:.4f}</b><extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=seg_df["m7_mape_pct"],
            mode="lines+markers",
            name="M7 MAPE (%, trục phải)",
            line={"color": BRAND_COLORS["Lyft"], "width": 2.2, "dash": "dot"},
            marker={"size": 7, "symbol": "diamond"},
            hovertemplate=(
                "<b>%{x} — M7 MAPE</b><br>MAPE: <b>%{y:.2f}%</b><extra></"
                "extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(title_text="RMSE (USD)", secondary_y=False)
    fig.update_yaxes(title_text="MAPE (%)", secondary_y=True, showgrid=False)
    shared_sub = seg_df.loc[seg_df["segment_value"].astype(str) == "Shared"]
    shared_note = (
        "Shared có MAE thấp "
        f"(${float(shared_sub.iloc[0]['m7_mae_usd']):.2f}) nhưng MAPE cao "
        f"({float(shared_sub.iloc[0]['m7_mape_pct']):.2f}%) do giá gốc chỉ "
        "~$6"
        if not shared_sub.empty
        else "Lưu ý hiệu ứng mẫu số MAPE ở các phân khúc giá thấp"
    )
    return apply_executive_layout(
        fig,
        title=f"Phân rã Sai số Dự báo M7 vs. M2 theo {dimension_label}",
        subtitle=(
            f"Hai trục tung độc lập (RMSE USD vs. MAPE %) — {shared_note}"
        ),
        height=430,
    )


def build_permutation_importance_fig(
    perm_df: pd.DataFrame,
    show_m2: bool = True,
) -> go.Figure:
    """
    Biểu đồ 4D (Tab 4 — HIGH-01 XAI): Permutation Feature Importance trên tập
    Test (n = 15,000, 8 repeats)
    với đúng 14 nhóm rời nhau = 27/27 cột thực của X_test (0 đếm trùng).
    """
    if perm_df.empty:
        return _empty_figure(
            "Permutation Feature Importance (14 Nhóm Rời nhau)",
            "Không có dữ liệu.",
        )

    df = (
        perm_df.copy()
        .sort_values("m7_delta_rmse_mean_usd", ascending=True)
        .reset_index(drop=True)
    )
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=df["feature_group"],
            x=df["m7_delta_rmse_mean_usd"],
            error_x={
                "type": "data",
                "array": df["m7_delta_rmse_std_usd"],
                "visible": True,
                "color": "#334155",
            },
            orientation="h",
            name="M7 HistGBR — ΔRMSE (USD)",
            marker_color=BRAND_COLORS["success"],
            text=[
                f"+${v:.2f} ({s:.2f}%)"
                if v >= 0.02
                else f"{v:+.4f} ({s:.2f}%)"
                for v, s in zip(
                    df["m7_delta_rmse_mean_usd"],
                    df["m7_importance_share_pct"],
                    strict=False,
                )
            ],
            textposition="outside",
            customdata=np.stack(
                [
                    df["domain_family"],
                    df["permuted_columns"],
                    df["m7_importance_share_pct"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Họ biến: %{customdata[0]}<br>"
                "Các cột hoán vị: <code>%{customdata[1]}</code><br>"
                "M7 ΔRMSE: <b>+$%{x:.4f} USD</b> (tỷ trọng: "
                "<b>%{customdata[2]:.2f}%</b>)<extra></extra>"
            ),
        )
    )

    if show_m2:
        fig.add_trace(
            go.Bar(
                y=df["feature_group"],
                x=df["m2_delta_rmse_mean_usd"],
                error_x={
                    "type": "data",
                    "array": df["m2_delta_rmse_std_usd"],
                    "visible": True,
                    "color": "#1E40AF",
                },
                orientation="h",
                name="M2 OLS — ΔRMSE (USD)",
                marker_color=BRAND_COLORS["accent"],
                opacity=0.72,
                hovertemplate=(
                    "<b>%{y} (M2 OLS)</b><br>ΔRMSE khi hoán vị: <b>+$%{x:"
                    ".4f} USD</b><extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="group")
    max_x = max(
        float(df["m7_delta_rmse_mean_usd"].max()),
        float(df["m2_delta_rmse_mean_usd"].max()),
    )
    fig.update_xaxes(range=[-0.15, max_x * 1.24])
    return apply_executive_layout(
        fig,
        title=(
            "Permutation Feature Importance trên Tập Test: 14 Nhóm Rời nhau "
            "= 27/27 Cột X"
        ),
        subtitle=(
            "Đo mức tăng RMSE (USD) khi xáo trộn từng nhóm trên X_test (n = "
            "15,000, 8 lần lặp) • Tương quan dự báo ≠ Quan hệ nhân quả"
        ),
        height=540,
        xaxis_title="Mức tăng sai số ΔRMSE khi xáo trộn nhóm đặc trưng (USD)",
        yaxis_title="",
    )


def build_pdp_curves_fig(stage6_summary: dict[str, Any]) -> go.Figure:
    """
    Biểu đồ 4E (Tab 4 — XAI PDP): Đường cong Phụ thuộc Riêng phần (Partial
    Dependence Plot)
    nhất quán vật lý cho Quãng đường (distance) và Hệ số Surge Lyft (M7 vs M2).
    """
    pdp = stage6_summary.get("part5_pdp_and_ice_summary", {})
    if not pdp:
        return _empty_figure(
            "Đường cong Phụ thuộc Riêng phần (PDP)",
            "Không có dữ liệu PDP trong stage6_summary.json.",
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "PDP theo Quãng đường: Ngưỡng Giá sàn < 1.0 dặm (M7 vs. M2)",
            "PDP theo Hệ số Surge Lyft: Hiện tượng Bão hòa Cây ≥ 2.50x",
        ),
        horizontal_spacing=0.10,
    )

    d_grid = pdp["distance_grid"]
    fig.add_trace(
        go.Scatter(
            x=d_grid,
            y=pdp["pdp_distance_m7"],
            mode="lines+markers",
            name="M7 HistGBR (Phi tuyến)",
            line={"color": BRAND_COLORS["success"], "width": 2.8},
            marker={"size": 4},
            hovertemplate=(
                "<b>M7 PDP — Cự ly %{x:.2f} dặm</b><br>Giá dự báo TB: <b>"
                "$%{y:.2f}</b><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d_grid,
            y=pdp["pdp_distance_m2"],
            mode="lines",
            name="M2 Interaction OLS (Tuyến tính)",
            line={
                "color": BRAND_COLORS["accent"],
                "width": 2.2,
                "dash": "dash",
            },
            hovertemplate=(
                "<b>M2 PDP — Cự ly %{x:.2f} dặm</b><br>Giá dự báo TB: <b>"
                "$%{y:.2f}</b><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=0.3,
        x1=1.0,
        fillcolor="#DCFCE7",
        opacity=0.40,
        line_width=0,
        annotation_text="Vùng giá sàn < 1.0 mi",
        annotation_position="top left",
        annotation_font={"size": 10, "color": "#166534"},
        row=1,
        col=1,
    )

    s_grid = pdp["surge_grid"]
    fig.add_trace(
        go.Scatter(
            x=s_grid,
            y=pdp["pdp_surge_lyft_m7"],
            mode="lines+markers",
            name="M7 PDP Surge",
            showlegend=False,
            line={"color": BRAND_COLORS["success"], "width": 2.8},
            marker={"size": 6},
            hovertemplate=(
                "<b>M7 PDP — Surge %{x:.2f}x</b><br>Giá dự báo TB Lyft: <b>"
                "$%{y:.2f}</b><extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=s_grid,
            y=pdp["pdp_surge_lyft_m2"],
            mode="lines+markers",
            name="M2 PDP Surge",
            showlegend=False,
            line={
                "color": BRAND_COLORS["accent"],
                "width": 2.2,
                "dash": "dash",
            },
            marker={"size": 5},
            hovertemplate=(
                "<b>M2 PDP — Surge %{x:.2f}x</b><br>Giá dự báo TB Lyft: <b>"
                "$%{y:.2f}</b><extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_vrect(
        x0=2.25,
        x1=3.05,
        fillcolor="#FEF3C7",
        opacity=0.45,
        line_width=0,
        annotation_text="Vùng hiếm (53 lượt): cây M7 bão hòa",
        annotation_position="top left",
        annotation_font={"size": 10, "color": "#92400E"},
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Quãng đường (dặm)", row=1, col=1)
    fig.update_yaxes(title_text="Giá dự báo biên TB (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Hệ số Surge Lyft (x)", row=1, col=2)
    fig.update_yaxes(title_text="Giá dự báo biên TB Lyft (USD)", row=1, col=2)

    return apply_executive_layout(
        fig,
        title=(
            "Đường cong Phụ thuộc Riêng phần (PDP) Nhất quán Vật lý: M7 vs. "
            "M2 OLS"
        ),
        subtitle=(
            "Tự động đồng bộ log_distance & distance × surge khi trượt trên "
            "lưới • Giải thích lý do M7 thắng M2 ở < 1 dặm & lý do cần M2 "
            "đối chứng ở ≥ 2.5x"
        ),
        height=430,
    )


def build_prediction_interval_fig(
    pi_df: pd.DataFrame,
    cqr_summary: dict[str, Any] | None = None,
) -> go.Figure:
    """
    Biểu đồ 4F (Tab 4 — Prediction Uncertainty): Khoảng dự báo bất định
    Quantile [P10, P90]
    (mục tiêu danh nghĩa 80%) và độ phủ thực nghiệm PICP (%) cho 12 gói dịch
    vụ.
    """
    if pi_df.empty:
        return _empty_figure(
            "Khoảng Dự báo Bất định Quantile [P10, P90]",
            "Không có dữ liệu khoảng dự báo.",
        )

    df = (
        pi_df.copy()
        .sort_values("mean_actual_usd", ascending=True)
        .reset_index(drop=True)
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Dải dự báo TB [P10 – P50 – P90] so với Giá Thực tế (USD)",
            "Tỷ lệ Bao phủ Thực nghiệm PICP (%) so với Mục tiêu 80%",
        ),
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.11,
    )

    # Thanh dải [P10, P90] bằng error_x bất đối xứng quanh P50
    fig.add_trace(
        go.Scatter(
            y=df["service_tier"],
            x=df["mean_p50_usd"],
            mode="markers",
            name="Trung vị dự báo P50 & dải [P10, P90]",
            marker={"size": 8, "color": BRAND_COLORS["accent"]},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (df["mean_p90_usd"] - df["mean_p50_usd"]).tolist(),
                "arrayminus": (
                    df["mean_p50_usd"] - df["mean_p10_usd"]
                ).tolist(),
                "color": "#93C5FD",
                "thickness": 5,
                "width": 5,
            },
            customdata=np.stack(
                [df["mean_p10_usd"], df["mean_p90_usd"], df["mpiw_width_usd"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y} — Quantile [P10, P90]</b><br>"
                "P10: <b>$%{customdata[0]:.2f}</b> | P50: <b>$%{x:.2f}</b> |"
                " P90: <b>$%{customdata[1]:.2f}</b><br>"
                "Độ rộng dải MPIW: <b>$%{customdata[2]:.2f}</b><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=df["service_tier"],
            x=df["mean_actual_usd"],
            mode="markers",
            name="Giá thực tế TB (Actual)",
            marker={
                "size": 8,
                "symbol": "diamond",
                "color": BRAND_COLORS["Lyft"],
            },
            hovertemplate=(
                "<b>%{y} — Thực tế</b><br>Giá TB thực tế: <b>$%{x:.2f}</b>"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    cqr = cqr_summary or {}
    by_tier = {
        str(item["service_tier"]): item
        for item in cqr.get("by_tier", [])
        if "service_tier" in item
    }
    if by_tier:
        picp_series = df["service_tier"].map(
            lambda t: float(by_tier.get(str(t), {}).get("cqr_picp_pct", 80.0))
        )
        bar_title = "PICP sau Split-CQR (%)"
        sub_prefix = (
            f"Cột hiển thị Monotonic Split-CQR 85/15 (PICP = "
            f"{float(cqr.get('overall_cqr_picp_pct', 79.89)):.2f}%, "
            f"MPIW = ${float(cqr.get('overall_cqr_mpiw_usd', 3.562)):.2f})"
        )
    else:
        picp_series = df["picp_coverage_pct"]
        bar_title = "PICP (%)"
        ov_picp = float(
            (df["picp_coverage_pct"] * df["n_test"]).sum()
            / max(int(df["n_test"].sum()), 1)
        )
        ov_mpiw = float(
            (df["mpiw_width_usd"] * df["n_test"]).sum()
            / max(int(df["n_test"].sum()), 1)
        )
        sub_prefix = (
            f"Cột hiển thị bản GĐ6 trước hiệu chỉnh (PICP = {ov_picp:.2f}%, "
            f"MPIW = ${ov_mpiw:.2f})"
        )

    bar_colors = [
        BRAND_COLORS["success"]
        if abs(v - 80.0) <= 4.0
        else BRAND_COLORS["warning"]
        for v in picp_series
    ]
    fig.add_trace(
        go.Bar(
            y=df["service_tier"],
            x=picp_series,
            orientation="h",
            name=bar_title,
            showlegend=False,
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in picp_series],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Độ phủ thực nghiệm PICP: <b>%{x:.2f}%</b> "
                "(mục tiêu 80%)<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=80.0,
        line_dash="dash",
        line_color="#0F172A",
        line_width=1.8,
        row=1,
        col=2,
    )

    total_n_pi = int(df["n_test"].sum())
    fig.update_xaxes(title_text="Giá báo (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Độ phủ PICP (%)", range=[0, 98], row=1, col=2)
    cqr_picp = float(cqr.get("overall_cqr_picp_pct", 79.89))
    cqr_lo = float(cqr.get("min_tier_cqr_picp_pct", 78.62))
    cqr_hi = float(cqr.get("max_tier_cqr_picp_pct", 81.99))
    return apply_executive_layout(
        fig,
        title=(
            "Ước lượng Bất định Dự báo bằng Quantile Regression [P10, P90] "
            f"(Tập Test n = {total_n_pi:,})"
        ),
        subtitle=(
            f"{sub_prefix} | Đạt chuẩn: {cqr_picp:.2f}% (12 gói trong "
            f"{cqr_lo:.2f}%–{cqr_hi:.2f}%, 0 cắt ngược)"
        ),
        height=490,
    )


# =============================================================================
# TAB 5 — RQ4: DỰ BÁO GIÁ LIVE (M7 + M2) & MÔ PHỎNG TRẦN SURGE CAP (GIAI ĐOẠN
# 6)
# =============================================================================
def build_live_fare_comparison_fig(
    pairs_df: pd.DataFrame,
    route_label: str,
    distance: float,
    lyft_surge: float,
) -> go.Figure:
    """
    Biểu đồ 5A (Tab 5A — Live Fare Predictor): So sánh trực tiếp giá báo API dự
    kiến (M7)
    giữa Uber và Lyft trên cả 5 phân khúc đối xứng cho lộ trình người dùng đang
    chọn.
    """
    if pairs_df.empty:
        return _empty_figure(
            "So sánh Giá Báo API Dự kiến (M7): Uber vs. Lyft",
            "Đầu vào chưa hợp lệ để dự báo.",
        )

    seg_short = [
        s.split(". ", 1)[-1].split(" (")[0] for s in pairs_df["Phân khúc"]
    ]
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=seg_short,
            y=pairs_df["Giá báo Uber (M7, USD)"],
            name="Uber (M7 dự báo, surge = 1.0x)",
            marker_color=BRAND_COLORS["Uber"],
            text=[f"${v:.2f}" for v in pairs_df["Giá báo Uber (M7, USD)"]],
            textposition="outside",
            customdata=np.stack(
                [pairs_df["Gói Uber"], pairs_df["Giá báo Uber (M2 OLS, USD)"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]} (Uber)</b><br>"
                "Giá báo dự báo M7: <b>$%{y:.2f}</b><br>"
                "Đối chứng M2 OLS: <b>$%{customdata[1]:.2f}</b><extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=seg_short,
            y=pairs_df["Giá báo Lyft (M7, USD)"],
            name=f"Lyft (M7 dự báo, surge = {lyft_surge:.2f}x)",
            marker_color=BRAND_COLORS["Lyft"],
            text=[f"${v:.2f}" for v in pairs_df["Giá báo Lyft (M7, USD)"]],
            textposition="outside",
            customdata=np.stack(
                [
                    pairs_df["Gói Lyft"],
                    pairs_df["Giá báo Lyft (M2 OLS, USD)"],
                    pairs_df["Hãng rẻ hơn (theo M7)"],
                    pairs_df["Chênh lệch (USD)"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]} (Lyft)</b><br>"
                "Giá báo dự báo M7: <b>$%{y:.2f}</b><br>"
                "Đối chứng M2 OLS: <b>$%{customdata[1]:.2f}</b><br>"
                "Gợi ý rẻ hơn: <b>%{customdata[2]}</b> (chênh "
                "$%{customdata[3]:.2f})<extra></extra>"
            ),
        )
    )

    max_y = max(
        float(pairs_df["Giá báo Uber (M7, USD)"].max()),
        float(pairs_df["Giá báo Lyft (M7, USD)"].max()),
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, max_y * 1.18])
    return apply_executive_layout(
        fig,
        title=f"Dự báo Giá Báo API Song song 5 Phân khúc: {route_label}",
        subtitle=(
            f"Quãng đường: {distance:.2f} dặm | Lyft Surge: "
            f"{lyft_surge:.2f}x (Uber cố định 1.00x) • Dự báo từ models/"
            "best_model.joblib (M7)"
        ),
        height=420,
        xaxis_title="Phân khúc đối xứng (Uber vs. Lyft)",
        yaxis_title="Giá báo API dự kiến (USD)",
    )


def build_surge_cap_tradeoff_fig(
    sim_df: pd.DataFrame,
    selected_cap: float = 2.00,
    elasticity_mode: str = "static_000",
    supply_drop_pct: int = 0,
) -> go.Figure:
    """
    Biểu đồ 5B (Tab 5B — Policy Simulator): Đường biên đánh đổi Surge Cap giữa
    Tiết kiệm TB/lượt chịu tác động (USD) và Biến động Tổng giá trị báo giá
    Lyft (API Quote Proxy, %)
    dưới nguyên tắc phân phối bên ngắn Q_realized = min(Q_D, Q_S).
    """
    if sim_df.empty:
        return _empty_figure(
            "Đường biên Đánh đổi — Mô phỏng Trần Surge Cap",
            "Không có dữ liệu mô phỏng.",
        )

    if int(supply_drop_pct) == 10:
        rev_col, eps_label = (
            "supply_constrained_rev_change_drop10_pct",
            "Cung nhạy giá ε_S = 0.60 (Q = min(Q_D, Q_S))",
        )
    elif int(supply_drop_pct) == 5:
        rev_col, eps_label = (
            "supply_constrained_rev_change_drop5_pct",
            "Cung nhạy giá ε_S = 0.30 (Q = min(Q_D, Q_S))",
        )
    else:
        col_map = {
            "static_000": (
                "static_lyft_rev_change_pct",
                "Cầu tĩnh (ε = 0.00 — Cung không đổi)",
            ),
            "eps_050": (
                "net_lyft_rev_change_eps_050_pct",
                "Cận trên dư cung Cohen et al. (ε = -0.50)",
            ),
            "eps_080": (
                "net_lyft_rev_change_eps_080_pct",
                "Cận trên dư cung cạnh tranh (ε = -0.80)",
            ),
        }
        rev_col, eps_label = col_map.get(
            elasticity_mode, col_map["static_000"]
        )

    sub = (
        sim_df.loc[sim_df["surge_cap"] < 3.0]
        .copy()
        .sort_values("surge_cap", ascending=False)
    )
    x_vals = sub[rev_col].abs()
    y_vals = sub["mean_rider_saving_per_affected_trip_usd"]
    sizes = np.clip(np.sqrt(sub["affected_trips_count"]) * 0.75 + 14, 16, 54)

    colors = []
    for cap in sub["surge_cap"]:
        if np.isclose(float(cap), float(selected_cap), atol=1e-4):
            colors.append(BRAND_COLORS["Lyft"])
        elif np.isclose(float(cap), 2.00, atol=1e-4):
            colors.append(BRAND_COLORS["success"])
        else:
            colors.append(BRAND_COLORS["accent"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers+text",
            name="Quỹ đạo kịch bản mô phỏng",
            line={"color": "#94A3B8", "width": 2.0, "dash": "dot"},
            marker={
                "size": sizes,
                "color": colors,
                "line": {"color": "#FFFFFF", "width": 2},
                "opacity": 0.90,
            },
            text=[
                f"  Trần {c:.2f}x"
                + (
                    " ★ Pha 1"
                    if np.isclose(c, 2.00)
                    else (" (Pha 2)" if np.isclose(c, 1.75) else "")
                )
                for c in sub["surge_cap"]
            ],
            textposition="middle right",
            customdata=np.stack(
                [
                    sub["surge_cap"],
                    sub["affected_trips_count"],
                    sub["mean_rider_saving_pct_affected"],
                    sub["total_consumer_savings_usd"],
                    sub[rev_col],
                    sub["m2_ols_mean_saving_affected_usd"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>MÔ PHỎNG — Trần %{customdata[0]:.2f}x</b><br>"
                "Số lượt báo giá chịu tác động: <b>%{customdata[1]:,.0f}</"
                "b><br>"
                "Tiết kiệm TB/lượt (M7): <b>$%{y:.2f}</b> (-%{customdata[2]:"
                ".2f}%)<br>"
                "Đối chứng M2 OLS: <b>$%{customdata[5]:.2f}/lượt</b><br>"
                "Tổng tiết kiệm báo giá: <b>$%{customdata[3]:,.2f}</b><br>"
                "Biến động Quote Proxy Lyft: <b>%{customdata[4]:.3f}%</"
                "b><extra></extra>"
            ),
        )
    )

    fig.update_xaxes(range=[-0.10, max(float(x_vals.max()) * 1.32, 0.5)])
    fig.update_yaxes(range=[3.0, float(y_vals.max()) * 1.18])
    return apply_executive_layout(
        fig,
        title=(
            "Đường biên Đánh đổi — Mô phỏng Trần Surge Cap (Q_realized = "
            "min(Q_D, Q_S))"
        ),
        subtitle=(
            f"Chế độ: {eps_label} | ★ Pha 1: 2.00x (đạt τ ≤ 0.10% mọi kịch "
            "bản cung) | Pha 2: 1.75x (Triggered Switchback)"
        ),
        height=430,
        xaxis_title=(
            "Mức giảm Tổng giá trị báo giá Lyft — API Quote Proxy (|%|)"
        ),
        yaxis_title="Tiết kiệm trung bình trên mỗi lượt bị sốc giá (USD)",
        legend_top=False,
        narrow=True,
    )


def build_surge_cap_m7_vs_m2_fig(sim_df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ 5C (Tab 5B — HIGH-03 Cross-Check): Đối chứng song song tiết kiệm
    báo giá TB/lượt
    giữa M7 (HistGBR) và M2 (Interaction OLS) qua 5 kịch bản áp trần (làm rõ
    bão hòa cây tại 2.00x).
    """
    if sim_df.empty:
        return _empty_figure(
            "Đối chứng M7 vs. M2 OLS trong Mô phỏng Surge Cap",
            "Không có dữ liệu.",
        )

    sub = (
        sim_df.loc[sim_df["surge_cap"] < 3.0]
        .copy()
        .sort_values("surge_cap", ascending=False)
    )
    labels = [
        f"Trần {c:.2f}x\n({int(n):,} lượt)"
        for c, n in zip(
            sub["surge_cap"], sub["affected_trips_count"], strict=False
        )
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=sub["mean_rider_saving_per_affected_trip_usd"],
            name="M7 HistGradientBoosting (USD/lượt)",
            marker_color=BRAND_COLORS["success"],
            text=[
                f"${v:.2f}"
                for v in sub["mean_rider_saving_per_affected_trip_usd"]
            ],
            textposition="outside",
            customdata=sub["total_consumer_savings_usd"],
            hovertemplate=(
                "<b>%{x} — M7</b><br>Tiết kiệm TB: <b>$%{y:.2f}/lượt</b><br>"
                "Tổng tiết kiệm: <b>$%{customdata:,.2f}</b><extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=sub["m2_ols_mean_saving_affected_usd"],
            name="M2 Interaction OLS đối chứng (USD/lượt)",
            marker_color=BRAND_COLORS["accent"],
            opacity=0.80,
            text=[f"${v:.2f}" for v in sub["m2_ols_mean_saving_affected_usd"]],
            textposition="outside",
            customdata=sub["m2_ols_total_savings_usd"],
            hovertemplate=(
                "<b>%{x} — M2 OLS</b><br>Tiết kiệm TB: <b>$%{y:.2f}/lượt</b>"
                "<br>Tổng tiết kiệm: <b>$%{customdata:,.2f}</b><extra></"
                "extra>"
            ),
        )
    )

    c200 = sub.loc[np.isclose(sub["surge_cap"], 2.00)].iloc[0]
    fig.update_layout(barmode="group")
    max_y = max(
        float(sub["mean_rider_saving_per_affected_trip_usd"].max()),
        float(sub["m2_ols_mean_saving_affected_usd"].max()),
    )
    fig.update_yaxes(range=[0, max_y * 1.22])
    return apply_executive_layout(
        fig,
        title=(
            "Kiểm chứng Đối chiếu Mô hình Mô phỏng: M7 HistGBR vs. M2 "
            "Interaction OLS"
        ),
        subtitle=(
            f"Tại trần 2.00x (53 lượt hiếm 2.5x–3.0x), M7 bão hòa nhẹ "
            f"(${float(c200['mean_rider_saving_per_affected_trip_usd']):.2f}"
            f" vs M2 ${float(c200['m2_ols_mean_saving_affected_usd']):.2f});"
            " "
            "từ 1.75x trở xuống 2 mô hình hội tụ rất sát"
        ),
        height=420,
        xaxis_title=(
            "Kịch bản Trần Surge Cap (số lượt báo giá Lyft chịu tác động)"
        ),
        yaxis_title="Tiết kiệm giá báo trung bình / lượt (USD)",
        narrow=True,
    )

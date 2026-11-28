"""
Module: dashboard/components/kpi_cards.py
Mục đích: Thành phần tái sử dụng cho thẻ KPI Executive và khung Insight 5 phần
chuẩn học thuật.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str,
    accent: str = "accent",
    badge_text: str | None = None,
    badge_class: str = "badge-verified",
) -> None:
    """
    Hiển thị 1 thẻ KPI có thanh nhấn màu thương hiệu, số căn tabular và nhãn
    nguồn gốc tùy chọn.
    """
    accent_class_map = {
        "accent": "kpi-accent-top",
        "uber": "kpi-uber-top",
        "lyft": "kpi-lyft-top",
        "emerald": "kpi-emerald-top",
        "amber": "kpi-amber-top",
    }
    top_cls = accent_class_map.get(accent, "kpi-accent-top")
    badge_html = (
        '<div style="margin-top:0.42rem;"><span class="badge-pill '
        f'{badge_class}">{html.escape(badge_text)}</span></div>'
        if badge_text
        else ""
    )
    st.markdown(
        f"""
        <div class="kpi-card {top_cls}">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}</div>
            <div class="kpi-sub">{html.escape(subtitle)}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(
    title: str,
    finding: str,
    evidence: str,
    method: str,
    business_meaning: str,
    limitation: str,
    accent: str = "blue",
    tag_text: str = "VERIFIED",
    tag_class: str = "badge-verified",
) -> None:
    """
    Hiển thị thẻ Insight định lượng đủ 5 phần bắt buộc theo chuẩn học thuật:
    1. Phát hiện | 2. Bằng chứng định lượng | 3. Phương pháp | 4. Ý nghĩa kinh
    doanh | 5. Giới hạn kết luận.
    """
    accent_cls = {
        "blue": "",
        "lyft": "lyft-accent",
        "emerald": "emerald-accent",
        "amber": "amber-accent",
    }.get(accent, "")

    st.markdown(
        f"""
        <div class="insight-card {accent_cls}">
            <div class="insight-title">
                {html.escape(title)}
                <span class="badge-pill {tag_class}"
                      style="margin-left:0.45rem;vertical-align:middle;">
                    {html.escape(tag_text)}
                </span>
            </div>
            <ul class="insight-list">
                <li><b>Phát hiện:</b> {html.escape(finding)}</li>
                <li><b>Bằng chứng định lượng:</b> {html.escape(evidence)}</li>
                <li><b>Phương pháp phân tích:</b> {html.escape(method)}</li>
                <li>
                    <b>Ý nghĩa kinh doanh:</b> {html.escape(business_meaning)}
                </li>
                <li><b>Giới hạn kết luận:</b> {html.escape(limitation)}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(cards: list[dict[str, Any]]) -> None:
    """Hiển thị một hàng nhiều thẻ KPI cân đều cột."""
    cols = st.columns(len(cards), gap="medium")
    for col, card in zip(cols, cards, strict=False):
        with col:
            render_kpi_card(**card)

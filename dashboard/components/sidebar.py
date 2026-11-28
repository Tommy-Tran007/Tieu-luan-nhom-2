"""
Module: dashboard/components/sidebar.py
Mục đích: Bộ lọc toàn cục (Global Filter Panel) ở thanh bên trái cho Dashboard
PRJ-EXT-01.
- Trả về đối tượng FilterParams chuẩn hóa để truyền sang
  dashboard/business_logic.py.
- Cung cấp nút đặt lại bộ lọc nhanh, hiển thị tỷ lệ mẫu đang lọc và ghi chú
  phạm vi tác dụng.
"""

from __future__ import annotations

import streamlit as st

from dashboard.business_logic import FilterParams
from dashboard.config import (
    ALL_TIERS_ORDERED,
    EMPIRICAL_DISTANCE_MAX,
    EMPIRICAL_DISTANCE_MIN,
    VALID_BOSTON_DISTRICTS,
    VALID_CAB_TYPES,
)


def _reset_sidebar_state() -> None:
    """Đưa toàn bộ widget Sidebar về trạng thái toàn mẫu mặc định."""
    st.session_state["flt_cabs"] = list(VALID_CAB_TYPES)
    st.session_state["flt_tiers"] = [tier for _, tier in ALL_TIERS_ORDERED]
    st.session_state["flt_sources"] = []
    st.session_state["flt_dist"] = (
        EMPIRICAL_DISTANCE_MIN,
        EMPIRICAL_DISTANCE_MAX,
    )
    st.session_state["flt_surge"] = (1.0, 3.0)
    st.session_state["flt_surge_only"] = False


def render_sidebar(total_rows: int, ram_mb: float) -> FilterParams:
    """Vẽ thanh lọc toàn cục và trả về FilterParams."""
    # Khởi tạo giá trị mặc định trong session_state ở lần tải đầu tiên
    if "flt_cabs" not in st.session_state:
        _reset_sidebar_state()

    with st.sidebar:
        st.markdown("### 🎛️ Bộ lọc Khám phá Dữ liệu")
        st.caption(
            "Áp dụng động cho **Tab 1 (Tổng quan)**, biểu đồ EDA **Tab 2 "
            "(RQ1)** và điểm đối chiếu **Tab 3 (RQ2)**. "
            "Các kết quả kiểm định thống kê & đánh giá tập Test (`Tab 4`, "
            "`Tab 5`) giữ cố định theo mẫu chuẩn để bảo toàn tính tái lập."
        )

        selected_cabs = st.multiselect(
            "1. Nền tảng gọi xe (`cab_type`)",
            options=list(VALID_CAB_TYPES),
            key="flt_cabs",
        )

        # Gợi ý gói dịch vụ thuộc các hãng đang chọn
        allowed_tiers = [
            tier for cab, tier in ALL_TIERS_ORDERED if cab in selected_cabs
        ]
        # Đồng bộ hóa lựa chọn gói hiện tại với hãng đang chọn nếu người dùng
        # bỏ 1 hãng
        valid_current_tiers = [
            t
            for t in st.session_state.get("flt_tiers", allowed_tiers)
            if t in allowed_tiers
        ]
        if not valid_current_tiers and allowed_tiers:
            valid_current_tiers = allowed_tiers
        st.session_state["flt_tiers"] = valid_current_tiers

        selected_tiers = st.multiselect(
            "2. Phân khúc dịch vụ (`name` — 12 gói)",
            options=allowed_tiers,
            key="flt_tiers",
        )

        selected_sources = st.multiselect(
            "3. Quận điểm đón (`source` — để trống = cả 12 quận)",
            options=list(VALID_BOSTON_DISTRICTS),
            key="flt_sources",
            placeholder="Toàn bộ 12 quận Boston",
        )

        dist_range = st.slider(
            "4. Quãng đường (`distance`, dặm)",
            min_value=float(EMPIRICAL_DISTANCE_MIN),
            max_value=float(EMPIRICAL_DISTANCE_MAX),
            step=0.05,
            key="flt_dist",
        )

        surge_range = st.slider(
            "5. Hệ số nhân Surge (`surge_multiplier`)",
            min_value=1.0,
            max_value=3.0,
            step=0.25,
            key="flt_surge",
            help=(
                "Ở mức mặc định [1.0, 3.0], 38,037 dòng Lyft khuyết surge "
                "(NaN) vẫn được giữ nguyên. Khi thu hẹp dải, chỉ lọc trên "
                "mẫu có surge quan sát thực."
            ),
        )

        surge_only = st.toggle(
            "Chỉ hiển thị lượt có Surge > 1.0x",
            key="flt_surge_only",
            help=(
                "Lưu ý: Trong dữ liệu Boston, chỉ nền tảng Lyft ghi nhận "
                "surge > 1.0 (6,280 lượt trên 92,384 lượt quan sát thực). "
                "Chọn riêng Uber + nút này sẽ trả về 0 dòng."
            ),
        )

        st.button(
            "↺ Đặt lại toàn bộ bộ lọc",
            on_click=_reset_sidebar_state,
            width="stretch",
        )

        st.divider()
        st.markdown("#### 🛡️ Hồ sơ Tệp Dữ liệu")
        st.markdown(
            f"- **Nguồn sạch:** `master_rideshare.parquet`\n"
            f"- **Quy mô gốc:** `{total_rows:,} × 32`\n"
            f"- **Bộ nhớ RAM:** `{ram_mb:.2f} MB` *(tối ưu Parquet)*\n"
            f"- **Múi giờ:** `America/New_York` *(EST)*\n"
            f"- **Nhóm 02:** Trần Thanh Huy & Lê Huy Thắng"
        )

    return FilterParams(
        cab_types=tuple(selected_cabs),
        service_tiers=tuple(selected_tiers),
        sources=tuple(selected_sources) if selected_sources else None,
        distance_min=float(dist_range[0]),
        distance_max=float(dist_range[1]),
        surge_min=float(surge_range[0]),
        surge_max=float(surge_range[1]),
        surge_only=bool(surge_only),
    )

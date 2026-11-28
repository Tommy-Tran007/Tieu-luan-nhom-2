"""
Module: dashboard/components/styles.py
Mục đích: Hệ thống CSS tinh gọn tạo giao diện Executive Light Theme sắc nét,
độ tương phản cao cho máy chiếu thuyết trình và các thẻ KPI / Insight / Badge
chuẩn học thuật.
"""

from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """Chèn bộ CSS nhất quán toàn ứng dụng Streamlit."""
    st.markdown(
        """
        <style>
        /* Khung trang sạch và khoảng đệm tối ưu cho màn hình rộng */
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }
        /* Thanh tiêu đề Hero Banner */
        .hero-banner {
            background: linear-gradient(
                120deg, #0F172A 0%, #1E293B 58%, #3B0764 100%
            );
            color: #F8FAFC;
            padding: 1.35rem 1.75rem;
            border-radius: 14px;
            margin-bottom: 1.15rem;
            border: 1px solid #334155;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.12);
        }
        .hero-title {
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.015em;
            color: #FFFFFF;
        }
        .hero-subtitle {
            font-size: 0.92rem;
            color: #CBD5E1;
            margin: 0;
            line-height: 1.45;
        }
        .hero-badges {
            margin-top: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .badge-pill {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .badge-verified {
            background: #DCFCE7; color: #166534; border: 1px solid #86EFAC;
        }
        .badge-assumption {
            background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D;
        }
        .badge-simulation {
            background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD;
        }
        .badge-dark {
            background: rgba(255,255,255,0.12);
            color: #F1F5F9;
            border: 1px solid rgba(255,255,255,0.22);
        }
        .badge-uber { background: #111827; color: #FFFFFF; }
        .badge-lyft { background: #E11D8A; color: #FFFFFF; }

        /* Thẻ KPI Executive */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            height: 100%;
            transition: border-color 0.15s ease;
        }
        .kpi-card:hover {
            border-color: #94A3B8;
        }
        .kpi-accent-top {
            border-top: 3.5px solid #2563EB;
        }
        .kpi-uber-top {
            border-top: 3.5px solid #111827;
        }
        .kpi-lyft-top {
            border-top: 3.5px solid #E11D8A;
        }
        .kpi-emerald-top {
            border-top: 3.5px solid #059669;
        }
        .kpi-amber-top {
            border-top: 3.5px solid #D97706;
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #64748B;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.52rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
            font-variant-numeric: tabular-nums;
        }
        .kpi-sub {
            font-size: 0.80rem;
            color: #475569;
            margin-top: 0.32rem;
            line-height: 1.35;
        }

        /* Thẻ Insight học thuật 5 ý */
        .insight-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .insight-card.lyft-accent { border-left-color: #E11D8A; }
        .insight-card.emerald-accent { border-left-color: #059669; }
        .insight-card.amber-accent { border-left-color: #D97706; }
        .insight-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.45rem;
        }
        .insight-list {
            margin: 0;
            padding-left: 1.15rem;
            font-size: 0.86rem;
            color: #334155;
            line-height: 1.5;
        }
        .insight-list li {
            margin-bottom: 0.22rem;
        }

        /* Hộp Kiểm toán chất lượng dữ liệu */
        .audit-box {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-top: 0.5rem;
        }
        /* Dải Phạm vi Dataset Executive (Section 1 Tab 1) */
        .exec-scope-bar {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-left: 5px solid #2563EB;
            border-radius: 12px;
            padding: 0.9rem 1.15rem;
            margin-bottom: 0.95rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }
        .exec-scope-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.7rem;
            margin-top: 0.5rem;
            font-size: 0.84rem;
            color: #334155;
        }
        /* Thẻ 3 Phát hiện cốt lõi Executive (Section 3 Tab 1) */
        .finding-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: 4px solid #2563EB;
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            height: 100%;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        .finding-card.f-lyft { border-top-color: #E11D8A; }
        .finding-card.f-emerald { border-top-color: #059669; }
        .finding-num {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 0.25rem;
        }
        .finding-headline {
            font-size: 0.96rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.4rem;
            line-height: 1.3;
        }
        .finding-body {
            font-size: 0.84rem;
            color: #334155;
            line-height: 1.48;
            margin: 0;
        }
        /* Hộp Kết luận Nghiên cứu Ngắn gọn (Section 5 Tab 1) */
        .exec-verdict-box {
            background: linear-gradient(
                135deg, #EFF6FF 0%, #F8FAFC 65%, #FDF2F8 100%
            );
            border: 1px solid #93C5FD;
            border-left: 5px solid #0F172A;
            border-radius: 12px;
            padding: 1.05rem 1.3rem;
            margin-top: 0.65rem;
            margin-bottom: 0.9rem;
            color: #1E293B;
            font-size: 0.90rem;
            line-height: 1.55;
        }
        .section-heading {
            font-size: 1.14rem;
            font-weight: 700;
            color: #0F172A;
            margin: 1.1rem 0 0.55rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 2px solid #E2E8F0;
        }

        /* Làm nổi bật thanh Tab */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            padding-top: 0.6rem !important;
            padding-bottom: 0.6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

"""
Script: scripts/check_thesis_audit.py
Mục đích: Kiểm toán tự động toàn diện Toàn văn Tiểu luận (.qmd & .pdf xuất trực
tiếp
qua Quarto/Typst, kèm bản dự phòng .docx). 100% trị số kỳ vọng được đọc ĐỘNG từ
reports/results/thesis_v5_verification.json, stage5_summary.json,
stage6_summary.json
và các bảng CSV kết quả (không gõ tay số cứng).

Lệnh chạy:
    python scripts/check_thesis_audit.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.thesis_content import (  # noqa: E402
    build_thesis_structure,
    load_verified_data,
)

DEFAULT_STUDENT_ID = "2521001060"
QMD_PATH = ROOT_DIR / "reports" / f"{DEFAULT_STUDENT_ID}_Nhom02_TieuLuan.qmd"
DOCX_PATH = ROOT_DIR / "reports" / f"{DEFAULT_STUDENT_ID}_Nhom02_TieuLuan.docx"
PDF_PATH = ROOT_DIR / "reports" / f"{DEFAULT_STUDENT_ID}_Nhom02_TieuLuan.pdf"
PREVIEW_DIR = ROOT_DIR / "reports" / "preview_pages"


def run_audit(student_id: str = DEFAULT_STUDENT_ID) -> int:
    qmd_path = ROOT_DIR / "reports" / f"{student_id}_Nhom02_TieuLuan.qmd"
    docx_path = ROOT_DIR / "reports" / f"{student_id}_Nhom02_TieuLuan.docx"
    pdf_path = ROOT_DIR / "reports" / f"{student_id}_Nhom02_TieuLuan.pdf"
    assert qmd_path.exists(), f"Thiếu tệp nguồn Quarto: {qmd_path}"
    assert docx_path.exists(), f"Thiếu tệp Word dự phòng: {docx_path}"
    assert pdf_path.exists(), f"Thiếu tệp PDF: {pdf_path}"

    data = load_verified_data()
    content = build_thesis_structure(data)
    v5 = data["v5"]
    s5 = data["s5"]
    tb = data["tables"]

    # 1. Executive Summary word count (250–350)
    es_words = len(content["executive_summary"].split())
    print(f"[CHECK 01] Executive Summary words: {es_words} (yêu cầu 250–350)")
    assert 250 <= es_words <= 350, (
        f"Executive Summary ngoài khoảng [250, 350]: {es_words}"
    )

    # 2. Đọc toàn bộ PDF bằng PyMuPDF & kiểm tra khung 25–45 trang nội dung
    # chính (Ch.1–5)
    pdf = fitz.open(str(pdf_path))
    n_pages = len(pdf)
    page_texts = [page.get_text("text") for page in pdf]
    full_text = "\n".join(page_texts)
    full_norm = re.sub(r"\s+", " ", full_text)

    def _is_heading_page(txt: str, heading_prefix: str) -> bool:
        for ln in txt.splitlines():
            s = ln.strip()
            if s.startswith(heading_prefix) and not re.search(
                r"(?:\.\s*){4,}", s
            ):
                return True
        return False

    ch1_page = next(
        i
        for i, t in enumerate(page_texts, 1)
        if _is_heading_page(t, "CHƯƠNG 1: TỔNG QUAN") and "MỤC LỤC" not in t
    )
    ref_page = next(
        i
        for i, t in enumerate(page_texts, 1)
        if i > ch1_page and _is_heading_page(t, "TÀI LIỆU THAM KHẢO")
    )
    main_chapters_pages = ref_page - ch1_page
    content_pages_excl_cover = n_pages - 2
    print(
        f"[CHECK 02] Tổng số trang PDF/Word: {n_pages} | Nội dung không tính "
        f"bìa: {content_pages_excl_cover}/45 trang | Nội dung chính "
        f"Chương 1–5: {main_chapters_pages} trang"
    )
    assert content_pages_excl_cover <= 45, (
        f"Tổng trang nội dung (trừ 2 bìa) vượt trần 45 trang: "
        f"{content_pages_excl_cover}"
    )
    assert 25 <= main_chapters_pages <= 45, (
        "Số trang nội dung chính Ch.1–5 ngoài khung [25, 45]: "
        f"{main_chapters_pages}"
    )

    ref_split = full_text.split("TÀI LIỆU THAM KHẢO")
    main_body_text = "TÀI LIỆU THAM KHẢO".join(ref_split[:-1])
    main_body_norm = re.sub(r"\s+", " ", main_body_text)

    # 3. Thông tin trang bìa bắt buộc
    cover_all = re.sub(r"\s+", " ", page_texts[0] + "\n" + page_texts[1])
    required_cover = [
        "TRƯỜNG ĐẠI HỌC TÀI CHÍNH",
        "KHOA KHOA HỌC DỮ LIỆU",
        "TRẦN MẠNH TƯỜNG",
        "2631101190501",
        "1",
        "Trần Thanh Huy",
        "2521001060",
        "2",
        "Lê Huy Thắng",
        "2521001092",
        "100%",
        "α₁ = 1.00",
        "α₂ = 1.00",
        "∑αᵢ = 2.00",
        "Tỷ lệ hoàn thành phần việc được giao",
    ]
    missing_cover = [k for k in required_cover if k not in cover_all]
    print(
        "[CHECK 03] Thông tin trang bìa (Pages 1–2): "
        f"{'PASS' if not missing_cover else f'MISSING {missing_cover}'}"
    )
    assert not missing_cover, f"Thiếu thông tin trang bìa: {missing_cover}"

    # 4. Kiểm tra 0 lỗi trường Word (TOC / LOT / LOF / Bookmark)
    bad_field_markers = [
        "Error! Bookmark not defined",
        "No table of contents entries found",
        "No table of figures entries found",
        "[Mục lục tự động sẽ cập nhật",
        "[Danh mục Bảng biểu tự động",
        "[Danh mục Hình vẽ tự động",
    ]
    found_bad_fields = [m for m in bad_field_markers if m in full_text]
    print(
        f"[CHECK 04] Lỗi trường Word/TOC: {len(found_bad_fields)} "
        f"({found_bad_fields})"
    )
    assert not found_bad_fields, (
        f"Phát hiện lỗi trường Word: {found_bad_fields}"
    )

    # 4B. Kiểm tra 0 font Calibri rò rỉ trong PDF (khóa Times New Roman đồng
    # nhất)
    calibri_pages = sorted(
        {
            p_idx
            for p_idx, page in enumerate(pdf, 1)
            for b in page.get_text("dict")["blocks"]
            if b["type"] == 0
            for ln in b["lines"]
            for sp in ln["spans"]
            if sp["text"].strip() and "Calibri" in sp["font"]
        }
    )
    print(
        f"[CHECK 04B] Rò rỉ font Calibri trên PDF: {len(calibri_pages)} "
        f"trang ({calibri_pages})"
    )
    assert not calibri_pages, (
        "Phát hiện tiêu đề/văn bản bị dính Calibri trên trang: "
        f"{calibri_pages}"
    )

    # 4C. Xác minh nguồn gốc Quarto (.qmd -> Typst/LaTeX PDF, không phải
    # Microsoft Word) & Danh mục Bảng/Hình
    pdf_meta = pdf.metadata or {}
    creator_str = (
        f"{pdf_meta.get('creator', '')} {pdf_meta.get('producer', '')}".strip()
    )
    is_quarto_engine = any(
        k in creator_str.lower() for k in ("typst", "tex", "quarto", "pandoc")
    )
    assert "microsoft" not in creator_str.lower() and is_quarto_engine, (
        "PDF không được biên dịch từ .qmd (Quarto/Typst/LaTeX), "
        f"creator={creator_str!r}"
    )
    front_pages = page_texts[: ch1_page - 1]
    lot_start = next(
        idx
        for idx, t in enumerate(front_pages)
        if _is_heading_page(t, "DANH MỤC BẢNG BIỂU") and "MỤC LỤC" not in t
    )
    lof_start = next(
        idx
        for idx, t in enumerate(front_pages)
        if _is_heading_page(t, "DANH MỤC HÌNH VẼ / BIỂU ĐỒ")
        and "MỤC LỤC" not in t
    )
    es_start = next(
        idx
        for idx, t in enumerate(front_pages)
        if _is_heading_page(t, "TÓM TẮT ĐỒ ÁN") and "MỤC LỤC" not in t
    )
    lot_pages = front_pages[lot_start:lof_start]
    lof_pages = front_pages[lof_start:es_start]
    lot_entries = re.findall(
        r"Bảng\s+[A-Za-z0-9]+\.\d+:", "\n".join(lot_pages)
    )
    lof_entries = re.findall(r"Hình\s+\d+\.\d+:", "\n".join(lof_pages))
    print(
        f"[CHECK 04C] Nguồn gốc Quarto (.qmd -> PDF): PASS ({creator_str}) | "
        f"Danh mục Bảng: {len(lot_entries)} mục | Danh mục Hình: "
        f"{len(lof_entries)} mục"
    )
    assert len(lot_entries) >= 18, (
        f"Danh mục Bảng biểu thiếu mục: {len(lot_entries)}"
    )
    assert len(lof_entries) == 17, (
        f"Danh mục Hình vẽ phải đủ 17 biểu đồ, hiện có: {len(lof_entries)}"
    )

    # 4D. Kiểm toán trình bày Typst: 0 trang mồ côi 'Nguồn:', 0 lặp ký hiệu
    # q̂q̂, khóa font mã nguồn Consolas (tránh vỡ dấu tiếng Việt
    # DejaVuSansMono)
    orphan_nguon_pages: list[int] = []
    for p_idx, txt_p in enumerate(page_texts, 1):
        lines_p = [
            ln.strip()
            for ln in txt_p.splitlines()
            if ln.strip() and not ln.strip().startswith("PRJ-EXT-01 —")
        ]
        if lines_p and lines_p[0].startswith("Nguồn:"):
            orphan_nguon_pages.append(p_idx)
    dup_qhat_pages = [
        p_idx
        for p_idx, txt_p in enumerate(page_texts, 1)
        if "q\u0302q\u0302" in txt_p
    ]
    pdf_fonts = sorted(
        {
            sp["font"]
            for page in pdf
            for b in page.get_text("dict")["blocks"]
            if b["type"] == 0
            for ln in b["lines"]
            for sp in ln["spans"]
            if sp["text"].strip()
        }
    )
    dejavu_leaks = (
        [f for f in pdf_fonts if "DejaVuSansMono" in f]
        if sys.platform == "win32"
        else []
    )
    print(
        "[CHECK 04D] Chất lượng dàn trang Quarto/Typst: mồ côi "
        f"Nguồn={len(orphan_nguon_pages)} "
        f"| lặp q̂={len(dup_qhat_pages)} | lỗi font mã tiếng "
        f"Việt={len(dejavu_leaks)}"
    )
    assert not orphan_nguon_pages, (
        "Phát hiện chú thích 'Nguồn:' bị tách mồ côi đầu trang: "
        f"{orphan_nguon_pages}"
    )
    assert not dup_qhat_pages, (
        f"Phát hiện lặp ký tự q̂q̂ trên trang: {dup_qhat_pages}"
    )
    assert not dejavu_leaks, (
        f"Khối mã dùng DejaVuSansMono làm vỡ dấu tiếng Việt: {dejavu_leaks}"
    )

    # 5. Chặn toàn bộ chuỗi lỗi cũ v3/v4 và thuật ngữ cấm
    forbidden_global = [
        "Chấp nhận H0",
        "+$-",
        "+$ -",
        "Raghavendra",
        "Stratified 80/20",
        "Doanh thu Tĩnh",
        "Doanh thu Ròng",
        "doanh thu ròng Lyft",
        "Cước phí",
        "cước phí",
        "~89 MB",
        "Northeastern, Boston Univ",
        "0.9531",
        "$2.0213",
        "56.53",
        "130,468",
        "3,354 cụm",
        "thesis_v4_verification.py",
        "thesis_v3_addons.py",
        "thesis_supplementary_stats.py",
        "m2_ols.joblib",
        "build_stage3..6",
        "Cohen 2016",
        "audit_v4",
        "00_nhật_ký",
        "dashboard/docs",
    ]
    forbidden_pct_standalone = ["7.87%", "6.86%", "7.52%"]
    apb_split = full_text.split("PHỤ LỤC B:")
    pre_apb_norm = re.sub(r"\s+", " ", "PHỤ LỤC B:".join(apb_split[:-1]))
    found_global = [s for s in forbidden_global if s in pre_apb_norm]
    found_global.extend(
        s
        for s in forbidden_pct_standalone
        if re.search(rf"(?<!\d){re.escape(s)}", pre_apb_norm)
    )
    print(
        "[CHECK 05] Chuỗi cấm / số liệu v4 cũ (Cover–Phụ lục A): "
        f"{len(found_global)} ({found_global})"
    )
    assert not found_global, f"Phát hiện chuỗi lỗi cũ: {found_global}"

    # 6. Kiểm tra 0 mã kiểm toán nội bộ trong phần Thân bài chính (Tóm tắt +
    # Chương 1–5)
    internal_jargon_body = [
        "CRIT-01",
        "CRIT-02",
        "CRIT-03",
        "HIGH-01",
        "HIGH-02",
        "HIGH-03",
        "MED-03",
        "GAP-MODEL-01",
        "R3a",
        "R3b",
        "bảo vệ Vòng 3",
        "Final Audit",
    ]
    found_jargon = [s for s in internal_jargon_body if s in main_body_norm]
    v5_leaks_body = re.findall(r"\bv5\b|thesis_v5", main_body_norm, flags=re.I)
    bracket_cites = re.findall(
        r"\[[A-Z][A-Za-z &.,;]+(?:19|20)\d{2}\]", main_body_norm
    )
    raw_md_hits = re.findall(r"\*\*|## \[", full_text)
    print(
        f"[CHECK 06] Mã kiểm toán nội bộ Thân bài: {len(found_jargon)} | "
        f"Rò nhãn nháp v5 (Cover–Ch.5): {len(v5_leaks_body)} | "
        f"Ngoặc vuông APA: {len(bracket_cites)} | Raw MD PDF: "
        f"{len(raw_md_hits)}"
    )
    assert not found_jargon, (
        f"Phát hiện mã kiểm toán nội bộ lọt vào Thân bài: {found_jargon}"
    )
    assert not v5_leaks_body, (
        f"Phát hiện rò nhãn nháp v5 trong Cover–Ch.5: {v5_leaks_body}"
    )
    assert not bracket_cites, (
        f"Phát hiện trích dẫn ngoặc vuông trái APA 7th: {bracket_cites}"
    )
    assert not raw_md_hits, (
        f"Phát hiện cú pháp Markdown thô lọt ra PDF: {raw_md_hits[:5]}"
    )

    # 7. Kiểm tra ĐỘNG 100% trị số cốt lõi v5 đọc từ JSON/CSV
    m7 = (
        tb["test"]
        .loc[tb["test"]["model_code"] == "M7_HistGradientBoosting"]
        .iloc[0]
    )
    m5_p = s5["part2_hyperparameter_tuning"]["M5_DecisionTree"]["best_params"]
    prov = v5["waterfall_and_selection_bias"]["master_provenance_breakdown"]
    st = v5["statistical_rigor_extensions"]
    cqr = v5["model_diagnostics_and_cqr"]["clean_monotonic_cqr"]
    sp = v5["spatial_fairness_and_surge_eda"]
    io_s = v5["policy_rq4_supply_power_and_io"][
        "io_benchmark_fair_same_table"
    ]["same_table_master_benchmark"]
    xl_sb = v5["crossing_distances_5_pairs"][2]["spec_b_train80_nonsurge"][
        "hc3"
    ]["d_star"]
    ux_sb = v5["crossing_distances_5_pairs"][1]["spec_b_train80_nonsurge"][
        "hc3"
    ]["d_star"]
    ex_all = next(
        r
        for r in s5["part3_ex_ante_experiments"]["pricing_without_surge"]
        if r["model_or_slice"] == "M7_ExAnte_All_Test"
    )
    ex_auc = s5["part3_ex_ante_experiments"]["lyft_surge_classifier"][
        "HistGB_Classifier"
    ]["roc_auc"]

    required_v5 = [
        f"{prov['total_master_rows']:,}",
        f"{prov['is_surge_original_1_total']:,}",
        f"{prov['lyft_surge_missing_nan']:,}",
        f"{prov['weather_valid_rain_obs']:,}",
        f"{sp['lyft_overall_surge_rate_pct']:.2f}%",
        f"{st['n_loc_hr_clusters_all']:,}",
        f"{st['n_citywide_hour_clusters_all']}",
        f"{st['heavy_rain_vs_dry']['exact_wcr_loc_hr_B1999']['p_wild']:.4f}",
        f"{st['h3b_lyft_surge_rain_cluster']['p_hour_cluster']:.4f}",
        f"{float(m7['test_r2']):.4f}",
        f"${float(m7['test_rmse']):.4f}",
        f"{float(m7['test_mape_pct']):.2f}%",
        f"min_samples_leaf = {m5_p['min_samples_leaf']}",
        f"min_samples_split = {m5_p['min_samples_split']}",
        f"{ex_all['r2']:.4f}",
        f"{ex_auc:.4f}",
        f"{cqr['overall_cqr_picp_pct']:.2f}%",
        f"{cqr['min_tier_cqr_picp_pct']:.2f}%",
        f"{cqr['max_tier_cqr_picp_pct']:.2f}%",
        f"{ux_sb:.2f}",
        f"{xl_sb:.2f}",
        f"{io_s['parquet_vs_csv_full_speedup']:.1f}",
        "$"
        f"{v5['policy_rq4_supply_power_and_io']['lyft_observed_baseline_quote_proxy_usd']:,.2f}",
        "min(Q_D, Q_S)",
        "Triggered",
        "thesis_v5_verification.py",
        "m2_interaction_ols.joblib",
        "18/18",
    ]
    missing_v5 = [u for u in required_v5 if u not in full_norm]
    print(
        f"[CHECK 07] {len(required_v5)} trị số v5 đọc động từ JSON/CSV: "
        f"{'PASS' if not missing_v5 else f'MISSING {missing_v5}'}"
    )
    assert not missing_v5, f"Thiếu trị số v5 bắt buộc: {missing_v5}"

    # 8. Khung tiểu mục projects.pdf Mục 4.2
    required_headings = [
        "LỜI CAM ĐOAN VÀ LỜI CẢM ƠN",
        "MỤC LỤC",
        "DANH MỤC CHỮ VIẾT TẮT",
        "DANH MỤC BẢNG BIỂU",
        "DANH MỤC HÌNH VẼ / BIỂU ĐỒ",
        "TÓM TẮT ĐỒ ÁN",
        "CHƯƠNG 1:",
        "1.1.",
        "1.2.",
        "1.3.",
        "1.4.",
        "1.5.",
        "CHƯƠNG 2:",
        "2.1.",
        "2.2.",
        "2.3.",
        "2.4.",
        "2.5.",
        "CHƯƠNG 3:",
        "3.1.",
        "3.2.",
        "3.3.",
        "3.4.",
        "3.5.",
        "3.6.",
        "CHƯƠNG 4:",
        "4.1.",
        "4.2.",
        "4.3.",
        "4.4.",
        "4.5.",
        "4.6.",
        "CHƯƠNG 5:",
        "5.1.",
        "5.2.",
        "5.3.",
        "5.4.",
        "TÀI LIỆU THAM KHẢO",
        "PHỤ LỤC A:",
        "PHỤ LỤC B:",
    ]
    missing_h = [h for h in required_headings if h not in full_text]
    print(
        f"[CHECK 08] Khung cấu trúc projects.pdf ({len(required_headings)} "
        f"mục): {'PASS' if not missing_h else f'MISSING {missing_h}'}"
    )
    assert not missing_h, f"Thiếu mục cấu trúc: {missing_h}"

    # 9. [M5 UPGRADE] Kiểm tra trang thưa / trang mồ côi trên MỌI TRANG (trừ
    # trang bìa & mục lục)
    empty_pages = []
    sparse_body_pages = []
    for idx, page in enumerate(pdf, 1):
        txt = page.get_text("text").strip()
        imgs = len(page.get_images())
        # Loại bỏ header/footer khi đo độ đầy nội dung trang
        body_lines = [
            ln.strip()
            for ln in txt.splitlines()
            if ln.strip()
            and not ln.strip().startswith("PRJ-EXT-01 —")
            and not ln.strip().startswith("Nhóm 02 — Lớp 25DDS01")
        ]
        body_chars = sum(len(ln) for ln in body_lines)
        if body_chars < 120 and imgs == 0:
            empty_pages.append((idx, body_chars, len(body_lines)))
        elif idx >= ch1_page and imgs == 0 and body_chars < 380:
            sparse_body_pages.append(
                (
                    idx,
                    body_chars,
                    len(body_lines),
                    body_lines[0][:50] if body_lines else "",
                )
            )
    print(
        f"[CHECK 09A] Trang trống (<120 ký tự, 0 hình): {len(empty_pages)} "
        f"({empty_pages})"
    )
    print(
        "[CHECK 09B] Trang thưa/mồ côi phần nội dung & phụ lục (<380 ký tự,"
        f" 0 hình): {len(sparse_body_pages)} ({sparse_body_pages})"
    )
    assert not empty_pages, f"Phát hiện trang trống: {empty_pages}"
    assert not sparse_body_pages, (
        f"Phát hiện trang thưa/mồ côi giữa/cuối bài: {sparse_body_pages}"
    )

    # 9C. Khóa cứng 17/17 biểu đồ luận văn trong PDF đạt thực tế >= 300 DPI
    # (không bị Word nén 200 DPI)
    expected_figs = sum(
        1
        for ch in content["chapters"]
        for sec in ch["sections"]
        for btype, _ in sec["blocks"]
        if btype == "figure"
    )
    fig_dpi_records: list[tuple[int, float, float, int, int]] = []
    for idx in range(ch1_page, n_pages + 1):
        for info in pdf[idx - 1].get_image_info(xrefs=True):
            x0, y0, x1, y1 = info["bbox"]
            eff_x = info["width"] / ((x1 - x0) / 72.0)
            eff_y = info["height"] / ((y1 - y0) / 72.0)
            fig_dpi_records.append(
                (idx, eff_x, eff_y, int(info["width"]), int(info["height"]))
            )
    low_dpi_figs = [r for r in fig_dpi_records if min(r[1], r[2]) < 300.0]
    min_eff_dpi = min((min(r[1], r[2]) for r in fig_dpi_records), default=0.0)
    max_eff_dpi = max((min(r[1], r[2]) for r in fig_dpi_records), default=0.0)
    print(
        "[CHECK 09C] Biểu đồ PDF >= 300 DPI: "
        f"{len(fig_dpi_records) - len(low_dpi_figs)}/{expected_figs} "
        f"(min = {min_eff_dpi:.1f} DPI, max = {max_eff_dpi:.1f} DPI)"
    )
    assert len(fig_dpi_records) == expected_figs, (
        f"Số biểu đồ trong thân bài PDF ({len(fig_dpi_records)}) != kỳ vọng "
        f"({expected_figs})"
    )
    assert not low_dpi_figs, (
        f"Phát hiện biểu đồ PDF dưới 300 DPI: {low_dpi_figs}"
    )

    # 10. Kiểm tra cú pháp Python Phụ lục A (main.py, config.py, 4 trích đoạn)
    # + đối soát nguyên văn Phụ lục B
    compile(content["appendix_a_main_code"], "main.py", "exec")
    compile(content["appendix_a_config_code"], "src/config.py", "exec")
    compile(content["appendix_a_code"], "<appendix_a_code>", "exec")
    assert (
        content["appendix_a_main_code"]
        == (ROOT_DIR / "main.py").read_text(encoding="utf-8").strip()
    )
    assert (
        content["appendix_a_config_code"]
        == (ROOT_DIR / "src/config.py").read_text(encoding="utf-8").strip()
    )
    assert content["appendix_b_verbatim"] in data["worklog_text"], (
        "Phụ lục B (B.3) không khớp nguyên văn ./nhật ký làm việc.md"
    )
    code_parity_pairs = [
        (
            "src/clean_raw_data.py",
            'df["is_surge_original"] = '
            'df["surge_multiplier"].notna().astype(np.int8)',
        ),
        (
            "src/stage3_pipeline_and_audit.py",
            'master_df["is_rain_sentinel"] = np.where(',
        ),
        (
            "src/stage3_pipeline_and_audit.py",
            '(master_df["weather_matched"] == 1) & master_df["rain"].notna(),',
        ),
        (
            "src/simulation.py",
            "affected_mask, np.minimum(q_demand_mid, q_supply_mod), 1.0",
        ),
        (
            "scripts/thesis_v5_verification.py",
            "cal_lo = np.minimum(m10.predict(X_cal), m50.predict(X_cal))",
        ),
    ]
    parity_misses = []
    for rel_f, needle in code_parity_pairs:
        src_txt = (ROOT_DIR / rel_f).read_text(encoding="utf-8")
        if needle not in src_txt or needle not in content["appendix_a_code"]:
            parity_misses.append((rel_f, needle))
    print(
        "[CHECK 10] Phụ lục A compile PASS & đối soát khớp code thật "
        f"({len(code_parity_pairs)}/{len(code_parity_pairs)}): "
        f"{'PASS' if not parity_misses else f'MISS {parity_misses}'}"
    )
    assert not parity_misses, f"Phụ lục A lệch mã nguồn thật: {parity_misses}"

    # 11. Trích dẫn APA 7th: 23/23 tài liệu được trích dẫn 2 chiều
    expected_in_text = [
        "Benjamini",
        "Breiman",
        "brllrb",
        "Cachon",
        "Cameron",
        "Castillo",
        "Chapman",
        "Chen",
        "Cohen",
        "Duan",
        "Friedman",
        "Hall",
        "Holm",
        "Ke",
        "Koenker",
        "Lakens",
        "MacKinnon",
        "McKinney",
        "Pedregosa",
        "Rochet",
        "Romano",
        "Schuirmann",
        "Seabold",
    ]
    assert "Weyl, E. G." in full_norm, "Thiếu tên chuẩn APA 7th 'Weyl, E. G.'"
    assert "Bassett, G., Jr." in full_norm, (
        "Thiếu hậu tố chuẩn APA 7th 'Bassett, G., Jr.'"
    )
    missing_cites = [c for c in expected_in_text if c not in main_body_text]
    print(
        "[CHECK 11] Đối soát 23/23 trích dẫn APA 7th trong thân bài: "
        f"{'PASS' if not missing_cites else f'MISSING {missing_cites}'}"
    )
    assert not missing_cites, (
        f"Tài liệu tham khảo chưa được trích dẫn trong bài: {missing_cites}"
    )

    # 12. Nhất quán dấu thập phân & Gatekeeper mã nguồn
    comma_pct_hits = re.findall(r"\d+,\d+%", main_body_text)
    assert not comma_pct_hits, (
        f"Phát hiện trộn dấu phẩy thập phân: {comma_pct_hits[:10]}"
    )
    hardcoded_hits: list[str] = []
    korean_hits: list[str] = []
    for folder in (
        ROOT_DIR / "src",
        ROOT_DIR / "scripts",
        ROOT_DIR / "dashboard",
    ):
        for pyf in folder.rglob("*.py"):
            txt_py = pyf.read_text(encoding="utf-8")
            if re.search(r"[A-Za-z]:[/\\]Users[/\\]", txt_py):
                hardcoded_hits.append(str(pyf.relative_to(ROOT_DIR)))
            if re.search(r"[\uac00-\ud7af]", txt_py):
                korean_hits.append(str(pyf.relative_to(ROOT_DIR)))
    print(
        "[CHECK 12] Dấu thập phân: 0 lỗi | Gatekeeper Hardcoded Paths: "
        f"{len(hardcoded_hits)} | Ký tự lạ: {len(korean_hits)}"
    )
    assert not hardcoded_hits, (
        f"Phát hiện đường dẫn cứng trong mã nguồn: {hardcoded_hits}"
    )
    assert not korean_hits, f"Phát hiện ký tự lạ trong mã nguồn: {korean_hits}"

    # 13. In bản đồ phân trang & xuất ảnh xem trước các trang trọng yếu
    print("\n=== BẢN ĐỒ PHÂN TRANG & ĐỘ ĐẦY TỪNG TRANG ===")
    key_pages: list[int] = [1, 2, 3, 4, ch1_page, ref_page, n_pages]
    for idx, page in enumerate(pdf, 1):
        txt = page.get_text("text")
        imgs = len(page.get_images())
        lines = [
            ln.strip()
            for ln in txt.splitlines()
            if ln.strip()
            and not ln.strip().startswith("PRJ-EXT-01 —")
            and not ln.strip().startswith("Nhóm 02 —")
        ]
        chars = sum(len(ln) for ln in lines)
        head = lines[0][:62] if lines else "(trống)"
        if (
            idx <= 10
            or chars < 700
            or any(
                k in txt
                for k in (
                    "CHƯƠNG ",
                    "TÀI LIỆU THAM KHẢO",
                    "PHỤ LỤC A:",
                    "PHỤ LỤC B:",
                    "Bảng 4.4:",
                    "Bảng 5.1:",
                )
            )
        ):
            print(
                f"  PDF p.{idx:2d} | chars={chars:4d} | imgs={imgs} | {head}"
            )
        if any(
            k in txt
            for k in (
                "TÓM TẮT ĐỒ ÁN",
                "Bảng 4.4:",
                "Bảng 4.8:",
                "Bảng 5.1:",
                "PHỤ LỤC A:",
                "PHỤ LỤC B:",
            )
        ):
            key_pages.append(idx)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in PREVIEW_DIR.glob("page_*.png"):
        old_png.unlink()
    for pno in sorted(set(key_pages))[:20]:
        pix = pdf[pno - 1].get_pixmap(dpi=130)
        pix.save(str(PREVIEW_DIR / f"page_{pno:02d}.png"))
    print(
        f"\n=== KET LUAN KIEM TOAN TIEU LUAN V5: 100% PASS ({n_pages} trang,"
        f" Ch.1-5 = {main_chapters_pages} trang) ==="
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm toán tự động toàn diện Toàn văn Tiểu luận v5."
    )
    parser.add_argument(
        "--student-id",
        default=DEFAULT_STUDENT_ID,
        help="MSSV đại diện cho tên tệp cần kiểm toán (mặc định: 2521001060).",
    )
    args = parser.parse_args()
    raise SystemExit(run_audit(student_id=args.student_id))

"""
Script: scripts/build_final_thesis.py
Mục đích: Tạo tự động Toàn văn Tiểu luận chuẩn Quarto Markdown (.qmd), biên
dịch PDF (.pdf)
trực tiếp từ .qmd qua Quarto/Typst và đồng bộ bản dự phòng Word (.docx) theo
mẫu trình bày
Đại học Tài chính – Marketing (UFM) và quy chuẩn projects.pdf (Mục 4.1–4.2) từ
cùng một nguồn
sự thật duy nhất (scripts/thesis_content.py).

Sản phẩm đầu ra:
- reports/2521001060_Nhom02_TieuLuan.qmd   (Bản nguồn Quarto Markdown chuẩn
  UFM, biên dịch trực tiếp PDF/HTML/DOCX)
- reports/2521001060_Nhom02_TieuLuan.pdf   (Bản PDF chính thức xuất trực tiếp
  từ .qmd qua Quarto/Typst, ảnh 300+ DPI)
- reports/2521001060_Nhom02_TieuLuan.docx  (Bản Word dự phòng đồng bộ, Mục
  lục/Danh mục Bảng/Hình cập nhật tự động)

Lệnh chạy:
    python scripts/build_final_thesis.py            # Xuất .qmd, biên dịch .pdf
    bằng Quarto và đồng bộ .docx
    python scripts/build_final_thesis.py --skip-docx # Chỉ xuất .qmd và biên
    dịch .pdf bằng Quarto
    python scripts/build_final_thesis.py --qmd-only # Chỉ xuất/cập nhật nhanh
    bản .qmd
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.thesis_content import (  # noqa: E402
    COVER_META,
    build_thesis_structure,
    load_verified_data,
)

DEFAULT_STUDENT_ID = "2521001060"
DOCX_OUT = ROOT_DIR / "reports" / f"{DEFAULT_STUDENT_ID}_Nhom02_TieuLuan.docx"
PDF_OUT = ROOT_DIR / "reports" / f"{DEFAULT_STUDENT_ID}_Nhom02_TieuLuan.pdf"
QMD_OUT = ROOT_DIR / "reports" / f"{DEFAULT_STUDENT_ID}_Nhom02_TieuLuan.qmd"
LOGO_PATH = ROOT_DIR / "reports" / "assets" / "ufm_logo.jpeg"

UFM_NAVY = RGBColor(0x1B, 0x36, 0x5D)
TEXT_BLACK = RGBColor(0x00, 0x00, 0x00)
SLATE_MUTED = RGBColor(0x47, 0x55, 0x69)


# Nguồn văn bản dùng chung 1 nơi duy nhất cho cả .qmd/.pdf và .docx (tránh lặp
# trùng)
def _subcover_deliverables(wm: dict[str, Any]) -> tuple[str, ...]:
    n_sess = int(wm["sessions_count"])
    n_src = int(wm["n_src_files"])
    n_scripts = int(wm["n_script_files"])
    return (
        f"Mã nguồn Python PEP 8 (≤ 79 ký tự): main.py, {n_src} module lõi ./"
        "src/, "
        f"{n_scripts} script xuất bản–kiểm chứng ./scripts/ và 4 notebook ./"
        "notebooks/",
        "Dashboard tương tác 5 Tab: dashboard/app.py (Streamlit) & reports/"
        "dashboard_interactive.html",
        f"Nhật ký làm việc P-D-M-V nhóm ({n_sess} phiên, phân công 50/50): "
        "./nhật ký làm việc.md",
    )


DECLARATION_PARAS: tuple[str, ...] = (
    "Nhóm 02 (Trần Thanh Huy — MSSV: 2521001060 và Lê Huy Thắng — MSSV: "
    "2521001092, Lớp 25DDS01) "
    "xin cam đoan bài tiểu luận kết thúc học phần Lập trình Python cho Khoa "
    "học Dữ liệu với đề tài "
    '"Phân tích Biểu hiện Định giá Động (Dynamic Surge Pricing) và Dự báo '
    "Giá báo API trên "
    'Nền tảng Gọi xe Công nghệ (Uber & Lyft Boston, MA)" là công trình thực '
    "hiện nghiêm túc của nhóm dưới "
    "sự hướng dẫn của ThS. Trần Mạnh Tường.",
    "Toàn bộ dữ liệu gốc (cab_rides.csv và weather.csv) được giữ nguyên vẹn "
    "chỉ-đọc và xác minh toàn vẹn "
    "qua mã băm SHA-256; mọi bảng số liệu, kiểm định thống kê, mô hình học "
    "máy, biểu đồ 300 DPI và Dashboard "
    "đều được tạo trực tiếp từ mã nguồn Python đính kèm trong kho lưu trữ. "
    "Quá trình sử dụng công cụ hỗ trợ "
    "lập trình tuân thủ nguyên tắc Human-in-the-Loop và quy trình P-D-M-V, "
    "được lưu vết trong nhật ký làm "
    "việc của nhóm. Nhóm xin chịu trách nhiệm học thuật về tính trung thực "
    "của toàn bộ nội dung báo cáo.",
)

THANKS_PARAS: tuple[str, ...] = (
    "Nhóm xin gửi lời cảm ơn chân thành đến ThS. Trần Mạnh Tường — Giảng "
    "viên giảng dạy học phần "
    "Lập trình Python cho Khoa học Dữ liệu tại Trường Đại học Tài chính – "
    "Marketing — đã truyền đạt những "
    "kiến thức nền tảng về quy trình CRISP-DM, kỹ thuật lập trình Python "
    "cho dữ liệu lớn và phương pháp "
    "P-D-M-V. Những yêu cầu khắt khe của thầy về khả năng tái lập, kiểm thử "
    "phá vỡ và kỷ luật kiểm toán "
    "dữ liệu đã giúp nhóm nhận diện và khắc phục nhiều hạn chế phương pháp "
    "trong quá trình nghiên cứu.",
    "Dù đã nỗ lực rà soát kỹ lưỡng và bổ sung nhiều kiểm định độ bền, bài "
    "tiểu luận chắc chắn vẫn còn "
    "những giới hạn nhất định về phạm vi dữ liệu và mô hình hóa. Nhóm rất "
    "mong nhận được ý kiến nhận xét "
    "và phản biện của giảng viên để tiếp tục hoàn thiện năng lực nghiên cứu "
    "khoa học dữ liệu.",
)

KEYWORDS_TEXT: str = (
    "Định giá động (Dynamic Surge Pricing); Giá báo API Uber và Lyft; Biểu "
    "giá cắt nhau; "
    "Kiểm định tương đương TOST; Sai số chuẩn cụm; Exact WCR Wild Cluster "
    "Bootstrap; HistGradientBoosting; "
    "Grouped Permutation Importance; Monotonic Split-CQR; Trần Surge Cap; "
    "API Quote Proxy."
)


def _appendix_a_intro(wm: dict[str, Any]) -> str:
    sid = wm.get("student_id", DEFAULT_STUDENT_ID)
    return (
        "Toàn bộ 100% mã nguồn Python chuẩn PEP 8 (ruff check 0 lỗi ở "
        "ngưỡng ≤ 79 ký tự trên toàn bộ "
        f"{wm['n_py_files']} tệp .py và 4 notebook .ipynb; gồm "
        f"{wm['total_src_loc']:,} dòng lõi pipeline "
        f"main.py + {wm['n_src_files']} module src/, tách độc lập "
        f"{wm['scripts_only_loc']:,} dòng ở "
        f"{wm['n_script_files']} script xuất bản–kiểm chứng scripts/, kèm "
        "Dashboard 5 Tab và "
        f"{wm['n_test_files']} tệp pytest) được bàn giao nguyên vẹn trong "
        f"gói nộp bài {sid}_Nhom02_TieuLuan/. Để tuân thủ chặt giới hạn "
        "tối đa 45 trang nội dung ngoài 2 trang bìa (bao gồm mở đầu, 5 "
        "chương chính, TLTK và Phụ lục A–B), Phụ lục A trình bày "
        "sơ đồ kiến trúc đầy đủ, bảng chỉ mục toàn bộ module kèm số dòng đo "
        "động, toàn văn nguyên bản "
        "main.py & src/config.py và bốn trích đoạn thuật toán lõi đối soát "
        "trực tiếp với mã nguồn thực thi."
    )


def _appendix_b_intro(wm: dict[str, Any]) -> str:
    n_sess = int(wm["sessions_count"])
    n_huy = int(wm["n_huy"])
    n_thang = int(wm["n_thang"])
    n_joint = int(wm["n_joint"])
    return (
        f"Nhật ký làm việc nhóm tại ./nhật ký làm việc.md ghi nhận {n_sess} "
        "phiên P-D-M-V "
        f"({wm['lines']:,} dòng, {wm['words']:,} từ) từ 01/10/2026 đến 28/"
        "11/2026 với phân công tổng thể 50/50 "
        f"({n_huy} phiên độc lập Trần Thanh Huy, {n_thang} phiên độc lập "
        f"Lê Huy Thắng và {n_joint} phiên đồng thực hiện; xem quy chiếu mã "
        "kiểm toán tại Bảng B.1 và tóm tắt tiến độ tại Bảng B.2)."
    )


# =============================================================================
# 1. CÁC HÀM TRỢ GIÚP OPENXML CHO WORD CHUẨN HỌC THUẬT
# =============================================================================
def _set_run_font(
    run: Any,
    font_name: str = "Times New Roman",
    size_pt: float = 13.0,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = TEXT_BLACK,
) -> None:
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    rFonts.set(qn("w:cs"), font_name)
    rFonts.set(qn("w:eastAsia"), font_name)


def _configure_section_a4(section: Any, with_border: bool = False) -> None:
    """
    Thiết lập khổ A4 (210x297 mm) và lề chuẩn projects.pdf: Trên 2, Dưới 2,
    Trái 3, Phải 2 cm.
    """
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.0)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)

    sectPr = section._sectPr
    old_borders = sectPr.find(qn("w:pgBorders"))
    if old_borders is not None:
        sectPr.remove(old_borders)
    if with_border:
        pgBorders = parse_xml(
            f'<w:pgBorders {nsdecls("w")} w:offsetFrom="text">'
            '  <w:top w:val="double" w:sz="12" w:space="18" w:color="1B365D"/>'
            '  <w:left w:val="double" w:sz="12" w:space="18" w:'
            'color="1B365D"/>'
            '  <w:bottom w:val="double" w:sz="12" w:space="18" w:'
            'color="1B365D"/>'
            '  <w:right w:val="double" w:sz="12" w:space="18" w:'
            'color="1B365D"/>'
            "</w:pgBorders>"
        )
        sectPr.append(pgBorders)


def _set_page_number_format(
    section: Any, fmt: str = "decimal", start: int = 1
) -> None:
    """
    Định dạng kiểu số trang (lowerRoman / decimal) và trang bắt đầu cho từng
    Section.
    """
    sectPr = section._sectPr
    old_num = sectPr.find(qn("w:pgNumType"))
    if old_num is not None:
        sectPr.remove(old_num)
    pgNum = OxmlElement("w:pgNumType")
    pgNum.set(qn("w:fmt"), fmt)
    pgNum.set(qn("w:start"), str(start))
    sectPr.append(pgNum)


def _add_field(
    paragraph: Any,
    instr: str,
    placeholder: str = "",
    font_size: float = 12.0,
    bold: bool = False,
) -> None:
    """
    Chèn Word Field chuẩn (PAGE, TOC, ...) để Microsoft Word tự cập nhật số
    trang.
    """
    r_begin = paragraph.add_run()
    _set_run_font(r_begin, size_pt=font_size, bold=bold)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    r_begin._r.append(fld_begin)

    r_instr = paragraph.add_run()
    _set_run_font(r_instr, size_pt=font_size, bold=bold)
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = f" {instr} "
    r_instr._r.append(instr_el)

    r_sep = paragraph.add_run()
    _set_run_font(r_sep, size_pt=font_size, bold=bold)
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    r_sep._r.append(fld_sep)

    if placeholder:
        r_val = paragraph.add_run(placeholder)
        _set_run_font(r_val, size_pt=font_size, bold=bold, color=SLATE_MUTED)

    r_end = paragraph.add_run()
    _set_run_font(r_end, size_pt=font_size, bold=bold)
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    r_end._r.append(fld_end)


def _set_cell_shading(cell: Any, fill_hex: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def _set_cell_margins(
    cell: Any, top: int = 60, bottom: int = 60, left: int = 85, right: int = 85
) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f"<w:tcMar {nsdecls('w')}>"
        f'  <w:top w:w="{top}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'  <w:left w:w="{left}" w:type="dxa"/>'
        f'  <w:right w:w="{right}" w:type="dxa"/>'
        "</w:tcMar>"
    )
    tcPr.append(tcMar)


def _set_table_borders(
    table: Any, color: str = "94A3B8", sz: str = "4"
) -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f"<w:tblBorders {nsdecls('w')}>"
        f'  <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:'
        f'color="{color}"/>'
        f'  <w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:right w:val="single" w:sz="{sz}" w:space="0" w:'
        f'color="{color}"/>'
        f'  <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:'
        f'color="{color}"/>'
        f'  <w:insideV w:val="single" w:sz="{sz}" w:space="0" w:'
        f'color="{color}"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)


def _set_row_header_and_nosplit(row: Any, is_header: bool = False) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
    if is_header:
        trPr.append(OxmlElement("w:tblHeader"))


def _add_p_bottom_border(
    paragraph: Any, color: str = "CBD5E1", sz: str = "4"
) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = parse_xml(
        f"<w:pBdr {nsdecls('w')}>"
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="3" w:'
        f'color="{color}"/>'
        "</w:pBdr>"
    )
    pPr.append(pBdr)


def _add_p_top_border(
    paragraph: Any, color: str = "CBD5E1", sz: str = "4"
) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = parse_xml(
        f"<w:pBdr {nsdecls('w')}>"
        f'  <w:top w:val="single" w:sz="{sz}" w:space="3" w:color="{color}"/>'
        "</w:pBdr>"
    )
    pPr.append(pBdr)


def _force_style_tnr(style: Any) -> None:
    """
    Ép tuyệt đối font Times New Roman cho style và xóa thuộc tính theme (tránh
    Calibri-Bold).
    """
    style.font.name = "Times New Roman"
    rPr = style.element.get_or_add_rPr()
    old_rf = rPr.find(qn("w:rFonts"))
    if old_rf is not None:
        rPr.remove(old_rf)
    rFonts = OxmlElement("w:rFonts")
    for k in ("ascii", "hAnsi", "cs", "eastAsia"):
        rFonts.set(qn(f"w:{k}"), "Times New Roman")
    rPr.insert(0, rFonts)


def _strip_auto_redefine(style: Any) -> None:
    """
    Xóa <w:autoRedefine/> trên style TOC để Word không phá vỡ căn trái và giãn
    dòng.
    """
    auto_red = style.element.find(qn("w:autoRedefine"))
    if auto_red is not None:
        style.element.remove(auto_red)


def _add_heading(
    doc: Document, text: str, level: int = 1, page_break_before: bool = False
) -> Any:
    """
    Chèn Heading 1/2 giữ nguyên outline TOC nhưng khóa cứng Times New Roman
    Bold (0 Calibri).
    """
    h = doc.add_heading("", level=level)
    if page_break_before:
        h.paragraph_format.page_break_before = True
    r = h.add_run(text)
    _set_run_font(
        r,
        font_name="Times New Roman",
        size_pt=14.0 if level == 1 else 13.0,
        bold=True,
        italic=(level >= 3),
        color=UFM_NAVY if level <= 2 else TEXT_BLACK,
    )
    return h


# =============================================================================
# 2. KHỞI TẠO STYLES CHUẨN UFM & PROJECTS.PDF
# =============================================================================
def _configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    _force_style_tnr(normal)
    normal.font.size = Pt(12)
    normal.font.color.rgb = TEXT_BLACK
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    h1 = doc.styles["Heading 1"]
    _force_style_tnr(h1)
    h1.font.size = Pt(14)
    h1.font.bold = True
    h1.font.color.rgb = UFM_NAVY
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.line_spacing = 1.35
    h1.paragraph_format.space_before = Pt(4)
    h1.paragraph_format.space_after = Pt(8)
    h1.paragraph_format.keep_with_next = True
    h1.paragraph_format.first_line_indent = Cm(0)

    h2 = doc.styles["Heading 2"]
    _force_style_tnr(h2)
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = UFM_NAVY
    h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h2.paragraph_format.line_spacing = 1.35
    h2.paragraph_format.space_before = Pt(7)
    h2.paragraph_format.space_after = Pt(3.5)
    h2.paragraph_format.keep_with_next = True
    h2.paragraph_format.first_line_indent = Cm(0)

    h3 = doc.styles["Heading 3"]
    _force_style_tnr(h3)
    h3.font.size = Pt(13)
    h3.font.bold = True
    h3.font.italic = True
    h3.font.color.rgb = TEXT_BLACK
    h3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h3.paragraph_format.line_spacing = 1.35
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(4)
    h3.paragraph_format.keep_with_next = True
    h3.paragraph_format.first_line_indent = Cm(0)

    tcap = doc.styles.add_style("TableCaption", 1)
    tcap.base_style = normal
    _force_style_tnr(tcap)
    tcap.font.size = Pt(12)
    tcap.font.bold = True
    tcap.font.color.rgb = TEXT_BLACK
    tcap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    tcap.paragraph_format.line_spacing = 1.25
    tcap.paragraph_format.space_before = Pt(8)
    tcap.paragraph_format.space_after = Pt(4)
    tcap.paragraph_format.first_line_indent = Cm(0)
    tcap.paragraph_format.keep_with_next = True

    fcap = doc.styles.add_style("FigureCaption", 1)
    fcap.base_style = normal
    _force_style_tnr(fcap)
    fcap.font.size = Pt(11.5)
    fcap.font.bold = False
    fcap.font.italic = True
    fcap.font.color.rgb = TEXT_BLACK
    fcap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fcap.paragraph_format.line_spacing = 1.25
    fcap.paragraph_format.space_before = Pt(3)
    fcap.paragraph_format.space_after = Pt(8)
    fcap.paragraph_format.first_line_indent = Cm(0)

    for toc_name in ("TOC 1", "TOC 2", "TOC 3"):
        if toc_name in doc.styles:
            ts = doc.styles[toc_name]
            _force_style_tnr(ts)
            _strip_auto_redefine(ts)
            ts.font.size = Pt(12)
            ts.font.bold = toc_name == "TOC 1"
            ts.font.color.rgb = TEXT_BLACK
            ts.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            ts.paragraph_format.line_spacing = 1.22
            ts.paragraph_format.space_before = Pt(1.5)
            ts.paragraph_format.space_after = Pt(2)


# =============================================================================
# 3. CÁC HÀM DỰNG THÀNH PHẦN NỘI DUNG (ĐOẠN VĂN, CÔNG THỨC, BẢNG, HÌNH, CODE)
# =============================================================================
def _add_body_p(doc: Document, text: str, indent: bool = True) -> Any:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(1.27) if indent else Cm(0)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6.0)
    r = p.add_run(text)
    _set_run_font(r, size_pt=12.0, bold=False)
    return p


def _add_formula_box(
    doc: Document, tokens: list[tuple[str, bool, str | None]]
) -> None:
    """
    Hiển thị công thức toán học căn giữa dùng chỉ số dưới/trên chuẩn Word (0
    lỗi glyph PDF).
    """
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.3
    for txt, is_italic, script in tokens:
        r = p.add_run(txt)
        _set_run_font(r, size_pt=13.0, italic=is_italic, color=UFM_NAVY)
        if script == "sub":
            r.font.subscript = True
        elif script == "sup":
            r.font.superscript = True


def _add_bullet_list(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.27)
        p.paragraph_format.first_line_indent = Cm(-0.45)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6.0)

        rb_dot = p.add_run("–  ")
        _set_run_font(rb_dot, size_pt=12.0, bold=True)
        if ":" in item and item.index(":") < 95:
            lead, rest = item.split(":", 1)
            r1 = p.add_run(lead + ":")
            _set_run_font(r1, size_pt=12.0, bold=True)
            r2 = p.add_run(rest)
            _set_run_font(r2, size_pt=12.0, bold=False)
        else:
            r = p.add_run(item)
            _set_run_font(r, size_pt=12.0, bold=False)


def _add_academic_table(
    doc: Document,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    aligns: list[str] | None = None,
    source: str = "",
    font_size: float = 9.8,
    compact: bool = False,
    keep_together: bool = False,
) -> None:
    """
    Dựng bảng học thuật chuẩn projects.pdf: tên bảng PHÍA TRÊN, nguồn PHÍA
    DƯỚI.
    """
    if caption:
        pc = doc.add_paragraph(style="TableCaption")
        if compact:
            pc.paragraph_format.space_before = Pt(4)
            pc.paragraph_format.space_after = Pt(2)
        r = pc.add_run(caption)
        _set_run_font(r, size_pt=12.0, bold=True)

    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    _set_table_borders(tbl, color="94A3B8", sz="4")

    align_map = {
        "L": WD_ALIGN_PARAGRAPH.LEFT,
        "C": WD_ALIGN_PARAGRAPH.CENTER,
        "R": WD_ALIGN_PARAGRAPH.RIGHT,
    }
    col_aligns = aligns or ["L"] * len(headers)

    # Giảm cỡ chữ nhẹ cho bảng nhiều cột (7–9 cột) để hiển thị sắc nét, không
    # bị gãy dòng xấu
    eff_fs = font_size if len(headers) <= 6 else min(font_size, 9.0)
    hdr_pad_v = 38 if compact else 65
    row_pad_v = 26 if compact else 50
    cell_line_sp = 1.08 if compact else 1.15

    hdr_row = tbl.rows[0]
    _set_row_header_and_nosplit(hdr_row, is_header=True)
    for j, htxt in enumerate(headers):
        cell = hdr_row.cells[j]
        cell.width = Cm(col_widths[j])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_shading(cell, "1B365D")
        _set_cell_margins(
            cell, top=hdr_pad_v, bottom=hdr_pad_v, left=60, right=60
        )
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = cell_line_sp
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(str(htxt))
        _set_run_font(
            r, size_pt=eff_fs, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF)
        )

    for i, rdata in enumerate(rows):
        row = tbl.rows[i + 1]
        _set_row_header_and_nosplit(row, is_header=False)
        fill = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        pin_row = keep_together and (i < len(rows) - 1)
        for j, val in enumerate(rdata):
            cell = row.cells[j]
            cell.width = Cm(col_widths[j])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_shading(cell, fill)
            _set_cell_margins(
                cell, top=row_pad_v, bottom=row_pad_v, left=60, right=60
            )
            p = cell.paragraphs[0]
            p.alignment = align_map.get(col_aligns[j], WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = cell_line_sp
            p.paragraph_format.first_line_indent = Cm(0)
            if pin_row:
                p.paragraph_format.keep_with_next = True
            r = p.add_run(str(val))
            _set_run_font(
                r, size_pt=eff_fs, bold=(j == 0 and len(headers) <= 5)
            )

    if source:
        ps = doc.add_paragraph()
        ps.alignment = WD_ALIGN_PARAGRAPH.LEFT
        ps.paragraph_format.first_line_indent = Cm(0)
        ps.paragraph_format.space_before = Pt(2 if compact else 3)
        ps.paragraph_format.space_after = Pt(4 if compact else 8)
        ps.paragraph_format.line_spacing = 1.25 if compact else 1.3
        r = ps.add_run(source)
        _set_run_font(
            r,
            size_pt=10.0 if compact else 10.5,
            italic=True,
            color=SLATE_MUTED,
        )
    else:
        sp = doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(4 if compact else 6)


def _add_figure(
    doc: Document, rel_path: str, caption: str, width_cm: float = 14.5
) -> None:
    """
    Chèn biểu đồ 300 DPI PNG kích thước rộng sắc nét, cân bằng số trang nội
    dung chuẩn 25–45 trang.
    """
    full_path = ROOT_DIR / rel_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy biểu đồ bắt buộc: {full_path}"
        )

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.first_line_indent = Cm(0)
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(str(full_path), width=Cm(width_cm))

    p_cap = doc.add_paragraph(style="FigureCaption")
    r_cap = p_cap.add_run(caption)
    _set_run_font(r_cap, size_pt=11.5, italic=True)


def _add_code_box(
    doc: Document,
    title: str,
    code_text: str,
    compact: bool = False,
) -> None:
    pt = doc.add_paragraph()
    pt.paragraph_format.first_line_indent = Cm(0)
    pt.paragraph_format.space_before = Pt(3 if compact else 6)
    pt.paragraph_format.space_after = Pt(1.5 if compact else 3)
    pt.paragraph_format.keep_with_next = True
    rt = pt.add_run(title)
    _set_run_font(
        rt, size_pt=10.5 if compact else 11.5, bold=True, color=UFM_NAVY
    )

    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    _set_table_borders(tbl, color="CBD5E1", sz="4")
    cell = tbl.rows[0].cells[0]
    cell.width = Cm(16.0)
    _set_cell_shading(cell, "F8FAFC")
    pad_v = 26 if compact else 90
    _set_cell_margins(cell, top=pad_v, bottom=pad_v, left=95, right=85)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.line_spacing = 1.0 if compact else 1.14
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(code_text)
    _set_run_font(
        r,
        font_name="Consolas",
        size_pt=7.0 if compact else 8.0,
        color=RGBColor(0x0F, 0x17, 0x2A),
    )

    if not compact:
        sp = doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(6)


def _add_inline_md_runs_docx(
    paragraph: Any,
    text: str,
    size_pt: float = 9.0,
) -> None:
    """
    Chuyển chuỗi Markdown nội dòng (**đậm**, *nghiêng*, `mã`, [nhãn](url)) sang
    Word runs.
    """
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")
    pos = 0
    for m in pattern.finditer(clean):
        if m.start() > pos:
            r = paragraph.add_run(clean[pos : m.start()])
            _set_run_font(r, size_pt=size_pt)
        tok = m.group(0)
        if tok.startswith("**") and tok.endswith("**"):
            inner = tok[2:-2].replace("`", "")
            r = paragraph.add_run(inner)
            _set_run_font(r, size_pt=size_pt, bold=True)
        elif tok.startswith("`") and tok.endswith("`"):
            r = paragraph.add_run(tok[1:-1])
            _set_run_font(
                r, font_name="Consolas", size_pt=max(7.5, size_pt - 0.8)
            )
        elif tok.startswith("*") and tok.endswith("*"):
            inner = tok[1:-1].replace("`", "")
            r = paragraph.add_run(inner)
            _set_run_font(r, size_pt=size_pt, italic=True)
        pos = m.end()
    if pos < len(clean):
        r = paragraph.add_run(clean[pos:])
        _set_run_font(r, size_pt=size_pt)


def _add_worklog_session_docx(doc: Document, title: str, md_text: str) -> None:
    """
    Hiển thị 100% nội dung một phiên nhật ký Markdown sang định dạng Word chuẩn
    (0 raw **).
    """
    pt = doc.add_paragraph()
    pt.paragraph_format.first_line_indent = Cm(0)
    pt.paragraph_format.space_before = Pt(6)
    pt.paragraph_format.space_after = Pt(3)
    pt.paragraph_format.keep_with_next = True
    _set_run_font(pt.add_run(title), size_pt=10.8, bold=True, color=UFM_NAVY)

    raw_lines = md_text.splitlines()
    idx = 0
    while idx < len(raw_lines):
        ln = raw_lines[idx]
        stripped = ln.strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            tbl_lines: list[str] = []
            while idx < len(raw_lines) and raw_lines[idx].strip().startswith(
                "|"
            ):
                tbl_lines.append(raw_lines[idx].strip())
                idx += 1
            parsed = [
                [c.strip() for c in row.strip("|").split("|")]
                for row in tbl_lines
                if not re.fullmatch(r"[\s|:\-]+", row)
            ]
            if len(parsed) >= 2:
                headers = [re.sub(r"[`*]", "", h) for h in parsed[0]]
                rows = [
                    [re.sub(r"[`*]", "", c) for c in r[: len(headers)]]
                    for r in parsed[1:]
                ]
                ncols = len(headers)
                if ncols == 4:
                    cw = [2.2, 4.4, 4.7, 4.7]
                elif ncols == 5:
                    cw = [1.8, 3.6, 4.3, 5.0, 1.3]
                else:
                    cw = [16.0 / ncols] * ncols
                _add_academic_table(
                    doc,
                    caption=(
                        "Bảng B.3: Đối soát trước và sau khắc phục các mã "
                        "kiểm toán (trích xuất nguyên văn Phiên 12)"
                    ),
                    headers=headers,
                    rows=rows,
                    col_widths=cw,
                    aligns=["L"] * ncols,
                    source=(
                        "Nguồn: Trích xuất nguyên văn Phiên 12 từ "
                        "./nhật ký làm việc.md."
                    ),
                    font_size=7.8,
                    compact=True,
                )
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.line_spacing = 1.22
        if stripped.startswith("## "):
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(4)
            _add_inline_md_runs_docx(
                p, f"**{stripped[3:].strip()}**", size_pt=9.6
            )
        elif stripped.startswith(("- ", "* ")):
            indent_lv = (len(ln) - len(ln.lstrip(" "))) // 2
            p.paragraph_format.left_indent = Cm(0.45 * (indent_lv + 1))
            p.paragraph_format.first_line_indent = Cm(-0.30)
            bullet = (
                "• " if indent_lv == 0 else ("– " if indent_lv == 1 else "▪ ")
            )
            _set_run_font(p.add_run(bullet), size_pt=8.8, bold=True)
            _add_inline_md_runs_docx(p, stripped[2:].strip(), size_pt=8.8)
        else:
            p.paragraph_format.first_line_indent = Cm(0)
            _add_inline_md_runs_docx(p, stripped, size_pt=8.8)
        idx += 1


# =============================================================================
# 4. DỰNG SECTION 1 (TRANG BÌA CHÍNH + TRANG THÔNG TIN NHÓM TINH GỌN UFM)
# =============================================================================
def _build_cover_pages(doc: Document, wm: dict[str, Any]) -> None:
    sec1 = doc.sections[0]
    _configure_section_a4(sec1, with_border=True)

    def _center_p(
        text: str,
        size: float = 13.0,
        bold: bool = False,
        color: RGBColor = TEXT_BLACK,
        before: float = 0,
        after: float = 6,
        line_sp: float = 1.35,
    ) -> Any:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(before)
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.line_spacing = line_sp
        r = p.add_run(text)
        _set_run_font(r, size_pt=size, bold=bold, color=color)
        return p

    # --- TRANG 1: BÌA CHÍNH ---
    _center_p(
        COVER_META["university"],
        size=14.5,
        bold=True,
        color=UFM_NAVY,
        before=8,
        after=4,
    )
    _center_p(
        COVER_META["faculty"],
        size=13.5,
        bold=True,
        color=UFM_NAVY,
        before=0,
        after=16,
    )

    if LOGO_PATH.exists():
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.first_line_indent = Cm(0)
        p_logo.paragraph_format.space_before = Pt(6)
        p_logo.paragraph_format.space_after = Pt(24)
        p_logo.add_run().add_picture(str(LOGO_PATH), width=Cm(3.6))

    _center_p(
        COVER_META["course_line"], size=15.0, bold=True, before=6, after=6
    )
    _center_p(
        COVER_META["course_name"],
        size=16.0,
        bold=True,
        color=UFM_NAVY,
        before=0,
        after=26,
    )

    _center_p("ĐỀ TÀI:", size=13.5, bold=True, before=6, after=6)
    _center_p(
        COVER_META["topic_title"],
        size=14.2,
        bold=True,
        color=UFM_NAVY,
        before=0,
        after=32,
        line_sp=1.45,
    )

    _center_p(
        f"GIẢNG VIÊN GIẢNG DẠY HỌC PHẦN: {COVER_META['lecturer']}",
        size=13.0,
        bold=True,
        before=10,
        after=8,
    )
    _center_p(
        f"MÃ LỚP HỌC PHẦN: {COVER_META['class_code']}",
        size=13.0,
        bold=True,
        before=0,
        after=8,
    )
    _center_p(
        f"NHÓM THỰC HIỆN: {COVER_META['group_name']} — LỚP "
        f"{COVER_META['student_class']}",
        size=13.0,
        bold=True,
        before=0,
        after=8,
    )
    _center_p(
        f"MÃ ĐỀ TÀI: {COVER_META['topic_code']}",
        size=12.5,
        bold=False,
        color=SLATE_MUTED,
        before=0,
        after=64,
    )

    _center_p(
        COVER_META["location_year"], size=13.0, bold=True, before=28, after=0
    )

    # --- TRANG 2: BÌA PHỤ & BẢNG PHÂN CÔNG NHÓM (TINH GỌN, KHÔNG LẶP THỪA, RÕ
    # TỶ LỆ HOÀN THÀNH) ---
    doc.add_page_break()

    _center_p(
        "THÔNG TIN NHÓM THỰC HIỆN VÀ PHÂN CÔNG NHIỆM VỤ",
        size=14.5,
        bold=True,
        color=UFM_NAVY,
        before=10,
        after=14,
    )
    _center_p(
        f"Học phần: {COVER_META['course_name']}  |  Mã lớp HP: "
        f"{COVER_META['class_code']}",
        size=11.0,
        bold=True,
        before=0,
        after=4,
    )
    _center_p(
        f"Giảng viên hướng dẫn: {COVER_META['lecturer']}  |  "
        f"{COVER_META['group_name']} (Lớp {COVER_META['student_class']})",
        size=11.5,
        bold=False,
        color=SLATE_MUTED,
        before=0,
        after=14,
    )

    member_rows = [
        [
            m["stt"],
            m["name"],
            m["mssv"],
            f"{m['completion']}\n({m['alpha_label']})",
            m["role"],
            "(Ký xác nhận)",
        ]
        for m in COVER_META["members"]
    ]
    _add_academic_table(
        doc,
        caption="",
        headers=[
            "STT",
            "HỌ VÀ TÊN",
            "MSSV",
            "Tỷ lệ hoàn thành phần việc được giao",
            "NỘI DUNG PHỤ TRÁCH CHÍNH",
            "Chữ ký",
        ],
        rows=member_rows,
        col_widths=[1.0, 2.9, 2.2, 2.3, 5.6, 2.0],
        aligns=["C", "L", "C", "C", "L", "C"],
        source=(
            "Ghi chú: Cột 'Tỷ lệ hoàn thành phần việc được giao' thể hiện "
            "mức độ hoàn tất nhiệm vụ "
            "phân công (không phải tỷ trọng chia điểm); hệ số đóng góp cá "
            "nhân theo Mục 5.4 đề cương là "
            "α₁ = 1.00, α₂ = 1.00 (∑αᵢ\u00a0=\u00a02.00; phân công 50/50 "
            f"qua {wm['sessions_count']} phiên P-D-M-V)."
        ),
        font_size=10.5,
    )

    p_repo = doc.add_paragraph()
    p_repo.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_repo.paragraph_format.first_line_indent = Cm(0)
    p_repo.paragraph_format.space_before = Pt(10)
    p_repo.paragraph_format.space_after = Pt(6)
    p_repo.paragraph_format.line_spacing = 1.35
    _set_run_font(
        p_repo.add_run("Hồ sơ mã nguồn và sản phẩm bàn giao đi kèm:\n"),
        size_pt=12.0,
        bold=True,
        color=UFM_NAVY,
    )
    _set_run_font(
        p_repo.add_run(
            "\n".join(f"– {item}" for item in _subcover_deliverables(wm))
        ),
        size_pt=11.0,
        color=TEXT_BLACK,
    )

    _center_p(
        COVER_META["location_year"], size=13.0, bold=True, before=38, after=0
    )


# =============================================================================
# 5. DỰNG SECTION 2 (PHẦN MỞ ĐẦU — ĐÁNH SỐ TRANG LA MÃ i, ii, iii...)
# =============================================================================
def _setup_footer_roman(section: Any) -> None:
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    ftr = section.footer
    p = ftr.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    _add_field(p, "PAGE", placeholder="i", font_size=12.0)


def _build_front_matter(doc: Document, content: dict[str, Any]) -> None:
    sec2 = doc.add_section(WD_SECTION.NEW_PAGE)
    _configure_section_a4(sec2, with_border=False)
    _set_page_number_format(sec2, fmt="lowerRoman", start=1)
    _setup_footer_roman(sec2)

    # 1. LỜI CAM ĐOAN & LỜI CẢM ƠN (vừa khít trang i)
    _add_heading(doc, "LỜI CAM ĐOAN VÀ LỜI CẢM ƠN", level=1)

    def _add_front_p(text: str) -> Any:
        p = _add_body_p(doc, text, indent=True)
        p.paragraph_format.line_spacing = 1.32
        p.paragraph_format.space_after = Pt(6)
        return p

    p_sub1 = doc.add_paragraph()
    p_sub1.paragraph_format.first_line_indent = Cm(0)
    p_sub1.paragraph_format.space_before = Pt(2)
    p_sub1.paragraph_format.space_after = Pt(4)
    _set_run_font(
        p_sub1.add_run("LỜI CAM ĐOAN"), size_pt=13.0, bold=True, color=UFM_NAVY
    )

    for para_txt in DECLARATION_PARAS:
        _add_front_p(para_txt)

    p_sub2 = doc.add_paragraph()
    p_sub2.paragraph_format.first_line_indent = Cm(0)
    p_sub2.paragraph_format.space_before = Pt(8)
    p_sub2.paragraph_format.space_after = Pt(4)
    _set_run_font(
        p_sub2.add_run("LỜI CẢM ƠN"), size_pt=13.0, bold=True, color=UFM_NAVY
    )

    for para_txt in THANKS_PARAS:
        _add_front_p(para_txt)

    p_sig = doc.add_paragraph()
    p_sig.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_sig.paragraph_format.space_before = Pt(8)
    p_sig.paragraph_format.space_after = Pt(0)
    p_sig.paragraph_format.line_spacing = 1.3
    _set_run_font(
        p_sig.add_run("TP. Hồ Chí Minh, tháng 11 năm 2026\n"),
        size_pt=12.5,
        italic=True,
    )
    _set_run_font(
        p_sig.add_run("Đại diện Nhóm 02 thực hiện\n"), size_pt=12.5, bold=True
    )
    _set_run_font(
        p_sig.add_run("Trần Thanh Huy — Lê Huy Thắng"),
        size_pt=12.5,
        bold=False,
    )

    # 2. MỤC LỤC TỰ ĐỘNG
    p_toc_title = doc.add_paragraph()
    p_toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_toc_title.paragraph_format.first_line_indent = Cm(0)
    p_toc_title.paragraph_format.page_break_before = True
    p_toc_title.paragraph_format.space_before = Pt(0)
    p_toc_title.paragraph_format.space_after = Pt(6)
    _set_run_font(
        p_toc_title.add_run("MỤC LỤC"), size_pt=14.0, bold=True, color=UFM_NAVY
    )

    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.first_line_indent = Cm(0)
    _add_field(
        p_toc,
        'TOC \\o "1-3" \\h \\z \\u',
        placeholder=(
            "[Mục lục tự động sẽ cập nhật số trang bởi Microsoft Word]"
        ),
        font_size=11.5,
    )

    # 3. DANH MỤC CHỮ VIẾT TẮT
    _add_heading(doc, "DANH MỤC CHỮ VIẾT TẮT", level=1, page_break_before=True)
    _add_academic_table(
        doc,
        caption="",
        headers=["Viết tắt", "Tiếng Việt", "Tiếng Anh"],
        rows=content["abbreviations"],
        col_widths=[2.4, 6.6, 7.0],
        aligns=["C", "L", "L"],
        source="",
        font_size=11.0,
        compact=True,
    )

    # 4. DANH MỤC BẢNG BIỂU
    _add_heading(doc, "DANH MỤC BẢNG BIỂU", level=1, page_break_before=True)
    p_lot = doc.add_paragraph()
    p_lot.paragraph_format.first_line_indent = Cm(0)
    _add_field(
        p_lot,
        'TOC \\h \\z \\t "TableCaption,1"',
        placeholder="[Danh mục Bảng biểu tự động cập nhật bởi Microsoft Word]",
        font_size=11.5,
    )

    # 5. DANH MỤC HÌNH VẼ / BIỂU ĐỒ
    _add_heading(
        doc, "DANH MỤC HÌNH VẼ / BIỂU ĐỒ", level=1, page_break_before=True
    )
    p_lof = doc.add_paragraph()
    p_lof.paragraph_format.first_line_indent = Cm(0)
    _add_field(
        p_lof,
        'TOC \\h \\z \\t "FigureCaption,1"',
        placeholder="[Danh mục Hình vẽ tự động cập nhật bởi Microsoft Word]",
        font_size=11.5,
    )

    # 6. TÓM TẮT ĐỒ ÁN (EXECUTIVE SUMMARY — ĐÚNG 250–350 TỪ)
    _add_heading(
        doc,
        "TÓM TẮT ĐỒ ÁN (EXECUTIVE SUMMARY)",
        level=1,
        page_break_before=True,
    )
    _add_body_p(doc, content["executive_summary"], indent=True)

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.first_line_indent = Cm(0)
    p_kw.paragraph_format.space_before = Pt(8)
    p_kw.paragraph_format.space_after = Pt(6)
    _set_run_font(
        p_kw.add_run("Từ khóa: "), size_pt=12.5, bold=True, color=UFM_NAVY
    )
    _set_run_font(p_kw.add_run(KEYWORDS_TEXT), size_pt=12.5, italic=True)


# =============================================================================
# 6. DỰNG SECTION 3 (5 CHƯƠNG CHÍNH + TÀI LIỆU THAM KHẢO + PHỤ LỤC — TRANG 1,
# 2, 3...)
# =============================================================================
def _setup_header_footer_arabic(section: Any) -> None:
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False

    hdr = section.header
    hp = hdr.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after = Pt(2)
    _add_p_bottom_border(hp, color="CBD5E1", sz="4")
    _set_run_font(
        hp.add_run(
            "PRJ-EXT-01 — Phân tích Định giá Động & Dự báo Giá báo API Uber "
            "vs. Lyft (Boston, MA)"
        ),
        size_pt=9.5,
        italic=True,
        color=SLATE_MUTED,
    )

    ftr = section.footer
    fp = ftr.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.paragraph_format.space_before = Pt(2)
    fp.paragraph_format.space_after = Pt(0)
    _add_p_top_border(fp, color="CBD5E1", sz="4")
    _set_run_font(
        fp.add_run("Nhóm 02 — Lớp 25DDS01 (UFM)    |    Trang "),
        size_pt=10.5,
        italic=True,
        color=SLATE_MUTED,
    )
    _add_field(fp, "PAGE", placeholder="1", font_size=11.5, bold=True)


def _build_main_body_and_appendices(
    doc: Document, content: dict[str, Any]
) -> None:
    sec3 = doc.add_section(WD_SECTION.NEW_PAGE)
    _configure_section_a4(sec3, with_border=False)
    _set_page_number_format(sec3, fmt="decimal", start=1)
    _setup_header_footer_arabic(sec3)

    # --- 5 CHƯƠNG CHÍNH ---
    for ch_idx, ch in enumerate(content["chapters"]):
        _add_heading(doc, ch["title"], level=1, page_break_before=(ch_idx > 0))

        for sec in ch["sections"]:
            _add_heading(doc, sec["heading"], level=2)
            for btype, payload in sec["blocks"]:
                if btype == "p":
                    _add_body_p(doc, payload, indent=True)
                elif btype == "formula":
                    _add_formula_box(doc, payload)
                elif btype == "bullets":
                    _add_bullet_list(doc, payload)
                elif btype == "table":
                    cap_txt = payload["caption"]
                    is_short_pin = len(payload["rows"]) <= 8
                    is_tbl_2_1 = cap_txt.startswith("Bảng 2.1:")
                    _add_academic_table(
                        doc,
                        caption=cap_txt,
                        headers=payload["headers"],
                        rows=payload["rows"],
                        col_widths=payload["col_widths"],
                        aligns=payload.get("aligns"),
                        source=payload.get("source", ""),
                        font_size=8.6
                        if is_tbl_2_1
                        else float(payload.get("font_size", 9.8)),
                        compact=is_tbl_2_1,
                        keep_together=is_short_pin,
                    )
                elif btype == "figure":
                    _add_figure(
                        doc,
                        payload["path"],
                        payload["caption"],
                        width_cm=float(payload.get("width_cm", 14.5)),
                    )

    # --- TÀI LIỆU THAM KHẢO (CHUẨN APA 7TH — IN NGHIÊNG TẠP CHÍ/VOLUME + THỤT
    # TREO 1.27 CM) ---
    h_ref = _add_heading(
        doc, "TÀI LIỆU THAM KHẢO", level=1, page_break_before=True
    )
    h_ref.paragraph_format.space_before = Pt(0)
    h_ref.paragraph_format.space_after = Pt(6)
    for ref_runs in content["references"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Cm(1.27)
        p.paragraph_format.first_line_indent = Cm(-1.27)
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6.0)
        for txt, is_ital in ref_runs:
            r = p.add_run(txt)
            _set_run_font(r, size_pt=12.0, bold=False, italic=is_ital)

    # --- PHỤ LỤC A: MÃ NGUỒN PYTHON CHUẨN PEP 8 VÀ KIẾN TRÚC HỆ THỐNG ---
    _add_heading(
        doc,
        "PHỤ LỤC A: MÃ NGUỒN PYTHON CHUẨN PEP 8 VÀ KIẾN TRÚC HỆ THỐNG",
        level=1,
        page_break_before=True,
    )
    wm = content["worklog_meta"]
    _add_body_p(doc, _appendix_a_intro(wm))
    _add_code_box(
        doc,
        "A.1. Sơ đồ kiến trúc kho mã nguồn và sản phẩm bàn giao",
        content["appendix_a_tree"],
        compact=True,
    )
    _add_academic_table(
        doc,
        caption=(
            "Bảng A.1: Chỉ mục đầy đủ toàn bộ module Python trong kho mã "
            "nguồn dự án PRJ-EXT-01"
        ),
        headers=[
            "Module / Thư mục",
            "Chức năng chính trong quy trình Khoa học Dữ liệu",
        ],
        rows=content["appendix_a_module_index"],
        col_widths=[5.4, 10.6],
        aligns=["L", "L"],
        source=(
            "Nguồn: Rà soát trực tiếp cấu trúc và đếm dòng động trên kho mã "
            "nguồn nộp bài "
            "(ruff PEP 8 ≤ 79 ký tự: 0 lỗi trên toàn bộ .py & .ipynb)."
        ),
        font_size=8.0,
        compact=True,
    )
    _add_code_box(
        doc,
        "A.2. Toàn văn mã nguồn điều phối toàn trình main.py (100% nguyên "
        "bản chuẩn PEP 8)",
        content["appendix_a_main_code"],
        compact=True,
    )
    _add_code_box(
        doc,
        "A.3. Toàn văn mã nguồn cấu hình đường dẫn & kiểm soát rò rỉ src/"
        "config.py (100% nguyên bản chuẩn PEP 8)",
        content["appendix_a_config_code"],
        compact=True,
    )
    _add_code_box(
        doc,
        "A.4. Bốn trích đoạn thuật toán cốt lõi khớp 100% mã nguồn thực thi "
        "(Bảo toàn Surge, Khí tượng EST, Surge Cap & Monotonic Split-CQR)",
        content["appendix_a_code"],
        compact=True,
    )

    # --- PHỤ LỤC B: HỒ SƠ KIỂM TOÁN NỘI BỘ VÀ NHẬT KÝ P-D-M-V ---
    _add_heading(
        doc,
        "PHỤ LỤC B: HỒ SƠ KIỂM TOÁN VÀ NHẬT KÝ P-D-M-V",
        level=1,
        page_break_before=True,
    )
    p_apb_intro = _add_body_p(doc, _appendix_b_intro(wm))
    p_apb_intro.paragraph_format.line_spacing = 1.35
    p_apb_intro.paragraph_format.space_after = Pt(6)
    _add_academic_table(
        doc,
        caption=(
            "Bảng B.1: Quy chiếu các mã kiểm toán nội bộ trong hồ sơ kỹ "
            "thuật sang biện pháp xử lý học thuật"
        ),
        headers=[
            "Mã kiểm toán",
            "Nội dung hạng mục đã rà soát và xử lý trong mã nguồn & tiểu luận",
        ],
        rows=content["appendix_b_audit_map"],
        col_widths=[2.9, 13.1],
        aligns=["C", "L"],
        source=(
            "Nguồn: src/clean_raw_data.py, src/simulation.py và hệ thống "
            "kiểm chứng thống kê hợp nhất (scripts/)."
        ),
        font_size=8.0,
        compact=True,
        keep_together=True,
    )
    _add_academic_table(
        doc,
        caption=(
            "Bảng B.2: Tóm tắt tiến độ các nhóm phiên làm việc P-D-M-V (01/"
            "10/2026 – 28/11/2026)"
        ),
        headers=[
            "Nhóm phiên",
            "Thời gian",
            "Thành quả kiểm chứng và Quyết định kỹ thuật cốt lõi",
        ],
        rows=content["appendix_b_milestones"],
        col_widths=[2.3, 2.8, 10.9],
        aligns=["C", "C", "L"],
        source="Nguồn: Trích xuất từ ./nhật ký làm việc.md.",
        font_size=8.0,
        compact=True,
        keep_together=True,
    )
    _add_worklog_session_docx(
        doc,
        "B.3. Trích đoạn mục tiêu Phiên 12 (Biên bản P-D-M-V, bảng đối "
        "soát & SHA-256 từ ./nhật ký làm việc.md)",
        content["appendix_b_verbatim"],
    )


# =============================================================================
# 7. BIÊN DỊCH PDF CHÍNH THỨC TỪ .QMD (QUARTO / TYPST) & CẬP NHẬT TOC CHO .DOCX
# =============================================================================
def _render_pdf_via_quarto(qmd_path: Path, pdf_path: Path) -> None:
    """
    Biên dịch trực tiếp reports/...TieuLuan.qmd -> .pdf bằng Quarto (Typst, ảnh
    300+ DPI gốc).
    """
    quarto_bin = shutil.which("quarto")
    if not quarto_bin:
        raise RuntimeError(
            "Không tìm thấy lệnh 'quarto' trong PATH. Hãy cài đặt Quarto "
            "(>= 1.4) để biên dịch "
            f"{qmd_path} sang PDF."
        )
    proc = subprocess.run(
        [quarto_bin, "render", qmd_path.name, "--to", "typst"],
        cwd=str(qmd_path.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Lỗi Quarto khi biên dịch {qmd_path} -> PDF:\nSTDOUT: "
            f"{proc.stdout}\nSTDERR: {proc.stderr}"
        )
    if not pdf_path.exists() or pdf_path.stat().st_size < 100_000:
        raise RuntimeError(
            f"Quarto hoàn tất nhưng không thấy tệp PDF hợp lệ tại: {pdf_path}"
        )
    print(
        f"QUARTO_TYPST_OK | RENDERED={pdf_path.name} "
        f"({pdf_path.stat().st_size / (1024**2):.2f} MiB)"
    )


def _update_docx_toc_via_word(docx_path: Path) -> None:
    """
    Cập nhật tự động Mục lục/Danh mục cho bản Word dự phòng (.docx) mà không
    ghi đè PDF Quarto.
    """
    if sys.platform != "win32":
        print(
            f"DOCX_SAVED_NO_WORD_COM | {docx_path} (bỏ qua cập nhật TOC "
            "Word trên non-Windows)"
        )
        return

    ps_script = f"""
$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
try {{
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $docPath = '{str(docx_path.resolve())}'
    $doc = $word.Documents.Open($docPath, $false, $false, $false)
    $doc.Repaginate()
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {{
        $doc.TablesOfContents.Item($i).Update()
    }}
    foreach ($sname in @('TOC 1', 'TOC 2', 'TOC 3')) {{
        try {{
            $st = $doc.Styles.Item($sname)
            $st.AutomaticallyUpdate = $false
            $st.Font.Name = 'Times New Roman'
            $st.ParagraphFormat.Alignment = 0
        }} catch {{}}
    }}
    $tabPos = $word.CentimetersToPoints(16.0)
    $rIndent = $word.CentimetersToPoints(0.75)
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {{
        $toc = $doc.TablesOfContents.Item($i)
        $isList = ($i -ge 2)
        foreach ($para in $toc.Range.Paragraphs) {{
            $pf = $para.Format
            $pf.Alignment = 0
            $pf.RightIndent = $rIndent
            $pf.TabStops.ClearAll()
            $pf.TabStops.Add($tabPos, 2, 1) | Out-Null
            $para.Range.Font.Name = 'Times New Roman'
            if ($isList) {{
                $para.Range.Font.Size = 11.5
                $para.Range.Font.Bold = $false
                $pf.LeftIndent = 0
                $pf.FirstLineIndent = 0
                $pf.LineSpacingRule = 5
                $pf.LineSpacing = $word.LinesToPoints(1.15)
                $pf.SpaceBefore = 0
                $pf.SpaceAfter = 2.0
            }} else {{
                $para.Range.Font.Size = 11.2
                $pf.LineSpacingRule = 5
                $pf.LineSpacing = $word.LinesToPoints(1.10)
                $pf.SpaceAfter = 1.0
                if ($para.Style.NameLocal -like '*1*') {{
                    $para.Range.Font.Bold = $true
                    $pf.LeftIndent = 0
                    $pf.SpaceBefore = 1.5
                }} else {{
                    $para.Range.Font.Bold = $false
                    $pf.LeftIndent = $word.CentimetersToPoints(0.55)
                    $pf.SpaceBefore = 0
                }}
            }}
        }}
    }}
    foreach ($f in $doc.Fields) {{
        if ($f.Type -ne 13) {{
            $f.Update() | Out-Null
        }}
    }}
    foreach ($sec in $doc.Sections) {{
        foreach ($hdr in $sec.Headers) {{
            $hdr.Range.Fields.Update() | Out-Null
        }}
        foreach ($ftr in $sec.Footers) {{
            $ftr.Range.Fields.Update() | Out-Null
        }}
    }}
    $doc.Repaginate()
    for ($pass = 1; $pass -le 2; $pass++) {{
        for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {{
            $doc.TablesOfContents.Item($i).UpdatePageNumbers()
        }}
        $doc.Repaginate()
    }}
    $doc.Save()
    $pages = $doc.ComputeStatistics(2)
    $tocs = $doc.TablesOfContents.Count
    $msg = 'WORD_DOCX_TOC_OK | TOC_COUNT=' + $tocs + ' | PAGES=' + $pages
    Write-Output $msg
}} finally {{
    if ($doc -ne $null) {{ $doc.Close($false) }}
    if ($word -ne $null) {{ $word.Quit() }}
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}}
"""
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_script,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        print(
            "WARN_DOCX_TOC: Bỏ qua cập nhật TOC Word (.docx đã lưu): "
            f"{proc.stderr.strip()}"
        )
        return
    print(proc.stdout.strip())


# =============================================================================
# 8. XUẤT BẢN NGUỒN QUARTO MARKDOWN (.qmd) ĐỒNG BỘ 100% VỚI BẢN WORD/PDF
# =============================================================================
def _md_escape(text: str) -> str:
    """
    Thoát các ký tự điều khiển Pandoc/Quarto để tránh kích hoạt math/italic/sub
    ngoài ý muốn.
    """
    s = str(text)
    s = s.replace("K_{6,6}", "K₆,₆")
    s = s.replace("\\", "\\\\")
    s = s.replace("$", "\\$")
    s = s.replace("*", "\\*")
    s = s.replace("_", "\\_")
    s = s.replace("~", "\\~")
    s = s.replace("^", "\\^")
    s = s.replace("--", "\\-\\-")
    # Khôi phục ký hiệu toán học có chủ đích sang LaTeX gọn đẹp
    s = s.replace("10\\^m", "$10^m$").replace("10\\^0.55", "$10^{0.55}$")
    s = s.replace("q\u0302", "$\\hat{q}$")
    s = s.replace("∈ {0.30, 0.60}", "$\\in \\{0.30, 0.60\\}$")
    s = s.replace("∈ {25×, 50×}", "$\\in \\{25\\times, 50\\times\\}$")
    s = s.replace("∈", "$\\in$")
    return s


def _md_cell(text: str) -> str:
    """
    Chuẩn hóa nội dung ô bảng Markdown (1 dòng gọn, thoát dấu | và ký tự đặc
    biệt).
    """
    s = str(text).replace("/\n", "/ ").replace("\n", " ").strip()
    return _md_escape(s).replace("|", "\\|")


def _col_widths_to_pct(
    col_widths: list[float] | None, n_cols: int
) -> list[int]:
    """
    Quy đổi danh sách độ rộng cột (cm) sang tỷ lệ % nguyên có tổng bằng 100 cho
    Quarto.
    """
    if not col_widths or len(col_widths) != n_cols:
        base = 100 // n_cols
        rem = 100 - base * n_cols
        return [base + (1 if i < rem else 0) for i in range(n_cols)]
    total = sum(float(w) for w in col_widths)
    raw = [float(w) / total * 100.0 for w in col_widths]
    pcts = [max(4, int(round(x))) for x in raw]
    diff = 100 - sum(pcts)
    if diff != 0:
        max_idx = max(range(n_cols), key=lambda i: pcts[i])
        pcts[max_idx] += diff
    return pcts


def _formula_tokens_to_latex(
    tokens: list[tuple[str, bool, str | None]],
) -> str:
    """
    Chuyển danh sách token công thức Word sang LaTeX chuẩn cho khối $$ ... $$
    của Quarto.
    """
    greek_map = {
        "β": r"\beta",
        "α": r"\alpha",
        "ε": r"\varepsilon",
        "τ": r"\tau",
        "Δ": r"\Delta",
        "χ": r"\chi",
        "−": "-",
        "×": r"\times",
        "≤": r"\le",
        "≥": r"\ge",
        "±": r"\pm",
    }

    def _convert_piece(txt: str) -> str:
        out = txt
        for src_ch, tex in greek_map.items():
            out = out.replace(src_ch, tex)
        return out

    parts: list[str] = []
    for txt, _is_italic, script in tokens:
        t = _convert_piece(txt)
        if script == "sub":
            if re.fullmatch(r"[A-Za-z0-9,]+", t):
                if "," in t:
                    left, right = t.split(",", 1)
                    parts.append(f"_{{{left},\\text{{{right}}}}}")
                elif len(t) > 1:
                    parts.append(f"_{{\\text{{{t}}}}}")
                else:
                    parts.append(f"_{{{t}}}")
            else:
                parts.append(f"_{{\\text{{{t}}}}}")
        elif script == "sup":
            parts.append("^{*}" if t == "*" else f"^{{{t}}}")
        else:
            parts.append(t)

    expr = "".join(parts).strip()
    expr = expr.replace("min(", r"\min\left(").replace("min( ", r"\min\left(")
    if r"\min\left(" in expr and expr.endswith(" )"):
        expr = expr[:-2] + r"\right)"
    elif r"\min\left(" in expr and expr.endswith(")"):
        expr = expr[:-1] + r"\right)"
    if " = (" in expr and ") / (" in expr and expr.endswith(")"):
        lhs, rhs = expr.split(" = (", 1)
        num, den = rhs[:-1].split(") / (", 1)
        expr = f"{lhs.strip()} = \\frac{{{num.strip()}}}{{{den.strip()}}}"
    return expr


def _slug_id(prefix: str, caption: str, fallback_idx: int) -> str:
    """
    Sinh mã định danh Quarto duy nhất (#tbl-2-1, #fig-3-0, #tbl-a-1...) từ tiêu
    đề Bảng/Hình.
    """
    m = re.search(r"(?:Bảng|Hình)\s+([A-Za-z0-9]+)\.([0-9]+)", caption)
    if m:
        return f"{prefix}-{m.group(1).lower()}-{m.group(2)}"
    return f"{prefix}-item-{fallback_idx}"


def _typst_escape(text: str) -> str:
    """
    Thoát ký tự điều khiển Typst để hiển thị nguyên văn trong khối nội dung
    Typst [...].
    """
    s = str(text)
    for ch in ("\\", "#", "$", "*", "_", "`", "<", ">", "@", "[", "]"):
        s = s.replace(ch, f"\\{ch}")
    s = s.replace("q\u0302", "$hat(q)$")
    s = s.replace("∈", "$in$")
    return s


def _typst_format_url(url: str) -> str:
    """
    Định dạng URL/DOI chuẩn Typst: không vỡ giữa tên miền hoặc mã định danh
    DOI.
    """
    escaped_href = url.replace("\\", "\\\\").replace('"', '\\"')
    if len(url) <= 64:
        return (
            f'#box(link("{escaped_href}")'
            f"[#text(hyphenate: false)[{_typst_escape(url)}]])"
        )
    parts = url.split("/")
    origin = "/".join(parts[:3]) + "/"
    tail_parts = parts[3:]
    tail_chunks: list[str] = []
    for idx_t, part in enumerate(tail_parts):
        esc_p = _typst_escape(part)
        if idx_t < len(tail_parts) - 1:
            tail_chunks.append(f"#box[{esc_p}/];#sym.zws;")
        else:
            tail_chunks.append(f"#box[{esc_p}];")
    disp = f"#box[{_typst_escape(origin)}];#sym.zws;" + "".join(tail_chunks)
    return f'#link("{escaped_href}")[#text(hyphenate: false)[{disp}]]'


def _typst_apa_entry(ref_runs: list[tuple[str, bool]]) -> str:
    """
    Chuyển 1 mục Tài liệu tham khảo APA 7th sang dòng nội dung Typst chuẩn
    12pt.
    """
    url_re = re.compile(r"(https?://\S+)")
    pieces: list[str] = []
    for txt, is_ital in ref_runs:
        if is_ital:
            pieces.append(f"#emph[{_typst_escape(txt)}];")
            continue
        pos = 0
        for m in url_re.finditer(txt):
            if m.start() > pos:
                pieces.append(_typst_escape(txt[pos : m.start()]))
            pieces.append(_typst_format_url(m.group(1)))
            pos = m.end()
        if pos < len(txt):
            pieces.append(_typst_escape(txt[pos:]))
    return "".join(pieces)


def _typst_wrap_code_text(text: str) -> str:
    """
    Chèn điểm ngắt dòng mềm (#sym.zws;) sau '_' và '/' trong mã nội dòng
    bằng một lượt thay thế duy nhất để tên tệp/hàm snake_case dài tự xuống
    dòng gọn trong cột bảng hẹp mà không cắt đứt số thập phân.
    """
    esc = _typst_escape(text)
    return re.sub(r"(\\_|/)", r"\1#sym.zws;", esc)


def _inline_md_to_typst(text: str, allow_code_wrap: bool = False) -> str:
    """
    Chuyển cú pháp Markdown nội dòng (**đậm**, *nghiêng*, `mã`, [nhãn](url))
    sang Typst sạch.
    """
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"`\1`", text)
    pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")
    out: list[str] = []
    pos = 0
    for m in pattern.finditer(clean):
        if m.start() > pos:
            plain = clean[pos : m.start()]
            out.append(
                _typst_wrap_code_text(plain)
                if (allow_code_wrap and "_" in plain)
                else _typst_escape(plain)
            )
        tok = m.group(0)
        if tok.startswith("**") and tok.endswith("**"):
            inner = tok[2:-2]
            if "`" in inner:
                sub = _inline_md_to_typst(
                    inner, allow_code_wrap=allow_code_wrap
                )
                out.append(f"#strong[{sub}];")
            else:
                out.append(f"#strong[{_typst_escape(inner)}];")
        elif tok.startswith("`") and tok.endswith("`"):
            code_body = (
                _typst_wrap_code_text(tok[1:-1])
                if allow_code_wrap
                else _typst_escape(tok[1:-1])
            )
            out.append(f"#text(font: mono-stack, size: 7.2pt)[{code_body}];")
        elif tok.startswith("*") and tok.endswith("*"):
            inner = tok[1:-1]
            if "`" in inner:
                sub = _inline_md_to_typst(
                    inner, allow_code_wrap=allow_code_wrap
                )
                out.append(f"#emph[{sub}];")
            else:
                out.append(f"#emph[{_typst_escape(inner)}];")
        pos = m.end()
    if pos < len(clean):
        tail = clean[pos:]
        out.append(
            _typst_wrap_code_text(tail)
            if (allow_code_wrap and "_" in tail)
            else _typst_escape(tail)
        )
    return "".join(out)


def _render_worklog_session_typst(md_text: str) -> list[str]:
    """
    Kết xuất 100% phiên nhật ký Markdown sang khung Typst định dạng chuẩn (0
    raw ** / ##).
    """
    out: list[str] = [
        "```{=typst}",
        "#block(",
        '  fill: rgb("#F8FAFC"),',
        '  stroke: 0.55pt + rgb("#CBD5E1"),',
        "  width: 100%,",
        "  inset: 7pt,",
        "  radius: 2.5pt,",
        "  breakable: true,",
        ")[",
        "  #set par(justify: false, leading: 0.48em, spacing: 0.56em)",
        '  #set text(size: 8.3pt, fill: rgb("#0F172A"))',
    ]
    raw_lines = md_text.splitlines()
    idx = 0
    while idx < len(raw_lines):
        ln = raw_lines[idx]
        stripped = ln.strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            tbl_lines: list[str] = []
            while idx < len(raw_lines) and raw_lines[idx].strip().startswith(
                "|"
            ):
                tbl_lines.append(raw_lines[idx].strip())
                idx += 1
            parsed = [
                [c.strip() for c in row.strip("|").split("|")]
                for row in tbl_lines
                if not re.fullmatch(r"[\s|:\-]+", row)
            ]
            if len(parsed) >= 2:
                headers = parsed[0]
                ncols = len(headers)
                if ncols == 4:
                    col_spec = "(12%, 28%, 30%, 30%)"
                elif ncols == 5:
                    col_spec = "(11%, 23%, 27%, 31%, 8%)"
                else:
                    col_spec = f"({', '.join(['1fr'] * ncols)})"
                out.append("  #v(0.2em)")
                out.append("  #[")
                out.append("    #show table: set text(size: 7.4pt)")
                out.append("    #set table(inset: (x: 3.5pt, y: 2.5pt))")
                out.append("    #set figure.caption(position: top)")
                out.append(
                    '    #figure(kind: "quarto-float-tbl", '
                    "supplement: none, caption: [Bảng B.3: Đối soát trước "
                    "và sau khắc phục các mã kiểm toán (trích xuất nguyên "
                    "văn Phiên 12)])["
                )
                out.append(f"      #table(columns: {col_spec},")
                hdr_cells = ", ".join(
                    f"[{_inline_md_to_typst(h, allow_code_wrap=True)}]"
                    for h in headers
                )
                out.append(f"        table.header({hdr_cells}),")
                for row in parsed[1:]:
                    padded = (row + [""] * ncols)[:ncols]
                    row_cells = ", ".join(
                        f"[{_inline_md_to_typst(c, allow_code_wrap=True)}]"
                        for c in padded
                    )
                    out.append(f"        {row_cells},")
                out.append("      )")
                out.append("    ]")
                out.append(
                    "    #block(above: 0.14em, below: 0.25em)[#set par("
                    "justify: false, leading: 0.45em); #text(size: 7.8pt, "
                    'style: "italic", fill: rgb("#475569"))[Nguồn: Trích '
                    "xuất nguyên văn Phiên 12 từ ./nhật ký làm việc.md.]]"
                )
                out.append("  ]")
                out.append("  #v(0.2em)")
            continue
        if stripped.startswith("## "):
            hdr_body = _inline_md_to_typst(stripped[3:].strip())
            out.append(
                '  #block(below: 0.45em)[#text(size: 9.2pt, weight: "bold",'
                f" fill: ufm-navy)[{hdr_body}]]"
            )
        elif stripped.startswith(("- ", "* ")):
            indent_lv = (len(ln) - len(ln.lstrip(" "))) // 2
            pad_left = (
                0.0 if indent_lv == 0 else (0.45 if indent_lv == 1 else 0.90)
            )
            bullet = (
                "•" if indent_lv == 0 else ("–" if indent_lv == 1 else "▪")
            )
            body = _inline_md_to_typst(stripped[2:].strip())
            out.append(f"  #pad(left: {pad_left:.2f}cm)[{bullet} {body}]")
        else:
            out.append(f"  {_inline_md_to_typst(stripped)}")
        idx += 1
    out.extend(["]", "```", ""])
    return out


def _qmd_table_lines(
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    aligns: list[str] | None = None,
    col_widths: list[float] | None = None,
    source: str = "",
    tbl_idx: int = 0,
    keep_together: bool = False,
) -> list[str]:
    """
    Dựng bảng Pandoc Pipe Table chuẩn Quarto (#tbl-...) để tự động sinh Danh
    mục Bảng biểu.
    """
    align_tok = {"L": ":---", "C": ":---:", "R": "---:"}
    col_aligns = aligns or ["L"] * len(headers)
    pcts = _col_widths_to_pct(col_widths, len(headers))
    widths_spec = 'tbl-colwidths="[' + ",".join(str(p) for p in pcts) + ']"'

    lines: list[str] = []
    if keep_together:
        lines.extend(["```{=typst}", "#block(breakable: false)[", "```", ""])
    lines.append("| " + " | ".join(_md_cell(h) for h in headers) + " |")
    lines.append(
        "| " + " | ".join(align_tok.get(a, ":---") for a in col_aligns) + " |"
    )
    for rdata in rows:
        lines.append("| " + " | ".join(_md_cell(v) for v in rdata) + " |")
    lines.append("")
    if caption:
        tid = _slug_id("tbl", caption, tbl_idx)
        lines.append(f": {_md_escape(caption)} {{#{tid} {widths_spec}}}")
    else:
        lines.append(f": {{{widths_spec}}}")
    if source:
        close_tok = "\n]" if keep_together else ""
        lines.extend(
            [
                "",
                "```{=typst}",
                "#block(above: 0.16em, below: 0.72em)[#set par(justify: "
                "false, leading: 0.50em); #text(size: 9.4pt, style: "
                '"italic", fill: '
                f'rgb("#475569"))[{_typst_escape(source)}]]{close_tok}',
                "```",
                "",
                '::: {.content-visible unless-format="typst"}',
                f"*{_md_escape(source)}*",
                ":::",
            ]
        )
    elif keep_together:
        lines.extend(["", "```{=typst}", "]", "```"])
    lines.append("")
    return lines


def _qmd_relative_asset(rel_from_root: str, qmd_dir: Path) -> str:
    """
    Quy đổi đường dẫn tài nguyên (tính từ ROOT_DIR) sang đường dẫn tương đối từ
    thư mục file .qmd.
    """
    abs_target = (ROOT_DIR / rel_from_root).resolve()
    try:
        return abs_target.relative_to(qmd_dir.resolve()).as_posix()
    except ValueError:
        return Path(rel_from_root).as_posix()


def export_thesis_qmd(
    content: dict[str, Any], qmd_path: Path = QMD_OUT
) -> Path:
    """
    Sinh tệp Quarto Markdown (.qmd) chuẩn UFM biên dịch trực tiếp PDF (Typst),
    HTML & DOCX.
    """
    qmd_path.parent.mkdir(parents=True, exist_ok=True)
    qmd_dir = qmd_path.parent
    logo_rel = _qmd_relative_asset("reports/assets/ufm_logo.jpeg", qmd_dir)
    wm = content["worklog_meta"]
    n_sess = int(wm["sessions_count"])

    lines: list[str] = [
        "---",
        f'title: "{COVER_META["topic_title"]}"',
        f'subtitle: "{COVER_META["course_line"]} — '
        f'{COVER_META["course_name"]} ({COVER_META["topic_code"]})"',
        "author:",
        '  - name: "Trần Thanh Huy (MSSV: 2521001060)"',
        '    affiliation: "Nhóm 02 — Lớp 25DDS01 (UFM)"',
        '  - name: "Lê Huy Thắng (MSSV: 2521001092)"',
        '    affiliation: "Nhóm 02 — Lớp 25DDS01 (UFM)"',
        'date: "2026-11-28"',
        "lang: vi",
        "number-sections: false",
        "tbl-cap-location: top",
        "fig-cap-location: bottom",
        "crossref:",
        '  fig-title: "Hình"',
        '  tbl-title: "Bảng"',
        "format:",
        "  typst:",
        "    toc: false",
        "    papersize: a4",
        "    margin:",
        "      top: 2cm",
        "      bottom: 2cm",
        "      left: 3cm",
        "      right: 2cm",
        '    mainfont: "Times New Roman"',
        '    monofont: "Consolas"',
        "    fontsize: 12pt",
        "    include-in-header:",
        "      text: |",
        '        #let ufm-navy = rgb("#1B365D")',
        '        #let slate-muted = rgb("#475569")',
        '        #let mono-stack = ("Consolas", "Courier New", "DejaVu Sans '
        'Mono")',
        "        #let Skylighting(fill: none, number: false, start: 1, "
        "sourcelines) = {",
        "          let blocks = []",
        "          let lnum = start - 1",
        "          for ln in sourcelines {",
        "            if number {",
        "              lnum = lnum + 1",
        "              blocks = blocks + box(width: if start + "
        "sourcelines.len() > 999 { 30pt } else { 24pt }, text(fill: "
        'rgb("#aaaaaa"), [ #lnum ]))',
        "            }",
        "            blocks = blocks + ln + EndLine()",
        "          }",
        "          block(",
        '            fill: rgb("#F8FAFC"),',
        '            stroke: 0.45pt + rgb("#CBD5E1"),',
        "            width: 100%,",
        "            inset: 4.5pt,",
        "            radius: 2pt,",
        "            breakable: true,",
        "            {",
        "              set par(justify: false, leading: 0.39em)",
        "              set text(font: mono-stack, size: 6.7pt)",
        "              show raw: set text(font: mono-stack, size: 6.7pt)",
        "              blocks",
        "            },",
        "          )",
        "        }",
        "        #let article(",
        "          title: none, subtitle: none, authors: none, keywords: (),"
        " date: none,",
        "          abstract-title: none, abstract: none, thanks: none, cols:"
        ' 1, lang: "vi",',
        "          region: none, font: none, fontsize: 12pt, title-size: "
        "1.5em, subtitle-size: 1.25em,",
        '          heading-family: none, heading-weight: "bold", heading-'
        'style: "normal",',
        "          heading-color: black, heading-line-height: 0.65em, "
        "mathfont: none, codefont: none,",
        "          linestretch: 1.35, sectionnumbering: none, linkcolor: "
        "none, citecolor: none,",
        "          filecolor: none, toc: false, toc_title: none, toc_depth: "
        "none, toc_indent: 1.5em, doc,",
        "        ) = {",
        "          set document(title: title)",
        "          set document(",
        "            author: authors.map(author => content-to-"
        'string(author.name)).join(", ", last: " & "),',
        "          ) if authors != none and authors != ()",
        '          set text(lang: lang, size: fontsize, font: ("Times New '
        'Roman", "Libertinus Serif"), fill: black)',
        "          set par(justify: true, leading: 0.70em, spacing: 1.20em)",
        "          show raw: set text(font: mono-stack)",
        "          show heading.where(level: 1): it => {",
        "            set align(center)",
        '            set text(size: 13.5pt, weight: "bold", fill: ufm-navy)',
        "            block(above: 0.35em, below: 0.85em, sticky: true, "
        "it.body)",
        "          }",
        "          show heading.where(level: 2): it => {",
        "            set align(left)",
        '            set text(size: 12.2pt, weight: "bold", fill: ufm-navy)',
        "            block(above: 0.90em, below: 0.45em, sticky: true, "
        "it.body)",
        "          }",
        "          show heading.where(level: 3): it => {",
        "            set align(left)",
        '            set text(size: 11.5pt, weight: "bold", style: "italic",'
        " fill: black)",
        "            block(above: 0.65em, below: 0.35em, sticky: true, "
        "it.body)",
        "          }",
        '          show figure.where(kind: "quarto-float-tbl"): set '
        "block(breakable: true)",
        "          set table.cell(breakable: false)",
        '          show figure.where(kind: "quarto-float-tbl"): set '
        "figure(numbering: none, supplement: none)",
        '          show figure.where(kind: "quarto-float-fig"): set '
        "figure(numbering: none, supplement: none)",
        '          show figure.where(kind: "quarto-float-tbl"): it => {',
        "            show figure.caption: cap => align(",
        "              left,",
        "              block(above: 0.16em, below: 0.26em, sticky: true, "
        'text(size: 10.6pt, weight: "bold", fill: black, cap.body)),',
        "            )",
        "            block(above: 0.62em, below: 0.20em, it)",
        "          }",
        '          show figure.where(kind: "quarto-float-fig"): it => {',
        "            show figure.caption: cap => align(",
        "              center,",
        "              block(above: 0.24em, below: 0.12em, text(size: "
        '10.2pt, style: "italic", fill: black, cap.body)),',
        "            )",
        "            block(above: 0.55em, below: 0.58em, it)",
        "          }",
        "          show table: set text(size: 8.2pt)",
        "          show table: set par(justify: false, leading: 0.46em)",
        "          set table(",
        "            inset: (x: 4pt, y: 3.2pt),",
        '            stroke: 0.45pt + rgb("#94A3B8"),',
        "            fill: (x, y) => if y == 0 { ufm-navy } else if "
        'calc.odd(y) { rgb("#F8FAFC") } else { white },',
        "          )",
        '          show table.cell.where(y: 0): set text(weight: "bold", '
        "fill: white)",
        "          show raw.where(block: true): it => block(",
        '            fill: rgb("#F8FAFC"),',
        '            stroke: 0.45pt + rgb("#CBD5E1"),',
        "            width: 100%,",
        "            inset: 5pt,",
        "            radius: 2pt,",
        "            breakable: true,",
        "            {",
        "              set par(justify: false, leading: 0.45em)",
        "              set text(font: mono-stack, size: 6.8pt, fill: "
        'rgb("#0F172A"))',
        "              show raw: set text(font: mono-stack, size: 6.8pt, "
        'fill: rgb("#0F172A"))',
        "              it",
        "            },",
        "          )",
        "          show outline.entry: it => {",
        "            set text(size: 11.8pt)",
        "            set par(leading: 0.48em, justify: false)",
        "            if it.element.func() == heading and it.level == 1 {",
        "              block(above: 0.62em, below: 0.28em, text(weight: "
        '"bold", link(it.element.location(), it.inner())))',
        "            } else {",
        "              block(above: 0.40em, below: 0.28em, pad(left: 3.1em, "
        "link(it.element.location(), [#h(-1.9em)#it.inner()])))",
        "            }",
        "          }",
        "          set outline.entry(fill: box(width: 1fr, pad(x: 0.35em, "
        'text(weight: "regular", fill: rgb("#475569"), '
        "repeat(gap: 0.28em)[.]))))",
        "          doc",
        "        }",
        "---",
        "",
        "```{=html}",
        "<style>",
        "body { font-family: 'Times New Roman', Georgia, serif; line-height:"
        " 1.65; color: #0f172a; }",
        "h1, h2 { color: #1B365D; }",
        ".ufm-meta { border: 2px double #1B365D; padding: 1.2rem 1.6rem; "
        "text-align: center; margin-bottom: 1.5rem; background: #f8fafc; "
        "border-radius: 4px; }",
        ".apa-ref p { padding-left: 2.2em; text-indent: -2.2em; margin-"
        "bottom: 0.55em; text-align: left; }",
        "table { font-size: 0.90rem; }",
        "table thead th { background-color: #1B365D; color: #ffffff; }",
        "pre code { font-size: 0.78rem; line-height: 1.35; }",
        "</style>",
        "```",
        "",
        "```{=typst}",
        "// --- SECTION 1: TRANG BÌA CHÍNH & BÌA PHỤ CHUẨN UFM (KHUNG VIỀN "
        "KÉP, KHÔNG ĐÁNH SỐ TRANG) ---",
        "#set page(",
        "  numbering: none,",
        "  header: none,",
        "  footer: none,",
        "  background: place(",
        "    top + left,",
        "    dx: 2.35cm,",
        "    dy: 1.35cm,",
        "    rect(",
        "      width: 17.3cm,",
        "      height: 27.0cm,",
        '      stroke: 1.6pt + rgb("#1B365D"),',
        "      inset: 3pt,",
        "      rect(width: 100%, height: 100%, stroke: 0.7pt + "
        'rgb("#1B365D")),',
        "    ),",
        "  ),",
        ")",
        "#align(center)[",
        "  #v(0.4cm)",
        '  #text(size: 14.5pt, weight: "bold", fill: '
        f'rgb("#1B365D"))[{COVER_META["university"]}]',
        "  #v(0.22cm)",
        '  #text(size: 13.5pt, weight: "bold", fill: '
        f'rgb("#1B365D"))[{COVER_META["faculty"]}]',
        "  #v(0.8cm)",
        f'  #image("{logo_rel}", width: 3.6cm)',
        "  #v(1.1cm)",
        f'  #text(size: 15pt, weight: "bold")[{COVER_META["course_line"]}]',
        "  #v(0.3cm)",
        '  #text(size: 16pt, weight: "bold", fill: '
        f'rgb("#1B365D"))[{COVER_META["course_name"]}]',
        "  #v(1.3cm)",
        '  #text(size: 13.5pt, weight: "bold")[ĐỀ TÀI:]',
        "  #v(0.35cm)",
        "  #set par(leading: 0.85em, justify: false)",
        '  #text(size: 14pt, weight: "bold", fill: '
        f'rgb("#1B365D"))[{COVER_META["topic_title"]}]',
        "  #v(1.6cm)",
        '  #text(size: 13pt, weight: "bold")[GIẢNG VIÊN GIẢNG DẠY HỌC PHẦN:'
        f" {COVER_META['lecturer']}]",
        "  #v(0.35cm)",
        '  #text(size: 13pt, weight: "bold")[MÃ LỚP HỌC PHẦN: '
        f"{COVER_META['class_code']}]",
        "  #v(0.35cm)",
        '  #text(size: 13pt, weight: "bold")[NHÓM THỰC HIỆN: '
        f"{COVER_META['group_name']} — LỚP {COVER_META['student_class']}]",
        "  #v(0.35cm)",
        '  #text(size: 12.2pt, fill: rgb("#475569"))[MÃ ĐỀ TÀI: '
        f"{COVER_META['topic_code']}]",
        "  #v(1fr)",
        f'  #text(size: 13pt, weight: "bold")[{COVER_META["location_year"]}]',
        "  #v(0.4cm)",
        "]",
        "#pagebreak()",
        "#align(center)[",
        "  #v(0.3cm)",
        '  #text(size: 14.5pt, weight: "bold", fill: rgb("#1B365D"))[THÔNG '
        "TIN NHÓM THỰC HIỆN VÀ PHÂN CÔNG NHIỆM VỤ]",
        "  #v(0.35cm)",
        '  #text(size: 11pt, weight: "bold")[Học phần: '
        f"{COVER_META['course_name']} #h(0.6em) | #h(0.6em) Mã lớp HP: "
        f"{COVER_META['class_code']}]",
        "  #v(0.2cm)",
        '  #text(size: 11pt, fill: rgb("#475569"))[Giảng viên hướng dẫn: '
        f"{COVER_META['lecturer']} #h(0.6em) | #h(0.6em) "
        f"{COVER_META['group_name']} (Lớp {COVER_META['student_class']})]",
        "  #v(0.5cm)",
        "]",
        "```",
        "",
        '::: {.content-visible when-format="html"}',
        "::: {.ufm-meta}",
        f"**{COVER_META['university']}**  ",
        f"**{COVER_META['faculty']}**",
        "",
        f'![]({logo_rel}){{width=95px fig-align="center"}}',
        "",
        f"**Giảng viên hướng dẫn:** {COVER_META['lecturer']} | **Mã lớp học "
        f"phần:** `{COVER_META['class_code']}`  ",
        f"**{COVER_META['group_name']} — Lớp "
        f"{COVER_META['student_class']}** | **Mã đề tài:** "
        f"`{COVER_META['topic_code']}`  ",
        f"*{COVER_META['location_year']}*",
        ":::",
        "",
        "# THÔNG TIN NHÓM THỰC HIỆN VÀ PHÂN CÔNG NHIỆM VỤ {.unnumbered "
        ".unlisted}",
        ":::",
        "",
    ]

    member_rows = [
        [
            m["stt"],
            m["name"],
            m["mssv"],
            f"{m['completion']} ({m['alpha_label']})",
            m["role"],
            "(Ký xác nhận)",
        ]
        for m in COVER_META["members"]
    ]
    lines.extend(
        _qmd_table_lines(
            caption="",
            headers=[
                "STT",
                "HỌ VÀ TÊN",
                "MSSV",
                "Tỷ lệ hoàn thành phần việc được giao",
                "NỘI DUNG PHỤ TRÁCH CHÍNH",
                "Chữ ký",
            ],
            rows=member_rows,
            aligns=["C", "L", "C", "C", "L", "C"],
            col_widths=[1.0, 2.9, 2.2, 2.3, 5.6, 2.0],
            source=(
                "Ghi chú: Cột 'Tỷ lệ hoàn thành phần việc được giao' thể "
                "hiện mức độ hoàn tất nhiệm vụ "
                "phân công (không phải tỷ trọng chia điểm); hệ số đóng góp "
                "cá nhân theo Mục 5.4 đề cương là "
                "α₁ = 1.00, α₂ = 1.00 (∑αᵢ = 2.00; phân công 50/50 qua "
                f"{n_sess} phiên P-D-M-V)."
            ),
        )
    )

    lines.extend(
        [
            "```{=typst}",
            "#[",
            "#set par(justify: false, leading: 0.65em, spacing: 0.55em)",
            "```",
            "",
            "**Hồ sơ mã nguồn và sản phẩm bàn giao đi kèm:**",
            "",
            *[f"- {_md_escape(item)}" for item in _subcover_deliverables(wm)],
            "",
            "```{=typst}",
            "]",
            "#v(1fr)",
            "#align(center, text(size: 13pt, weight: "
            f'"bold")[{COVER_META["location_year"]}])',
            "#v(0.4cm)",
            "#pagebreak()",
            "// --- SECTION 2: PHẦN MỞ ĐẦU (ĐÁNH SỐ TRANG LA MÃ i, ii, "
            "iii...) ---",
            "#set page(",
            "  background: none,",
            '  numbering: "i",',
            "  header: none,",
            "  footer: context align(right, text(size: 11pt, "
            'counter(page).display("i"))),',
            ")",
            "#counter(page).update(1)",
            "```",
            "",
            "# LỜI CAM ĐOAN VÀ LỜI CẢM ƠN {.unnumbered}",
            "",
            "**LỜI CAM ĐOAN**",
            "",
            *[
                x
                for p_txt in DECLARATION_PARAS
                for x in (_md_escape(p_txt), "")
            ],
            "**LỜI CẢM ƠN**",
            "",
            *[x for p_txt in THANKS_PARAS for x in (_md_escape(p_txt), "")],
            "```{=typst}",
            "#v(0.4em)",
            "#align(right)[",
            '  #text(style: "italic")[TP. Hồ Chí Minh, tháng 11 năm 2026] \\',
            '  #text(weight: "bold")[Đại diện Nhóm 02 thực hiện] \\',
            "  Trần Thanh Huy — Lê Huy Thắng",
            "]",
            "#pagebreak()",
            "#heading(level: 1, outlined: false)[MỤC LỤC]",
            "#outline(title: none, depth: 2, indent: 1.2em)",
            "#pagebreak()",
            "```",
            "",
            '::: {.content-visible when-format="html"}',
            "*TP. Hồ Chí Minh, tháng 11 năm 2026*  ",
            "**Đại diện Nhóm 02 thực hiện:** Trần Thanh Huy — Lê Huy Thắng",
            ":::",
            "",
            "# DANH MỤC CHỮ VIẾT TẮT {.unnumbered}",
            "",
        ]
    )

    # Dùng scope typst #[ ... ] giữ cỡ chữ chuẩn học thuật 11.5pt cho bảng
    # Chữ viết tắt (ngắt trang tự nhiên kèm lặp header nếu vượt 1 trang)
    lines.extend(
        [
            "```{=typst}",
            "#[",
            "#show table: set text(size: 11.5pt)",
            "#set table(inset: (x: 5pt, y: 3.8pt))",
            "```",
            "",
        ]
    )
    lines.extend(
        _qmd_table_lines(
            caption="",
            headers=["Viết tắt", "Tiếng Việt", "Tiếng Anh"],
            rows=content["abbreviations"],
            aligns=["C", "L", "L"],
            col_widths=[2.4, 6.6, 7.0],
        )
    )
    lines.extend(
        [
            "```{=typst}",
            "]",
            "#pagebreak()",
            "```",
            "",
            "# DANH MỤC BẢNG BIỂU {.unnumbered}",
            "",
            "```{=typst}",
            "#block({",
            "  show outline.entry: it => {",
            "    set text(size: 11.5pt)",
            "    set par(leading: 0.42em, justify: false)",
            "    block(above: 0.58em, below: 0.32em, pad(left: 4.5em, "
            "link(it.element.location(), [#h(-4.5em)#it.inner()])))",
            "  }",
            '  outline(title: none, target: figure.where(kind: "quarto-'
            'float-tbl"))',
            "})",
            "#pagebreak()",
            "```",
            "",
            "# DANH MỤC HÌNH VẼ / BIỂU ĐỒ {.unnumbered}",
            "",
            "```{=typst}",
            "#block({",
            "  show outline.entry: it => {",
            "    set text(size: 11.5pt)",
            "    set par(leading: 0.42em, justify: false)",
            "    block(above: 0.58em, below: 0.32em, pad(left: 4.6em, "
            "link(it.element.location(), [#h(-4.6em)#it.inner()])))",
            "  }",
            '  outline(title: none, target: figure.where(kind: "quarto-'
            'float-fig"))',
            "})",
            "#pagebreak()",
            "```",
            "",
            "# TÓM TẮT ĐỒ ÁN (EXECUTIVE SUMMARY) {.unnumbered}",
            "",
            _md_escape(content["executive_summary"]),
            "",
            f"**Từ khóa:** *{_md_escape(KEYWORDS_TEXT)}*",
            "",
            "```{=typst}",
            "#pagebreak()",
            "// --- SECTION 3: 5 CHƯƠNG CHÍNH + TÀI LIỆU THAM KHẢO + PHỤ "
            "LỤC (ĐÁNH SỐ Ả RẬP 1, 2, 3...) ---",
            "#set page(",
            '  numbering: "1",',
            "  header-ascent: 24%,",
            "  footer-descent: 28%,",
            "  header: align(right)[",
            '    #set text(size: 8.8pt, style: "italic", fill: '
            'rgb("#475569"))',
            "    PRJ-EXT-01 — Phân tích Định giá Động & Dự báo Giá báo API "
            "Uber vs. Lyft (Boston, MA)",
            "    #v(-0.55em)",
            '    #line(length: 100%, stroke: 0.45pt + rgb("#CBD5E1"))',
            "  ],",
            "  footer: context align(right)[",
            '    #line(length: 100%, stroke: 0.45pt + rgb("#CBD5E1"))',
            "    #v(-0.50em)",
            '    #set text(size: 9.8pt, style: "italic", fill: '
            'rgb("#475569"))',
            "    Nhóm 02 — Lớp 25DDS01 (UFM) #h(0.9em) | #h(0.9em) Trang "
            '#text(weight: "bold", style: "normal", fill: black, '
            'counter(page).display("1"))',
            "  ],",
            ")",
            "#counter(page).update(1)",
            "```",
            "",
        ]
    )

    # --- 5 CHƯƠNG CHÍNH ---
    tbl_counter = 0
    fig_counter = 0
    for ch_idx, ch in enumerate(content["chapters"]):
        if ch_idx > 0:
            lines.extend(["{{< pagebreak >}}", ""])
        lines.extend([f"# {_md_escape(ch['title'])}", ""])
        for sec in ch["sections"]:
            lines.extend([f"## {_md_escape(sec['heading'])}", ""])
            for btype, payload in sec["blocks"]:
                if btype == "p":
                    lines.extend([_md_escape(payload), ""])
                elif btype == "formula":
                    tex = _formula_tokens_to_latex(payload)
                    lines.extend(["$$", tex, "$$", ""])
                elif btype == "bullets":
                    for item in payload:
                        if ":" in item and item.index(":") < 110:
                            lead, rest = item.split(":", 1)
                            lines.append(
                                f"- **{_md_escape(lead)}:**{_md_escape(rest)}"
                            )
                        else:
                            lines.append(f"- {_md_escape(item)}")
                    lines.append("")
                elif btype == "table":
                    tbl_counter += 1
                    lines.extend(
                        _qmd_table_lines(
                            caption=payload["caption"],
                            headers=payload["headers"],
                            rows=payload["rows"],
                            aligns=payload.get("aligns"),
                            col_widths=payload.get("col_widths"),
                            source=payload.get("source", ""),
                            tbl_idx=tbl_counter,
                            keep_together=(
                                len(payload["rows"]) <= 13
                                and not payload["caption"].startswith(
                                    "Bảng 3.5:"
                                )
                            ),
                        )
                    )
                elif btype == "figure":
                    fig_counter += 1
                    fid = _slug_id("fig", payload["caption"], fig_counter)
                    rel_img = _qmd_relative_asset(payload["path"], qmd_dir)
                    width_pct = min(
                        100,
                        max(
                            65,
                            int(
                                round(
                                    float(payload.get("width_cm", 14.5))
                                    / 16.0
                                    * 100
                                )
                            ),
                        ),
                    )
                    lines.extend(
                        [
                            f"![{_md_escape(payload['caption'])}]({rel_img})"
                            f"{{#{fid} width={width_pct}% fig-"
                            f'align="center"}}',
                            "",
                        ]
                    )

    # --- TÀI LIỆU THAM KHẢO (23/23 CHUẨN APA 7TH — 12PT, AFTER 6PT, THỤT TREO
    # 1.27 CM, 0 VỠ DOI) ---
    lines.extend(
        [
            "{{< pagebreak >}}",
            "",
            "# TÀI LIỆU THAM KHẢO {.unnumbered}",
            "",
            "```{=typst}",
            "#[",
            "#set par(hanging-indent: 1.27cm, justify: false, leading: "
            "0.70em, spacing: 1.20em)",
            "#set text(size: 12pt)",
            "",
        ]
    )
    for ref_runs in content["references"]:
        lines.extend([_typst_apa_entry(ref_runs), ""])
    lines.extend(
        [
            "]",
            "```",
            "",
            '::: {.content-visible unless-format="typst"}',
            "::: {.apa-ref}",
        ]
    )
    for ref_runs in content["references"]:
        seg_out: list[str] = []
        for txt, is_ital in ref_runs:
            esc = _md_escape(txt)
            if is_ital and esc.strip():
                lead_ws = esc[: len(esc) - len(esc.lstrip())]
                trail_ws = esc[len(esc.rstrip()) :]
                seg_out.append(f"{lead_ws}*{esc.strip()}*{trail_ws}")
            else:
                seg_out.append(esc)
        lines.extend(["".join(seg_out), ""])
    lines.extend([":::", ":::", ""])

    # --- PHỤ LỤC A ---
    lines.extend(
        [
            "{{< pagebreak >}}",
            "",
            "# PHỤ LỤC A: MÃ NGUỒN PYTHON CHUẨN PEP 8 VÀ KIẾN TRÚC HỆ THỐNG "
            "{.unnumbered}",
            "",
            _md_escape(_appendix_a_intro(wm)),
            "",
            "## A.1. Sơ đồ kiến trúc kho mã nguồn và sản phẩm bàn giao "
            "{.unnumbered .unlisted}",
            "",
            "```text",
            content["appendix_a_tree"],
            "```",
            "",
        ]
    )
    tbl_counter += 1
    lines.extend(
        _qmd_table_lines(
            caption=(
                "Bảng A.1: Chỉ mục đầy đủ toàn bộ module Python trong kho "
                "mã nguồn dự án PRJ-EXT-01"
            ),
            headers=[
                "Module / Thư mục",
                "Chức năng chính trong quy trình Khoa học Dữ liệu",
            ],
            rows=content["appendix_a_module_index"],
            aligns=["L", "L"],
            col_widths=[5.2, 10.8],
            source=(
                "Nguồn: Rà soát trực tiếp cấu trúc và đếm dòng động trên "
                "kho mã nguồn nộp bài "
                "(ruff PEP 8 ≤ 79 ký tự: 0 lỗi trên toàn bộ .py & .ipynb)."
            ),
            tbl_idx=tbl_counter,
        )
    )
    lines.extend(
        [
            "## A.2. Toàn văn mã nguồn điều phối toàn trình `main.py` (100% "
            "nguyên bản chuẩn PEP 8) {.unnumbered .unlisted}",
            "",
            "```python",
            content["appendix_a_main_code"],
            "```",
            "",
            "## A.3. Toàn văn mã nguồn cấu hình đường dẫn & kiểm soát rò rỉ "
            "`src/config.py` (100% nguyên bản chuẩn PEP 8) {.unnumbered "
            ".unlisted}",
            "",
            "```python",
            content["appendix_a_config_code"],
            "```",
            "",
            "## A.4. Bốn trích đoạn thuật toán cốt lõi khớp 100% mã nguồn "
            "thực thi (Bảo toàn Surge, Khí tượng EST, Surge Cap & Monotonic "
            "Split-CQR) {.unnumbered .unlisted}",
            "",
            "```python",
            content["appendix_a_code"],
            "```",
            "",
        ]
    )

    # --- PHỤ LỤC B ---
    lines.extend(
        [
            "{{< pagebreak >}}",
            "",
            "# PHỤ LỤC B: HỒ SƠ KIỂM TOÁN VÀ NHẬT KÝ P-D-M-V {.unnumbered}",
            "",
            _md_escape(_appendix_b_intro(wm)),
            "",
        ]
    )
    tbl_counter += 1
    lines.extend(
        _qmd_table_lines(
            caption=(
                "Bảng B.1: Quy chiếu các mã kiểm toán nội bộ trong hồ sơ kỹ "
                "thuật sang biện pháp xử lý học thuật"
            ),
            headers=[
                "Mã kiểm toán",
                "Nội dung hạng mục đã rà soát và xử lý trong mã nguồn & "
                "tiểu luận",
            ],
            rows=content["appendix_b_audit_map"],
            aligns=["C", "L"],
            col_widths=[3.0, 13.0],
            source=(
                "Nguồn: src/clean_raw_data.py, src/simulation.py và hệ "
                "thống kiểm chứng thống kê hợp nhất (scripts/)."
            ),
            tbl_idx=tbl_counter,
            keep_together=True,
        )
    )
    tbl_counter += 1
    lines.extend(
        _qmd_table_lines(
            caption=(
                "Bảng B.2: Tóm tắt tiến độ các nhóm phiên làm việc P-D-M-V "
                "(01/10/2026 – 28/11/2026)"
            ),
            headers=[
                "Nhóm phiên",
                "Thời gian",
                "Thành quả kiểm chứng và Quyết định kỹ thuật cốt lõi",
            ],
            rows=content["appendix_b_milestones"],
            aligns=["C", "C", "L"],
            col_widths=[2.3, 3.0, 10.7],
            source="Nguồn: Trích xuất từ ./nhật ký làm việc.md.",
            tbl_idx=tbl_counter,
            keep_together=True,
        )
    )
    lines.extend(
        [
            "## B.3. Trích đoạn mục tiêu Phiên 12 (Biên bản P-D-M-V, "
            "bảng đối soát & SHA-256 từ `./nhật ký làm việc.md`) "
            "{.unnumbered .unlisted}",
            "",
        ]
    )
    lines.extend(_render_worklog_session_typst(content["appendix_b_verbatim"]))
    lines.extend(
        [
            '::: {.content-visible unless-format="typst"}',
            content["appendix_b_verbatim"],
            ":::",
            "",
        ]
    )

    qmd_text = "\n".join(lines)
    qmd_path.write_text(qmd_text, encoding="utf-8", newline="\n")
    print(
        f"DA TAO BAN QUARTO (.qmd): {qmd_path} "
        f"({qmd_path.stat().st_size / 1024:.1f} KB)"
    )
    return qmd_path


def build_final_thesis(
    qmd_only: bool = False,
    skip_docx: bool = False,
    student_id: str = DEFAULT_STUDENT_ID,
) -> tuple[Path, Path, Path]:
    """
    Quy trình tạo trọn bộ Tiểu luận Quarto (.qmd), PDF (.pdf từ Quarto) và Word
    dự phòng (.docx).
    """
    docx_out = ROOT_DIR / "reports" / f"{student_id}_Nhom02_TieuLuan.docx"
    pdf_out = ROOT_DIR / "reports" / f"{student_id}_Nhom02_TieuLuan.pdf"
    qmd_out = ROOT_DIR / "reports" / f"{student_id}_Nhom02_TieuLuan.qmd"

    data = load_verified_data()
    content = build_thesis_structure(data, student_id=student_id)

    export_thesis_qmd(content, qmd_out)
    if qmd_only:
        return docx_out, pdf_out, qmd_out

    _render_pdf_via_quarto(qmd_out, pdf_out)

    if not skip_docx:
        doc = Document()
        _configure_styles(doc)

        _build_cover_pages(doc, content["worklog_meta"])
        _build_front_matter(doc, content)
        _build_main_body_and_appendices(doc, content)

        docx_out.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(docx_out))
        _update_docx_toc_via_word(docx_out)
        print(
            f"DA TAO BAN WORD DU PHONG (.docx): {docx_out} "
            f"({docx_out.stat().st_size / (1024**2):.2f} MiB)"
        )

    print(
        f"DA XUAT BAN PDF TU QUARTO (.pdf): {pdf_out} "
        f"({pdf_out.stat().st_size / (1024**2):.2f} MiB)"
    )
    return docx_out, pdf_out, qmd_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Xuất bản Tiểu luận chuẩn UFM (.qmd -> .pdf qua Quarto/Typst & "
            "đồng bộ .docx) từ dữ liệu v5."
        )
    )
    parser.add_argument(
        "--qmd-only",
        action="store_true",
        help=(
            "Chỉ xuất/cập nhật tệp Quarto Markdown (.qmd) mà không biên "
            "dịch PDF/DOCX."
        ),
    )
    parser.add_argument(
        "--skip-docx",
        action="store_true",
        help=(
            "Xuất .qmd và biên dịch .pdf bằng Quarto, bỏ qua bước dựng bản "
            ".docx dự phòng."
        ),
    )
    parser.add_argument(
        "--student-id",
        default=DEFAULT_STUDENT_ID,
        help="MSSV đại diện cho tên tệp đầu ra (mặc định: 2521001060).",
    )
    args = parser.parse_args()
    build_final_thesis(
        qmd_only=args.qmd_only,
        skip_docx=args.skip_docx,
        student_id=args.student_id,
    )

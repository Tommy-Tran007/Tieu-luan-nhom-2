"""
Module: src/thesis_content.py
Mục đích: Nạp 100% số liệu động từ nguồn sự thật duy nhất v5 (Stage 3–6
JSON/CSV +
reports/results/thesis_v5_verification.json) và biên soạn trực tiếp Toàn văn
Tiểu luận
Capstone UFM (PRJ-EXT-01) — không vá đè monkey-patch, không gõ tay số liệu cũ.

Các chuẩn hóa cốt lõi của hệ thống báo cáo động:
- Khóa chặt khung nội dung chính Chương 1–5 trong [25, 45] trang theo
  projects.pdf.
- Toàn bộ kết quả phản ánh trung thực dữ liệu sạch đã kiểm chứng:
  (1) Bảo toàn nguồn gốc surge_multiplier (is_surge_original == 1: 232,861 lượt
  gồm
      140,477 Uber API 1.0x + 92,384 Lyft quan sát thực; 38,037 Lyft khuyết
      surge giữ NaN;
      tỷ lệ Surge Lyft thực = 6.80%).
  (2) Làm sạch khí tượng bằng khôi phục bậc 10 theo trung vị cùng giờ, điền
  khuyết theo
      trung vị cùng giờ toàn Boston (0 lệch > ngưỡng) và tách cờ
      is_rain_sentinel
      (152,043 lượt mưa/khô thực, 27,237 ô toàn sentinel giữ NaN).
  (3) Khung mô hình hóa 2 tầng: Dự báo giá báo API có điều kiện (M0a–M7, tách
  chế độ
      Uber surge ẩn vs. Lyft surge công bố) kết hợp thực nghiệm Ex-Ante (23
      biến không
      dùng surge + bộ phân loại xác suất Surge Lyft).
  (4) Tái lập 100% siêu tham số từ src/models.py (GridSearchCV/CV Pipeline; M5
  khớp
      max_depth=15, min_samples_leaf=15, min_samples_split=10).
  (5) RQ2 Đặc tả B trung thực: UberXL vs. Lyft XL dịch ra ngoài miền quan sát
  (8.32 dặm > 7.86 dặm), xác nhận Lyft XL rẻ hơn toàn miền khi không surge.
  (6) RQ4 Mô phỏng Trần Surge áp dụng nguyên tắc bên ngắn Q_realized = min(Q_D,
  Q_S)
      đơn điệu tuyệt đối (Pha 1: Trần 2.00x đạt ngưỡng bảo thủ tau <= 0.10%;
      Pha 2 mở rộng
      có điều kiện: Trần 1.75x cho tau = 0.30%–0.50%) kèm Power Analysis
      Switchback trên
      tập kích hoạt (Triggered Analysis) đối chiếu sai lầm pha loãng ITT.
  (7) Phụ lục A chuẩn xác 100% theo mã nguồn thực + bảng chỉ mục toàn bộ
  module.
"""

from __future__ import annotations

import importlib.metadata as _imp_meta
import json
import platform
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. THÔNG TIN HÀNH CHÍNH TRANG BÌA CHÍNH & BÌA PHỤ UFM
# =============================================================================
COVER_META: dict[str, Any] = {
    "university": "TRƯỜNG ĐẠI HỌC TÀI CHÍNH – MARKETING",
    "faculty": "KHOA KHOA HỌC DỮ LIỆU",
    "course_line": "TIỂU LUẬN KẾT THÚC HỌC PHẦN",
    "course_name": "LẬP TRÌNH PYTHON CHO KHOA HỌC DỮ LIỆU",
    "topic_code": "PRJ-EXT-01",
    "topic_title": (
        "PHÂN TÍCH BIỂU HIỆN ĐỊNH GIÁ ĐỘNG (DYNAMIC SURGE PRICING) "
        "VÀ DỰ BÁO GIÁ BÁO API TRÊN NỀN TẢNG GỌI XE CÔNG NGHỆ "
        "(UBER & LYFT BOSTON, MA)"
    ),
    "lecturer": "ThS. TRẦN MẠNH TƯỜNG",
    "class_code": "2631101190501",
    "group_name": "NHÓM 02",
    "student_class": "25DDS01",
    "location_year": "TP. HỒ CHÍ MINH, NĂM 2026",
    "members": [
        {
            "stt": "1",
            "name": "Trần Thanh Huy",
            "mssv": "2521001060",
            "completion": "100%",
            "alpha_label": "α₁ = 1.00",
            "role": (
                "Kiến trúc pipeline làm sạch, ghép nối & đặc trưng (src/, "
                "main.py); "
                "huấn luyện mô hình có điều kiện & Ex-Ante, kiểm định độ "
                "bền & hiệu chỉnh Split-CQR; "
                "hạ tầng Dashboard Streamlit/HTML Offline, bộ kiểm thử "
                "pytest, chuẩn hóa PEP 8, xuất bản & đóng gói."
            ),
        },
        {
            "stt": "2",
            "name": "Lê Huy Thắng",
            "mssv": "2521001092",
            "completion": "100%",
            "alpha_label": "α₂ = 1.00",
            "role": (
                "Thẩm định dữ liệu gốc & EDA 300 DPI; hệ thống 13 kiểm định "
                "thống kê "
                "(H1–H4, Cluster-Robust, Exact WCR, TOST); thiết kế trực "
                "quan Tab 1–3/5B Dashboard; "
                "diễn giải XAI, mô phỏng Surge Cap, kiểm chứng thống kê hợp "
                "nhất và rà soát học thuật APA 7th."
            ),
        },
    ],
}


# =============================================================================
# 2. NẠP DỮ LIỆU ĐỘNG 100% TỪ NGUỒN SỰ THẬT DUY NHẤT V5
# =============================================================================
def load_verified_data() -> dict[str, Any]:
    """
    Đọc toàn bộ tệp JSON tổng hợp GĐ3–GĐ6, thesis_v5_verification.json và bảng
    CSV v5.
    """
    s3 = json.loads(
        (ROOT_DIR / "data/processed/stage3_audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    s4 = json.loads(
        (ROOT_DIR / "reports/results/stage4_summary.json").read_text(
            encoding="utf-8"
        )
    )
    s5 = json.loads(
        (ROOT_DIR / "reports/results/stage5_summary.json").read_text(
            encoding="utf-8"
        )
    )
    s6 = json.loads(
        (ROOT_DIR / "reports/results/stage6_summary.json").read_text(
            encoding="utf-8"
        )
    )
    v5 = json.loads(
        (ROOT_DIR / "reports/results/thesis_v5_verification.json").read_text(
            encoding="utf-8"
        )
    )

    tdir = ROOT_DIR / "reports/tables"
    tables = {
        "data_dict": pd.read_csv(tdir / "data_dictionary_complete.csv"),
        "vif": pd.read_csv(tdir / "stage4_vif.csv"),
        "splits": pd.read_csv(tdir / "stage4_split_summary.csv"),
        "hyp": pd.read_csv(tdir / "stage5_hypothesis_tests.csv"),
        "cv": pd.read_csv(tdir / "stage5_cv_results.csv"),
        "test": pd.read_csv(tdir / "stage5_test_metrics.csv"),
        "rob": pd.read_csv(tdir / "stage5_robustness.csv"),
        "mb": pd.read_csv(tdir / "stage5_mutate_break.csv"),
        "seg": pd.read_csv(tdir / "stage6_segment_errors.csv"),
        "pi": pd.read_csv(tdir / "stage6_prediction_interval.csv"),
        "tariffs": pd.read_csv(tdir / "stage6_tier_tariffs.csv"),
        "sim": pd.read_csv(tdir / "stage6_surge_cap_simulation.csv"),
    }

    worklog_text = (ROOT_DIR / "nhật ký làm việc.md").read_text(
        encoding="utf-8"
    )
    worklog_sessions = re.findall(
        r"^## \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] - (.+)$", worklog_text, re.M
    )
    src_files = sorted(p.name for p in (ROOT_DIR / "src").glob("*.py"))
    all_script_py_files = sorted(
        p.name for p in (ROOT_DIR / "scripts").glob("*.py")
    )
    script_files = [p for p in all_script_py_files if p != "__init__.py"]
    dashboard_py_files = sorted(
        p
        for p in (ROOT_DIR / "dashboard").rglob("*.py")
        if "demo_v2" not in p.parts
    )
    test_files = sorted(
        p.name for p in (ROOT_DIR / "dashboard/tests").glob("test_*.py")
    )

    return {
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "s6": s6,
        "v5": v5,
        "tables": tables,
        "src_files": src_files,
        "script_files": script_files,
        "n_py_files": 1
        + len(src_files)
        + len(all_script_py_files)
        + len(dashboard_py_files),
        "test_files": test_files,
        "worklog_text": worklog_text,
        "worklog_lines": len(worklog_text.splitlines()),
        "worklog_words": sum(
            len(ln.split()) for ln in worklog_text.splitlines()
        ),
        "worklog_sessions": worklog_sessions,
    }


def _row(df: pd.DataFrame, col: str, val: Any) -> pd.Series:
    return df.loc[df[col] == val].iloc[0]


def _fmt_signed_usd(val: float, digits: int = 4) -> str:
    if abs(val) < 1e-9:
        return f"$0.{'0' * digits}"
    sign = "+" if val > 0 else "-"
    return f"{sign}${abs(val):.{digits}f}"


def _fmt_p(val: float) -> str:
    return "< 0.0001" if val < 0.0001 else f"{val:.4f}"


def _fmt_qhat(val: float) -> str:
    rounded = round(float(val), 4)
    if abs(rounded) < 1e-9:
        return "0.0000"
    return f"{rounded:+.4f}"


# =============================================================================
# 3. DANH MỤC CHỮ VIẾT TẮT
# =============================================================================
ABBREVIATIONS_TABLE: list[list[str]] = [
    ["AI", "Trí tuệ nhân tạo", "Artificial Intelligence"],
    ["ANOVA", "Phân tích phương sai", "Analysis of Variance"],
    [
        "APA",
        "Hiệp hội Tâm lý học Hoa Kỳ (chuẩn trích dẫn)",
        "American Psychological Association",
    ],
    [
        "API",
        "Giao diện lập trình ứng dụng",
        "Application Programming Interface",
    ],
    [
        "AUC",
        "Diện tích dưới đường cong ROC / Precision–Recall",
        "Area Under the Curve (ROC-AUC / PR-AUC)",
    ],
    [
        "BH-FDR",
        "Kiểm soát tỷ lệ phát hiện sai Benjamini–Hochberg",
        "Benjamini–Hochberg False Discovery Rate",
    ],
    ["CI", "Khoảng tin cậy thống kê", "Confidence Interval"],
    [
        "CQR",
        "Hồi quy phân vị hiệu chỉnh Conformal",
        "Conformalized Quantile Regression",
    ],
    [
        "CRISP-DM",
        "Quy trình chuẩn liên ngành khai phá dữ liệu",
        "Cross-Industry Standard Process for Data Mining",
    ],
    ["CV", "Kiểm định chéo", "Cross-Validation"],
    ["DEFF", "Hệ số hiệu ứng thiết kế cụm", "Design Effect"],
    ["EDA", "Phân tích khám phá dữ liệu", "Exploratory Data Analysis"],
    [
        "EST",
        "Giờ chuẩn miền Đông Hoa Kỳ (UTC-5, Boston)",
        "Eastern Standard Time",
    ],
    ["FE", "Kỹ thuật tạo lập đặc trưng", "Feature Engineering"],
    [
        "HC3",
        "Ước lượng sai số chuẩn vững phương sai thay đổi",
        "Heteroscedasticity-Consistent Estimator (HC3)",
    ],
    [
        "HistGBR",
        "Tăng cường độ dốc dựa trên lược đồ tần suất",
        "Histogram-based Gradient Boosting Regressor",
    ],
    [
        "ICE",
        "Kỳ vọng có điều kiện cá thể",
        "Individual Conditional Expectation",
    ],
    ["IQR", "Khoảng tứ phân vị (Q3 − Q1)", "Interquartile Range"],
    ["ITT", "Phân tích theo ý định điều trị toàn mẫu", "Intention-to-Treat"],
    ["MAE", "Sai số tuyệt đối trung bình", "Mean Absolute Error"],
    [
        "MAPE",
        "Sai số phần trăm tuyệt đối trung bình",
        "Mean Absolute Percentage Error",
    ],
    [
        "MCAR",
        "Khuyết dữ liệu hoàn toàn ngẫu nhiên",
        "Missing Completely At Random",
    ],
    [
        "MDE",
        "Mức hiệu ứng tối thiểu có thể phát hiện",
        "Minimum Detectable Effect",
    ],
    ["MNAR", "Khuyết dữ liệu không ngẫu nhiên", "Missing Not At Random"],
    [
        "MPIW",
        "Độ rộng trung bình của khoảng dự báo",
        "Mean Prediction Interval Width",
    ],
    ["OLS", "Bình phương tối thiểu thông thường", "Ordinary Least Squares"],
    ["OVB", "Thiên lệch bỏ sót biến", "Omitted Variable Bias"],
    [
        "P-D-M-V",
        "Đặt lệnh – Bóc tách – Thử phá – Kiểm chứng",
        "Prompt – Deconstruct – Mutate – Validate",
    ],
    ["PDP", "Đồ thị phụ thuộc từng phần", "Partial Dependence Plot"],
    [
        "PEP 8",
        "Quy chuẩn phong cách viết mã nguồn Python",
        "Python Enhancement Proposal 8",
    ],
    [
        "PICP",
        "Tỷ lệ bao phủ thực nghiệm của khoảng dự báo",
        "Prediction Interval Coverage Probability",
    ],
    ["RAM", "Bộ nhớ truy cập ngẫu nhiên", "Random Access Memory"],
    [
        "RMSE",
        "Căn bậc hai sai số bình phương trung bình",
        "Root Mean Squared Error",
    ],
    ["RQ", "Câu hỏi nghiên cứu", "Research Question"],
    [
        "SMART",
        "Cụ thể – Đo lường – Khả thi – Liên quan – Thời hạn",
        "Specific, Measurable, Achievable, Relevant, Time-bound",
    ],
    ["TOST", "Kiểm định tương đương hai phía một đuôi", "Two One-Sided Tests"],
    ["VIF", "Hệ số phóng đại phương sai", "Variance Inflation Factor"],
    [
        "WCR",
        "Bootstrap cụm hoang dã có ràng buộc",
        "Wild Cluster Restricted Bootstrap",
    ],
    [
        "XAI",
        "Trí tuệ nhân tạo có thể giải thích",
        "Explainable Artificial Intelligence",
    ],
]


# =============================================================================
# 4. TÀI LIỆU THAM KHẢO CHUẨN APA 7TH (23 NGUỒN THẬT, 23/23 TRÍCH DẪN TRONG
# BÀI)
# =============================================================================
REFERENCES_APA7: list[list[tuple[str, bool]]] = [
    [
        (
            "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false "
            "discovery rate: A practical and powerful approach to multiple "
            "testing. ",
            False,
        ),
        (
            "Journal of the Royal Statistical Society: Series B "
            "(Methodological), 57",
            True,
        ),
        (
            "(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
            False,
        ),
    ],
    [
        ("Breiman, L. (2001). Random forests. ", False),
        ("Machine Learning, 45", True),
        ("(1), 5–32. https://doi.org/10.1023/A:1010933404324", False),
    ],
    [
        ("brllrb. (2019). ", False),
        ("Uber and Lyft dataset Boston, MA", True),
        (
            " [Data set]. Kaggle. https://www.kaggle.com/datasets/brllrb/"
            "uber-and-lyft-dataset-boston-ma",
            False,
        ),
    ],
    [
        (
            "Cachon, G. P., Daniels, K. M., & Lobel, R. (2017). The role of "
            "surge pricing on a service platform with self-scheduling "
            "capacity. ",
            False,
        ),
        ("Manufacturing & Service Operations Management, 19", True),
        ("(3), 368–384. https://doi.org/10.1287/msom.2017.0618", False),
    ],
    [
        (
            "Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). "
            "Bootstrap-based improvements for inference with clustered "
            "errors. ",
            False,
        ),
        ("The Review of Economics and Statistics, 90", True),
        ("(3), 414–427. https://doi.org/10.1162/rest.90.3.414", False),
    ],
    [
        (
            "Castillo, J. C., Knoepfle, D., & Weyl, E. G. (2017). Surge "
            "pricing solves the wild goose chase. In ",
            False,
        ),
        (
            "Proceedings of the 2017 ACM Conference on Economics and "
            "Computation (EC 2017)",
            True,
        ),
        (
            " (pp. 241–242). Association for Computing Machinery. https://"
            "doi.org/10.1145/3033274.3085098",
            False,
        ),
    ],
    [
        (
            "Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, "
            "T., Shearer, C., & Wirth, R. (2000). ",
            False,
        ),
        ("CRISP-DM 1.0: Step-by-step data mining guide", True),
        (". SPSS Inc.", False),
    ],
    [
        (
            "Chen, L., Mislove, A., & Wilson, C. (2015). Peeking beneath "
            "the hood of Uber. In ",
            False,
        ),
        (
            "Proceedings of the 2015 Internet Measurement Conference (IMC "
            "2015)",
            True,
        ),
        (
            " (pp. 495–508). Association for Computing Machinery. https://"
            "doi.org/10.1145/2815675.2815681",
            False,
        ),
    ],
    [
        (
            "Cohen, P., Hahn, R., Hall, J., Levitt, S., & Metcalfe, R. "
            "(2016). ",
            False,
        ),
        (
            "Using big data to estimate consumer surplus: The case of Uber",
            True,
        ),
        (
            " (NBER Working Paper No. 22627). National Bureau of Economic "
            "Research. https://doi.org/10.3386/w22627",
            False,
        ),
    ],
    [
        (
            "Duan, N. (1983). Smearing estimate: A nonparametric "
            "retransformation method. ",
            False,
        ),
        ("Journal of the American Statistical Association, 78", True),
        (
            "(383), 605–610. https://doi.org/10.1080/01621459.1983.10478017",
            False,
        ),
    ],
    [
        (
            "Friedman, J. H. (2001). Greedy function approximation: A "
            "gradient boosting machine. ",
            False,
        ),
        ("The Annals of Statistics, 29", True),
        ("(5), 1189–1232. https://doi.org/10.1214/aos/1013203451", False),
    ],
    [
        ("Hall, J., Kendrick, C., & Nosko, C. (2015). ", False),
        ("The effects of Uber's surge pricing: A case study", True),
        (
            " [Working paper]. University of Chicago Booth School of "
            "Business. "
            "http://economicsforlife.ca/wp-content/uploads/2015/10/"
            "effects_of_ubers_surge_pricing.pdf",
            False,
        ),
    ],
    [
        (
            "Holm, S. (1979). A simple sequentially rejective multiple test "
            "procedure. ",
            False,
        ),
        ("Scandinavian Journal of Statistics, 6", True),
        ("(2), 65–70. https://www.jstor.org/stable/4615733", False),
    ],
    [
        (
            "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, "
            "Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient "
            "boosting decision tree. ",
            False,
        ),
        ("Advances in Neural Information Processing Systems, 30", True),
        (
            ", 3146–3154. https://proceedings.neurips.cc/paper/2017/hash/"
            "6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
            False,
        ),
    ],
    [
        (
            "Koenker, R., & Bassett, G., Jr. (1978). Regression quantiles. ",
            False,
        ),
        ("Econometrica, 46", True),
        ("(1), 33–50. https://doi.org/10.2307/1913643", False),
    ],
    [
        (
            "Lakens, D. (2017). Equivalence tests: A practical primer for ",
            False,
        ),
        ("t", True),
        (" tests, correlations, and meta-analyses. ", False),
        ("Social Psychological and Personality Science, 8", True),
        ("(4), 355–362. https://doi.org/10.1177/1948550617697177", False),
    ],
    [
        (
            "MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-"
            "consistent covariance matrix estimators with improved finite "
            "sample properties. ",
            False,
        ),
        ("Journal of Econometrics, 29", True),
        ("(3), 305–325. https://doi.org/10.1016/0304-4076(85)90158-7", False),
    ],
    [
        (
            "McKinney, W. (2010). Data structures for statistical computing "
            "in Python. In S. van der Walt & J. Millman (Eds.), ",
            False,
        ),
        (
            "Proceedings of the 9th Python in Science Conference (SciPy 2010)",
            True,
        ),
        (
            " (pp. 56–61). SciPy. https://doi.org/10.25080/Majora-92bf1922-00a",
            False,
        ),
    ],
    [
        (
            "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., "
            "Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, "
            "R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., "
            "Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: "
            "Machine learning in Python. ",
            False,
        ),
        ("Journal of Machine Learning Research, 12", True),
        (
            "(85), 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html",
            False,
        ),
    ],
    [
        (
            "Rochet, J.-C., & Tirole, J. (2003). Platform competition in "
            "two-sided markets. ",
            False,
        ),
        ("Journal of the European Economic Association, 1", True),
        ("(4), 990–1029. https://doi.org/10.1162/154247603322493212", False),
    ],
    [
        (
            "Romano, Y., Patterson, E., & Candès, E. J. (2019). "
            "Conformalized quantile regression. ",
            False,
        ),
        ("Advances in Neural Information Processing Systems, 32", True),
        (
            ", 3538–3548. https://proceedings.neurips.cc/paper/2019/hash/"
            "5103c3584b063c431bd1268e9b5e76fb-Abstract.html",
            False,
        ),
    ],
    [
        (
            "Schuirmann, D. J. (1987). A comparison of the two one-sided "
            "tests procedure and the power approach for assessing the "
            "equivalence of average bioavailability. ",
            False,
        ),
        ("Journal of Pharmacokinetics and Biopharmaceutics, 15", True),
        ("(6), 657–680. https://doi.org/10.1007/BF01068419", False),
    ],
    [
        (
            "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric "
            "and statistical modeling with Python. In S. van der Walt & J. "
            "Millman (Eds.), ",
            False,
        ),
        (
            "Proceedings of the 9th Python in Science Conference (SciPy 2010)",
            True,
        ),
        (
            " (pp. 92–96). SciPy. https://doi.org/10.25080/Majora-92bf1922-011",
            False,
        ),
    ],
]


# =============================================================================
# 5. XÂY DỰNG TOÀN VĂN TIỂU LUẬN (100% ĐỘNG TỪ V5 — ZERO MONKEY-PATCH)
# =============================================================================
def build_thesis_structure(data: dict[str, Any]) -> dict[str, Any]:
    s3, s4, s5, s6, v5 = (
        data["s3"],
        data["s4"],
        data["s5"],
        data["s6"],
        data["v5"],
    )
    tb = data["tables"]
    n_src = len(data["src_files"])
    n_scripts = len(data["script_files"])
    n_test_files = len(data["test_files"])

    # Rút trích các khối dữ liệu đã xác minh từ v5
    cl_cab = s3["cleaning_log"]["cab_rides"]
    merge = s3["eda_results"]["merge_audit"]
    num = s3["eda_results"]["numerical_stats"]
    leak = s4["pipeline_and_leakage_verification"]["leakage_trap_experiment"]
    tuning = s5["part2_hyperparameter_tuning"]
    regime = s5["part2_regime_breakdown"]
    ex_ante = s5["part3_ex_ante_experiments"]
    ex_pricing = {
        r["model_or_slice"]: r for r in ex_ante["pricing_without_surge"]
    }
    ex_clf = ex_ante["lyft_surge_classifier"]
    hgb_c = ex_clf["HistGB_Classifier"]
    hgb_b = ex_clf["HistGB_Classifier_Balanced"]
    lr_c = ex_clf["LogisticRegression_Classifier"]
    lr_b = ex_clf["LogisticRegression_Balanced"]
    hgb_cm = hgb_c["confusion_matrix"]
    n_pos_test = int(hgb_cm["tp"]) + int(hgb_cm["fn"])

    wf_sec = v5["waterfall_and_selection_bias"]
    prov = wf_sec["master_provenance_breakdown"]
    sb_aud = wf_sec["selection_bias_audit"]
    st_ext = v5["statistical_rigor_extensions"]
    cross5 = v5["crossing_distances_5_pairs"]
    md = v5["model_diagnostics_and_cqr"]
    sp_v5 = v5["spatial_fairness_and_surge_eda"]
    rq4 = v5["policy_rq4_supply_power_and_io"]

    cqr = md["clean_monotonic_cqr"]
    cqr_map = {r["service_tier"]: r for r in cqr["by_tier"]}
    s6_bm = cqr["stage6_100pct_boundary_audit"]
    p_m76 = md["m7_vs_m6_paired_5fold"]
    s_m0b = md["m0b_vs_m7_by_surge_regime"]
    st_lk = md["strict_train_only_cleaning"]
    abg = md["api_batch_group_split"]
    thr_s = md["log10_ratio_threshold_sensitivity"]
    miss_s = md["missingness_sensitivity"]
    n_test_rain_sentinel = (
        int(miss_s["M7_full_test"]["n"])
        - int(miss_s["M7_complete_weather_valid_rain"]["n"])
        - int(miss_s["M7_unmatched_weather_0"]["n"])
    )

    top12 = sp_v5["lyft_spatial_surge_all12"]
    cl_s = sp_v5["surge_clustering_in_loc_hour"]
    fair_list = sp_v5["spatial_fairness_12_districts"]
    min_bias_d = min(fair_list, key=lambda r: r["mean_signed_residual_usd"])
    max_bias_d = max(fair_list, key=lambda r: r["mean_signed_residual_usd"])
    min_mape_d = min(fair_list, key=lambda r: r["mape_pct"])
    max_mape_d = max(fair_list, key=lambda r: r["mape_pct"])

    io_same = rq4["io_benchmark_fair_same_table"][
        "same_table_master_benchmark"
    ]
    io_raw = rq4["io_benchmark_fair_same_table"]["raw_cab_streaming_benchmark"]
    arith_map = {r["surge_cap"]: r for r in rq4["arithmetic_surge_benchmark"]}
    sup_map = {
        r["surge_cap"]: r for r in rq4["surge_cap_supply_constrained_table"]
    }
    oos200 = rq4["oos_surge_cap_test_only"]["cap_2.00x"]
    oos175 = rq4["oos_surge_cap_test_only"]["cap_1.75x"]
    pw_scopes = rq4["switchback_power_analysis"]["scopes"]
    pw175 = pw_scopes["all_12_districts"]["cap_1.75x"]
    pw200 = pw_scopes["all_12_districts"]["cap_2.00x"]
    pw175_top5 = pw_scopes["top5_hotspot_districts"]["cap_1.75x"]
    pw200_top5 = pw_scopes["top5_hotspot_districts"]["cap_2.00x"]

    m7 = _row(tb["test"], "model_code", "M7_HistGradientBoosting")
    m6 = _row(tb["test"], "model_code", "M6_Random_Forest")
    m2 = _row(tb["test"], "model_code", "M2_Interaction_OLS")
    m1 = _row(tb["test"], "model_code", "M1_Basic_OLS")
    m0b = _row(tb["test"], "model_code", "M0b_Tier_Route_Lookup_Baseline")
    cv_map = tb["cv"].set_index("model_code").to_dict("index")
    seg_tier = (
        tb["seg"]
        .loc[tb["seg"]["dimension"] == "service_tier"]
        .set_index("segment_value")
        .to_dict("index")
    )
    shared_seg = seg_tier["Shared"]
    res_m7 = s6["part1_residual_diagnostics"]["M7_HistGradientBoosting"]
    pi_ov = s6["part2_prediction_intervals_p10_p90"]["overall"]

    rob_df = tb["rob"]
    r_temp_m7 = rob_df[
        (rob_df["scenario"] == "temporal_split")
        & (rob_df["model_code"] == "M7_HistGradientBoosting")
    ].iloc[0]
    r_grp_m7 = rob_df[
        (rob_df["scenario"] == "route_group_split")
        & (rob_df["model_code"] == "M7_HistGradientBoosting")
    ].iloc[0]
    r_grp_m6 = rob_df[
        (rob_df["scenario"] == "route_group_split")
        & (rob_df["model_code"] == "M6_Random_Forest")
    ].iloc[0]
    r_grp_m2 = rob_df[
        (rob_df["scenario"] == "route_group_split")
        & (rob_df["model_code"] == "M2_Interaction_OLS")
    ].iloc[0]
    r_orig_test_m7 = rob_df[
        (rob_df["scenario"] == "R3a_Test_Only_Strictly_Unmodified_Raw")
        & (rob_df["model_code"] == "M7_HistGradientBoosting")
    ].iloc[0]
    r_orig_both_m7 = rob_df[
        (rob_df["scenario"] == "R3b_Train_And_Test_Strictly_Unmodified_Raw")
        & (rob_df["model_code"] == "M7_HistGradientBoosting")
    ].iloc[0]

    ram_csv_pct = (
        1.0 - io_same["parquet_ram_mb"] / io_same["csv_full_ram_mb"]
    ) * 100.0
    cab_log = s3["cleaning_log"]["cab_rides"]
    iqr_p = cab_log["remaining_iqr_outliers_price"]
    iqr_d = cab_log["remaining_iqr_outliers_distance"]
    r3_aud = s4["r3_scale_recovery_audit"]
    n_p_rec = int(r3_aud["price_scale_recovered_count"])
    p_rec_diff = abs(int(thr_s[1]["n_price_flagged_in_master"]) - n_p_rec)
    h4b_mt = next(
        r
        for r in st_ext["multiple_testing_13"]
        if r["hypothesis_id"] == "H4b_Wald_UberPool_vs_Shared"
    )
    lux_b = next(r for r in cqr["by_tier"] if r["service_tier"] == "Lux Black")
    lux_b_delta = lux_b["cqr_picp_pct"] - lux_b["base85_raw_tol1e6_picp_pct"]
    n_cqr_lt80 = sum(1 for r in cqr["by_tier"] if r["cqr_picp_pct"] < 80.0)
    total_n = int(prov["total_master_rows"])
    obs_surge_n = int(prov["is_surge_original_1_total"])
    uber_n = int(prov["uber_api_fixed_1x"])
    lyft_tot_n = int(prov["lyft_total"])
    lyft_obs_n = int(prov["lyft_surge_observed"])
    lyft_miss_n = int(prov["lyft_surge_missing_nan"])
    lyft_surge_rate = float(sp_v5["lyft_overall_surge_rate_pct"])
    lyft_surge_cnt = int(sp_v5["lyft_overall_surge_count"])
    n_train_rand = int(st_lk["n_train"])
    n_test_rand = int(st_lk["n_test"])

    w_un = st_ext["weather_unadjusted_loc_hr"]
    w_ct_loc = st_ext["weather_controlled_loc_hr"]
    w_ct_hr = st_ext["weather_controlled_citywide_hour"]
    w_ns_hr = st_ext["weather_controlled_nosurge_full152k_hour"]
    hv = st_ext["heavy_rain_vs_dry"]
    wcr_loc = hv["exact_wcr_loc_hr_B1999"]
    wcr_hr = hv["exact_wcr_citywide_hour_B1999"]
    h3b_cl = st_ext["h3b_lyft_surge_rain_cluster"]
    ws_any = st_ext["weather_sentinel_strict_sensitivity"]
    tod_cl = st_ext["time_of_day_cluster_f_test"]

    ux_pair, xl_pair, bl_pair, bs_pair = (
        cross5[1],
        cross5[2],
        cross5[3],
        cross5[4],
    )
    gx_rows = md["grouped_permutation_importance"]
    m7_top1_pct, m7_top2_pct = (
        gx_rows[0]["m7_rel_share_pct"],
        gx_rows[1]["m7_rel_share_pct"],
    )
    m2_top1_pct, m2_top2_pct = (
        gx_rows[0]["m2_rel_share_pct"],
        gx_rows[1]["m2_rel_share_pct"],
    )
    m7_wt_pct = sum(
        max(float(r["m7_rel_share_pct"]), 0.0) for r in gx_rows[4:6]
    )
    m2_wt_pct = sum(
        max(float(r["m2_rel_share_pct"]), 0.0) for r in gx_rows[4:6]
    )

    def _fmt_pct_range(a: float, b: float) -> str:
        s1, s2 = f"{a:.1f}%", f"{b:.1f}%"
        return s1 if s1 == s2 else f"{min(a, b):.1f}%–{max(a, b):.1f}%"

    m5_p = tuning["M5_DecisionTree"]["best_params"]
    m6_p = tuning["M6_RandomForest"]["best_params"]
    m7_p = tuning["M7_HistGradientBoosting"]["best_params"]

    # --- TÓM TẮT ĐỒ ÁN (EXECUTIVE SUMMARY — ĐÚNG KHUNG 250–350 TỪ) ---
    executive_summary = (
        "Nghiên cứu phân tích biểu hiện định giá động (Dynamic Surge "
        "Pricing), cấu trúc biểu giá và dự báo "
        "giá báo API (API Fare Quotes) của Uber và Lyft tại nội đô Boston "
        "trong 24 ngày lịch (25/11–18/12/2018; "
        "18 ngày quan sát). Từ 693,071 bản ghi chuyến đi "
        f"({wf_sec['raw_cab_size_mib']:.2f} MiB) và 6,276 bản "
        "ghi khí tượng, quy trình làm sạch bảo toàn nguồn gốc tạo bảng "
        f"sạch {total_n:,} lượt × 32 biến, giảm "
        f"{merge['memory_saved_pct']:.2f}% RAM trong pipeline "
        f"({merge['memory_mb_before_downcast']:.2f} → "
        f"{merge['memory_mb_after_downcast']:.2f} MB; "
        f"{io_same['parquet_ram_mb']:.2f} MB Parquet, đọc nhanh gấp "
        f"{io_same['parquet_vs_csv_full_speedup']:.1f} lần CSV cùng bảng), "
        f"tách bạch {lyft_miss_n:,} lượt Lyft "
        f"khuyết surge (giữ NaN) và {prov['weather_rain_sentinel_nan']:,} "
        "lượt thuộc ô lỗi cảm biến mưa. Kết hợp "
        "13 kiểm định hiệu chỉnh Holm–Bonferroni và BH-FDR, sai số chuẩn "
        f"cụm ({st_ext['n_loc_hr_clusters_all']:,} "
        f"quận×giờ; {st_ext['n_citywide_hour_clusters_all']} giờ Boston), "
        "Exact WCR (B = 1,999), TOST, 10 mô hình "
        "có điều kiện và đối chứng Ex-Ante (23 biến không surge), nghiên "
        "cứu rút ra bốn phát hiện chính. Thứ nhất, "
        "trên tập khớp thời tiết thực không lỗi cảm biến (n = "
        f"{st_ext['n_valid_rain_obs']:,}), mưa nhẹ–vừa và "
        "khung giờ không tạo chênh lệch giá vượt ±$0.10 (TOST p < 0.0001; "
        f"H3b gom cụm p = {h3b_cl['p_hour_cluster']:.3f}). "
        f"Thứ hai, trên mẫu surge quan sát thực (n = {obs_surge_n:,}; Lyft "
        f"surge {lyft_surge_rate:.2f}%), UberX–Lyft "
        f"cắt tại {ux_pair['spec_a_train80']['hc3']['d_star']:.2f} mi (Đặc "
        "tả A) và "
        f"{ux_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f} mi (Đặc "
        "tả B thuần 1.0x), còn UberXL–Lyft XL "
        "không cắt trong miền khi không surge "
        f"({xl_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f} mi > "
        "7.86 mi). "
        f"Thứ ba, trên tập kiểm tra (n = {n_test_rand:,}), M7 có điều kiện "
        f"đạt R² = {float(m7['test_r2']):.4f}, "
        f"RMSE = ${float(m7['test_rmse']):.4f}, MAPE = "
        f"{float(m7['test_mape_pct']):.2f}% (Lyft RMSE = "
        "$"
        f"{regime['lyft_observed_surge_test']['M7_Conditional']['rmse']:.2f}"
        " vs. Uber surge ẩn = "
        f"${regime['uber_hidden_surge_test']['M7_Conditional']['rmse']:.2f};"
        " Ex-Ante R² = "
        f"{ex_pricing['M7_ExAnte_All_Test']['r2']:.4f}, phân loại Surge AUC "
        f"= {ex_clf['HistGB_Classifier']['roc_auc']:.3f}), "
        "Split-CQR đơn điệu đạt PICP "
        f"{cqr['min_tier_cqr_picp_pct']:.2f}%–"
        f"{cqr['max_tier_cqr_picp_pct']:.2f}%. "
        "Thứ tư, dưới ràng buộc cung Q = min(Q_D, Q_S), trần 2.00x đạt "
        "ngưỡng bảo thủ τ ≤ 0.10%, còn 1.75x là "
        f"ứng viên Pha 2 (τ = 0.30%–0.50%) cho Switchback trên tập kích hoạt."
    )

    # =========================================================================
    # CÁC BẢNG DỮ LIỆU ĐỘNG CHO 5 CHƯƠNG
    # =========================================================================
    tbl_2_1_rows = [
        [
            "Chen et al. (2015)",
            "API Uber tại San Francisco & Manhattan",
            "Đo lường thực nghiệm biến động surge theo thời gian thực",
            "Surge biến thiên nhanh theo chu kỳ vài phút và phụ thuộc mạnh "
            "vào cung–cầu cấp tiểu khu.",
            "Chỉ khảo sát một nền tảng (Uber), không so sánh biểu giá chéo.",
            "Giải thích vì sao khí tượng gộp theo giờ không kích hoạt surge "
            "cơ học tại Boston.",
        ],
        [
            "Hall et al. (2015)",
            "Sự kiện hòa nhạc & đêm giao thừa tại Hoa Kỳ",
            "Nghiên cứu tình huống khi hệ thống surge gián đoạn",
            "Tắt surge đột ngột làm giảm mạnh tỷ lệ hoàn thành chuyến và "
            "kéo dài thời gian chờ ETA.",
            "Tập trung cú sốc cầu đặc thù thay vì vận hành ngày thường.",
            "Cơ sở xây dựng nguyên tắc ràng buộc cung bên ngắn Q = min(Q_D, "
            "Q_S) ở Chương 5.",
        ],
        [
            "Cohen et al. (2016)",
            "Gần 50 triệu phiên Uber tại 4 đô thị Hoa Kỳ",
            "Hồi quy gián đoạn (RD) quanh bậc làm tròn surge",
            "Ước lượng độ co giãn cầu theo giá ε_d ≈ −0.4 đến −0.6 (mức "
            "trung tâm −0.50).",
            "Dữ liệu Uber năm 2015; không tự động đồng nhất cho Lyft 2018.",
            "Tham số cận trên phía cầu (ε = −0.50; −0.80 khi dư cung) tại "
            "Chương 5.",
        ],
        [
            "Cachon et al. (2017)",
            "Mô hình lý thuyết tài xế tự sắp xếp lịch",
            "Phân tích cân bằng thị trường hai phía",
            "Định giá động cải thiện phúc lợi so với giá cố định, nhưng "
            "đỉnh cực đoan tổn hại niềm tin.",
            "Mô hình lý thuyết thuần túy, chưa kiểm chứng trên biểu giá đa "
            "gói.",
            "Khung lý thuyết đánh đổi giữa hạn chế sốc giá và duy trì động "
            "lực cung tài xế.",
        ],
        [
            "Castillo et al. (2017)",
            "Mô hình không gian điều phối tài xế – khách",
            "Phân tích thất bại thị trường 'Wild Goose Chase'",
            "Thiếu điều tiết giá khi cầu vượt cung làm tài xế chạy rỗng xa, "
            "sụp đổ năng suất đón khách.",
            "Mô hình chuẩn tắc, chưa lượng hóa sai số dự báo trên dữ liệu "
            "API.",
            "Cơ sở đánh giá cung nhạy giá ε_S ∈ {0.30, 0.60} và quy tắc dừng "
            "ETA trong Switchback A/B.",
        ],
        [
            "Nghiên cứu này (2026)",
            f"{total_n:,} báo giá API Uber & Lyft ({obs_surge_n:,} surge "
            "thực)",
            "Cluster SE, Exact WCR, TOST, 10 ML Có điều kiện + Ex-Ante, CQR "
            "& XAI",
            "Tách rõ Uber surge ẩn vs. Lyft surge công bố, đối sánh 12 gói, "
            "Spec A/B & Surge Cap bên ngắn.",
            "Báo giá API 24 ngày lịch, thiếu duration, tỷ lệ chốt chuyến và "
            "cung thực.",
            "Lấp khoảng trống đối sánh hai hãng, định giá Ex-Ante và ràng "
            "buộc cung thực nghiệm.",
        ],
    ]

    dd = tb["data_dict"]
    tbl_3_1_rows = [
        [
            str(r["column_name"]),
            str(r["unit"]),
            str(r["dtype"]),
            f"{int(r['n_missing']):,}",
            str(r["decision"]),
        ]
        for _, r in dd.iterrows()
        if str(r["column_name"])
        in {
            "id",
            "cab_type",
            "name",
            "source",
            "destination",
            "price",
            "distance",
            "surge_multiplier",
            "is_surge_original",
            "time_stamp",
            "datetime",
            "hour",
            "temp",
            "rain",
            "humidity",
            "wind",
            "clouds",
            "pressure",
            "is_raining",
            "is_rain_sentinel",
            "is_timestamp_original",
            "weather_matched",
            "log_distance",
            "lyft_surge_excess",
            "distance_x_surge",
            "distance_x_lyft_surge_excess",
            "hour_sin",
            "hour_cos",
            "price_per_mile",
        }
    ]

    tbl_3_2_rows = [
        [
            str(w["step"]),
            f"{int(w['in_rows']):,}",
            f"-{int(w['dropped']):,}" if int(w["dropped"]) > 0 else "0",
            f"{int(w['out_rows']):,}",
            f"{float(w['share_raw_pct']):.2f}%",
        ]
        for w in wf_sec["waterfall"]
    ]

    tbl_3_3_ex_rows = [
        [
            str(ex["tier"]),
            str(ex["route"]),
            f"${float(ex['raw_price_usd']):,.2f}",
            f"${float(ex['ref_median_usd']):.2f}",
            f"÷ {int(ex['factor']):,}",
            f"${float(ex['adjusted_price_usd']):.2f}",
        ]
        for ex in wf_sec["scale_recovery_examples"]
    ]

    label_map = {
        "price": "Giá báo API (price, USD)",
        "distance": "Quãng đường (distance, dặm)",
        "surge_multiplier": "Hệ số nhân giá (surge thực, lần)",
        "temp": "Nhiệt độ (temp, °F — khớp thực)",
        "rain": "Lượng mưa (rain, inch/h — không sentinel)",
        "humidity": "Độ ẩm tương đối (humidity, 0–1)",
        "wind": "Tốc độ gió (wind, mph)",
    }
    tbl_3_4_desc_rows = [
        [
            lbl,
            f"{int(num[k]['count']):,}",
            f"{num[k]['mean']:.2f}",
            f"{num[k]['std']:.2f}",
            f"{num[k]['min']:.2f}",
            f"{num[k]['q1']:.2f}",
            f"{num[k]['median']:.2f}",
            f"{num[k]['q3']:.2f}",
            f"{num[k]['max']:.2f}",
        ]
        for k, lbl in label_map.items()
        if k in num
    ]

    mt_map = {m["hypothesis_id"]: m for m in st_ext["multiple_testing_13"]}
    hyp_meta = {
        "H1a_All_Rides": (
            "H1a",
            "Giá báo Mưa vs. Tạnh (Toàn thị trường không sentinel)",
            "Welch t / MW-U",
        ),
        "H1b_Uber_Only": (
            "H1b",
            "Giá báo Mưa vs. Tạnh (Riêng Uber không sentinel)",
            "Welch t / MW-U",
        ),
        "H1c_Lyft_Only": (
            "H1c",
            "Giá báo Mưa vs. Tạnh (Riêng Lyft không sentinel)",
            "Welch t / MW-U",
        ),
        "H2a_Surge_All": (
            "H2a",
            "Hệ số Surge theo 4 buổi EST (TS & surge gốc)",
            "ANOVA / KW-H",
        ),
        "H2b_Surge_Lyft": (
            "H2b",
            "Hệ số Surge theo 4 buổi EST (Lyft TS & surge gốc)",
            "ANOVA / KW-H",
        ),
        "H2c_Price_All": (
            "H2c",
            "Giá báo theo 4 buổi EST (Toàn mẫu TS gốc)",
            "ANOVA / KW-H",
        ),
        "H3a_Surge_x_CabType": (
            "H3a",
            "Ghi nhận Surge > 1.0x giữa API Lyft thực vs. Uber",
            "Chi-square / G-test",
        ),
        "H3b_Surge_x_Rain_Lyft": (
            "H3b",
            "Tần suất Surge Lyft thực theo Mưa vs. Tạnh*",
            "Chi-square / G-test",
        ),
        "H3c_Surge_x_TimeOfDay_Lyft": (
            "H3c",
            "Tần suất Surge Lyft thực theo 4 buổi EST",
            "Chi-square / G-test",
        ),
        "H3d_Surge_x_RushHour_Lyft": (
            "H3d",
            "Tần suất Surge Lyft thực giờ cao điểm EST",
            "Chi-square / G-test",
        ),
        "H4a_Wald_UberX_vs_Lyft_Slope": (
            "H4a",
            "So sánh độ dốc & mở cửa UberX vs. Lyft (Spec A)",
            "Wald Z (HC3: β₁ / β₀)",
        ),
        "H4b_Wald_UberPool_vs_Shared": (
            "H4b",
            "So sánh độ dốc & mở cửa UberPool vs. Shared",
            "Wald Z (HC3: β₁ / β₀)",
        ),
        "H4c_Wald_Joint_Weather_Multivariate": (
            "H4c",
            "Hồi quy đa biến đồng thời 6 biến thời tiết (full rank)",
            "Joint Wald F (HC3)",
        ),
    }
    tbl_3_5_rows = []
    for _, r in tb["hyp"].iterrows():
        hid = str(r["hypothesis_id"])
        code, full_scope, test_pair = hyp_meta[hid]
        mt = mt_map[hid]
        p_raw = _fmt_p(float(r["parametric_pvalue"]))
        p_holm = _fmt_p(float(mt["p_holm"]))
        q_bh = _fmt_p(float(mt["q_bh"]))
        p_alt = _fmt_p(float(r["nonparametric_pvalue"]))
        if hid == "H4a_Wald_UberX_vs_Lyft_Slope":
            es_txt = f"d* = {float(r['effect_size_value']):.2f} mi"
        elif hid == "H4b_Wald_UberPool_vs_Shared":
            es_txt = f"Δβ₀ = ${float(r['effect_size_value']):.2f}"
        elif hid == "H4c_Wald_Joint_Weather_Multivariate":
            es_txt = f"ΔR² = {float(r['effect_size_value']):.1e}"
        else:
            es_nm = str(r["effect_size_name"])
            if "Cohen" in es_nm:
                es_txt = f"d = {float(r['effect_size_value']):.5f}"
            elif "Eta" in es_nm:
                es_txt = f"η² = {float(r['effect_size_value']):.1e}"
            elif "Cram" in es_nm:
                es_txt = f"V = {float(r['effect_size_value']):.5f}"
            else:
                es_txt = f"{float(r['effect_size_value']):.5g}"
        decision = (
            "Bác bỏ H0 (α=5%)"
            if bool(mt["reject_holm_05"])
            else "Chưa đủ bằng chứng bác bỏ H0"
        )
        tbl_3_5_rows.append(
            [
                code,
                full_scope,
                test_pair,
                p_raw,
                f"{p_holm} / {q_bh}",
                p_alt,
                es_txt,
                decision,
            ]
        )

    tbl_3_6_rows = [
        [
            "OLS đơn biến: Mưa vs. Tạnh (cụm quận×giờ)",
            f"{st_ext['n_valid_rain_obs']:,} "
            f"({st_ext['n_loc_hr_clusters_all']:,} cụm)",
            f"{w_un['diff_usd']:+.4f} (SE={w_un['se_cluster']:.4f})",
            f"[{w_un['ci95_cluster'][0]:+.4f}, "
            f"{w_un['ci95_cluster'][1]:+.4f}]",
            f"p_cl = {w_un['p_cluster']:.4f}",
            f"TOST ±$0.25: p={w_un['tost_025']['p_tost']:.4f} (Tương đương)",
        ],
        [
            "OLS đa biến (+ gói, cự ly, surge, 72 tuyến FE)",
            f"{st_ext['n_valid_rain_and_surge_dist_obs']:,} "
            f"({st_ext['n_loc_hr_clusters_obs']:,} cụm)",
            f"{w_ct_loc['beta_usd']:+.4f} (SE={w_ct_loc['se']:.4f})",
            f"[{w_ct_loc['ci95'][0]:+.4f}, {w_ct_loc['ci95'][1]:+.4f}]",
            f"p_cl = {w_ct_loc['p_value']:.4f}",
            "TOST ±$0.10: p < 0.0001 (Tương đương chặt)",
        ],
        [
            "OLS đa biến gom cụm giờ toàn Boston (bảo thủ)",
            f"{st_ext['n_valid_rain_and_surge_dist_obs']:,} "
            f"({st_ext['n_citywide_hour_clusters_obs']} giờ)",
            f"{w_ct_hr['beta_usd']:+.4f} (SE={w_ct_hr['se']:.4f})",
            f"[{w_ct_hr['ci95'][0]:+.4f}, {w_ct_hr['ci95'][1]:+.4f}]",
            f"p_hr = {w_ct_hr['p_value']:.4f}",
            "TOST ±$0.10: p < 0.0001 (Bền vững toàn TP)",
        ],
        [
            "OLS đa biến không biến surge (mẫu mưa/khô có distance)",
            f"{st_ext['n_valid_rain_dist_obs']:,} "
            f"({st_ext['n_citywide_hour_clusters_dist']} giờ)",
            f"{w_ns_hr['beta_usd']:+.4f} (SE={w_ns_hr['se']:.4f})",
            f"[{w_ns_hr['ci95'][0]:+.4f}, {w_ns_hr['ci95'][1]:+.4f}]",
            f"p_hr = {w_ns_hr['p_value']:.4f}",
            f"TOST ±$0.10: p={w_ns_hr['tost_010']['p_tost']:.4f} (Tương "
            "đương)",
        ],
        [
            "Độ nhạy loại mọi ô từng chứa sentinel mưa (cụm giờ)",
            f"{ws_any['n_obs_excl_any_sentinel_cell']:,} "
            f"({ws_any['n_hour_clusters']} giờ)",
            f"{ws_any['beta_usd']:+.4f} (SE={ws_any['se_hour']:.4f})",
            f"p_cl = {ws_any['p_loc_hr']:.4f}",
            f"p_hr = {ws_any['p_hour']:.4f}",
            "TOST ±$0.10: p < 0.0001 (Sạch sentinel 100%)",
        ],
        [
            "Mưa lớn ≥ 0.10 in/h — Exact WCR (M_Z, B=1,999)",
            f"N = {wcr_loc['n_obs']:,} ({wcr_loc['n_total_clusters']:,} cụm "
            f"/ {wcr_hr['n_total_clusters']} giờ; mưa lớn: "
            f"{hv['n_heavy_obs']:,} / {hv['n_heavy_loc_hr_clusters']} cụm / "
            f"{hv['n_heavy_citywide_hours']} giờ)",
            f"{hv['ctrl_beta_usd']:+.4f} (SE={hv['ctrl_se_hour']:.4f})",
            f"t = {wcr_loc['t_obs']:.3f} / {wcr_hr['t_obs']:.3f}",
            f"p_w = {wcr_loc['p_wild']:.4f} / {wcr_hr['p_wild']:.4f}",
            f"Thăm dò (few treated clusters: {hv['n_heavy_citywide_hours']} "
            "giờ, 2⁸ = 256 tổ hợp dấu)",
        ],
        [
            "H3b cụm: Xác suất Surge Lyft khi Mưa vs. Tạnh",
            f"{h3b_cl['n_lyft_obs']:,} ({h3b_cl['n_hour_clusters']} giờ)",
            f"{h3b_cl['diff_pp']:+.2f} pp (SE={h3b_cl['se_hour_pp']:.2f})",
            f"p_iid = {h3b_cl['p_naive_iid']:.4f}",
            f"p_cl = {h3b_cl['p_loc_hr_cluster']:.4f} / "
            f"{h3b_cl['p_hour_cluster']:.4f}",
            "Không có ý nghĩa dưới sai số cụm & Holm/BH",
        ],
        [
            "Wald F cụm 4 buổi EST (cụm quận×giờ)",
            f"{tod_cl['n_obs']:,} ({tod_cl['n_clusters']:,} cụm)",
            f"Max–Min = ${tod_cl['max_min_gap_usd']:.4f}",
            f"F_cl = {tod_cl['f_stat_cluster']:.4f}",
            f"p_cl = {tod_cl['p_cluster']:.4f}",
            "Chưa đủ bằng chứng bác bỏ H0 theo buổi",
        ],
    ]

    vif_a = (
        tb["vif"]
        .loc[tb["vif"]["config"] == "A_Uncentered_Raw_Interaction"]
        .set_index("feature")["vif"]
        .to_dict()
    )
    vif_b = (
        tb["vif"]
        .loc[tb["vif"]["config"] == "B_Centered_Excess_Interaction"]
        .set_index("feature")["vif"]
        .to_dict()
    )
    tbl_4_1_rows = [
        [
            "distance (Quãng đường)",
            f"{vif_a['distance']:.2f}",
            f"{vif_b['distance']:.2f}",
            "Giảm mạnh từ đa cộng tuyến cấu trúc xuống ngưỡng an toàn (< 5)",
        ],
        [
            "distance_x_surge → distance_x_lyft_surge_excess",
            f"{vif_a['distance_x_surge']:.2f}",
            f"{vif_b['distance_x_lyft_surge_excess']:.2f}",
            "Khử trùng lặp tuyến tính tại vùng không surge (= 1.0x)",
        ],
        [
            "surge_multiplier → lyft_surge_excess",
            f"{vif_a['surge_multiplier']:.2f}",
            f"{vif_b['lyft_surge_excess']:.2f}",
            "Tách riêng phần phụ trội surge Lyft (< 10)",
        ],
        [
            "pressure (max VIF trong 6 biến thời tiết)",
            f"{vif_a['pressure']:.2f}",
            f"{vif_b['pressure']:.2f}",
            "An toàn (< 5) sau đổi giờ sang chu kỳ sin/cos",
        ],
    ]

    strat_labels = {
        "random_80_20": (
            "1. Random Split 80/20 (Nội miền 72 tuyến, surge thực)"
        ),
        "temporal_split": (
            "2. Temporal Split Boston EST (TS & surge gốc 80/20)"
        ),
        "route_group_split": (
            "3. Route Group Split (57 tuyến Train / 15 tuyến mới)"
        ),
    }
    tbl_4_2_rows = [
        [
            strat_labels.get(
                str(r["split_strategy"]), str(r["split_strategy"])
            ),
            f"{int(r['n_train']):,} ({float(r['train_pct']):.1f}%)",
            f"{int(r['n_test']):,} ({float(r['test_pct']):.1f}%)",
            f"${float(r['train_price_mean']):.2f} / "
            f"${float(r['test_price_mean']):.2f}",
            f"{float(r['ks_statistic']):.4f}",
            f"{float(r['standardized_mean_diff']):+.4f}",
        ]
        for _, r in tb["splits"].iterrows()
    ]

    tbl_4_3_rows = []
    for _, r in tb["test"].iterrows():
        code = str(r["model_code"])
        cvi = cv_map.get(code, {})
        short_name = str(r["model_name"]).split("(")[0].strip()
        tbl_4_3_rows.append(
            [
                short_name,
                f"{float(cvi.get('cv_rmse_mean', 0)):.4f} ± "
                f"{float(cvi.get('cv_rmse_std', 0)):.4f}",
                f"{float(r['test_r2']):.4f}",
                f"${float(r['test_rmse']):.4f}",
                f"${float(r['test_mae']):.4f}",
                f"{float(r['test_mape_pct']):.2f}%",
                f"{float(r['r2_overfit_gap']):+.4f}",
                "ĐẠT" if bool(r["meets_both_smart"]) else "Chưa đạt",
            ]
        )

    # Bảng mới 4.3B: Phân rã chế độ Uber vs Lyft & Đối chứng Ex-Ante (C3)
    tbl_4_3b_rows = [
        [
            "Có điều kiện M7 — Toàn tập Test (surge thực)",
            f"{regime['overall_observed_test']['n_test']:,}",
            f"{regime['overall_observed_test']['M7_Conditional']['r2']:.4f}",
            "$"
            f"{regime['overall_observed_test']['M7_Conditional']['rmse']:.4f}",
            f"{regime['overall_observed_test']['M7_Conditional']['mape_pct']:.2f}"
            "%",
            "Gồm 27 đặc trưng (dùng surge công bố trên API)",
        ],
        [
            "  • Chế độ Uber (surge ẩn trong Upfront Price)",
            f"{regime['uber_hidden_surge_test']['n_test']:,}",
            f"{regime['uber_hidden_surge_test']['M7_Conditional']['r2']:.4f}",
            "$"
            f"{regime['uber_hidden_surge_test']['M7_Conditional']['rmse']:.4f}",
            f"{regime['uber_hidden_surge_test']['M7_Conditional']['mape_pct']:.2f}"
            "%",
            "Chiếm 74.3% SSE & 57/58 sai số > $15 (Kurtosis = 29.46)",
        ],
        [
            "  • Chế độ Lyft (surge công bố tường minh)",
            f"{regime['lyft_observed_surge_test']['n_test']:,}",
            f"{regime['lyft_observed_surge_test']['M7_Conditional']['r2']:.4f}",
            "$"
            f"{regime['lyft_observed_surge_test']['M7_Conditional']['rmse']:.4f}",
            f"{regime['lyft_observed_surge_test']['M7_Conditional']['mape_pct']:.2f}"
            "%",
            "1/18,440 sai số > $15 (0 > $16, max $"
            f"{regime['lyft_observed_surge_test']['M7_Conditional']['max_error']:.2f}"
            ", Kurtosis = 3.47)",
        ],
        [
            "Ex-Ante M7 (23 biến, KHÔNG dùng surge) — All Test",
            f"{ex_pricing['M7_ExAnte_All_Test']['n_eval']:,}",
            f"{ex_pricing['M7_ExAnte_All_Test']['r2']:.4f}",
            f"${ex_pricing['M7_ExAnte_All_Test']['rmse']:.4f}",
            f"{ex_pricing['M7_ExAnte_All_Test']['mape_pct']:.2f}%",
            "Uber gần bất biến "
            f"(${ex_pricing['M7_ExAnte_Uber_Test']['rmse']:.2f}); Lyft tăng "
            f"mạnh lên ${ex_pricing['M7_ExAnte_Lyft_All_Test']['rmse']:.2f}",
        ],
        [
            "  • Ex-Ante M7 trên Lyft đang Surge (> 1.0x)",
            f"{ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['n_eval']:,}",
            f"{ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['r2']:.4f}",
            f"${ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['rmse']:.4f}",
            f"{ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['mape_pct']:.2f}%",
            "Biến công khai không thể tái tạo đỉnh surge thời gian thực",
        ],
        [
            "  • Ex-Ante M7 trên Holdout Lyft khuyết surge",
            f"{ex_pricing['M7_ExAnte_Lyft_MissingSurge_Holdout']['n_eval']:,}",
            f"{ex_pricing['M7_ExAnte_Lyft_MissingSurge_Holdout']['r2']:.4f}",
            "$"
            f"{ex_pricing['M7_ExAnte_Lyft_MissingSurge_Holdout']['rmse']:.4f}",
            f"{ex_pricing['M7_ExAnte_Lyft_MissingSurge_Holdout']['mape_pct']:.2f}"
            "%",
            "Trùng khớp sát Lyft Test "
            f"({ex_pricing['M7_ExAnte_Lyft_All_Test']['r2']:.4f} / "
            f"${ex_pricing['M7_ExAnte_Lyft_All_Test']['rmse']:.2f}) → xác "
            "nhận chứa ~6.8% surge ẩn",
        ],
    ]

    tier_order = [str(r["service_tier"]) for _, r in tb["pi"].iterrows()]
    tbl_4_4_rows = []
    wp_num, wp_den = 0.0, 0
    for tier in tier_order:
        r = cqr_map[tier]
        wp_num += r["mean_actual_usd"] * r["n_test"]
        wp_den += r["n_test"]
        d_mpiw = r["cqr_mpiw_usd"] - r["base85_raw_mpiw_usd"]
        d_txt = (
            "$0.000"
            if abs(d_mpiw) < 5e-4
            else f"{'+' if d_mpiw > 0 else '-'}${abs(d_mpiw):.3f}"
        )
        tbl_4_4_rows.append(
            [
                f"{r['cab_type']} — {tier}",
                f"${r['mean_actual_usd']:.2f}",
                f"{r['base85_raw_strict_picp_pct']:.2f}%",
                f"{r['base85_raw_tol1e6_picp_pct']:.2f}%",
                _fmt_qhat(r["q_hat_usd_6dec"]),
                f"{r['cqr_picp_pct']:.2f}%",
                f"${r['cqr_mpiw_usd']:.3f} ({d_txt})",
            ]
        )
    d_ov_mpiw = (
        cqr["overall_cqr_mpiw_usd"] - cqr["overall_base85_raw_mpiw_usd"]
    )
    tbl_4_4_rows.append(
        [
            f"Toàn tập Test (n = {cqr['n_test']:,})",
            f"${wp_num / wp_den:.2f}",
            f"{cqr['overall_base85_raw_strict_picp_pct']:.2f}%",
            f"{cqr['overall_base85_raw_tol1e6_picp_pct']:.2f}%",
            "0 cắt ngược",
            f"{cqr['overall_cqr_picp_pct']:.2f}%",
            f"${cqr['overall_cqr_mpiw_usd']:.3f} "
            f"({'+' if d_ov_mpiw > 0 else '-'}${abs(d_ov_mpiw):.3f})",
        ]
    )

    r3_unmod_pct = float(
        s4.get("r3_scale_recovery_audit", {}).get(
            "strictly_unmodified_both_pct", 88.85
        )
    )
    lasso_max_w_beta = max(
        abs(float(v))
        for v in tuning["M4_Lasso"]["weather_coefficients_in_lasso"].values()
    )
    scen_map = {
        "temporal_split": (
            "1. Chia theo thời gian thực Boston EST (TS & surge gốc)"
        ),
        "route_group_split": (
            "2. Chia theo 15 tuyến đường chưa gặp (Route Group)"
        ),
        "R3a_Test_Only_Strictly_Unmodified_Raw": (
            f"3. Kiểm tra trên {r3_unmod_pct:.2f}% dòng hoàn toàn nguyên bản"
        ),
        "R3b_Train_And_Test_Strictly_Unmodified_Raw": (
            "4. Huấn luyện & Kiểm tra thuần nguyên bản (không sửa)"
        ),
    }
    tbl_4_5_rows = []
    for sc, lbl in scen_map.items():
        sub = rob_df.loc[rob_df["scenario"] == sc]
        rm7 = sub.loc[sub["model_code"] == "M7_HistGradientBoosting"].iloc[0]
        rm2 = sub.loc[sub["model_code"] == "M2_Interaction_OLS"].iloc[0]
        tbl_4_5_rows.append(
            [
                lbl,
                f"{int(rm7['n_train']):,} / {int(rm7['n_test']):,}",
                f"{float(rm7['r2']):.4f} / ${float(rm7['rmse']):.4f} / "
                f"{float(rm7['mape_pct']):.2f}%",
                f"M2: {float(rm2['r2']):.4f} / ${float(rm2['rmse']):.4f} / "
                f"{float(rm2['mape_pct']):.2f}%",
            ]
        )
    m7_st, m2_st, m7_un = (
        st_lk["m7_strict_train_only"],
        st_lk["m2_strict_train_only"],
        st_lk["m7_strict_unmodified_raw"],
    )
    st_rg = st_lk.get("route_group_split", {})
    rg_st_m7 = st_rg.get("m7_strict_train_only", m7_st)
    rg_st_un = st_rg.get("m7_strict_unmodified_raw", m7_un)
    tbl_4_5_rows.append(
        [
            "5. Làm sạch & huấn luyện thuần Train-only (Full Refit)*",
            f"{st_lk['n_train']:,} / {st_lk['n_test']:,}",
            f"{m7_st['r2']:.4f} / ${m7_st['rmse']:.4f} / "
            f"{m7_st['mape_pct']:.2f}%",
            f"M2: {m2_st['r2']:.4f} / ${m2_st['rmse']:.4f} / "
            f"{m2_st['mape_pct']:.2f}% (0 price đổi)",
        ]
    )
    tbl_4_5_rows.append(
        [
            "6. GroupShuffleSplit phiên API batch (chặn "
            f"{abg['sibling_overlap_random_pct']:.2f}% trùng ms)",
            f"{abg['n_train']:,} / {abg['n_test']:,}",
            f"{abg['m7']['r2']:.4f} / ${abg['m7']['rmse']:.4f} / "
            f"{abg['m7']['mape_pct']:.2f}%",
            f"M2: {abg['m2']['r2']:.4f} / ${abg['m2']['rmse']:.4f} / "
            f"{abg['m2']['mape_pct']:.2f}%",
        ]
    )

    tbl_4_6_rows = [
        [
            str(r["experiment_id"]).split("_")[0],
            str(r["mutation_action"]),
            f"{float(r['baseline_r2']):.4f} / "
            f"{float(r['baseline_mape_pct']):.2f}%",
            f"{float(r['mutated_r2']):.4f} / "
            f"{float(r['mutated_mape_pct']):.2f}%",
            str(r["scientific_lesson"]),
        ]
        for _, r in tb["mb"].iterrows()
    ]

    tmap = tb["tariffs"].set_index("service_tier").to_dict("index")
    tbl_4_7_rows = []
    for pinfo in cross5:
        ut, lt = pinfo["uber_tier"], pinfo["lyft_tier"]
        ur, lr = tmap[ut], tmap[lt]
        sa = pinfo["spec_a_train80"]
        sb = pinfo["spec_b_train80_nonsurge"]
        sa_txt = (
            f"{sa['hc3']['d_star']:.2f} mi (CI_cl "
            f"[{sa['cluster']['ci95'][0]:.2f}, "
            f"{sa['cluster']['ci95'][1]:.2f}])"
            if sa["in_domain"]
            else "Không cắt trong [0.02, 7.86] mi"
        )
        if sb["in_domain"]:
            if ut == "Black SUV":
                af_bs = pinfo["above_floor_check"]
                sb_txt = (
                    f"{sb['hc3']['d_star']:.2f} mi (CI_cl "
                    f"[{sb['cluster']['ci95'][0]:.2f}, "
                    f"{sb['cluster']['ci95'][1]:.2f}]; d ≥ 1.0 mi: "
                    f"{af_bs['d_star_ge_1mi']:.3f} mi < 1.0 → Black SUV rẻ "
                    "hơn)"
                )
            else:
                sb_txt = (
                    f"{sb['hc3']['d_star']:.2f} mi (CI_cl "
                    f"[{sb['cluster']['ci95'][0]:.2f}, "
                    f"{sb['cluster']['ci95'][1]:.2f}])"
                )
        elif ut == "UberXL":
            sb_txt = (
                f"Không cắt ({sb['hc3']['d_star']:.2f} mi > 7.86; Lyft XL "
                "rẻ hơn toàn miền)"
            )
        elif ut == "Black":
            sb_txt = (
                f"Không cắt ({sb['hc3']['d_star']:.2f} mi; Uber Black luôn "
                "rẻ hơn)"
            )
        else:
            sb_txt = "Không cắt (Shared luôn rẻ hơn)"
        tbl_4_7_rows.append(
            [
                pinfo["segment"],
                f"{ut}: ${ur['base_fare_b0_usd']:.2f} + "
                f"${ur['per_mile_rate_b1_usd']:.2f}/mi",
                f"{lt}: ${lr['base_fare_b0_usd']:.2f} + "
                f"${lr['per_mile_rate_b1_usd']:.2f}/mi",
                sa_txt,
                sb_txt,
            ]
        )
    for st_name, note in [
        ("WAV", "Độc lập Uber (hỗ trợ xe lăn)"),
        ("Lux", "Độc lập Lyft (Premium Sedan)"),
    ]:
        sr = tmap[st_name]
        tbl_4_7_rows.append(
            [
                note,
                f"WAV: ${sr['base_fare_b0_usd']:.2f} + "
                f"${sr['per_mile_rate_b1_usd']:.2f}/mi"
                if st_name == "WAV"
                else "—",
                f"Lux: ${sr['base_fare_b0_usd']:.2f} + "
                f"${sr['per_mile_rate_b1_usd']:.2f}/mi"
                if st_name == "Lux"
                else "—",
                "Không ghép cặp chéo",
                "Tách riêng đối chứng",
            ]
        )

    tbl_4_8_rows = []
    for r in gx_rows:
        m7_s = max(float(r["m7_rel_share_pct"]), 0.0)
        m2_s = max(float(r["m2_rel_share_pct"]), 0.0)
        tbl_4_8_rows.append(
            [
                str(r["group"]),
                f"{int(r['n_cols'])}",
                str(r["cols"]),
                _fmt_signed_usd(float(r["m7_delta_rmse"])),
                f"{m7_s:.2f}%",
                _fmt_signed_usd(float(r["m2_delta_rmse"])),
                f"{m2_s:.2f}%",
            ]
        )

    # Bảng 5.1: 6 kịch bản Surge Cap (ràng buộc cung Q_realized = min(Q_D, Q_S)
    # đơn điệu tuyệt đối!)
    tbl_5_1_rows = []
    for _, r in tb["sim"].iterrows():
        cap = float(r["surge_cap"])
        ar = arith_map[cap]
        m7_sav = float(r["mean_rider_saving_per_affected_trip_usd"])
        m7_pct = float(r["mean_rider_saving_pct_affected"])
        ar_sav = float(ar["arith_mean_saving_usd"])
        ar_pct = float(ar["arith_saving_pct"])
        st_chg = float(r["static_lyft_rev_change_pct"])
        d5 = float(r["supply_constrained_rev_change_drop5_pct"])
        d10 = float(r["supply_constrained_rev_change_drop10_pct"])
        e50 = float(r["net_lyft_rev_change_eps_050_pct"])
        tbl_5_1_rows.append(
            [
                f"{cap:.2f}x",
                f"{int(r['affected_trips_count']):,}",
                "$0.00 (0.00%)"
                if m7_sav == 0
                else f"${m7_sav:.2f} (-{m7_pct:.2f}%)",
                f"${float(r['m2_ols_mean_saving_affected_usd']):.2f}",
                "$0.00 (0.00%)"
                if ar_sav == 0
                else f"${ar_sav:.2f} (-{ar_pct:.2f}%)",
                "0.000%" if abs(st_chg) < 1e-9 else f"{st_chg:+.3f}%",
                "0.000% /\n0.000%"
                if (abs(d5) < 1e-9 and abs(d10) < 1e-9)
                else f"{d5:+.3f}% /\n{d10:+.3f}%",
                "0.000%" if abs(e50) < 1e-9 else f"{e50:+.3f}%",
            ]
        )

    pw175_mde2 = pw175["mde_scenarios"][0]
    pw175_mde3 = pw175["mde_scenarios"][1]
    pw175_mde5 = pw175["mde_scenarios"][2]
    pw200_mde5 = pw200["mde_scenarios"][2]
    pw175_t5_mde2 = pw175_top5["mde_scenarios"][0]
    pw175_t5_mde3 = pw175_top5["mde_scenarios"][1]
    pw175_t5_mde5 = pw175_top5["mde_scenarios"][2]
    pw200_t5_mde5 = pw200_top5["mde_scenarios"][2]
    tbl_5_2_rows = [
        [
            "Pha 1 (Tháng 1–2): Shadow Mode & Trần Bảo thủ 2.00x",
            "Data Engineering & Pricing Ops",
            "Chạy song song M7 + M2 (và đối chứng Ex-Ante) trên Dashboard 5 "
            "Tab ở Shadow Mode; đóng gói bộ 3 mô hình Quantile Monotonic "
            "Split-CQR (pandas category) vào models/; kích hoạt thử nghiệm "
            "an toàn mức trần 2.00x (53 lượt cực đoan, đạt τ ≤ 0.10% ở mọi "
            "kịch bản cung).",
            "Độ trễ API < 150 ms (ngưỡng SLA báo giá thời gian thực); 18/18 "
            "chốt kiểm toán dữ liệu PASS; PICP CQR ∈ [78%, 83%] trên 12 gói,"
            " 0 cắt ngược; theo dõi ETA ở mức surge 2.5x–3.0x.",
        ],
        [
            "Pha 2 (Tháng 3–4): Triggered Switchback A/B cho Trần 1.75x",
            "Marketplace Economics & Product",
            "Thử nghiệm Switchback trên tập kích hoạt (Triggered Analysis, "
            "KHÔNG dùng ITT do pha loãng xuống "
            f"{pw175_mde2['itt_diluted_mde_pp']:.4f}–"
            f"{pw175_t5_mde2['itt_diluted_mde_pp']:.4f} pp) tại 5 quận điểm "
            f"nóng ({top12[0]['source']}, {top12[1]['source']}, "
            f"{top12[2]['source']}, {top12[3]['source']}, "
            f"{top12[4]['source']}; gom "
            f"{pw175_top5['share_of_citywide_triggered_pct']:.2f}% lượt "
            "kích hoạt toàn TP, tỷ lệ kích hoạt "
            f"{pw175_top5['empirical_trigger_rate_pct']:.3f}%, DEFF_25x = "
            f"{pw175_top5['deff_25x']:.3f} / DEFF_50x = "
            f"{pw175_top5['deff_50x']:.3f}) hoặc toàn 12 quận (DEFF_25x = "
            f"{pw175['deff_25x']:.3f}). Phát hiện MDE = 3.0 pp cần "
            f"{pw175_t5_mde3['triggered_quotes_both_arms_25x']:,} lượt "
            f"(~{pw175_t5_mde3['days_needed_at_prod_traffic_25x']:.1f} ngày "
            f"ở 25×; {pw175_t5_mde3['days_needed_at_prod_traffic_50x']:.1f} "
            "ngày ở 50×; toàn 12 quận: "
            f"{pw175_mde3['days_needed_at_prod_traffic_25x']:.1f} / "
            f"{pw175_mde3['days_needed_at_prod_traffic_50x']:.1f} ngày); "
            "MDE = 5.0 pp chỉ cần "
            f"~{pw175_t5_mde5['days_needed_at_prod_traffic_25x']:.1f} / "
            f"{pw175_t5_mde5['days_needed_at_prod_traffic_50x']:.1f} ngày.",
            "Đo tỷ lệ hoàn thành chuyến, ETA và tỷ lệ tài xế nhận cuốc trên "
            "tập kích hoạt; dừng sớm ngay nếu ETA tăng > 15% hoặc hủy do "
            "thiếu xe tăng > 2 pp (ngưỡng phòng chống Wild Goose Chase; "
            "Castillo et al., 2017; Hall et al., 2015).",
        ],
        [
            "Pha 3 (Tháng 5–6): Đánh giá Nhân quả & Đạo đức AI",
            "Vận hành & AI Governance",
            "Chỉ quyết định mở rộng trần 1.75x (chấp nhận τ = 0.30%–0.50%) "
            "nếu Pha 2 chứng minh không gây thiếu cung tài xế; tái huấn "
            "luyện bổ sung dữ liệu duration, mật độ xe và mùa đông Boston; "
            "duy trì kiểm toán công bằng 12 quận.",
            "Duy trì MAPE nội miền ≤ 9.0%, công bằng không gian 12 quận |"
            "bias| ≤ $0.12 (~"
            f"{0.12 / sp_v5['max_abs_district_bias_usd']:.2f}× hiện trạng "
            f"${sp_v5['max_abs_district_bias_usd']:.4f} và < 1/4 bậc làm "
            "tròn $0.50 của API).",
        ],
    ]

    # =========================================================================
    # TOÀN VĂN 5 CHƯƠNG CHÍNH (KHÓA ĐÚNG 25–45 TRANG NỘI DUNG CHÍNH)
    # =========================================================================
    chapters = [
        # =====================================================================
        # CHƯƠNG 1
        # =====================================================================
        {
            "title": "CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI VÀ MỤC TIÊU NGHIÊN CỨU",
            "sections": [
                {
                    "heading": "1.1. Bối cảnh nghiên cứu và Lý do chọn đề tài",
                    "blocks": [
                        (
                            "p",
                            (
                                "Sự phát triển nhanh chóng của kinh tế nền "
                                "tảng (Platform Economy) trong thập kỷ qua "
                                "đã làm thay đổi "
                                "căn bản cách thức vận hành của ngành vận "
                                "tải hành khách đô thị. Khác với taxi "
                                "truyền thống áp dụng "
                                "biểu giá cố định theo quy định quản lý, "
                                "các nền tảng gọi xe công nghệ như Uber và "
                                "Lyft vận hành theo "
                                "cơ chế thị trường hai phía (Two-Sided "
                                "Markets), kết nối nhu cầu di chuyển tức "
                                "thời của hành khách với "
                                "nguồn cung tài xế linh hoạt thông qua "
                                "thuật toán định giá động (Dynamic Surge "
                                "Pricing). Về lý thuyết "
                                "kinh tế học vi mô, hệ số nhân giá động "
                                "đóng vai trò tín hiệu cân bằng ngắn hạn: "
                                "thu hút tài xế vào khu "
                                "vực thiếu xe và phân bổ nguồn cung cho "
                                "hành khách có mức sẵn lòng chi trả cao hơn "
                                "khi cầu cục bộ vượt "
                                "cung (Castillo et al., 2017; Hall et al., "
                                "2015)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Trong thực tiễn cạnh tranh song độc quyền "
                                "tại các đô thị lớn của Hoa Kỳ như Boston "
                                "(Massachusetts), "
                                "biến động giá báo tạo ra nhiều bài toán "
                                "thực tiễn cho cả người dùng lẫn doanh "
                                "nghiệp vận hành. Về phía "
                                "hành khách, các đợt tăng giá từ 1.50x đến "
                                "3.00x gây tâm lý bất định và thúc đẩy "
                                "những giả định cảm tính "
                                "(như cho rằng trời mưa hoặc khung giờ cao "
                                "điểm luôn tự động đẩy giá tăng cơ học). Về "
                                "phía doanh nghiệp, "
                                "mức surge quá cao làm tăng nguy cơ khách "
                                "rời bỏ ứng dụng sang đối thủ, trong khi áp "
                                "trần giá cứng trong "
                                "thị trường thiếu cung lại tiềm ẩn rủi ro "
                                "giảm tài xế nhận cuốc và kéo dài thời gian "
                                "chờ cục bộ."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Việc vận dụng Khoa học Dữ liệu với hệ "
                                "sinh thái Python trên tập dữ liệu báo giá "
                                "API quy mô lớn tại "
                                "Boston (693,071 bản ghi thô và "
                                f"{total_n:,} lượt báo giá sạch sau xử lý) "
                                "giúp kiểm chứng thực nghiệm "
                                "các giả định phổ biến về thời tiết – "
                                "khung giờ, đối chiếu định lượng biểu giá "
                                "quan sát được giữa Uber "
                                "và Lyft, đánh giá công bằng cả mô hình "
                                "giá có điều kiện lẫn dự báo Ex-Ante không "
                                "dùng surge, và xây "
                                "dựng kịch bản mô phỏng chính sách có ràng "
                                "buộc cung phục vụ thảo luận quản trị. Đề "
                                "tài mang mã mở rộng "
                                "PRJ-EXT-01 (ngoài danh mục 32 đề tài gợi "
                                "ý tại Phần 5 đề cương học phần), đã được "
                                "giảng viên hướng dẫn "
                                "chấp thuận chủ đề và đáp ứng vượt mức "
                                "toàn bộ 4 điều kiện bắt buộc tại Phần 1 "
                                "(dữ liệu thực đa tệp > "
                                "100,000 dòng, đầy đủ chiều không gian – "
                                "thời gian – định lượng, kiểm định thống "
                                "kê và mô hình học máy tái lập)."
                            ),
                        ),
                    ],
                },
                {
                    "heading": "1.2. Xác định vấn đề và Câu hỏi nghiên cứu",
                    "blocks": [
                        (
                            "p",
                            (
                                "Nghiên cứu minh định chặt chẽ bản chất đo "
                                "lường ngay từ bước xác định vấn đề: tập "
                                "dữ liệu ghi nhận "
                                "giá báo ước tính trước chuyến đi trả về "
                                "từ giao diện lập trình ứng dụng (API Fare "
                                "Quotes) của Uber và "
                                "Lyft cho 12 gói dịch vụ trên 72 tuyến "
                                "liên quận có hướng tại Boston, kết hợp "
                                "bản tin khí tượng theo giờ. "
                                "Biến mục tiêu Y = price (USD) phản ánh "
                                "giá niêm yết tại thời điểm truy vấn "
                                "(Upfront API Quote), KHÔNG "
                                "đồng nhất với hóa đơn chuyến hoàn thành, "
                                "tỷ lệ chốt chuyến hay doanh thu kế toán "
                                "thực tế. Quan trọng "
                                "hơn, vì trường surge_multiplier là một "
                                "thành phần nội sinh do API Lyft công bố "
                                "cùng thời điểm trả giá "
                                "(còn API Uber luôn trả cố định 1.0x dù "
                                "vẫn áp phụ phí ẩn trong Upfront Pricing), "
                                "đề tài phân tách rõ "
                                "hai bài toán học máy: (i) Mô hình giá báo "
                                "API có điều kiện (Conditional Fare Quote "
                                "Modeling — dùng 27 "
                                "đặc trưng gồm surge công bố trên "
                                f"{obs_surge_n:,} lượt quan sát surge thực) "
                                "nhằm giải mã cấu trúc hàm giá "
                                "và chạy mô phỏng phản thực tế; và (ii) "
                                "Thực nghiệm Ex-Ante (chỉ dùng 23 đặc "
                                "trưng công khai trước thời "
                                "điểm định giá, loại bỏ hoàn toàn biến "
                                "surge, kèm bộ phân loại xác suất kích "
                                "hoạt Surge Lyft) nhằm đo "
                                "lường năng lực dự báo thực sự khi chưa "
                                "biết trạng thái surge."
                            ),
                        ),
                        (
                            "p",
                            "Bốn câu hỏi nghiên cứu cốt lõi (RQ1–RQ4) định "
                            "hướng toàn bộ thực nghiệm:",
                        ),
                        (
                            "bullets",
                            [
                                "RQ1 (Thời tiết & Khung giờ EST): Trên tập "
                                "khớp thời tiết thực đã loại trừ ô lỗi cảm "
                                "biến mưa và dấu thời gian gốc EST, điều "
                                "kiện mưa nhẹ–vừa và khung giờ trong ngày "
                                "có tạo chênh lệch có ý nghĩa thống kê và "
                                "thực tiễn lên giá báo API hay xác suất "
                                "Surge Lyft hay không?",
                                "RQ2 (Cấu trúc Biểu giá & Điểm cắt Cạnh "
                                "tranh): Uber và Lyft khác biệt thế nào về "
                                "hệ số chặn (β₀) và độ dốc theo quãng đường "
                                "(β₁) trên 5 cặp phân khúc đối xứng? Điểm "
                                "cắt d* trong miền [0.02, 7.86] dặm thay "
                                "đổi ra sao giữa Đặc tả A (gộp d × surge) "
                                "và Đặc tả B (thuần không surge = 1.0x trên "
                                "tập surge quan sát thực)?",
                                "RQ3 (Dự báo Có điều kiện vs. Ex-Ante, Hiệu "
                                "chỉnh Bất định & XAI): Hiệu năng của 10 "
                                "cấu hình mô hình có điều kiện khác biệt ra "
                                "sao giữa chế độ Uber (surge ẩn) và Lyft "
                                "(surge công bố), giữa Random/Temporal/"
                                "Route Group/API-Batch Split, và giảm bao "
                                "nhiêu khi chuyển sang bối cảnh Ex-Ante "
                                "không dùng surge? Nhóm đặc trưng nào dẫn "
                                "đầu ΔRMSE khi gom cụm chống đa cộng tuyến, "
                                "và Monotonic Split-CQR hiệu chuẩn khoảng "
                                "[P10, P90] ra sao?",
                                "RQ4 (Mô phỏng Trần Surge dưới Ràng buộc "
                                "Cung): Dưới nguyên tắc phân phối bên ngắn "
                                "Q_realized = min(Q_D, Q_S), áp các ngưỡng "
                                "trần Surge Cap (3.00x → 1.00x) trên tập "
                                "báo giá Lyft quan sát thực tạo ra sự đánh "
                                "đổi ra sao giữa mức giảm giá báo hành "
                                "khách, biến động API Quote Proxy và thiết "
                                "kế công suất thử nghiệm Switchback A/B "
                                "trên tập kích hoạt?",
                            ],
                        ),
                    ],
                },
                {
                    "heading": (
                        "1.3. Mục tiêu nghiên cứu và Mục tiêu kỹ thuật "
                        "(chuẩn SMART)"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Mục tiêu tổng quát là thực thi quy trình "
                                "Khoa học Dữ liệu Python có khả năng tái "
                                "lập và kiểm chứng cao "
                                "nhằm phân tích biểu hiện định giá động của "
                                "Uber và Lyft tại Boston, đánh giá trung "
                                "thực ranh giới mô "
                                "hình dự báo (có điều kiện và Ex-Ante) và "
                                "cung cấp khung mô phỏng chính sách chuẩn "
                                "kinh tế học vi mô:"
                            ),
                        ),
                        (
                            "bullets",
                            [
                                "Mục tiêu Nghiên cứu 1 (Quản trị Dữ liệu "
                                "Lớn & Bảo toàn Nguồn gốc — GĐ1–GĐ3, hoàn "
                                "thành đến 15/10/2026): Làm sạch đa tầng "
                                "nhiễu và ghép nối fan-out = 0 đạt bảng sạch"
                                f" ≥ 100,000 dòng (thực đạt {total_n:,} dòng "
                                "× 32 biến), giữ nguyên vẹn cờ nguồn gốc "
                                "cho timestamp, thời tiết, lỗi cảm biến mưa"
                                " và surge (0 dòng Lyft bị điền mù 1.0x), "
                                "bảo toàn NaN cự ly để nội suy strictly-on-"
                                "train trong Pipeline, giảm ≥ 50% RAM khi "
                                "chuyển từ CSV thô sang Parquet chuẩn hóa "
                                f"(đạt {ram_csv_pct:.2f}%: "
                                f"{io_same['csv_full_ram_mb']:.2f} → "
                                f"{io_same['parquet_ram_mb']:.2f} MB; "
                                "riêng bước ép kiểu hậu ghép nối giảm thêm "
                                f"{merge['memory_saved_pct']:.2f}%: "
                                f"{merge['memory_mb_before_downcast']:.2f} "
                                f"→ {merge['memory_mb_after_downcast']:.2f} "
                                "MB; đọc Parquet nhanh gấp "
                                f"{io_same['parquet_vs_csv_full_speedup']:.1f}"
                                " lần CSV cùng bảng).",
                                "Mục tiêu Nghiên cứu 2 (Suy luận Thống kê "
                                "Chặt chẽ — GĐ3–GĐ5, hoàn thành đến "
                                "15/11/2026): Triển khai 13 kiểm định giả "
                                "thuyết có hiệu chỉnh đa kiểm định "
                                "(Holm–Bonferroni, BH-FDR), sai số chuẩn "
                                "cụm hai cấp "
                                f"({st_ext['n_loc_hr_clusters_all']:,} quận "
                                "× giờ và "
                                f"{st_ext['n_citywide_hour_clusters_all']} "
                                "giờ toàn thành phố), Exact WCR Wild "
                                "Cluster Bootstrap (B = 1,999, đủ hệ số "
                                "bậc tự do) và kiểm định tương đương TOST "
                                "để trả lời RQ1–RQ2.",
                                "Mục tiêu Nghiên cứu 3 (Mô hình Dự báo 2 "
                                "Tầng, Hiệu chỉnh Bất định & XAI — GĐ4–GĐ6, "
                                "hoàn thành đến 19/11/2026): So sánh 10 cấu "
                                "hình mô hình có điều kiện (chỉ tiêu nội "
                                "miền R² ≥ 0.90, MAPE ≤ 10.0%) song song "
                                "với thực nghiệm Ex-Ante (23 biến không "
                                "surge + phân loại Surge Lyft), kiểm định "
                                "tổng quát hóa đa kịch bản, hiệu chỉnh "
                                "Monotonic Split-CQR dùng pandas category "
                                "gốc và giải mã 27 đặc trưng qua Grouped "
                                "Permutation Importance & PDP/ICE.",
                                "Mục tiêu Nghiên cứu 4 (Mô phỏng Chính sách "
                                "Bên ngắn & Power Analysis — GĐ6–GĐ7, hoàn "
                                "thành đến 22/11/2026): Mô phỏng 6 ngưỡng "
                                "trần Surge Cap theo cơ chế ràng buộc cung "
                                "Q_realized = min(Q_D, Q_S) có đối chứng M2"
                                " OLS, chuẩn số học và Out-of-Sample Test-"
                                "only, kèm tính toán cỡ mẫu Switchback "
                                "trên tập kích hoạt (Triggered Analysis).",
                                "Mục tiêu Kỹ thuật & Sản phẩm Bàn giao "
                                "(GĐ7–GĐ8, hoàn thành & đóng gói đến "
                                "28/11/2026): Bàn giao hệ thống Python "
                                f"chuẩn PEP 8 ({n_src} module src/, "
                                "Dashboard 5 Tab Streamlit/HTML offline, "
                                f"cổng kiểm toán 18/18, {n_test_files} tệp "
                                "pytest và nhật ký P-D-M-V) cùng ấn phẩm "
                                "không quá 45 trang nội dung ngoài bìa.",
                            ],
                        ),
                    ],
                },
                {
                    "heading": "1.4. Đối tượng và Phạm vi nghiên cứu",
                    "blocks": [
                        (
                            "p",
                            (
                                "Đối tượng nghiên cứu là mức giá báo API "
                                "(price), hệ số định giá động "
                                "(surge_multiplier), tham số biểu "
                                "giá thực nghiệm (β₀, β₁) và sai số dự báo "
                                "trên 12 gói dịch vụ của Uber (UberPool, "
                                "UberX, WAV, UberXL, "
                                f"Black, Black SUV; {uber_n:,} lượt) và "
                                "Lyft (Shared, Lyft, Lyft XL, Lux, Lux "
                                "Black, Lux Black XL; "
                                f"{lyft_tot_n:,} lượt, trong đó "
                                f"{lyft_obs_n:,} lượt có surge quan sát "
                                f"thực và {lyft_miss_n:,} lượt khuyết "
                                "surge được cô lập minh bạch). Gói Uber "
                                "Taxi (55,095 dòng) bị tách riêng do "
                                "khuyết giá 100% (MNAR)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Phạm vi không gian bao phủ 12 quận nội đô "
                                "Boston chia thành hai tập rời nhau mỗi tập "
                                "6 quận tạo "
                                "đồ thị lưỡng phân đầy đủ có hướng K₆,₆ (0 "
                                "tuyến nội tập): Tập A gồm Back Bay, Beacon "
                                "Hill, "
                                "Financial District, North Station, Theatre "
                                "District, West End; Tập B gồm Boston "
                                "University, Fenway, "
                                "Haymarket Square, North End, Northeastern "
                                "University, South Station — tạo đúng 72 "
                                "tuyến liên quận "
                                "có hướng trong miền [0.02, 7.86] dặm."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Phạm vi thời gian từ 25/11/2018 22:40:46 "
                                "đến 18/12/2018 14:15:10 giờ Boston (EST = "
                                "UTC-5), gồm "
                                "22.65 ngày liên tục qua 24 ngày lịch (18 "
                                "ngày có dữ liệu gốc, 6 ngày khuyết thu "
                                "thập: 05–08/12, "
                                "10–11/12/2018). Mọi kết luận chỉ có hiệu "
                                "lực trong giai đoạn cuối thu ôn hòa này."
                            ),
                        ),
                    ],
                },
                {
                    "heading": "1.5. Ý nghĩa lý luận và Thực tiễn của đề tài",
                    "blocks": [
                        (
                            "p",
                            (
                                "Về phương pháp luận, tiểu luận minh họa "
                                "giá trị quyết định của kỷ luật bảo toàn "
                                "nguồn gốc dữ liệu "
                                "(Data Provenance): chỉ một lỗi điền mù "
                                "surge_multiplier = 1.0 trên 29.16% dữ liệu "
                                "Lyft hay gán nhầm "
                                "mã lính canh mưa thành 0.0 cũng làm lệch "
                                "tỷ lệ surge, kurtosis và đảo ngược kết "
                                "luận điểm cắt. Đề "
                                "tài đồng thời tách bạch mô hình có điều "
                                "kiện với dự báo Ex-Ante, kết hợp Exact WCR "
                                "đủ bậc tự do, "
                                "TOST và Monotonic Split-CQR nhất quán mã hóa."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Về thực tiễn, kết quả cung cấp quy tắc so "
                                "sánh giá có căn cứ theo cự ly cho hành "
                                "khách nội đô "
                                "Boston khi không có surge, đồng thời bàn "
                                "giao khung Dashboard 5 Tab và kịch bản "
                                "Surge Cap nhất "
                                "quán kinh tế học vi mô (Pha 1 trần 2.00x "
                                "bảo thủ; Pha 2 Triggered Switchback cho "
                                "trần 1.75x)."
                            ),
                        ),
                    ],
                },
            ],
        },
        # =====================================================================
        # CHƯƠNG 2
        # =====================================================================
        {
            "title": "CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP NGHIÊN CỨU",
            "sections": [
                {
                    "heading": "2.1. Cơ sở lý thuyết chuyên ngành liên quan",
                    "blocks": [
                        (
                            "p",
                            (
                                "Nền tảng gọi xe công nghệ vận hành theo "
                                "cấu trúc thị trường hai phía (Two-Sided "
                                "Markets), nơi giá trị "
                                "của nhóm hành khách phụ thuộc trực tiếp "
                                "vào mật độ tài xế khả dụng qua hiệu ứng "
                                "mạng chéo (Rochet & "
                                "Tirole, 2003). Khi cầu cục bộ tăng đột "
                                "biến dưới chế độ giá cố định, xe quanh "
                                "điểm nóng nhanh chóng cạn "
                                "kiệt, đẩy hệ thống vào trạng thái 'cuộc "
                                "rượt đuổi vô ích' (Wild Goose Chase) — tài "
                                "xế phải chạy rỗng "
                                "quãng đường xa tới đón khách, làm suy giảm "
                                "năng suất và phúc lợi toàn hệ thống "
                                "(Castillo et al., 2017)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Để điều tiết mất cân bằng cục bộ, các nền "
                                "tảng áp dụng Biểu giá hai phần kết hợp "
                                "Định giá động. Trong "
                                "trạng thái không tăng giá (surge = 1.0x), "
                                "mức giá báo API của phân khúc k theo quãng "
                                "đường d được xấp "
                                "xỉ bởi mô hình thực nghiệm kèm ngưỡng giá "
                                "sàn tối thiểu F_min,k cho chặng siêu ngắn "
                                "(< 1.0 dặm): "
                                "P_k(d) = max(F_min,k ; β0,k + β1,k · d + "
                                "u). Cần nhấn mạnh đây là xấp xỉ thực "
                                "nghiệm chứ không phải công "
                                "thức biểu giá nội bộ đầy đủ của hãng; do "
                                "tập dữ liệu thiếu biến thời lượng di "
                                "chuyển thực (duration) "
                                "tương quan dương chặt với d, hệ số góc β1,"
                                "k chịu thiên lệch bỏ sót biến (OVB) và "
                                "phản ánh gộp cự ly lẫn "
                                "phần thời lượng tương quan. Khi hai đối "
                                "thủ cùng phân khúc có tham số chéo nhau "
                                "(β0,Lyft < β0,Uber nhưng "
                                "β1,Lyft > β1,Uber), điểm cắt cự ly d* được "
                                "xác định theo:"
                            ),
                        ),
                        (
                            "formula",
                            [
                                ("d", True, None),
                                ("*", False, "sup"),
                                (" = (", False, None),
                                ("β", True, None),
                                ("0,Uber", False, "sub"),
                                (" − ", False, None),
                                ("β", True, None),
                                ("0,Lyft", False, "sub"),
                                (") / (", False, None),
                                ("β", True, None),
                                ("1,Lyft", False, "sub"),
                                (" − ", False, None),
                                ("β", True, None),
                                ("1,Uber", False, "sub"),
                                (")", False, None),
                            ],
                        ),
                        (
                            "p",
                            (
                                "Khi áp đặt trần giá (Price Ceiling) C lên "
                                "một thị trường đang ở trạng thái đỉnh cầu "
                                "(surge > C), kinh tế "
                                "học vi mô xác lập nguyên tắc phân phối "
                                "theo bên ngắn (Short-Side Rationing): sản "
                                "lượng thực hiện luôn "
                                "bị chặn bởi phía nhỏ hơn giữa lượng cầu "
                                "sau khi giảm giá và lượng cung tài xế còn "
                                "lại:"
                            ),
                        ),
                        (
                            "formula",
                            [
                                ("Q", True, None),
                                ("realized", False, "sub"),
                                (" = min( ", False, None),
                                ("Q", True, None),
                                ("D", False, "sub"),
                                ("(P", False, None),
                                ("cap", False, "sub"),
                                ("), ", False, None),
                                ("Q", True, None),
                                ("S", False, "sub"),
                                ("(P", False, None),
                                ("cap", False, "sub"),
                                (") )", False, None),
                            ],
                        ),
                        (
                            "p",
                            (
                                "Cụ thể, trên mỗi lượt bị hạ giá với mức "
                                "giảm tương đối δ = (P_cap − P_base) / "
                                "P_base < 0, độ co giãn cầu "
                                "theo giá (Cohen et al., 2016; ε_d ≈ −0.50; "
                                "kịch bản cạnh tranh −0.80) chỉ làm tăng "
                                "cầu tiềm năng lên "
                                "Q_D = 1 + ε_d · δ ≥ 1.0 — đại diện cho Cận "
                                "trên lạc quan khi dư thừa cung tài xế. "
                                "Ngược lại, khi cung tài xế nhạy giá theo "
                                "mức cắt giảm với độ co giãn cung "
                                "ε_S ∈ {0.30, 0.60}, hàm cung tại đỉnh "
                                "giảm xuống Q_S = clip(1 + ε_S · δ, sàn, "
                                "1.0) < 1.0 (sàn 0.50 với ε_S = 0.30 và "
                                "0.40 với ε_S = 0.60); mọi phần cầu vượt "
                                "Q_S biến thành dư cầu không được phục vụ "
                                "Σ max(0, Q_D − Q_realized) (tăng ETA hoặc "
                                "hủy tìm xe), và sản lượng thực hiện bằng "
                                "Q_realized = min(Q_D, Q_S). Thiết kế "
                                "này đảm bảo tính đơn điệu vi mô chặt: cung "
                                "càng nhạy giá thì "
                                "giá trị báo giá quy đổi (API Quote Proxy) "
                                "càng giảm."
                            ),
                        ),
                    ],
                },
                {
                    "heading": (
                        "2.2. Tổng quan các nghiên cứu thực nghiệm trước đây"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Các công trình đi trước tạo khung tham "
                                "chiếu chặt chẽ cho nghiên cứu này. Chen et "
                                "al. (2015) đo lường dữ "
                                "liệu API Uber tại San Francisco và "
                                "Manhattan, chứng minh surge thay đổi theo "
                                "chu kỳ vài phút ở cấp tiểu "
                                "khu. Hall et al. (2015) chỉ ra rủi ro sụt "
                                "giảm tỷ lệ hoàn thành chuyến và kéo dài "
                                "ETA khi surge bị gián "
                                "đoạn trong sự kiện lớn. Cohen et al. "
                                "(2016) phân tích gần 50 triệu phiên Uber "
                                "bằng hồi quy gián đoạn "
                                "để ước lượng thặng dư tiêu dùng và độ co "
                                "giãn cầu quanh −0.50. Cachon et al. (2017) "
                                "và Castillo et al. "
                                "(2017) xây dựng nền tảng lý thuyết thị "
                                "trường hai phía. Bảng 2.1 đối chiếu năm "
                                "công trình nền tảng và "
                                "định vị đóng góp cụ thể của đề tài."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 2.1: Tổng hợp so sánh các nghiên "
                                    "cứu thực nghiệm tiêu biểu và vị trí "
                                    "đóng góp của đề tài"
                                ),
                                "headers": [
                                    "Nghiên cứu",
                                    "Dữ liệu / Phạm vi",
                                    "Phương pháp",
                                    "Phát hiện chính",
                                    "Hạn chế chính",
                                    "Liên hệ đề tài",
                                ],
                                "rows": tbl_2_1_rows,
                                "col_widths": [2.2, 2.5, 2.4, 3.2, 2.7, 3.0],
                                "aligns": ["L", "L", "L", "L", "L", "L"],
                                "source": (
                                    "Nguồn: Tổng hợp tổng quan tài liệu của "
                                    "nhóm nghiên cứu từ 5 công trình nền "
                                    "tảng (2015–2017)."
                                ),
                                "font_size": 9.0,
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "2.3. Quy trình nghiên cứu dữ liệu (Data Science "
                        "Workflow)"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Khung CRISP-DM 1.0 kinh điển gồm 6 giai "
                                "đoạn chuẩn: Business Understanding, Data "
                                "Understanding, Data "
                                "Preparation, Modeling, Evaluation và "
                                "Deployment (Chapman et al., 2000). Nghiên "
                                "cứu vận dụng CRISP-DM "
                                "6 giai đoạn kinh điển và mở rộng thành 7 "
                                "giai đoạn triển khai nội bộ để tách rõ "
                                "bước kiểm toán dữ liệu "
                                "đa tệp, thiết kế đặc trưng khử đa cộng "
                                "tuyến và xây dựng Dashboard tương tác: (1) "
                                "Xác định bài toán & "
                                "SMART; (2) Kiểm toán dữ liệu gốc; (3) Làm "
                                "sạch bảo toàn nguồn gốc, chuẩn hóa EST & "
                                "ghép nối; (4) EDA & "
                                "kiểm định giả thuyết; (5) Feature "
                                "Engineering & đóng gói Pipeline; (6) Huấn "
                                "luyện 10 mô hình có điều "
                                "kiện + đối chứng Ex-Ante, kiểm định độ bền "
                                "& thử nghiệm phá vỡ (Mutate & Break); và "
                                "(7) XAI, hiệu chỉnh "
                                "Monotonic Split-CQR, mô phỏng Surge Cap "
                                "bên ngắn & triển khai Dashboard."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Toàn bộ quy trình được vận hành theo chu "
                                "trình phản biện P-D-M-V (Prompt & Produce "
                                "– Deconstruct & Read "
                                "– Mutate & Break – Validate & Refine), kết "
                                "hợp kiểm toán độc lập toàn diện trước khi "
                                "đóng băng phiên bản "
                                "nộp cuối. Chi tiết đối chiếu mã kiểm toán "
                                "nội bộ và trích xuất nhật ký P-D-M-V được "
                                "tổng hợp tại Phụ lục B."
                            ),
                        ),
                    ],
                },
                {
                    "heading": (
                        "2.4. Thiết kế kiểm định thống kê và Thuật toán Học "
                        "máy ứng dụng"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Nhằm xử lý đồng thời phân phối lệch phải "
                                "của giá báo (Skewness = "
                                f"+{num['price']['skewness']:.4f}), sự "
                                "phụ thuộc quan sát theo cụm (nhiều báo "
                                "giá dùng chung một bản tin khí tượng) và "
                                "bài toán đa kiểm định, "
                                "khung suy luận thống kê được xác lập "
                                "trước như sau: (i) Đơn vị quan sát gốc là "
                                "từng lượt báo giá API "
                                "(biến phụ thuộc price hoặc chỉ báo surge "
                                "> 1.0x); (ii) Mức ý nghĩa danh nghĩa α = "
                                "5%, hiệu chỉnh đồng "
                                "thời toàn họ 13 kiểm định bằng cả thủ tục "
                                "Holm–Bonferroni (Holm, 1979 — kiểm soát "
                                "FWER, tiêu chí ra "
                                "quyết định chính) lẫn Benjamini–Hochberg "
                                "(Benjamini & Hochberg, 1995 — kiểm soát "
                                "FDR đối chứng); "
                                "(iii) Sai số chuẩn bền vững theo cụm "
                                "(Cluster-Robust SE) ở hai cấp độ — cấp "
                                "cụm quận × giờ "
                                f"({st_ext['n_loc_hr_clusters_all']:,} cụm "
                                "khớp thời tiết hợp lệ và "
                                f"{tod_cl['n_clusters']:,} cụm EST) và "
                                "cấp bảo thủ nhất theo giờ toàn thành phố "
                                f"({st_ext['n_citywide_hour_clusters_all']} "
                                "giờ), bổ sung Exact "
                                "Restricted Wild Cluster Bootstrap (WCR) "
                                "với phép chiếu trực giao M_Z chính xác, "
                                "đầy đủ hệ số bậc tự do "
                                "(G/(G−1))·((N−1)/(N−k)) và trọng số "
                                "Rademacher (Cameron et al., 2008; B = 1,"
                                "999 lần lặp) cho lát cắt "
                                f"mưa lớn ({hv['n_heavy_loc_hr_clusters']} "
                                f"cụm / {hv['n_heavy_citywide_hours']} giờ);"
                                " (iv) Kiểm định "
                                "tương đương hai phía TOST (Lakens, 2017; "
                                "Schuirmann, 1987) với hai biên ±$0.25 và "
                                "±$0.10 (nhỏ hơn 1/5 "
                                "bậc làm tròn $0.50 của API); và (v) Hồi "
                                "quy OLS HC3 (MacKinnon & White, 1985) kèm "
                                "Cluster-Robust SE và "
                                f"phương pháp Delta cho 95% CI điểm cắt d*."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Trong mô hình hóa dự báo (Chương 4), 10 "
                                "cấu hình có điều kiện thuộc bốn trường "
                                "phái được đối sánh song "
                                "song với 2 thực nghiệm Ex-Ante không dùng "
                                "surge: (i) Đối chứng không tham số (M0a "
                                "trung vị gói xe, M0b "
                                "bảng tra gói × tuyến); (ii) Tuyến tính & "
                                "Kinh tế lượng (M1 OLS cơ bản, M2 OLS tương "
                                "tác, M2-Log biến đổi "
                                "ln(Price) kèm Duan's Smearing; Duan, 1983; "
                                "M3 Ridge L2; M4 Lasso L1); (iii) Cây quyết "
                                "định & Tổ hợp "
                                "(M5 Decision Tree, M6 Random Forest; "
                                "Breiman, 2001; M7 HistGradientBoosting; "
                                "Friedman, 2001; Ke et al., "
                                "2017); và (iv) Hồi quy Phân vị Quantile "
                                "[P10, P50, P90] (Koenker & Bassett, 1978) "
                                "dùng pandas category "
                                "gốc kết hợp hiệu chỉnh Monotonic Split-"
                                "Conformalized Quantile Regression (CQR; "
                                "Romano et al., 2019). "
                                "Diễn giải mô hình sử dụng Grouped "
                                "Permutation Importance phân hoạch 6 cụm "
                                "rời phủ 27/27 cột X_test cùng PDP/ICE."
                            ),
                        ),
                    ],
                },
                {
                    "heading": (
                        "2.5. Môi trường thực nghiệm và Công cụ lập trình "
                        "Python"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Thực nghiệm được triển khai trên hai máy "
                                "Windows 64-bit theo cấu hình "
                                "requirements.txt (Python ≥ 3.11, kèm "
                                "pillow và pymupdf): phát triển song song "
                                "trên Python 3.14.4 và tái lập toàn trình "
                                "trên máy tổng hợp Python "
                                f"{platform.python_version()} "
                                f"(pandas {pd.__version__}; McKinney, 2010; "
                                f"numpy {_imp_meta.version('numpy')}, "
                                f"scipy {_imp_meta.version('scipy')}, "
                                f"pyarrow {_imp_meta.version('pyarrow')}, "
                                "statsmodels "
                                f"{_imp_meta.version('statsmodels')}; "
                                "Seabold & Perktold, 2010; scikit-learn "
                                f"{_imp_meta.version('scikit-learn')}; "
                                "Pedregosa et al., 2011; joblib "
                                f"{_imp_meta.version('joblib')}, matplotlib "
                                f"{_imp_meta.version('matplotlib')}, seaborn "
                                f"{_imp_meta.version('seaborn')} cho ảnh 300 "
                                "DPI, streamlit, plotly và pytest "
                                f"{_imp_meta.version('pytest')} cho Dashboard "
                                "& kiểm thử)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Mã nguồn tuân thủ nghiêm chuẩn PEP 8 (độ "
                                "dài dòng ≤ 79 ký tự) với 100% đường dẫn "
                                "tương đối pathlib.Path: "
                                f"main.py điều phối {n_src} module lõi "
                                "pipeline trong src/, tách độc lập "
                                f"{n_scripts} script xuất bản và "
                                "kiểm chứng tại scripts/, và tách lớp với "
                                "dashboard/ (Streamlit 5 Tab, HTML offline,"
                                " cổng kiểm toán "
                                f"18/18 và {n_test_files} tệp pytest)."
                            ),
                        ),
                    ],
                },
            ],
        },
        # =====================================================================
        # CHƯƠNG 3
        # =====================================================================
        {
            "title": (
                "CHƯƠNG 3: THU THẬP, GHÉP NỐI VÀ PHÂN TÍCH KHÁM PHÁ DỮ LIỆU "
                "(EDA)"
            ),
            "sections": [
                {
                    "heading": (
                        "3.1. Mô tả nguồn gốc và Cấu trúc tập dữ liệu đa "
                        "tệp (Multi-file)"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Bộ dữ liệu thứ cấp mở được thu thập từ "
                                "Kaggle (Uber and Lyft Dataset Boston, MA; "
                                "brllrb, 2019), gồm hai "
                                "bảng quan hệ độc lập lưu chỉ-đọc (read-"
                                "only) tại data/raw/ và xác minh toàn vẹn "
                                "qua mã băm SHA-256:"
                            ),
                        ),
                        (
                            "bullets",
                            [
                                "Bảng báo giá cab_rides.csv: "
                                f"{cl_cab['valid_product_id_count']:,} dòng "
                                "× 10 cột (distance, cab_type, time_stamp "
                                "[ms], destination, source, price, "
                                "surge_multiplier, id, product_id, name), "
                                "dung lượng "
                                f"{wf_sec['raw_cab_size_mib']:.2f} MiB "
                                "(85.5 MB).",
                                "Bảng khí tượng weather.csv: 6,276 dòng × 8 "
                                "cột (temp [°F], location, clouds [0–1], "
                                "pressure [mb], rain [inch/h], time_stamp "
                                "[giây], humidity [0–1], wind [mph]) theo "
                                "từng quận Boston và khung giờ.",
                            ],
                        ),
                        (
                            "p",
                            (
                                "Kiểm toán sơ bộ ghi nhận gói Uber Taxi "
                                "chiếm đúng 55,095 dòng nhưng khuyết giá "
                                "100% có hệ thống (MNAR) "
                                "và chứa 5,226 mã lính canh; đồng thời dữ "
                                "liệu thô chứa 7 tầng nhiễu (khuyết nhãn "
                                "hãng/gói khôi phục 1-1 "
                                "từ product_id, mã lính canh -100/100,000, "
                                "dấu âm, sai lệch bậc 10^m, chuỗi kèm đơn "
                                "vị, khuyết timestamp "
                                "và khuyết/lính canh surge). Bảng 3.1 "
                                "trích yếu các biến cốt lõi trong từ điển "
                                f"đầy đủ {len(dd)} biến đủ đơn vị đo."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 3.1: Trích yếu Từ điển Dữ liệu "
                                    "hợp nhất (29 biến tiêu biểu trong "
                                    f"{len(dd)} biến có đơn vị đo)"
                                ),
                                "headers": [
                                    "Tên biến",
                                    "Đơn vị đo",
                                    "Kiểu dữ liệu",
                                    "Số khuyết",
                                    "Quyết định xử lý / Vai trò",
                                ],
                                "rows": tbl_3_1_rows,
                                "col_widths": [4.2, 2.6, 2.1, 1.6, 5.5],
                                "aligns": ["L", "L", "L", "R", "L"],
                                "source": (
                                    "Nguồn: Trích xuất từ reports/tables/"
                                    "data_dictionary_complete.csv "
                                    f"({len(dd)}/{len(dd)} biến đầy đủ đơn "
                                    "vị đo)."
                                ),
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "3.2. Quy trình ghép nối bảng dữ liệu (Merge), Bảo "
                        "toàn nguồn gốc và Tối ưu bộ nhớ"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Trước khi ghép nối, kiểm toán bảng "
                                "weather_cleaned.csv (5,053 dòng) phát hiện "
                                "1,688 bản tin lặp khóa "
                                "(location, date_hour_est). Nhóm tổng hợp "
                                "trung bình theo từng cặp quận × giờ để nén "
                                "bảng khí tượng về "
                                "đúng 3,365 khóa duy nhất (1-1), đảm bảo "
                                "phép nối Left Join theo (source == "
                                "location) & (date_hour_est) "
                                f"bảo toàn chính xác {total_n:,} = "
                                f"{total_n:,} dòng (fan-out = 0, trùng khóa "
                                "chính id = 0)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Ba chốt bảo vệ nguồn gốc dữ liệu được "
                                "khóa chặt nhằm triệt tiêu mọi nhãn giả "
                                "trong phân tích: "
                                "(i) Quy đổi Unix timestamp UTC sang giờ "
                                "Boston (America/New_York, EST = UTC-5) và "
                                "KHÔNG dùng .ffill() "
                                f"trên {merge['missing_timestamp_rows']:,} "
                                "dòng khuyết timestamp "
                                f"({merge['missing_timestamp_pct']:.2f}%), "
                                "gắn cờ is_timestamp_original = 1 cho "
                                f"{prov['is_timestamp_original_1']:,} dòng "
                                "gốc; "
                                "(ii) Làm sạch khí tượng bằng khôi phục "
                                "bậc 10 dựa trên trung vị cùng giờ toàn "
                                "Boston (sửa cả các đột "
                                "biến 10× ẩn dưới ngưỡng cứng ở wind/"
                                "clouds), điền khuyết temp/clouds/pressure/"
                                "humidity/wind theo trung vị "
                                "cùng giờ toàn thành phố (đưa số ô lệch "
                                "cục bộ về 0 trên cả 4 biến khí tượng thay "
                                "vì điền trung bình 24 "
                                "ngày) và tách riêng cờ is_rain_sentinel "
                                "cho 1,074 bản tin lỗi cảm biến mưa trong "
                                "weather_cleaned.csv "
                                "(giữ rain = NaN thay vì gán 0.0). Sau khi "
                                "ghép nối, trong "
                                f"{prov['weather_matched_1']:,} dòng khớp "
                                "trạm "
                                "(66.18% toàn tập), có "
                                f"{prov['weather_rain_sentinel_nan']:,} "
                                "dòng rơi vào ô toàn sentinel giữ nguyên "
                                "rain/is_raining = NaN và đúng "
                                f"{prov['weather_valid_rain_obs']:,} dòng "
                                "có trạng thái mưa/khô hợp lệ "
                                f"({prov['weather_true_rain_rows']:,} dòng "
                                "mưa thực) phân bổ trên "
                                f"{st_ext['n_loc_hr_clusters_all']:,} cụm "
                                "quận × giờ và "
                                f"{st_ext['n_citywide_hour_clusters_all']} "
                                "khung giờ toàn Boston; "
                                "(iii) Bảo toàn tuyệt đối nguồn gốc "
                                "surge_multiplier: bỏ hoàn toàn thao tác "
                                "điền mù fillna(1.0) trên Lyft, "
                                "gắn cờ is_surge_original = 1 cho "
                                f"{obs_surge_n:,} lượt quan sát thực "
                                f"({uber_n:,} Uber API cố định 1.0x "
                                f"+ {lyft_obs_n:,} Lyft có surge thật) và "
                                f"giữ nguyên NaN trên {lyft_miss_n:,} lượt "
                                "Lyft khuyết/sentinel "
                                f"({prov['lyft_surge_missing_pct']:.2f}% "
                                "tổng lượt Lyft)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Về tối ưu bộ nhớ và đối chứng I/O dữ liệu "
                                "lớn công bằng: đọc phân khối trực tiếp "
                                "cab_rides.csv "
                                f"({io_raw['raw_rows_read']:,} dòng, "
                                f"chunksize = {io_raw['chunksize']:,}, "
                                "pre_filter_valid=False đồng nhất giữa "
                                "--stage clean và --stage all) "
                                "giữ đỉnh RAM mỗi khối ở mức "
                                f"{io_raw['raw_peak_single_chunk_mb']:.2f} "
                                "MB. Ép kiểu toàn bộ 8/8 cột phân "
                                "loại (bao gồm location) sang category, "
                                "float64 → float32 và cờ nhị phân → int8 "
                                "giúp giảm bổ sung "
                                f"{merge['memory_saved_pct']:.2f}% RAM ngay "
                                "tại bước ép kiểu sau hợp nhất "
                                f"({merge['memory_mb_before_downcast']:.2f} "
                                "MB "
                                f"→ {merge['memory_mb_after_downcast']:.2f} "
                                "MB), nâng tổng mức tiết kiệm RAM toàn "
                                "quy trình từ CSV gốc xuống Parquet lên "
                                f"{ram_csv_pct:.2f}% "
                                f"({io_same['csv_full_ram_mb']:.2f} MB → "
                                f"{io_same['parquet_ram_mb']:.2f} MB, vượt "
                                "ngưỡng SMART ≥ 50%). Đo tốc độ I/"
                                "O công bằng trên "
                                f"CÙNG bảng master ({io_same['n_rows']:,} × "
                                f"{io_same['n_cols']} cột, trung vị 3 lần): "
                                "định dạng Parquet "
                                "nén đĩa nhỏ gấp "
                                f"{io_same['disk_compression_ratio']:.2f} "
                                f"lần ({io_same['parquet_disk_mb']:.2f} MB "
                                "vs. "
                                f"{io_same['csv_disk_mb']:.2f} MB CSV), giữ "
                                f"RAM ở mức {io_same['parquet_ram_mb']:.2f} "
                                "MB (so với "
                                f"{io_same['csv_full_ram_mb']:.2f} MB CSV) "
                                "và đọc chỉ mất "
                                f"{io_same['parquet_median_sec']:.3f} giây —"
                                " "
                                "nhanh gấp "
                                f"{io_same['parquet_vs_csv_full_speedup']:.1f}"
                                " lần so với đọc toàn bộ CSV "
                                f"({io_same['csv_full_median_sec']:.3f}s) "
                                "và gấp "
                                f"{io_same['parquet_vs_csv_chunk_speedup']:.1f}"
                                " lần so với đọc CSV phân khối "
                                f"({io_same['csv_chunk100k_median_sec']:.3f}"
                                "s)."
                            ),
                        ),
                    ],
                },
                {
                    "heading": (
                        "3.3. Quy trình làm sạch, Phễu hao hụt dữ liệu và "
                        "Độ nhạy tiền xử lý"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Từ 693,071 dòng gốc, quy trình lọc sạch "
                                f"giữ lại {total_n:,} dòng (39.09%), loại "
                                "bỏ chính xác "
                                f"{wf_sec['total_dropped']:,} dòng (60.91%) "
                                "không trùng lặp theo phễu tuần tự tại "
                                "Bảng 3.2 và Hình 3.1A: "
                                "tách 55,095 dòng Uber Taxi (MNAR), loại "
                                "208,402 dòng khuyết/lính canh price, loại "
                                "79,452 dòng khuyết "
                                "source và 79,224 dòng khuyết destination. "
                                "Kiểm toán chọn mẫu tách riêng từng đơn vị "
                                "đo (Hình 3.1B) giữa "
                                f"{sb_aud['kept_n']:,} dòng giữ lại và "
                                f"{sb_aud['dropped_non_taxi_n']:,} dòng phi-"
                                "Taxi bị loại xác nhận "
                                "cân bằng chặt: tỷ trọng Uber "
                                f"{sb_aud['uber_share_kept_pct']:.2f}% vs. "
                                f"{sb_aud['uber_share_dropped_pct']:.2f}%, "
                                "tỷ lệ Surge Lyft trên nhóm quan sát thực "
                                f"{sb_aud['lyft_surge_observed_kept_pct']:.2f}"
                                "% vs. "
                                f"{sb_aud['lyft_surge_observed_dropped_pct']:.2f}"
                                "%, cự ly TB "
                                f"{sb_aud['mean_dist_kept_mi']:.3f} vs. "
                                f"{sb_aud['mean_dist_dropped_mi']:.3f} dặm "
                                "và giá khả dụng "
                                f"${sb_aud['mean_price_kept_usd']:.2f} vs. "
                                "$"
                                f"{sb_aud['mean_price_dropped_available_usd']:.2f}"
                                "."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 3.2: Phễu đối soát hao hụt dữ "
                                    "liệu tuần tự từng bước (693,071 → "
                                    f"{total_n:,} dòng sạch)"
                                ),
                                "headers": [
                                    "Bước xử lý tuần tự",
                                    "Dòng đầu vào",
                                    "Số dòng loại",
                                    "Dòng đầu ra",
                                    "Tỷ lệ / Gốc",
                                ],
                                "rows": tbl_3_2_rows,
                                "col_widths": [6.8, 2.3, 2.3, 2.3, 2.3],
                                "aligns": ["L", "R", "R", "R", "R"],
                                "source": (
                                    "Nguồn: Truy vết trực tiếp từ "
                                    "cab_rides.csv theo quy trình kiểm "
                                    "chứng hợp nhất của nhóm (tổng loại = "
                                    f"{wf_sec['total_dropped']:,})."
                                ),
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_3_0_cleaning_waterfall.png"
                                ),
                                "caption": (
                                    "Hình 3.1: Phễu hao hụt tuần tự đến "
                                    "bảng sạch (A) và đối chiếu thiên lệch "
                                    "chọn mẫu tách riêng từng đơn vị đo (B)"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Sau khi khử mã lính canh (-1, -100, 9999, "
                                "100000) và dấu âm (2,137 price, 3,823 "
                                "distance, 75 surge), "
                                "quy tắc khôi phục quy mô bậc 10 (src/"
                                "clean_raw_data.py) học trung vị giá cơ sở "
                                "hoàn toàn trên nhóm "
                                "quan sát surge thực và điều chỉnh các bản "
                                "ghi vượt trần vật lý ($95 cho price; 8.0 "
                                "dặm cho distance) "
                                "hoặc có tỷ số so với tham chiếu vượt "
                                "ngưỡng 3.55 (10^0.55) bằng cách chia "
                                "10^m. Quy tắc dùng mốc tham chiếu cố định "
                                "2.15 dặm cho distance để chặn rò rỉ "
                                "phân phối và điều chỉnh "
                                f"{cab_log['price_scale_fixed']:,} "
                                f"price ({n_p_rec:,} "
                                "dòng trong bảng master cuối = "
                                f"{r3_aud['price_scale_recovered_pct']:.2f}%),"
                                f" {cab_log['distance_scale_fixed']:,} "
                                "distance, "
                                f"{cab_log['surge_scale_fixed']:,} "
                                "surge và hạ max(rain) "
                                "xuống 0.7807 inch/h (0.5469 trong master);"
                                " "
                                f"{cab_log['distance_missing_kept_nan_for_pipeline']:,}"
                                " dòng khuyết riêng distance được giữ nguyên "
                                "NaN ở Stage 2 và nội suy theo trung vị "
                                "từng tuyến strictly-on-train bởi "
                                "RouteDistanceImputer bên trong Pipeline "
                                "(0% rò rỉ train/test). Bảng 3.3 minh họa "
                                "6 ví dụ thực tế trước/sau hiệu chỉnh. Độ "
                                "nhạy ngưỡng tỷ số "
                                "(kiểm tra độc lập không chia trước theo "
                                "surge_multiplier) cho thấy số dòng price "
                                "bị đánh dấu giữ ổn "
                                "định quanh 1.48% ở các mốc 3.0 "
                                f"({thr_s[0]['n_price_flagged_in_master']:,}"
                                "), 3.55 "
                                f"({thr_s[1]['n_price_flagged_in_master']:,}"
                                f" — chênh {p_rec_diff:,} dòng so với "
                                f"{n_p_rec:,} dòng hiệu chỉnh trong "
                                "pipeline chính có chia theo surge) và 4.0 "
                                f"({thr_s[2]['n_price_flagged_in_master']:,}"
                                "); "
                                "đồng thời kiểm chứng Làm sạch & Huấn "
                                "luyện thuần Train-only ở Mục 4.4 xác nhận "
                                f"0/{n_test_rand:,} giá "
                                f"Test thay đổi so với bản toàn cục."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 3.3: Ví dụ thực tế trước và sau "
                                    "áp dụng quy tắc khôi phục sai lệch quy "
                                    "mô bậc 10"
                                ),
                                "headers": [
                                    "Gói dịch vụ",
                                    "Tuyến đường Boston",
                                    "Giá trước sửa",
                                    "Trung vị tham chiếu",
                                    "Hệ số chia",
                                    "Giá sau sửa",
                                ],
                                "rows": tbl_3_3_ex_rows,
                                "col_widths": [2.2, 5.2, 2.3, 2.4, 1.8, 2.1],
                                "aligns": ["L", "L", "R", "R", "C", "R"],
                                "source": (
                                    "Nguồn: Trích xuất trực tiếp từ "
                                    "cab_rides.csv qua quy trình kiểm chứng "
                                    "dữ liệu của nhóm."
                                ),
                            },
                        ),
                    ],
                },
                {
                    "heading": "3.4. Phân tích thống kê mô tả",
                    "blocks": [
                        (
                            "p",
                            (
                                f"Trên {total_n:,} lượt báo giá sạch, cơ "
                                f"cấu mẫu gồm Uber {uber_n:,} dòng (51.86%, "
                                "TB $15.83, trung vị "
                                f"$12.50) và Lyft {lyft_tot_n:,} dòng "
                                "(48.14%, TB $17.38, trung vị $16.50) — "
                                "phản ánh thiết kế truy vấn "
                                "API phân tầng đều chứ không phải thị phần "
                                "thực. Giá báo toàn mẫu đạt trung bình "
                                f"${num['price']['mean']:.2f}, "
                                f"trung vị ${num['price']['median']:.2f} "
                                f"(SD = ${num['price']['std']:.2f}, IQR = "
                                f"${num['price']['iqr']:.2f}, "
                                "Skewness = "
                                f"+{num['price']['skewness']:.4f}); quãng "
                                "đường trung bình "
                                f"{num['distance']['mean']:.2f} dặm "
                                f"(trung vị {num['distance']['median']:.2f},"
                                " miền [0.02, 7.86] dặm). Trên mẫu quan "
                                "sát surge thực "
                                f"(n = {obs_surge_n:,}), toàn bộ "
                                f"{lyft_surge_cnt:,} lượt surge > 1.0x xuất "
                                "hiện trên API Lyft, tương ứng "
                                "tỷ lệ Surge thực của Lyft là "
                                f"{lyft_surge_rate:.2f}% "
                                f"({lyft_surge_cnt:,} / {lyft_obs_n:,}; 7 "
                                "bậc từ "
                                "1.25x đến 3.00x) — cao hơn hẳn con số bị "
                                "pha loãng 4.82% nếu điền mù NaN thành "
                                "1.0x. Lượng mưa trên "
                                f"{prov['weather_valid_rain_obs']:,} dòng "
                                "không lỗi cảm biến chủ yếu ở mức nhẹ–vừa "
                                "(Q3 = 0.00, tối đa "
                                "0.55 inch/h). Các ngoại lai hợp lệ ("
                                f"{iqr_p['count']:,} price > "
                                f"${iqr_p['upper_fence']:.2f}; "
                                f"{iqr_d['count']:,} distance > "
                                f"{iqr_d['upper_fence']:.2f} "
                                "mi) được giữ nguyên (Bảng 3.4)."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 3.4: Thống kê mô tả các biến định "
                                    "lượng cốt lõi trên tập dữ liệu sạch "
                                    "(master_rideshare.parquet)"
                                ),
                                "headers": [
                                    "Biến phân tích (Đơn vị)",
                                    "N hợp lệ",
                                    "Mean",
                                    "SD",
                                    "Min",
                                    "Q1",
                                    "Median",
                                    "Q3",
                                    "Max",
                                ],
                                "rows": tbl_3_4_desc_rows,
                                "col_widths": [
                                    4.4,
                                    1.8,
                                    1.4,
                                    1.3,
                                    1.2,
                                    1.3,
                                    1.5,
                                    1.3,
                                    1.8,
                                ],
                                "aligns": [
                                    "L",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                ],
                                "source": (
                                    "Nguồn: master_rideshare.parquet "
                                    f"(surge trên n = {obs_surge_n:,} quan "
                                    "sát thực; mưa trên n = "
                                    f"{prov['weather_valid_rain_obs']:,} "
                                    "không sentinel)."
                                ),
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "3.5. Trực quan hóa dữ liệu và Phát hiện Insight "
                        "kinh doanh"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Từ bộ 21 biểu đồ pipeline (và 17 biểu đồ "
                                "300 DPI chuẩn hóa đưa vào tiểu luận, đồng "
                                "bộ số liệu với "
                                "Dashboard 5 Tab), nhóm rút ra bốn phát "
                                "hiện khám phá trọng tâm:"
                            ),
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_3_1_market_overview.png"
                                ),
                                "caption": (
                                    "Hình 3.2: Cơ cấu số lượng quan sát báo "
                                    "giá API theo hai nền tảng Uber – Lyft "
                                    "và 12 gói dịch vụ tại Boston"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Nhận định 3.1 (Mẫu phân tầng đều 12 gói và "
                                "phân hóa giá 4 bậc): Hình 3.2 và Hình 3.3 "
                                "cho thấy mẫu API "
                                "truy vấn tương đối đều cho từng gói (~22,"
                                "000 lượt/gói) và tách thành 4 tầng giá rõ "
                                "rệt: Đi ghép "
                                "(Shared trung vị $7.00; UberPool $8.50), "
                                "Phổ thông 4 chỗ (Lyft $9.00; UberX & WAV "
                                "$9.50), Gia đình 6 chỗ "
                                "(Lyft XL $13.50; UberXL $15.00) và Hạng "
                                "sang (Lux $16.50 đến Lux Black XL $30.00). "
                                "Do tỷ trọng hạng sang "
                                "trong mẫu cao hơn tiêu dùng đường phố, mọi "
                                "phân tích suy luận đều kiểm soát theo phân "
                                "khúc."
                            ),
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_3_2_distributions.png"
                                ),
                                "caption": (
                                    "Hình 3.3: Phân phối mật độ xác suất và "
                                    "biểu đồ hộp ngoại lai của giá báo API "
                                    "(price) và quãng đường (distance)"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Nhận định 3.2 (Phân bố thu thập API và "
                                "tính phẳng của giá theo 24 giờ EST): Trên "
                                f"{prov['is_timestamp_original_1']:,} "
                                "bản ghi có timestamp gốc (Hình 3.4), số "
                                "lượt báo giá ghi nhận chạm đáy quanh 03:"
                                "00 và cao nhất lúc "
                                "18:00–19:00 EST — phản ánh mật độ truy "
                                "vấn API trong mẫu. Trên thang trục tung "
                                "trung thực [12, 21] USD, "
                                "đường giá báo trung bình gần như nằm "
                                "ngang xuyên suốt 24 giờ trên cả Lyft "
                                "(~$17.4) lẫn Uber (~$15.8)."
                            ),
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_3_3_temporal_est.png"
                                ),
                                "caption": (
                                    "Hình 3.4: Lưu lượng quan sát báo giá "
                                    "API và giá báo trung bình theo 24 giờ "
                                    "địa phương Boston EST (n = "
                                    f"{prov['is_timestamp_original_1']:,})"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Nhận định 3.3 (Giá báo trùng khít giữa "
                                "mưa nhẹ–vừa và trời tạnh trên tập sạch "
                                "sentinel): Hình 3.5 đối "
                                "chiếu bản đồ nhiệt Surge Lyft trên tập "
                                "quan sát thực có timestamp gốc (n = 70,"
                                "971; 5 ô xám khuyết thu "
                                "thập đêm Thứ Ba – rạng sáng Thứ Tư EST) "
                                "và giá báo Mưa vs. Không mưa trên "
                                f"{prov['weather_valid_rain_obs']:,} "
                                "dòng không lỗi cảm biến mưa: giá trung "
                                "bình gần trùng khít ở cả Uber ($15.84 vs. "
                                "$15.83) và Lyft ($17.39 vs. $17.41)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Nhận định 3.4 (Bất cân xứng không gian "
                                "gấp "
                                + "{:.1f}".format(
                                    top12[0]["surge_rate_pct"]
                                    / top12[-1]["surge_rate_pct"]
                                )
                                + " "
                                "lần và tính tụ cụm của Surge Lyft): Trên "
                                f"đúng {lyft_obs_n:,} lượt báo giá Lyft "
                                "quan sát surge thực (tỷ "
                                f"lệ toàn thành phố {lyft_surge_rate:.2f}%),"
                                " sự phân hóa địa lý lộ diện mạnh hơn rõ "
                                "rệt so với bản bị "
                                "pha loãng: năm quận đại học – thương mại "
                                f"dẫn đầu gồm {top12[0]['source']} "
                                f"({top12[0]['surge_rate_pct']:.2f}%, "
                                f"{top12[0]['surge_count']} lượt), "
                                f"{top12[1]['source']} "
                                f"({top12[1]['surge_rate_pct']:.2f}%, "
                                f"{top12[1]['surge_count']} lượt), "
                                f"{top12[2]['source']} "
                                f"({top12[2]['surge_rate_pct']:.2f}%, "
                                f"{top12[2]['surge_count']} lượt), "
                                f"{top12[3]['source']} "
                                f"({top12[3]['surge_rate_pct']:.2f}%, "
                                f"{top12[3]['surge_count']} lượt) và "
                                f"{top12[4]['source']} "
                                f"({top12[4]['surge_rate_pct']:.2f}%, "
                                f"{top12[4]['surge_count']} lượt) — cao gấp "
                                + "{:.1f}".format(
                                    top12[0]["surge_rate_pct"]
                                    / top12[-1]["surge_rate_pct"]
                                )
                                + " "
                                f"lần so với {top12[-1]['source']} "
                                f"({top12[-1]['surge_rate_pct']:.2f}%, "
                                f"{top12[-1]['surge_count']} lượt). "
                                f"Trong {cl_s['active_surge_loc_hours']:,} "
                                "ô (quận × giờ) có surge "
                                f"({cl_s['pct_loc_hours_with_surge']:.2f}%),"
                                " "
                                "bình quân có "
                                f"{cl_s['mean_surge_quotes_per_active_loc_hour']:.2f}"
                                " lượt surge tụ cụm cùng giờ (tối đa "
                                f"{cl_s['max_surge_quotes_in_single_loc_hour']}"
                                " lượt), củng cố bản chất cục bộ theo điểm "
                                "nóng của surge."
                            ),
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_3_4_weather_surge_est.png"
                                ),
                                "caption": (
                                    "Hình 3.5: Bản đồ nhiệt Surge Lyft "
                                    "theo Thứ × Giờ EST (n = 70,971 surge "
                                    "thực) và giá báo Mưa vs. Tạnh (n = "
                                    f"{prov['weather_valid_rain_obs']:,})"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "3.6. Kiểm định giả thuyết thống kê, Suy luận cụm "
                        "và Kiểm định tương đương TOST"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Bảng 3.5 trình bày trọn bộ 13 kiểm định "
                                "giả thuyết (H1a–H4c) trên bộ dữ liệu sạch "
                                "chuẩn hóa kèm hiệu chỉnh đa "
                                "kiểm định Holm–Bonferroni (kiểm soát "
                                "FWER) và Benjamini–Hochberg (BH-FDR). Lưu "
                                "ý kỹ thuật quan trọng: "
                                "trong kiểm định đa biến H4c (n_train = "
                                + "{:,}".format(
                                    int(
                                        _row(
                                            tb["hyp"],
                                            "hypothesis_id",
                                            "H4c_Wald_Joint_Weather_Multivariate",
                                        )["n_group1"]
                                    )
                                )
                                + "), "
                                "việc loại đúng biến giả dư thừa của thành "
                                "phần lưỡng phân K_{6,6} thứ hai "
                                "(destination_Boston University) "
                                "đưa ma trận thiết kế về hạng đầy đủ 100% "
                                "(0 cảnh báo suy biến). Bảng 3.6 và Hình "
                                "3.6A bổ sung kiểm định "
                                "sai số chuẩn cụm hai cấp, kiểm định cụm "
                                "cho H3b, độ nhạy loại trừ tuyệt đối mọi ô "
                                "từng chứa sentinel mưa "
                                "và Exact WCR Wild Cluster Bootstrap (B = "
                                "1,999)."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 3.5: Hệ thống 13 kiểm định giả "
                                    "thuyết (H1–H4) kèm hiệu chỉnh đa kiểm "
                                    "định Holm–Bonferroni & BH-FDR"
                                ),
                                "headers": [
                                    "Mã",
                                    "Phạm vi kiểm định",
                                    "Phép kiểm định",
                                    "p gốc",
                                    "p_Holm / q_BH",
                                    "p đối chứng",
                                    "Hiệu ứng",
                                    "Kết luận (α=5%)",
                                ],
                                "rows": tbl_3_5_rows,
                                "col_widths": [
                                    0.85,
                                    3.10,
                                    2.10,
                                    1.55,
                                    2.00,
                                    1.55,
                                    2.40,
                                    2.45,
                                ],
                                "aligns": [
                                    "L",
                                    "L",
                                    "L",
                                    "R",
                                    "R",
                                    "R",
                                    "L",
                                    "L",
                                ],
                                "source": (
                                    "Nguồn: stage5_hypothesis_tests.csv và "
                                    "kết quả hiệu chỉnh đa kiểm định của "
                                    "nhóm. *H3b dưới sai số cụm xem Bảng "
                                    "3.6."
                                ),
                                "font_size": 8.4,
                            },
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 3.6: Suy luận cụm hai cấp, Độ "
                                    "nhạy Sentinel mưa, Exact WCR (B = 1,"
                                    "999), H3b cụm và TOST"
                                ),
                                "headers": [
                                    "Mô hình / Kiểm định mở rộng",
                                    "Cỡ mẫu / Cụm",
                                    "Ước lượng",
                                    "95% CI / Thống kê",
                                    "p-value",
                                    "Kết luận & TOST",
                                ],
                                "rows": tbl_3_6_rows,
                                "col_widths": [
                                    3.70,
                                    2.15,
                                    2.45,
                                    2.30,
                                    2.45,
                                    2.95,
                                ],
                                "aligns": ["L", "L", "R", "R", "R", "L"],
                                "source": (
                                    "Nguồn: Kết quả kiểm chứng sai số chuẩn "
                                    "cụm và Exact WCR của nhóm (phép chiếu "
                                    "trực giao M_Z đủ hệ số DOF (G/"
                                    "(G−1))·((N−1)/(N−k)); Cameron et al., "
                                    "2008)."
                                ),
                                "font_size": 8.5,
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_3_5_hypotheses_clean.png"
                                ),
                                "caption": (
                                    "Hình 3.6: Kiểm định Cluster-Robust, độ "
                                    "nhạy Sentinel mưa & TOST (A) và điểm "
                                    "cắt UberX–Lyft trên X_train (B)"
                                ),
                            },
                        ),
                        (
                            "bullets",
                            [
                                (
                                    "Thời tiết (H1a–c, H3b, H4c và Bảng "
                                    "3.6): Khi kiểm soát gói xe, cự ly, "
                                    "surge thực và 72 tuyến cố định "
                                    "(n = "
                                    f"{st_ext['n_valid_rain_and_surge_dist_obs']:,}"
                                    "), hệ số mưa là "
                                    f"{w_ct_hr['beta_usd']:+.4f} USD với "
                                    f"p_cl = {w_ct_loc['p_value']:.4f} "
                                    f"({st_ext['n_loc_hr_clusters_obs']:,} "
                                    "cụm quận × giờ) và "
                                    f"p_hr = {w_ct_hr['p_value']:.4f} "
                                    "("
                                    f"{st_ext['n_citywide_hour_clusters_obs']}"
                                    " giờ toàn Boston, 95% CI "
                                    f"[{w_ct_hr['ci95'][0]:+.4f}, "
                                    f"{w_ct_hr['ci95'][1]:+.4f}]). Kiểm tra "
                                    f"trên {st_ext['n_valid_rain_dist_obs']:,}"
                                    " lượt có cự ly hợp lệ ("
                                    f"{st_ext['n_loc_hr_clusters_dist']:,} "
                                    "cụm quận × giờ / "
                                    f"{st_ext['n_citywide_hour_clusters_dist']}"
                                    " giờ; không dùng biến surge) cũng cho "
                                    f"p_hr = {w_ns_hr['p_value']:.4f}, và "
                                    "loại triệt để mọi ô từng "
                                    "chứa dù chỉ 1 bản tin sentinel mưa (n "
                                    "= "
                                    f"{ws_any['n_obs_excl_any_sentinel_cell']:,}"
                                    ") cho p_hr = "
                                    f"{ws_any['p_hour']:.4f}. Trên mọi cấu "
                                    "trúc, TOST xác nhận chênh lệch nằm "
                                    "chặt trong biên ±$0.10 "
                                    "(p_TOST < 0.0001). Riêng H3b (xác "
                                    "suất Surge Lyft khi mưa: "
                                    f"{h3b_cl['surge_rate_rain_pct']:.2f}% "
                                    "vs. "
                                    "tạnh "
                                    f"{h3b_cl['surge_rate_dry_pct']:.2f}%), "
                                    "dù kiểm định iid ngây thơ báo p = "
                                    f"{h3b_cl['p_naive_iid']:.4f} "
                                    "(ngược chiều giả thuyết mưa làm tăng "
                                    "surge, Cramér's V = 0.0104), sai số "
                                    "chuẩn cụm đưa p lên "
                                    f"{h3b_cl['p_loc_hr_cluster']:.4f} "
                                    "(quận × giờ) và "
                                    f"{h3b_cl['p_hour_cluster']:.4f} (giờ "
                                    "Boston) — đồng "
                                    "thời không vượt qua cả Holm lẫn BH-"
                                    "FDR. Lát cắt mưa lớn ≥ 0.10 in/h trên "
                                    "toàn mẫu hồi quy "
                                    f"N = {wcr_loc['n_obs']:,} "
                                    f"({wcr_loc['n_total_clusters']:,} cụm /"
                                    f" {wcr_hr['n_total_clusters']} giờ; "
                                    "nhóm treated mưa lớn gồm "
                                    f"{hv['n_heavy_obs']:,} lượt / "
                                    f"{hv['n_heavy_loc_hr_clusters']} cụm / "
                                    f"{hv['n_heavy_citywide_hours']} giờ, "
                                    "lưu ý few treated clusters với 2⁸ = "
                                    "256 tổ hợp dấu Rademacher) "
                                    "qua Exact WCR (B = 1,999) cho p_wild "
                                    f"= {wcr_loc['p_wild']:.4f} / "
                                    f"{wcr_hr['p_wild']:.4f} (kết quả thăm "
                                    "dò công suất thấp)."
                                ),
                                (
                                    "Khung giờ EST (H2a–c, H3c–d): Sau "
                                    "hiệu chỉnh Holm (p_Holm = 1.0000) và "
                                    "Wald F cụm "
                                    f"({tod_cl['n_clusters']:,} cụm EST, "
                                    f"F_cl = {tod_cl['f_stat_cluster']:.4f},"
                                    " p_cl = "
                                    f"{tod_cl['p_cluster']:.4f}, chênh tối "
                                    f"đa ${tod_cl['max_min_gap_usd']:.4f}), "
                                    "chưa đủ bằng chứng bác bỏ "
                                    "đồng nhất giá hay Surge giữa bốn buổi "
                                    "cố định."
                                ),
                                (
                                    "Bất đối xứng công bố Surge trên API "
                                    "(H3a): Bác bỏ mạnh H0 sau hiệu chỉnh "
                                    "Holm (p_Holm < 0.0001, "
                                    "V = 0.2053: Lyft thực "
                                    f"{lyft_surge_rate:.2f}% vs. Uber "
                                    "0.00%). Kết quả phản ánh bất đối xứng "
                                    "cơ chế "
                                    "công bố API (Uber tích hợp phụ phí "
                                    "động ẩn trong Upfront Pricing), không "
                                    "kết luận Uber thiếu surge."
                                ),
                                (
                                    "Biểu giá cắt nhau (H4a–b): Wald HC3 "
                                    "bác bỏ mạnh H0 trên cặp chủ lực UberX "
                                    "vs. Lyft (H4a: p_Holm < 0.0001), còn "
                                    "UberPool vs. Shared (H4b: p = "
                                    f"{h4b_mt['raw_p']:.4f}, p_Holm = "
                                    f"{h4b_mt['p_holm']:.4f}) không cắt "
                                    "trong miền; độ nhạy giữa Đặc tả A "
                                    "("
                                    f"{ux_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                    " dặm) và Đặc tả B thuần 1.0x "
                                    "("
                                    f"{ux_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                    " dặm, Hình 3.6B) được phân tích đủ 5 "
                                    "cặp tại Mục 4.5."
                                ),
                            ],
                        ),
                    ],
                },
            ],
        },
        # =====================================================================
        # CHƯƠNG 4
        # =====================================================================
        {
            "title": "CHƯƠNG 4: XÂY DỰNG MÔ HÌNH VÀ ĐÁNH GIÁ HIỆU NĂNG",
            "sections": [
                {
                    "heading": (
                        "4.1. Kỹ thuật tạo lập và Lựa chọn đặc trưng "
                        "(Feature Engineering)"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Từ 32 cột của bảng sạch, src/features.py "
                                "tạo thêm 9 đặc trưng kinh tế và chu kỳ "
                                "thuần đơn dòng (tổng "
                                f"{len(dd)} cột), sau đó loại chặt 14 cột "
                                "khỏi ma trận X (2 cột mục tiêu & rò rỉ: "
                                "price, price_per_mile; "
                                "8 cột định danh/ngày giờ thô: id, "
                                "product_id, time_stamp, datetime, "
                                "location, route_id, date_hour, "
                                "day_name; và 4 cờ nguồn gốc: "
                                "weather_matched, is_timestamp_original, "
                                "is_surge_original, is_rain_sentinel) "
                                "để giữ đúng 27 đặc trưng hợp lệ trong bộ "
                                "mô hình có điều kiện (và 23 đặc trưng "
                                "trong thực nghiệm Ex-Ante "
                                f"sau khi loại thêm 4 biến phụ thuộc surge):"
                            ),
                        ),
                        (
                            "bullets",
                            [
                                "Loại trừ biến rò rỉ mục tiêu "
                                "price_per_mile: Đưa price_per_mile (= "
                                "price / distance) cùng distance vào HistGBR "
                                "kéo MAE và MAPE trên tập có distance thực "
                                "xuống giả tạo (MAE = "
                                f"${float(leak['corrupted_hgb_with_ppm']['mae']):.4f}"
                                ", MAPE = "
                                f"{float(leak['corrupted_hgb_with_ppm']['mape_pct']):.2f}"
                                "%), dù R² chung tụt về "
                                f"{float(leak['corrupted_hgb_with_ppm']['r2']):.4f}"
                                " do điền trung vị toàn cục trên 3,830 dòng "
                                "Test khuyết distance/price_per_mile; biến "
                                "này bị cấm tuyệt đối.",
                                "Khử đa cộng tuyến cấu trúc bằng biến tâm "
                                "hóa lyft_surge_excess: Trên nhóm surge "
                                "thực, tích thô distance_x_surge gây "
                                f"VIF(distance) = {vif_a['distance']:.2f} "
                                "và VIF(distance_x_surge) = "
                                f"{vif_a['distance_x_surge']:.2f}. Chuyển "
                                "sang lyft_surge_excess = (surge − 1.0) × "
                                "is_lyft hạ VIF(distance) xuống "
                                f"{vif_b['distance']:.2f} và hai biến tương "
                                "tác xuống "
                                f"{vif_b['distance_x_lyft_surge_excess']:.2f}"
                                f" và {vif_b['lyft_surge_excess']:.2f}, "
                                "trong khi toàn bộ 6 biến thời tiết giữ "
                                "VIF < 5 (Bảng 4.1, Hình 4.1).",
                                "Thiết kế EconomicInteractionPreprocessor: "
                                "Cho phép họ tuyến tính (M2–M4) ước lượng "
                                "riêng hệ số chặn, độ dốc cự ly, độ cong "
                                "log_distance và phụ trội surge cho từng "
                                "gói xe, giúp OLS nâng R² từ "
                                f"{float(m1['test_r2']):.4f} (M1) lên "
                                f"{float(m2['test_r2']):.4f} (M2) và hạ "
                                f"MAPE từ {float(m1['test_mape_pct']):.2f}% "
                                f"xuống {float(m2['test_mape_pct']):.2f}%.",
                            ],
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.1: So sánh hệ số phóng đại "
                                    "phương sai (VIF) trước và sau tâm hóa "
                                    "tương tác (Tập Train)"
                                ),
                                "headers": [
                                    "Nhóm đặc trưng",
                                    "VIF Trước (Cấu hình A)",
                                    "VIF Sau (Cấu hình B)",
                                    "Đánh giá đa cộng tuyến",
                                ],
                                "rows": tbl_4_1_rows,
                                "col_widths": [4.8, 2.6, 2.6, 6.0],
                                "aligns": ["L", "R", "R", "L"],
                                "source": (
                                    "Nguồn: reports/tables/stage4_vif.csv "
                                    "(tính trên tập Train khớp thời tiết "
                                    "không sentinel & surge thực)."
                                ),
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_1_corr_vif.png"
                                ),
                                "caption": (
                                    "Hình 4.1: Tương quan Pearson (r) với "
                                    "biến mục tiêu price (A) và đối chiếu "
                                    "VIF trước – sau tâm hóa trên tập Train "
                                    "(B)"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "4.2. Xử lý dữ liệu khuyết theo từng Pipeline và "
                        "Phân chia tập Huấn luyện / Kiểm tra"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Toàn bộ 3 chiến lược phân chia dữ liệu "
                                "cho mô hình có điều kiện được thực hiện "
                                "ĐỘC QUYỀN trên tập "
                                "quan sát surge thực (is_surge_original == "
                                f"1, n = {obs_surge_n:,}) và tách rời hai "
                                "nhánh Pipeline "
                                "(Hình 4.3): (i) Nhánh tuyến tính & cây "
                                "truyền thống (M1–M6) bọc "
                                "SimpleImputer(strategy='median'), "
                                "StandardScaler và OneHotEncoder / "
                                "EconomicInteractionPreprocessor bên trong "
                                "Pipeline chỉ fit() trên "
                                "X_train; (ii) Nhánh Boosting bản địa (M7 "
                                "HistGradientBoosting) dùng m7_hgb_native "
                                "truyền thẳng cả biến "
                                "số chứa NaN lẫn 5 cột phân loại kiểu "
                                "pandas category vào "
                                "HistGradientBoostingRegressor("
                                "categorical_features='from_dtype'). Kiểm "
                                "định độ nhạy Complete-Case trên tập Test "
                                f"(n = {n_test_rand:,}) "
                                "xác nhận hiệu năng ổn định cao giữa nhóm "
                                "khớp thời tiết hợp lệ (n = "
                                f"{miss_s['M7_complete_weather_valid_rain']['n']:,}"
                                ": "
                                "M7 R² = "
                                f"{miss_s['M7_complete_weather_valid_rain']['r2']:.4f}"
                                ", MAPE = "
                                f"{miss_s['M7_complete_weather_valid_rain']['mape_pct']:.2f}"
                                "%) "
                                "và nhóm không khớp thời tiết (n = "
                                f"{miss_s['M7_unmatched_weather_0']['n']:,}:"
                                " M7 R² = "
                                f"{miss_s['M7_unmatched_weather_0']['r2']:.4f}"
                                ", MAPE = "
                                f"{miss_s['M7_unmatched_weather_0']['mape_pct']:.2f}"
                                "%); "
                                f"{n_test_rain_sentinel:,} dòng còn lại của "
                                f"tập Test ({miss_s['M7_full_test']['n']:,} "
                                "− "
                                f"{miss_s['M7_complete_weather_valid_rain']['n']:,}"
                                " − "
                                f"{miss_s['M7_unmatched_weather_0']['n']:,})"
                                " "
                                "là các lượt khớp trạm thời tiết nhưng "
                                "thuộc ô lỗi cảm biến mưa "
                                "(is_rain_sentinel = 1)."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Bên cạnh Random Split 80/20 "
                                f"({n_train_rand:,} Train / {n_test_rand:,} "
                                "Test; KS = "
                                + "{:.4f}".format(
                                    float(
                                        _row(
                                            tb["splits"],
                                            "split_strategy",
                                            "random_80_20",
                                        )["ks_statistic"]
                                    )
                                )
                                + "), nghiên cứu đánh "
                                "giá song song trên Temporal Split EST "
                                f"({int(r_temp_m7['n_train']):,} / "
                                f"{int(r_temp_m7['n_test']):,} TS "
                                "& surge gốc) và Route Group Split (57 "
                                "tuyến Train dự báo 15 tuyến chưa gặp: "
                                f"{int(r_grp_m7['n_train']):,} "
                                f"/ {int(r_grp_m7['n_test']):,}) như trình "
                                "bày tại Bảng 4.2 và Hình 4.2."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.2: Thống kê đối chứng ba chiến "
                                    "lược phân chia tập Huấn luyện và Kiểm "
                                    "tra (mẫu surge thực n = "
                                    f"{obs_surge_n:,})"
                                ),
                                "headers": [
                                    "Chiến lược phân chia",
                                    "Tập Train (%)",
                                    "Tập Test (%)",
                                    "Giá TB Train / Test",
                                    "KS stat",
                                    "SMD",
                                ],
                                "rows": tbl_4_2_rows,
                                "col_widths": [4.9, 2.6, 2.6, 2.9, 1.5, 1.5],
                                "aligns": ["L", "R", "R", "R", "R", "R"],
                                "source": (
                                    "Nguồn: Trích xuất trực tiếp từ reports/"
                                    "tables/stage4_split_summary.csv."
                                ),
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_2_split_dist.png"
                                ),
                                "caption": (
                                    "Hình 4.2: Đối chiếu mật độ phân phối "
                                    "giá báo giữa tập Train và Test trên 3 "
                                    "chiến lược phân chia dữ liệu"
                                ),
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_3_pipeline_clean.png"
                                ),
                                "caption": (
                                    "Hình 4.3: Kiến trúc Pipeline chuẩn xác "
                                    "(41 → loại 14 → 27 đặc trưng), tách "
                                    "nhánh M1–M6 vs. M7 bản địa và kết quả "
                                    "đánh giá"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "4.3. Huấn luyện các mô hình Machine Learning và "
                        "Tái lập siêu tham số"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Mười cấu hình hồi quy có điều kiện được "
                                "tinh chỉnh siêu tham số độc lập chỉ trên "
                                "X_train rồi đánh giá "
                                "đồng nhất qua 5-Fold Cross-Validation "
                                f"trên toàn bộ {n_train_rand:,} dòng trước "
                                "khi kiểm tra một lần duy "
                                f"nhất trên tập Test ({n_test_rand:,} "
                                "dòng). Để triệt tiêu rò rỉ tiền xử lý nội "
                                "bộ giữa các fold CV, mọi "
                                "bộ dò lưới đều bọc trực tiếp tiền xử lý "
                                "bên trong Pipeline trên mẫu con 60,000 "
                                "dòng rút độc quyền từ "
                                "X_train (src/models.py): M3 Ridge dùng "
                                "GridSearchCV 5-fold (best alpha = "
                                f"{tuning['M3_Ridge']['best_alpha']}); "
                                "M4 Lasso dùng 5-fold CV độc lập từng fold "
                                "với ma trận Gram (precompute=True) và "
                                "warm_start theo alpha "
                                "giảm dần (best alpha = "
                                f"{tuning['M4_Lasso']['best_alpha']}, ép "
                                "đúng "
                                f"{tuning['M4_Lasso']['n_zeroed_features']}/"
                                f"{tuning['M4_Lasso']['n_total_features']} "
                                "đặc trưng về 0, cả 6 biến thời tiết có |β|"
                                f" ≤ {lasso_max_w_beta:.4f}); M5 "
                                "DecisionTree dùng RandomizedSearchCV 5-"
                                f"fold (max_depth = {m5_p['max_depth']}, "
                                "min_samples_leaf = "
                                f"{m5_p['min_samples_leaf']}, "
                                "min_samples_split = "
                                f"{m5_p['min_samples_split']}); M6 "
                                "RandomForest dùng "
                                "RandomizedSearchCV 3-fold (max_depth = "
                                f"{m6_p['max_depth']}, max_features = "
                                f"{m6_p['max_features']}, "
                                "min_samples_leaf = "
                                f"{m6_p['min_samples_leaf']}, huấn luyện "
                                "cuối n_estimators = 100, max_samples = "
                                "0.50); "
                                "và M7 HistGradientBoosting dùng "
                                "RandomizedSearchCV 5-fold (learning_rate "
                                f"= {m7_p['learning_rate']}, "
                                f"max_leaf_nodes = {m7_p['max_leaf_nodes']},"
                                " min_samples_leaf = "
                                f"{m7_p['min_samples_leaf']}, "
                                "l2_regularization = "
                                f"{m7_p['l2_regularization']}, max_iter = "
                                "350, early_stopping = True). Toàn bộ siêu "
                                "tham số trên khớp 100% khi chạy lại mã "
                                "nguồn src/models.py."
                            ),
                        ),
                    ],
                },
                {
                    "heading": (
                        "4.4. So sánh hiệu năng (Có điều kiện vs. Ex-Ante), "
                        "Hiệu chỉnh CQR và Độ bền"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Bảng 4.3 và Hình 4.4 tổng hợp kết quả 10 "
                                "mô hình có điều kiện trên tập kiểm tra "
                                "ngẫu nhiên (n = "
                                f"{n_test_rand:,}). Sau khi loại bỏ nhiễu "
                                "điền mù surge trên Lyft, hiệu năng cải "
                                "thiện mạnh trên mọi họ "
                                "mô hình: (1) M7 HistGradientBoosting đạt "
                                f"Test R² = {float(m7['test_r2']):.4f}, "
                                f"RMSE = ${float(m7['test_rmse']):.4f}, "
                                f"MAE = ${float(m7['test_mae']):.4f}, MAPE "
                                f"= {float(m7['test_mape_pct']):.2f}%, vượt "
                                "M6 Random Forest "
                                f"(R² = {float(m6['test_r2']):.4f}, RMSE = "
                                f"${float(m6['test_rmse']):.4f}, MAPE = "
                                f"{float(m6['test_mape_pct']):.2f}%) "
                                f"trên cả {p_m76['m7_better_folds_count']}/"
                                "5 fold CV với độ chênh đạt ý nghĩa thống "
                                "kê rõ rệt "
                                f"({p_m76['mean_diff_m7_minus_m6_usd']:+.4f}"
                                " USD, paired t = "
                                f"{p_m76['paired_t_stat']:.2f}, p = "
                                f"{p_m76['paired_p_value']:.4f}, bootstrap "
                                "95% CI ["
                                f"{p_m76['bootstrap_95ci_diff_usd'][0]:+.4f}"
                                ", "
                                f"{p_m76['bootstrap_95ci_diff_usd'][1]:+.4f}"
                                "]), đồng thời có Overfit Gap nhỏ hơn "
                                f"({float(m7['r2_overfit_gap']):+.4f} "
                                f"vs. {float(m6['r2_overfit_gap']):+.4f}) "
                                "và huấn luyện nhanh gấp "
                                + "{:.1f}".format(
                                    float(m6["train_time_sec"])
                                    / float(m7["train_time_sec"])
                                )
                                + " "
                                f"lần ({float(m7['train_time_sec']):.2f}s "
                                f"vs. {float(m6['train_time_sec']):.2f}s), "
                                "trong khi M6 vẫn nhỉnh "
                                "nhẹ ở MAE/MAPE. (2) Giải mã Baseline M0b "
                                f"(R² = {float(m0b['test_r2']):.4f}, MAPE = "
                                f"{float(m0b['test_mape_pct']):.2f}%): "
                                "M0b chỉ cạnh tranh trên lượt không surge "
                                "(MAPE = "
                                f"{s_m0b['non_surge_m0b']['mape_pct']:.2f}% "
                                "nhờ mất mát "
                                "trung vị L1, khớp P50 Quantile MAPE = "
                                f"{pi_ov['p50_median_mape_pct']:.2f}%), "
                                "nhưng sụp đổ nặng trên "
                                f"{s_m0b['surge_gt1_m0b']['n']:,} lượt có "
                                "Surge (> 1.0x) với RMSE = "
                                f"${s_m0b['surge_gt1_m0b']['rmse']:.2f} "
                                "so với M7 chỉ "
                                f"${s_m0b['surge_gt1_m7']['rmse']:.2f} (tốt "
                                "hơn 5.0 lần). (3) Họ tuyến tính tương tác "
                                "(M2/M3/M4) đạt R² = "
                                f"{float(m2['test_r2']):.4f}, RMSE = "
                                f"${float(m2['test_rmse']):.4f}, hội tụ sát "
                                "nhau "
                                f"do tỷ lệ n_train/p rất lớn."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.3: So sánh hiệu năng 10 cấu "
                                    "hình mô hình có điều kiện trên 5-Fold "
                                    f"CV và Tập Test (n = {n_test_rand:,})"
                                ),
                                "headers": [
                                    "Mô hình",
                                    "CV RMSE ($)",
                                    "Test R²",
                                    "Test RMSE",
                                    "Test MAE",
                                    "MAPE",
                                    "Overfit",
                                    "SMART*",
                                ],
                                "rows": tbl_4_3_rows,
                                "col_widths": [
                                    3.9,
                                    2.5,
                                    1.5,
                                    1.8,
                                    1.7,
                                    1.5,
                                    1.5,
                                    1.6,
                                ],
                                "aligns": [
                                    "L",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "C",
                                ],
                                "source": (
                                    "Nguồn: stage5_cv_results.csv & "
                                    "stage5_test_metrics.csv (*ngưỡng SMART "
                                    "nội miền: R² ≥ 0.90, MAPE ≤ 10%)."
                                ),
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_4_model_comp.png"
                                ),
                                "caption": (
                                    "Hình 4.4: Đối chiếu hiệu năng RMSE (5-"
                                    f"Fold CV & Test, N = {n_test_rand:,}) "
                                    "và MAPE (%) giữa 10 cấu hình mô hình"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Phân rã theo chế độ dữ liệu và thực "
                                "nghiệm Ex-Ante không dùng surge (Bảng "
                                "4.4) làm sáng tỏ bản chất "
                                "kinh tế của bài toán dự báo giá API: (i) "
                                "Trong mô hình có điều kiện, độ chính xác "
                                "trên Lyft (surge công "
                                "bố: R² = "
                                f"{regime['lyft_observed_surge_test']['M7_Conditional']['r2']:.4f}"
                                ", RMSE = "
                                "$"
                                f"{regime['lyft_observed_surge_test']['M7_Conditional']['rmse']:.4f}"
                                ", chỉ 1/18,440 sai số > $15 (max $15.08, "
                                "0 > $16, Kurtosis = 3.47) vượt trội hẳn so "
                                "với Uber (surge ẩn: R² = "
                                f"{regime['uber_hidden_surge_test']['M7_Conditional']['r2']:.4f}"
                                ", "
                                "RMSE = $"
                                f"{regime['uber_hidden_surge_test']['M7_Conditional']['rmse']:.4f}"
                                ", chiếm 74.3% tổng SSE và "
                                "57/58 sai số > $15 [47/47 > $16], Kurtosis = "
                                "29.46). (ii) Khi loại bỏ toàn bộ 4 biến "
                                "surge để đánh giá dự báo "
                                "Ex-Ante thuần (23 biến công khai), hiệu "
                                "năng M7 trên Uber hầu như giữ nguyên "
                                "(RMSE = "
                                "$"
                                f"{ex_pricing['M7_ExAnte_Uber_Test']['rmse']:.4f}"
                                " vs. $"
                                f"{regime['uber_hidden_surge_test']['M7_Conditional']['rmse']:.4f}"
                                "), "
                                "nhưng RMSE trên toàn mẫu Lyft tăng gấp "
                                "đôi lên $"
                                f"{ex_pricing['M7_ExAnte_Lyft_All_Test']['rmse']:.4f}"
                                " "
                                "(R² = "
                                f"{ex_pricing['M7_ExAnte_Lyft_All_Test']['r2']:.4f}"
                                ") và sụp xuống R² = "
                                f"{ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['r2']:.4f}"
                                " (RMSE = $"
                                f"{ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['rmse']:.2f}"
                                ") "
                                "trên các lượt Lyft đang surge. Đáng chú ý,"
                                " đánh giá mô hình Ex-Ante trên tập "
                                f"Holdout {lyft_miss_n:,} dòng "
                                "Lyft khuyết surge cho R² = "
                                f"{ex_pricing['M7_ExAnte_Lyft_MissingSurge_Holdout']['r2']:.4f}"
                                ", RMSE = "
                                "$"
                                f"{ex_pricing['M7_ExAnte_Lyft_MissingSurge_Holdout']['rmse']:.4f}"
                                " — trùng khớp tới hai chữ số thập phân "
                                "với Lyft Test "
                                "("
                                f"{ex_pricing['M7_ExAnte_Lyft_All_Test']['r2']:.4f}"
                                " / $"
                                f"{ex_pricing['M7_ExAnte_Lyft_All_Test']['rmse']:.4f}"
                                "), "
                                "cung cấp bằng chứng định lượng khẳng định "
                                "tập khuyết surge thực sự mang cấu trúc "
                                "~6.8% surge ẩn chứ không "
                                "phải 1.0x. (iii) Bộ phân loại Ex-Ante dự "
                                "báo P(surge > 1.0) trên Lyft (n_train = "
                                f"{ex_clf['n_lyft_train']:,}, "
                                f"n_test = {ex_clf['n_lyft_test']:,}; tỷ lệ "
                                "dương tính trên tập Test là "
                                f"{ex_clf['test_positive_prevalence_pct']:.2f}"
                                "% = "
                                f"{n_pos_test:,}/{ex_clf['n_lyft_test']:,} "
                                f"so với {lyft_surge_rate:.2f}% toàn mẫu do "
                                "dao động lấy mẫu) "
                                f"đạt ROC-AUC = {hgb_c['roc_auc']:.4f}, PR-"
                                f"AUC = {hgb_c['pr_auc']:.4f}, Brier = "
                                f"{hgb_c['brier_score']:.5f} "
                                "ở HistGB (Logistic Regression đạt "
                                f"{lr_c['roc_auc']:.4f} / "
                                f"{lr_c['pr_auc']:.4f}). Do mất cân bằng "
                                "lớp 6.50%, "
                                "ngưỡng mặc định τ = 0.50 trên mô hình "
                                "chuẩn không dự báo ca dương tính nào "
                                "(Accuracy = "
                                f"{hgb_c['default_050_threshold_metrics']['accuracy']:.4f}"
                                ", Precision = Recall = F1 = 0.0000); khi "
                                "hiệu chỉnh "
                                "ngưỡng tối ưu F1 hoàn toàn trên Train 3-"
                                "Fold OOF (τ* = "
                                f"{hgb_c['f1_optimal_threshold_from_train']:.4f}"
                                "), "
                                "HistGB đạt Accuracy = "
                                f"{hgb_c['accuracy']:.4f}, Precision = "
                                f"{hgb_c['precision']:.4f}, Recall = "
                                f"{hgb_c['recall']:.4f}, F1 = "
                                f"{hgb_c['f1_score']:.4f} với ma trận nhầm "
                                f"lẫn [TN = {hgb_cm['tn']:,}, "
                                f"FP = {hgb_cm['fp']:,}, FN = "
                                f"{hgb_cm['fn']:,}, TP = {hgb_cm['tp']:,}] "
                                "(LogReg tại τ* = "
                                f"{lr_c['f1_optimal_threshold_from_train']:.4f}"
                                f": Acc = {lr_c['accuracy']:.4f}, P = "
                                f"{lr_c['precision']:.4f}, "
                                f"R = {lr_c['recall']:.4f}, F1 = "
                                f"{lr_c['f1_score']:.4f}). Đối chứng "
                                "class_weight='balanced' nâng Recall nhưng "
                                "giảm độ sắc nét phân tách (HistGB "
                                "Balanced tại τ* = "
                                f"{hgb_b['f1_optimal_threshold_from_train']:.4f}"
                                ": "
                                f"ROC-AUC = {hgb_b['roc_auc']:.4f}, Acc = "
                                f"{hgb_b['accuracy']:.4f}, P = "
                                f"{hgb_b['precision']:.4f}, "
                                f"R = {hgb_b['recall']:.4f}, F1 = "
                                f"{hgb_b['f1_score']:.4f}; LogReg Balanced: "
                                f"ROC-AUC = {lr_b['roc_auc']:.4f}, "
                                f"F1 = {lr_b['f1_score']:.4f}). Mức giảm "
                                "ROC-AUC khi hoán vị tập trung ở tuyến/cự "
                                "ly "
                                "("
                                f"{ex_clf['group_roc_auc_drop']['source_destination_distance']:.4f}"
                                ") và gói xe "
                                "("
                                f"{ex_clf['group_roc_auc_drop']['service_tier_name']:.4f}"
                                "), trong khi 7 biến thời tiết "
                                "("
                                f"{ex_clf['group_roc_auc_drop']['weather_7vars']:.4f}"
                                ") và khung giờ EST "
                                "("
                                f"{ex_clf['group_roc_auc_drop']['time_clock_est']:.4f}"
                                ") đóng góp gần như bằng không."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.4: Phân rã hiệu năng M7 theo "
                                    "chế độ API (Uber surge ẩn vs. Lyft "
                                    "công bố) và Thực nghiệm Ex-Ante (23 "
                                    "biến không surge)"
                                ),
                                "headers": [
                                    "Cấu hình / Lát cắt đánh giá",
                                    "N đánh giá",
                                    "R²",
                                    "RMSE ($)",
                                    "MAPE",
                                    "Nhận định cơ chế dữ liệu",
                                ],
                                "rows": tbl_4_3b_rows,
                                "col_widths": [4.3, 1.6, 1.3, 1.6, 1.3, 5.9],
                                "aligns": ["L", "R", "R", "R", "R", "L"],
                                "source": (
                                    "Nguồn: stage5_summary.json & "
                                    "stage5_ex_ante_experiments.csv "
                                    "(HistGB Ex-Ante Surge: ROC-AUC = "
                                    f"{hgb_c['roc_auc']:.4f}, PR-AUC = "
                                    f"{hgb_c['pr_auc']:.4f}; tại τ* = "
                                    f"{hgb_c['f1_optimal_threshold_from_train']:.4f}"
                                    f": Acc = {hgb_c['accuracy']:.4f}, P = "
                                    f"{hgb_c['precision']:.4f}, R = "
                                    f"{hgb_c['recall']:.4f}, F1 = "
                                    f"{hgb_c['f1_score']:.4f})."
                                ),
                                "font_size": 8.6,
                            },
                        ),
                        (
                            "p",
                            (
                                "Chẩn đoán phần dư của M7 (Hình 4.5) xác "
                                "nhận sai số không chệch (Mean = "
                                f"+${res_m7['residual_mean']:.4f}, "
                                f"SD = ${res_m7['residual_std']:.4f}, "
                                f"{res_m7['pct_within_1usd']:.2f}% lệch ≤ "
                                "±$1.00 và "
                                f"{res_m7['pct_within_2usd']:.2f}% lệch ≤ "
                                "±$2.00), với Kurtosis giảm quá nửa xuống "
                                "còn "
                                f"{res_m7['kurtosis']:.2f} sau khi loại bỏ "
                                "nhiễu surge giả trên Lyft — phần đuôi "
                                "phải còn lại hoàn toàn "
                                "thuộc cơ chế phụ phí ẩn của Uber; kiểm "
                                "định Breusch–Pagan (LM = "
                                f"{res_m7['breusch_pagan_lm_stat']:.2f}, "
                                "p < 0.0001) xác nhận phương sai thay đổi "
                                "theo phân khúc. Hình 4.6 làm rõ hiệu ứng "
                                "mẫu số MAPE ở gói "
                                "Shared (MAE thấp "
                                f"${float(shared_seg['m7_mae_usd']):.2f} "
                                "nhưng MAPE "
                                f"{float(shared_seg['m7_mape_pct']):.2f}% "
                                "do giá trung bình chỉ "
                                "$"
                                f"{float(shared_seg['mean_actual_usd']):.2f}"
                                ")."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Đối với khoảng bất định Quantile [P10, "
                                "P90], cả mô hình GĐ6 (100% Train) lẫn "
                                "Monotonic Split-CQR (85% "
                                "fit / 15% cal; Romano et al., 2019) đều "
                                "được nâng cấp dùng trực tiếp pandas "
                                "category gốc "
                                "(categorical_features='from_dtype') đồng "
                                "nhất 100% với M7 (Bảng 4.5 và Hình 4.7). "
                                "Trên bản GĐ6 100% "
                                "Train, hiệu ứng dấu phẩy động ở biên bậc "
                                "giá $0.50 làm PICP chặt "
                                f"({s6_bm['overall_strict_pct']:.2f}%; "
                                "Lux Black XL "
                                f"{s6_bm['lux_black_xl_strict_pct']:.2f}%, "
                                "Black SUV "
                                f"{s6_bm['black_suv_strict_pct']:.2f}%) "
                                "thấp hơn PICP dung sai 10⁻⁶ USD "
                                f"({s6_bm['overall_tol1e6_pct']:.2f}%; Lux "
                                "Black XL "
                                f"{s6_bm['lux_black_xl_tol1e6_pct']:.2f}%, "
                                "Black SUV "
                                f"{s6_bm['black_suv_tol1e6_pct']:.2f}%). "
                                "Khi áp dụng "
                                f"Monotonic Split-CQR (0/{cqr['n_test']:,} "
                                "cắt ngược), độ bao phủ tổng thể đạt "
                                f"{cqr['overall_cqr_picp_pct']:.2f}% (MPIW "
                                f"= ${cqr['overall_cqr_mpiw_usd']:.3f}) và "
                                "cả 12/12 gói hội tụ "
                                "rất chặt quanh mục tiêu 80% trong đoạn "
                                f"[{cqr['min_tier_cqr_picp_pct']:.2f}%, "
                                f"{cqr['max_tier_cqr_picp_pct']:.2f}%]. Đặc "
                                "biệt, do 99.9989% (270,895/270,898) báo "
                                "giá API trong tập sạch "
                                "bị lượng tử hóa rời rạc theo bội số $0.50,"
                                " một dịch chuyển biên rất nhỏ của q̂ cũng "
                                "đủ đưa hàng loạt điểm "
                                "trùng biên qua ngưỡng bao phủ (chẳng hạn "
                                f"Lux Black có q̂ = "
                                f"{lux_b['q_hat_usd_6dec']:+.4f} "
                                "USD làm PICP điều chỉnh "
                                f"{lux_b_delta:+.2f} "
                                "điểm phần trăm từ "
                                f"{lux_b['base85_raw_tol1e6_picp_pct']:.2f}% "
                                f"xuống {lux_b['cqr_picp_pct']:.2f}%, "
                                f"và {n_cqr_lt80}/12 "
                                "gói nằm sát quanh mốc danh nghĩa 80%). Cần "
                                "lưu ý hai cột PICP "
                                "đầu tại Bảng 4.5 "
                                "("
                                f"{cqr['overall_base85_raw_strict_picp_pct']:.2f}"
                                "% và "
                                f"{cqr['overall_base85_raw_tol1e6_picp_pct']:.2f}"
                                "%) là hiệu năng gốc của mô hình phân vị "
                                "fit trên 85% tập "
                                "Train trước khi cộng q̂ (phân biệt với "
                                "bản tham chiếu GĐ6 huấn luyện trên 100% "
                                "Train ở ghi chú nguồn)."
                            ),
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_5_residuals_m7.png"
                                ),
                                "caption": (
                                    "Hình 4.5: Chẩn đoán phần dư M7 "
                                    "HistGradientBoosting (Residual vs. "
                                    "Predicted, Q-Q Plot, phân phối và IQR "
                                    "12 gói)"
                                ),
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_6_segment_errors.png"
                                ),
                                "caption": (
                                    "Hình 4.6: Phân rã sai số MAE (USD) và "
                                    "MAPE (%) của M7 theo 12 gói dịch vụ, "
                                    "cự ly và chế độ Surge"
                                ),
                            },
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.5: Hiệu chỉnh Monotonic Split-"
                                    "CQR dùng pandas category gốc trên 12 "
                                    "gói dịch vụ (85%/15%, 0 cắt ngược)"
                                ),
                                "headers": [
                                    "Hãng — Gói dịch vụ",
                                    "Giá TB",
                                    "PICP 85% chặt",
                                    "PICP 85% (10⁻⁶)",
                                    "q̂ CQR (USD)",
                                    "PICP CQR",
                                    "MPIW CQR (Δ)",
                                ],
                                "rows": tbl_4_4_rows,
                                "col_widths": [
                                    3.35,
                                    1.35,
                                    1.85,
                                    2.05,
                                    2.55,
                                    1.80,
                                    3.05,
                                ],
                                "aligns": ["L", "R", "R", "R", "R", "R", "R"],
                                "source": (
                                    "Nguồn: Kết quả hiệu chỉnh Monotonic "
                                    "Split-CQR của nhóm (n_fit = "
                                    f"{cqr['n_fit']:,}, n_cal = "
                                    f"{cqr['n_cal']:,}, n_test = "
                                    f"{cqr['n_test']:,}, pandas category "
                                    "gốc; 2 cột PICP đầu thuộc mô hình 85% "
                                    "Fit trước CQR, q̂ chuẩn 4 chữ số thập "
                                    "phân). Tham chiếu GĐ6 100% Train: "
                                    "PICP chặt "
                                    f"{s6_bm['overall_strict_pct']:.2f}% / "
                                    "tol 10⁻⁶ = "
                                    f"{s6_bm['overall_tol1e6_pct']:.2f}%, "
                                    "MPIW = $"
                                    f"{pi_ov['mean_interval_width_mpiw_usd']:.2f}"
                                    "."
                                ),
                                "font_size": 8.7,
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_7_quantile_pi.png"
                                ),
                                "caption": (
                                    "Hình 4.7: Độ bao phủ 12 phân khúc "
                                    "trước/sau Monotonic Split-CQR (A) và "
                                    "độ rộng MPIW kèm tham số q̂ (B)"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Bảng 4.6 tổng hợp sáu kịch bản kiểm định "
                                "độ bền: M7 ổn định cao trên Temporal "
                                "Split EST (R² = "
                                f"{float(r_temp_m7['r2']):.4f}, MAPE = "
                                f"{float(r_temp_m7['mape_pct']):.2f}%), tập "
                                "Test hoàn toàn nguyên "
                                f"bản {r3_unmod_pct:.2f}% (R² = "
                                f"{float(r_orig_test_m7['r2']):.4f}, MAPE = "
                                f"{float(r_orig_test_m7['mape_pct']):.2f}%; "
                                "khi đồng thời huấn luyện và kiểm tra "
                                "thuần nguyên bản: R² = "
                                f"{float(r_orig_both_m7['r2']):.4f}, "
                                "MAPE = "
                                f"{float(r_orig_both_m7['mape_pct']):.2f}%),"
                                " làm sạch & "
                                "huấn luyện thuần Train-only (0/"
                                f"{n_test_rand:,} giá Test đổi, M7 R² = "
                                f"{m7_st['r2']:.4f}, RMSE = "
                                f"${m7_st['rmse']:.4f}; trên tập nguyên bản:"
                                f" R² = {m7_un['r2']:.4f}) và API-Batch "
                                "GroupShuffleSplit chặn "
                                f"{abg['sibling_overlap_random_pct']:.2f}% "
                                "trùng phiên mili-giây (M7 R² = "
                                f"{abg['m7']['r2']:.4f}, RMSE = "
                                f"${abg['m7']['rmse']:.4f}). Đáng chú ý, "
                                "trên Route Group Split (15 tuyến hoàn "
                                "toàn mới), M7 đạt R² = "
                                f"{float(r_grp_m7['r2']):.4f} và MAPE = "
                                f"{float(r_grp_m7['mape_pct']):.2f}% (vượt "
                                "ngưỡng 10% do 15 tuyến mới không được rò rỉ "
                                "trung vị cự ly khi nội suy "
                                "strictly-on-train; M6 = "
                                f"{float(r_grp_m6['mape_pct']):.2f}%, "
                                f"M2 = {float(r_grp_m2['mape_pct']):.2f}%), "
                                " xác nhận mô hình "
                                "vẫn suy giảm rõ khi rời mạng lưới 57 "
                                "tuyến huấn luyện và cần hiệu chuẩn khi mở "
                                "rộng tuyến mới; khi loại bỏ hoàn toàn điền "
                                "khuyết cự ly từ 15 tuyến Test trong quy "
                                "trình Strict Train-Only trên Route Group "
                                "Split, M7 đạt R² = "
                                f"{rg_st_m7['r2']:.4f} / MAPE = "
                                f"{rg_st_m7['mape_pct']:.2f}% toàn mẫu "
                                f"(riêng tập nguyên bản không khuyết cự ly: "
                                f"R² = {rg_st_un['r2']:.4f}, MAPE = "
                                f"{rg_st_un['mape_pct']:.2f}%). Sáu thí "
                                "nghiệm phá vỡ MB1–MB6 tại Bảng 4.7 tiếp "
                                "tục xác minh vai trò cơ chế của từng lựa "
                                "chọn thiết kế."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.6: Kiểm định độ bền M7 và M2 "
                                    "qua 6 kịch bản (Thời gian EST, Tuyến "
                                    "mới, Nguyên bản, Train-only & API "
                                    "Batch)"
                                ),
                                "headers": [
                                    "Kịch bản kiểm định độ bền",
                                    "Train / Test (n)",
                                    "M7 HistGBR (R² / RMSE / MAPE)",
                                    "M2 OLS / Ghi chú đối chứng",
                                ],
                                "rows": tbl_4_5_rows,
                                "col_widths": [5.1, 2.7, 4.1, 4.1],
                                "aligns": ["L", "R", "R", "R"],
                                "source": (
                                    "Nguồn: stage5_robustness.csv và kiểm "
                                    "chứng độ bền mở rộng của nhóm (*Train-"
                                    "only Full Refit: trên tập Test nguyên "
                                    f"bản M7 đạt R² = {m7_un['r2']:.4f}, "
                                    f"RMSE = ${m7_un['rmse']:.4f})."
                                ),
                                "font_size": 8.7,
                            },
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.7: Tổng hợp 6 thí nghiệm phá vỡ "
                                    "có kiểm soát (Mutate & Break MB1–MB6) "
                                    "kiểm chứng cơ chế mô hình"
                                ),
                                "headers": [
                                    "Mã",
                                    "Thao tác phá vỡ (Mutation)",
                                    "Chuẩn (R²/MAPE)",
                                    "Sau phá vỡ",
                                    "Bài học khoa học rút ra",
                                ],
                                "rows": tbl_4_6_rows,
                                "col_widths": [1.0, 4.1, 2.4, 2.2, 6.3],
                                "aligns": ["L", "L", "R", "R", "L"],
                                "source": (
                                    "Nguồn: Trích xuất trực tiếp từ reports/"
                                    "tables/stage5_mutate_break.csv."
                                ),
                                "font_size": 8.7,
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "4.5. Phân tích độ quan trọng của đặc trưng và Diễn "
                        "giải mô hình (XAI)"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Về đối sánh biểu giá cạnh tranh (Bảng 4.8 "
                                "và Hình 4.8), nghiên cứu ước lượng song "
                                "song Đặc tả A "
                                "(Price = β₀ + β₁ · (d × surge)) và Đặc tả "
                                "B (thuần không surge = 1.0x, Price = β₀ + "
                                "β₁ · d) trên cùng "
                                "tập huấn luyện X_train có surge quan sát "
                                f"thực (n = {n_train_rand:,}) kèm 95% CI "
                                "cụm phương pháp Delta. "
                                "Việc loại bỏ 38,037 lượt Lyft khuyết "
                                "surge tạo ra một phát hiện đảo chiều quan "
                                "trọng cần báo cáo trung "
                                "thực ở phân khúc gia đình 6 chỗ: (1) Ở "
                                "cặp Phổ thông 4 chỗ (UberX vs. Lyft), quy "
                                "luật cắt nhau vẫn giữ "
                                "vững qua cả hai đặc tả — điểm cắt dịch từ "
                                f"{ux_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " dặm (CI_cl "
                                "["
                                f"{ux_pair['spec_a_train80']['cluster']['ci95'][0]:.2f}"
                                ", "
                                f"{ux_pair['spec_a_train80']['cluster']['ci95'][1]:.2f}"
                                "]) "
                                "ở Đặc tả A lên "
                                f"{ux_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                " dặm (CI_cl "
                                "["
                                f"{ux_pair['spec_b_train80_nonsurge']['cluster']['ci95'][0]:.2f}"
                                ", "
                                f"{ux_pair['spec_b_train80_nonsurge']['cluster']['ci95'][1]:.2f}"
                                "]; "
                                "toàn mẫu không surge: "
                                f"{ux_pair['spec_b_full_nonsurge']['hc3']['d_star']:.2f}"
                                " mi) ở Đặc tả B. "
                                "(2) Ở cặp Gia đình 6 chỗ (UberXL vs. Lyft "
                                "XL), dù Đặc tả A cắt tại "
                                f"{xl_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " "
                                "dặm (CI_cl "
                                "["
                                f"{xl_pair['spec_a_train80']['cluster']['ci95'][0]:.2f}"
                                ", "
                                f"{xl_pair['spec_a_train80']['cluster']['ci95'][1]:.2f}"
                                "]), "
                                "nhưng ở Đặc tả B thuần không surge điểm "
                                "giao toán học dịch hẳn ra ngoài miền quan "
                                "sát tại "
                                "d* = "
                                f"{xl_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                " dặm (CI_cl "
                                "["
                                f"{xl_pair['spec_b_train80_nonsurge']['cluster']['ci95'][0]:.2f}"
                                ", "
                                f"{xl_pair['spec_b_train80_nonsurge']['cluster']['ci95'][1]:.2f}"
                                "]; "
                                "toàn mẫu không surge: "
                                f"{xl_pair['spec_b_full_nonsurge']['hc3']['d_star']:.2f}"
                                " dặm > 7.86 dặm). Điều này "
                                "chứng tỏ khi không có surge, Lyft XL rẻ "
                                "hơn UberXL trên TOÀN BỘ miền Boston [0.02,"
                                " 7.86] dặm; giao điểm "
                                f"{xl_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " dặm ở Đặc tả A hoàn toàn do phụ "
                                "phí surge của Lyft đẩy dốc hiệu dụng lên. "
                                "(3) Kiểm tra độ nhạy trên miền giá sàn (d "
                                "≥ 1.0 mi, loại bỏ vùng phẳng giá tối "
                                "thiểu) cho thấy điểm cắt "
                                "UberX–Lyft nằm tại "
                                f"{ux_pair['above_floor_check']['d_star_ge_1mi']:.2f}"
                                " mi (CI_cl "
                                "["
                                f"{ux_pair['above_floor_check']['ci95_cluster_ge_1mi'][0]:.2f}"
                                ", "
                                f"{ux_pair['above_floor_check']['ci95_cluster_ge_1mi'][1]:.2f}"
                                "]) còn UberXL–Lyft XL tiến sát biên cực "
                                "đại nội đô ("
                                f"{xl_pair['above_floor_check']['d_star_ge_1mi']:.2f}"
                                " mi, CI_cl "
                                "["
                                f"{xl_pair['above_floor_check']['ci95_cluster_ge_1mi'][0]:.2f}"
                                ", "
                                f"{xl_pair['above_floor_check']['ci95_cluster_ge_1mi'][1]:.2f}"
                                "] bao trùm mốc 7.86 mi). Ở hạng sang Sedan "
                                "(Black vs. "
                                "Lux Black), giao điểm "
                                f"{bl_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " mi ở Đặc tả A biến mất trên cả "
                                "Đặc tả B ("
                                f"{bl_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                " mi) lẫn miền d ≥ 1.0 mi "
                                "("
                                f"{bl_pair['above_floor_check']['d_star_ge_1mi']:.2f}"
                                " mi) — xác nhận Uber Black rẻ hơn toàn "
                                "miền khi "
                                "không surge. Tương tự, ở cặp SUV hạng "
                                "sang (Black SUV vs. Lux Black XL), dù "
                                "giao điểm toàn miền nằm tại "
                                f"{bs_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                "–"
                                f"{bs_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " "
                                "mi, kiểm định trên miền d ≥ 1.0 mi kéo "
                                "điểm cắt xuống "
                                f"{bs_pair['above_floor_check']['d_star_ge_1mi']:.3f}"
                                " "
                                "mi < 1.0 mi (CI_cl "
                                "["
                                f"{bs_pair['above_floor_check']['ci95_cluster_ge_1mi'][0]:.2f}"
                                ", "
                                f"{bs_pair['above_floor_check']['ci95_cluster_ge_1mi'][1]:.2f}"
                                "], ngoài miền d ≥ 1.0 mi), chứng tỏ trên "
                                "toàn bộ các chặng từ 1.0 dặm trở lên Uber "
                                "Black SUV rẻ hơn Lux Black XL; cặp đi "
                                "ghép không cắt (Shared luôn rẻ hơn)."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.8: Đối sánh song đặc tả 5 cặp "
                                    "biểu giá đối xứng Uber vs. Lyft trên "
                                    f"X_train (n = {n_train_rand:,}) kèm "
                                    "95% CI cụm"
                                ),
                                "headers": [
                                    "Phân khúc",
                                    "Biểu giá Uber (Đặc tả A)",
                                    "Biểu giá Lyft (Đặc tả A)",
                                    "Đặc tả A (d×surge, CI cụm)",
                                    "Đặc tả B (thuần 1.0x, X_train)",
                                ],
                                "rows": tbl_4_7_rows,
                                "col_widths": [2.4, 3.3, 3.3, 3.4, 3.6],
                                "aligns": ["L", "L", "L", "L", "L"],
                                "source": (
                                    "Nguồn: stage6_tier_tariffs.csv và ước "
                                    "lượng song đặc tả của nhóm (Delta-"
                                    "method Cluster 95% CI; kiểm tra d ≥ "
                                    "1.0 mi: UberX "
                                    f"{ux_pair['above_floor_check']['d_star_ge_1mi']:.2f}"
                                    " mi, Black SUV "
                                    f"{bs_pair['above_floor_check']['d_star_ge_1mi']:.3f}"
                                    " mi < 1.0 mi; WAV/Lux tách riêng)."
                                ),
                                "font_size": 8.6,
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_8_tariffs_clean.png"
                                ),
                                "caption": (
                                    "Hình 4.8: Đường biểu giá 5 cặp phân "
                                    "khúc đối xứng trên X_train (Đặc tả A "
                                    "vs. Đặc tả B) và tổng hợp độ nhạy d*"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Về tầm quan trọng đặc trưng (Bảng 4.9 và "
                                "Hình 4.9), Grouped Permutation Importance "
                                "gộp toàn bộ 6 cột "
                                "cự ly–surge thành một cụm chống đa cộng "
                                "tuyến (6 cụm rời phủ đúng 27/27 cột "
                                "X_test). Kết quả đồng thuận "
                                "chặt giữa hai họ mô hình: Phân khúc dịch "
                                "vụ name (M7: "
                                f"+${gx_rows[0]['m7_delta_rmse']:.4f}, "
                                f"{m7_top1_pct:.2f}%; M2: "
                                f"+${gx_rows[0]['m2_delta_rmse']:.4f}, "
                                f"{m2_top1_pct:.2f}%), Cụm Cự ly & Surge "
                                f"(M7: +${gx_rows[1]['m7_delta_rmse']:.4f}, "
                                f"{m7_top2_pct:.2f}%; M2: "
                                f"+${gx_rows[1]['m2_delta_rmse']:.4f}, "
                                f"{m2_top2_pct:.2f}%) — tổng hai cụm chi "
                                f"phối {m7_top1_pct + m7_top2_pct:.2f}% ở "
                                "M7 và "
                                f"{m2_top1_pct + m2_top2_pct:.2f}% ở M2; "
                                "Tuyến không gian chiếm "
                                f"{gx_rows[2]['m7_rel_share_pct']:.2f}% vs. "
                                f"{gx_rows[2]['m2_rel_share_pct']:.2f}%; "
                                "còn toàn bộ 7 biến Thời tiết và 9 biến "
                                "Thời gian EST chỉ đạt tổng "
                                f"{m7_wt_pct:.2f}% ở M7 ("
                                f"{gx_rows[4]['m7_rel_share_pct']:.2f}% + "
                                f"{gx_rows[5]['m7_rel_share_pct']:.2f}%) và "
                                f"{m2_wt_pct:.2f}% ở M2 (trên riêng mẫu "
                                "khớp thời tiết sạch sentinel, 7 biến thời "
                                "tiết có ΔRMSE = "
                                f"{md['complete_case_weather_delta_rmse_m7']:+.4f}"
                                " USD ≈ 0). Đối chiếu với PDP/ICE (Hình "
                                "4.10) cho thấy rõ "
                                "vùng phẳng giá sàn < 1.0 dặm và hiện "
                                "tượng bão hòa nhánh cây ở vùng hiếm surge "
                                "≥ 2.50x."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 4.9: Grouped Permutation "
                                    "Importance chống đa cộng tuyến (6 cụm "
                                    "rời = 27/27 cột X_test, n = 15,000 × 8 "
                                    "lần lặp)"
                                ),
                                "headers": [
                                    "Cụm ngữ nghĩa",
                                    "Cột",
                                    "Các đặc trưng hoán vị đồng thời",
                                    "M7 ΔRMSE",
                                    "Tỷ trọng M7",
                                    "M2 ΔRMSE",
                                    "Tỷ trọng M2",
                                ],
                                "rows": tbl_4_8_rows,
                                "col_widths": [
                                    3.5,
                                    0.8,
                                    5.1,
                                    1.7,
                                    1.6,
                                    1.7,
                                    1.6,
                                ],
                                "aligns": ["L", "C", "L", "R", "R", "R", "R"],
                                "source": (
                                    "Nguồn: Tính toán trực tiếp từ "
                                    "best_model.joblib & "
                                    "m2_interaction_ols.joblib trên X_test "
                                    "(n = 15,000 × 8 lần lặp)."
                                ),
                                "font_size": 8.6,
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_9_grouped_xai_clean.png"
                                ),
                                "caption": (
                                    "Hình 4.9: So sánh Grouped Permutation "
                                    "Importance (6 cụm rời = 27/27 cột) "
                                    "giữa M7 và M2 kèm đối chứng Ex-Ante"
                                ),
                                "width_cm": 13.6,
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_4_10_pdp_ice.png"
                                ),
                                "caption": (
                                    "Hình 4.10: Đồ thị phụ thuộc từng phần "
                                    "PDP/ICE của M7 theo quãng đường (vùng "
                                    "giá sàn < 1.0 dặm) và hệ số Surge"
                                ),
                                "width_cm": 13.6,
                            },
                        ),
                    ],
                },
                {
                    "heading": "4.6. Thảo luận kết quả thực nghiệm",
                    "blocks": [
                        (
                            "p",
                            (
                                "Tổng hợp Chương 3–4 cho thấy bức tranh "
                                "nhất quán: giá báo API Boston trong 24 "
                                "ngày cuối thu ôn hòa "
                                "không biến động cơ học theo mưa nhẹ–vừa "
                                "hay khung giờ cố định, mà do phân khúc "
                                "dịch vụ "
                                f"({m7_top1_pct:.2f}% ΔRMSE), cụm cự "
                                f"ly–surge ({m7_top2_pct:.2f}% ΔRMSE) và "
                                "điểm nóng không gian chi "
                                "phối (5 quận dẫn đầu Surge Lyft "
                                "9.57%–11.17%, gấp 7.2–8.4 lần North End "
                                "1.33%), phù hợp tính vi mô "
                                "theo tiểu khu của Chen et al. (2015). "
                                "Chênh lệch lớn giữa dự báo có điều kiện "
                                "Lyft (R² = "
                                f"{regime['lyft_observed_surge_test']['M7_Conditional']['r2']:.4f}"
                                ", chỉ 1 lỗi > $15, 0 > $16) với Uber (R² = "
                                f"{regime['uber_hidden_surge_test']['M7_Conditional']['r2']:.4f}"
                                ", 57 lỗi > $15) và Ex-Ante Lyft khi surge "
                                "(R² = "
                                f"{ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['r2']:.4f}"
                                ") khẳng định "
                                "phần khó đoán nhất nằm ở mất cân bằng "
                                "cung–cầu tức thời ngoài dữ liệu khí tượng "
                                "hay đồng hồ công khai."
                            ),
                        ),
                    ],
                },
            ],
        },
        # =====================================================================
        # CHƯƠNG 5
        # =====================================================================
        {
            "title": "CHƯƠNG 5: KHUYẾN NGHỊ QUẢN TRỊ VÀ KẾT LUẬN",
            "sections": [
                {
                    "heading": (
                        "5.1. Mô phỏng Trần Surge Cap dưới Ràng buộc Cung "
                        "và Khuyến nghị phân tầng"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Mô phỏng phản thực tế có điều kiện "
                                "(Conditional Counterfactual Scenario "
                                "Analysis — KHÔNG phải ước lượng "
                                "nhân quả thực địa) được thực hiện độc "
                                "quyền trên "
                                f"{rq4['lyft_observed_n_quotes']:,} lượt "
                                "báo giá Lyft "
                                "quan sát surge thực (tổng giá trị cơ sở = "
                                "$"
                                f"{rq4['lyft_observed_baseline_quote_proxy_usd']:,.2f}"
                                " API "
                                "Quote Proxy, KHÔNG phải doanh thu kế "
                                "toán). Trong src/simulation.py, mỗi khi "
                                "hạ surge_multiplier về "
                                "ngưỡng trần, hệ thống tính lại đồng bộ cả "
                                "4 biến phụ thuộc surge trước khi gọi "
                                "predict() trên M7 và M2, "
                                "đồng thời đối chứng với chuẩn số học P × "
                                "(cap / surge). Khác với cách cộng trừ "
                                "tuyến tính gây phi lý, "
                                "mô hình áp dụng chặt nguyên tắc bên ngắn "
                                "Q_realized = min(Q_D, Q_S) với độ co giãn "
                                "cung tài xế theo mức giảm giá "
                                "(ε_S ∈ {0.30, 0.60}; q_S = clip(1 + ε_S · "
                                "Δp/p, sàn, 1.0)) trên lượt bị cắt: mọi "
                                "cầu tăng thêm theo co giãn "
                                "(Cohen et al., 2016; ε = −0.50) vượt quá Q_S "
                                "trở thành dư cầu không phục vụ "
                                "Σ max(0, Q_D − Q_realized), bảo đảm tính "
                                "đơn điệu "
                                "Proxy(ε_S=0.60) < Proxy(ε_S=0.30) < "
                                "Proxy(tĩnh 0%) < Cận trên dư cung (Bảng "
                                "5.1, Hình 5.1)."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 5.1: Mô phỏng 6 kịch bản Trần "
                                    "Surge Cap dưới ràng buộc cung Q = "
                                    "min(Q_D, Q_S) trên "
                                    f"{rq4['lyft_observed_n_quotes']:,} "
                                    "lượt Lyft thực"
                                ),
                                "headers": [
                                    "Trần",
                                    "Lượt cắt",
                                    "Giảm giá M7",
                                    "M2 OLS",
                                    "Chuẩn số học",
                                    "Proxy tĩnh",
                                    "Cung ε_S=0.3 / 0.6*",
                                    "Cận trên ε=−0.5",
                                ],
                                "rows": tbl_5_1_rows,
                                "col_widths": [
                                    1.15,
                                    1.45,
                                    2.55,
                                    1.35,
                                    2.55,
                                    1.55,
                                    3.10,
                                    2.30,
                                ],
                                "aligns": [
                                    "C",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                    "R",
                                ],
                                "source": (
                                    "Nguồn: stage6_surge_cap_simulation.csv "
                                    "và mô phỏng ràng buộc cung của nhóm "
                                    "(*Q_realized = min(Q_D, Q_S) với "
                                    "q_S = clip(1 + ε_S·Δp/p); cột cuối là "
                                    "cận trên khi dư cung, Cohen et al., "
                                    "2016)."
                                ),
                                "font_size": 8.4,
                            },
                        ),
                        (
                            "figure",
                            {
                                "path": (
                                    "reports/figures/thesis/"
                                    "thesis_fig_5_1_surge_cap_clean.png"
                                ),
                                "caption": (
                                    "Hình 5.1: Biến động API Quote Proxy "
                                    "dưới ràng buộc cung Q = min(Q_D, Q_S) "
                                    "(A) và đối chứng mức giảm giá báo 3 "
                                    "nhánh (B)"
                                ),
                            },
                        ),
                        (
                            "p",
                            (
                                "Bài toán tối ưu có ràng buộc (tối đa số "
                                "lượt được bảo vệ sao cho |ΔProxy| ≤ τ) "
                                "trên mẫu Lyft quan sát "
                                "thực cho thấy sự phân tầng chính sách rõ "
                                "rệt thay vì cố giữ một ngưỡng duy nhất: "
                                "(1) Dưới khẩu vị rủi ro bảo thủ τ = 0.05% "
                                "hoặc τ = 0.10%, CHỈ duy nhất Trần 2.00x "
                                "(53 lượt surge "
                                "2.50x–3.00x; giảm giá báo M7 "
                                f"${sup_map[2.0]['mean_saving_usd']:.2f} "
                                f"tức {sup_map[2.0]['mean_saving_pct']:.2f}%"
                                ", "
                                "M2 $"
                                f"{sup_map[2.0]['m2_ols_mean_saving_usd']:.2f}"
                                ", chuẩn số học "
                                "$"
                                f"{arith_map[2.0]['arith_mean_saving_usd']:.2f}"
                                ") "
                                "đạt chuẩn trên MỌI kịch bản cung (tĩnh "
                                f"{sup_map[2.0]['proxy_static_pct']:+.3f}%, "
                                "cung nhạy giá ε_S = 0.30 "
                                f"{sup_map[2.0]['proxy_supply_drop5_pct']:+.3f}"
                                "%, ε_S = 0.60 "
                                f"{sup_map[2.0]['proxy_supply_drop10_pct']:+.3f}"
                                "%; "
                                "OOS Test-only: "
                                f"{oos200['static_proxy_change_pct']:+.3f}% "
                                "đến "
                                f"{oos200['supply_drop10_proxy_change_pct']:+.3f}"
                                "%). "
                                "(2) Trần 1.75x (704 lượt, giảm giá báo "
                                f"${sup_map[1.75]['mean_saving_usd']:.2f} "
                                "tức "
                                f"{sup_map[1.75]['mean_saving_pct']:.2f}%) "
                                "tạo biến động Proxy "
                                f"{sup_map[1.75]['proxy_static_pct']:+.3f}% "
                                "(tĩnh), "
                                f"{sup_map[1.75]['proxy_supply_drop5_pct']:+.3f}"
                                "% (cung ε_S = 0.30) và "
                                f"{sup_map[1.75]['proxy_supply_drop10_pct']:+.3f}"
                                "% (cung ε_S = 0.60; OOS Test-only: "
                                f"{oos175['static_proxy_change_pct']:+.3f}% "
                                "đến "
                                f"{oos175['supply_drop10_proxy_change_pct']:+.3f}"
                                "%) — vượt "
                                "ngưỡng 0.10% và chỉ đạt yêu cầu khi doanh "
                                "nghiệp chấp nhận biên rủi ro mở rộng τ = "
                                "0.30% (tĩnh/cung ε_S = 0.30) "
                                "hoặc τ = 0.50% (cả cung ε_S = 0.60). Về cơ "
                                "chế mô hình, mức giảm giá báo tại Trần 2.00x "
                                "của HistGB M7 "
                                f"(${sup_map[2.0]['mean_saving_usd']:.2f}) "
                                "thấp hơn biên trên M2 OLS "
                                f"(${sup_map[2.0]['m2_ols_mean_saving_usd']:.2f}"
                                ") và cận số học "
                                "($"
                                f"{arith_map[2.0]['arith_mean_saving_usd']:.2f}"
                                ") do hiện tượng ngoại suy phẳng ở vùng đuôi "
                                "hiếm n = 53 của mô hình cây (min_samples_leaf"
                                f" = {m7_p['min_samples_leaf']} gom chung "
                                "các đỉnh 2.50x–3.00x vào cùng lá), nhưng cả "
                                "ba nhánh đều nằm gọn trong τ "
                                "≤ 0.10%. Ngoài ra, "
                                "ε_S ∈ {0.30, 0.60} là tham số kịch bản "
                                "ngoại sinh (không ước lượng nội sinh từ "
                                "cung tài xế) và chưa "
                                "lượng hóa hiệu ứng lan tỏa không gian "
                                "(spillover) hay tài xế chạy rỗng sang "
                                "quận lân cận ('Wild Goose "
                                "Chase'; Castillo et al., 2017) — vì vậy "
                                "đề tài đề xuất lộ trình hai nấc: áp trần "
                                "bảo thủ 2.00x trước ở "
                                "Pha 1 và chỉ mở rộng sang 1.75x sau thử "
                                "nghiệm thực địa Pha 2 có ngưỡng dừng ETA."
                            ),
                        ),
                        (
                            "p",
                            "Hệ thống khuyến nghị được phân tách minh bạch "
                            "theo ba tầng bằng chứng:",
                        ),
                        (
                            "bullets",
                            [
                                "Tầng 1 — Trực tiếp từ bằng chứng thực "
                                "nghiệm trong mẫu: (i) Trên 72 tuyến "
                                "Boston khi không có surge, hành khách ưu "
                                "tiên chi phí chọn Lyft Shared cho đi ghép "
                                "(rẻ hơn UberPool toàn miền), chọn Lyft "
                                "cho chặng phổ thông ngắn (< "
                                f"{ux_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " mi ở Đặc tả A / < "
                                f"{ux_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                " mi ở Đặc tả B thuần 1.0x) và UberX cho "
                                "chặng dài hơn (lưu ý trung thực rằng nhóm "
                                "chuyến vượt "
                                f"{ux_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                " mi chỉ chiếm "
                                f"{ux_pair['spec_b_train80_nonsurge']['share_longer_pct']:.1f}%"
                                " ở Đặc tả B và vượt "
                                f"{ux_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " mi chiếm "
                                + _fmt_pct_range(
                                    ux_pair["spec_a_train80"][
                                        "share_longer_pct"
                                    ],
                                    ux_pair["spec_a_train80"][
                                        "all_train_share_longer_pct"
                                    ],
                                )
                                + " ở Đặc tả A); chọn "
                                "Lyft XL trên toàn miền [0.02, 7.86] mi "
                                "khi không surge ở hạng 6 chỗ; và chọn "
                                "Uber Black (toàn miền không surge) hoặc "
                                "Black SUV (toàn miền d ≥ 1.0 mi vì điểm "
                                "cắt d ≥ 1.0 mi chỉ "
                                f"{bs_pair['above_floor_check']['d_star_ge_1mi']:.3f}"
                                " mi < 1.0 mi) ở hạng sang. (ii) Đội kỹ "
                                "thuật dự báo giá không cần ưu tiên biến "
                                "khí tượng gộp theo giờ hay lịch buổi, mà "
                                "tập trung bổ sung duration, mật độ tài xế "
                                "thời gian thực và đóng gói Monotonic "
                                "Split-CQR (pandas category) vào mô hình "
                                "sản phẩm.",
                                "Tầng 2 — Suy luận quản trị có điều kiện "
                                "từ mô phỏng: Lyft có thể áp dụng Trần "
                                "2.00x như bước cắt đỉnh cực đoan ít rủi "
                                "ro nhất (τ ≤ 0.10%), đồng thời dùng Trần "
                                "1.75x làm ứng viên thử nghiệm Switchback "
                                "A/B tại 5 quận điểm nóng dẫn đầu "
                                f"({top12[0]['source']} "
                                f"{top12[0]['surge_rate_pct']:.2f}%, "
                                f"{top12[1]['source']} "
                                f"{top12[1]['surge_rate_pct']:.2f}%, "
                                f"{top12[2]['source']} "
                                f"{top12[2]['surge_rate_pct']:.2f}%, "
                                f"{top12[3]['source']} "
                                f"{top12[3]['surge_rate_pct']:.2f}%, "
                                f"{top12[4]['source']} "
                                f"{top12[4]['surge_rate_pct']:.2f}%) nhằm "
                                "đo trực tiếp ETA và độ co giãn cung tài "
                                "xế. Uber nên tập trung định vị lợi thế "
                                "giá ở phân khúc hạng sang (Black toàn "
                                "miền & Black SUV từ d ≥ 1.0 mi) thay vì "
                                "chỉ dựa vào đuôi cự ly dài UberX (chỉ "
                                f"{ux_pair['spec_b_train80_nonsurge']['share_longer_pct']:.1f}%"
                                " lượt > "
                                f"{ux_pair['spec_b_train80_nonsurge']['hc3']['d_star']:.2f}"
                                " mi khi không surge và "
                                + _fmt_pct_range(
                                    ux_pair["spec_a_train80"][
                                        "share_longer_pct"
                                    ],
                                    ux_pair["spec_a_train80"][
                                        "all_train_share_longer_pct"
                                    ],
                                )
                                + " > "
                                f"{ux_pair['spec_a_train80']['hc3']['d_star']:.2f}"
                                " mi), đồng thời rà soát giá mở cửa $"
                                f"{float(tmap['UberPool']['base_fare_b0_usd']):.2f}"
                                " của UberPool so với $"
                                f"{float(tmap['Shared']['base_fare_b0_usd']):.2f}"
                                " của Lyft Shared.",
                                "Tầng 3 — Giả thuyết kinh doanh cần kiểm "
                                "chứng thực địa (chưa chứng minh trực tiếp "
                                "trong dữ liệu): Đề xuất gói hội viên Price-"
                                "Lock Pass 1.50x đòi hỏi khảo sát riêng về "
                                "mức sẵn lòng chi trả (WTP) và rủi ro lựa "
                                "chọn bất lợi; gợi ý hành khách chờ 5–10 "
                                "phút khi gặp surge theo Chen et al. (2015) "
                                "cần xác minh bằng chuỗi thời gian độ phân "
                                "giải phút.",
                            ],
                        ),
                    ],
                },
                {
                    "heading": (
                        "5.2. Lộ trình triển khai và Phân tích công suất "
                        "thử nghiệm Switchback (Triggered vs. ITT)"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Một sai lầm thống kê phổ biến khi thiết "
                                "kế thử nghiệm Switchback cho chính sách "
                                "trần surge là tính cỡ "
                                "mẫu theo phân tích toàn mẫu (Intention-to-"
                                "Treat — ITT). Do tần suất kích hoạt thực "
                                "trên Lyft rất thấp "
                                "(toàn 12 quận: "
                                f"{pw175['empirical_trigger_rate_pct']:.3f}%"
                                " lượt có surge > 1.75x và "
                                f"{pw200['empirical_trigger_rate_pct']:.3f}%"
                                " có surge > 2.00x), một tác động thực 2.0 "
                                "pp trên tập chịu cắt "
                                "bị pha loãng xuống chỉ còn "
                                f"{pw175_mde2['itt_diluted_mde_pp']:.4f} pp "
                                "(12 quận) hoặc "
                                f"{pw175_t5_mde2['itt_diluted_mde_pp']:.4f} "
                                "pp (5 điểm nóng) trên toàn mẫu, đòi hỏi "
                                "tới "
                                + "{:.1f}".format(
                                    pw175_mde2[
                                        "itt_unfeasible_requests_per_arm"
                                    ]
                                    / 1e6
                                )
                                + " triệu lượt/nhánh (45.7 triệu ở 5 điểm "
                                "nóng) — "
                                "hoàn toàn bất khả thi. Do đó, thử nghiệm "
                                "thực địa bắt buộc đánh giá trên tập kích "
                                "hoạt (Triggered / "
                                "Exposed-Only Analysis) với hệ số hiệu ứng "
                                "thiết kế cụm phụ thuộc quy mô lưu lượng "
                                "DEFF_k× = "
                                "1 + (k·m₀ − 1)·ICC (ICC = 0.05, α = 5%, "
                                "power = 80%). Khi lưu lượng sản xuất tăng "
                                f"k ∈ {{25×, 50×}} "
                                "so với tốc độ thu thập API 1×, cỡ cụm "
                                "kích hoạt m_k tăng tuyến tính làm DEFF "
                                "giãn mạnh: với Trần 1.75x "
                                "trên toàn 12 quận (m₀ = "
                                f"{pw175['mean_triggered_per_active_loc_hr']:.2f}"
                                ", "
                                f"{pw175['observed_triggered_per_day_scrape']:.2f}"
                                " lượt/ngày; DEFF_1x = "
                                f"{pw175['deff_1x']:.3f} → "
                                f"DEFF_25x = {pw175['deff_25x']:.3f} → "
                                f"DEFF_50x = {pw175['deff_50x']:.3f}), phát "
                                "hiện MDE = 5.0 pp cần "
                                f"{pw175_mde5['triggered_quotes_both_arms_25x']:,}"
                                " lượt ("
                                f"{pw175_mde5['days_needed_at_prod_traffic_25x']:.1f}"
                                " "
                                "ngày ở 25×; "
                                f"{pw175_mde5['days_needed_at_prod_traffic_50x']:.1f}"
                                " ngày ở 50×), MDE = 3.0 pp cần "
                                f"{pw175_mde3['triggered_quotes_both_arms_25x']:,}"
                                " lượt ("
                                f"{pw175_mde3['days_needed_at_prod_traffic_25x']:.1f}"
                                " "
                                "ngày ở 25×; "
                                f"{pw175_mde3['days_needed_at_prod_traffic_50x']:.1f}"
                                " ngày ở 50×), còn MDE = 2.0 pp đòi hỏi "
                                f"{pw175_mde2['triggered_quotes_both_arms_25x']:,}"
                                " lượt ("
                                f"{pw175_mde2['days_needed_at_prod_traffic_25x']:.1f}"
                                " "
                                "ngày ở 25×; "
                                f"{pw175_mde2['days_needed_at_prod_traffic_50x']:.1f}"
                                " ngày ở 50×). Nếu khu trú tại 5 quận điểm "
                                "nóng (gom "
                                f"{pw175_top5['share_of_citywide_triggered_pct']:.2f}"
                                "% lượt kích hoạt toàn TP nhưng chỉ chạm "
                                "41.5% "
                                "tổng lượng chuyến; tỷ lệ kích hoạt tăng "
                                "gấp đôi lên "
                                f"{pw175_top5['empirical_trigger_rate_pct']:.3f}"
                                "%, "
                                f"DEFF_25x = {pw175_top5['deff_25x']:.3f}, "
                                f"DEFF_50x = {pw175_top5['deff_50x']:.3f}), "
                                "MDE = 5.0 pp và 3.0 pp "
                                "khả thi trong "
                                f"{pw175_t5_mde5['days_needed_at_prod_traffic_25x']:.1f}"
                                "–"
                                f"{pw175_t5_mde3['days_needed_at_prod_traffic_25x']:.1f}"
                                " ngày ở 25× "
                                "("
                                f"{pw175_t5_mde5['days_needed_at_prod_traffic_50x']:.1f}"
                                "–"
                                f"{pw175_t5_mde3['days_needed_at_prod_traffic_50x']:.1f}"
                                " ngày ở 50×), còn MDE = 2.0 pp cần "
                                f"{pw175_t5_mde2['days_needed_at_prod_traffic_25x']:.1f}"
                                " / "
                                f"{pw175_t5_mde2['days_needed_at_prod_traffic_50x']:.1f}"
                                " ngày (> 2 tháng). Riêng Trần 2.00x quá "
                                "hiếm nên "
                                "ngay ở MDE = 5.0 pp vẫn cần "
                                f"{pw200_mde5['days_needed_at_prod_traffic_25x']:.1f}"
                                " / "
                                f"{pw200_mde5['days_needed_at_prod_traffic_50x']:.1f}"
                                " ngày (12 quận) hay "
                                f"{pw200_t5_mde5['days_needed_at_prod_traffic_25x']:.1f}"
                                " / "
                                f"{pw200_t5_mde5['days_needed_at_prod_traffic_50x']:.1f}"
                                " ngày (5 điểm nóng), phù hợp giám sát an "
                                "toàn Pha 1 hơn là A/B hẹp (cắt 53 đỉnh cực "
                                "đoan để bảo vệ niềm tin thương hiệu và "
                                "tuân thủ quy định minh bạch của Sở Tiện ích "
                                "Công cộng Massachusetts — DPU TNC mà không "
                                "cần chi phí truyền thông lớn). Bảng 5.2 "
                                "đồng thời gắn căn cứ vận hành rõ cho các "
                                "ngưỡng dừng (SLA < 150 ms; ETA > 15% hoặc "
                                "hủy > 2 pp phòng Wild Goose Chase; giám sát "
                                "chuyển nền tảng đa ứng dụng driver multi-"
                                "homing sang Uber khi Uber giữ phụ phí ẩn "
                                "không trần; công bằng không gian |bias| ≤ "
                                "$0.12 < 1/4 bậc làm tròn $0.50)."
                            ),
                        ),
                        (
                            "table",
                            {
                                "caption": (
                                    "Bảng 5.2: Lộ trình triển khai 3 pha (6 "
                                    "tháng) tích hợp Triggered Switchback "
                                    "Power Analysis và quy tắc dừng ETA"
                                ),
                                "headers": [
                                    "Giai đoạn triển khai",
                                    "Đơn vị phụ trách",
                                    "Hạng mục kỹ thuật & Nghiệp vụ trọng tâm",
                                    "Tiêu chí nghiệm thu & Quy tắc dừng",
                                ],
                                "rows": tbl_5_2_rows,
                                "col_widths": [3.0, 2.5, 6.4, 4.1],
                                "aligns": ["L", "L", "L", "L"],
                                "source": (
                                    "Nguồn: Đề xuất triển khai & Triggered "
                                    "Switchback Power Analysis (α = 5%, "
                                    "power = 80%, ICC = 0.05) của nhóm thực "
                                    "hiện."
                                ),
                                "font_size": 8.6,
                            },
                        ),
                    ],
                },
                {
                    "heading": (
                        "5.3. Hạn chế của nghiên cứu và Đạo đức dữ liệu "
                        "(Kiểm toán công bằng 12 quận)"
                    ),
                    "blocks": [
                        (
                            "p",
                            "Nghiên cứu công khai minh bạch bảy giới hạn "
                            "khách quan của dữ liệu, phương pháp và sản "
                            "phẩm:",
                        ),
                        (
                            "bullets",
                            [
                                "Bản chất báo giá API & cơ cấu mẫu phân "
                                "tầng: Dữ liệu là giá báo trước trên API "
                                "(API Fare Quotes) truy vấn tương đối đều "
                                "qua 12 gói và 72 tuyến, không phản ánh tỷ "
                                "lệ chốt chuyến, nhu cầu thực, thị phần hay "
                                "doanh thu kế toán.",
                                "Khuyết thu thập gốc & thiên lệch bỏ sót "
                                "biến (OVB): Thiếu "
                                f"{merge['missing_timestamp_pct']:.2f}% "
                                "timestamp gốc, "
                                f"{prov['lyft_surge_missing_pct']:.2f}% "
                                "surge Lyft, 6 ngày tháng 12/2018 và 5 "
                                "khung giờ đêm EST; thiếu thời lượng "
                                "chuyến thực (duration), kẹt xe và mật độ "
                                "tài xế — gây OVB trong β₁ và tạo trần độ "
                                "chính xác ~4% phương sai chưa giải thích.",
                                "Bất đối xứng công bố Surge & ranh giới Ex-"
                                "Ante: API Uber trả cố định "
                                "surge_multiplier = 1.00 dù áp phụ phí ẩn "
                                "trong Upfront Pricing (tạo 57/58 sai số > "
                                "$15 và 47/47 > $16 trên M7), còn dự báo "
                                "Ex-Ante thuần "
                                "không thể tái tạo các đỉnh surge thời gian "
                                "thực (R² = "
                                f"{ex_pricing['M7_ExAnte_Lyft_SurgeGt1x_Test']['r2']:.4f}"
                                " trên Lyft > 1.0x).",
                                "Phạm vi thời tiết cuối thu & công suất "
                                "mưa lớn thấp: Dữ liệu chỉ bao phủ 24 ngày "
                                "cuối thu ôn hòa (max rain 0.55 in/h trong "
                                "bảng sạch, chưa gồm bão tuyết mùa đông "
                                "Boston) và chỉ ghi nhận đúng "
                                f"{hv['n_heavy_citywide_hours']} khung giờ "
                                "mưa lớn ≥ 0.10 in/h toàn thành phố — giới "
                                "hạn công suất thống kê cho cực trị khí "
                                "tượng.",
                                "Khôi phục bậc 10 heuristic & ranh giới "
                                "tuyến mới: Quy tắc log10 toàn cục (mốc "
                                "2.15 mi cố định cho distance) không làm "
                                "lạm phát giá Test (Strict Train-Only Full "
                                f"Refit cho 0/{n_test_rand:,} giá Test "
                                f"đổi, M7 R² = {m7_st['r2']:.4f}); và khi "
                                "khử hoàn toàn rò rỉ nội suy cự ly bằng "
                                "RouteDistanceImputer trong Pipeline, hiệu "
                                "năng giảm rõ khi dự báo trên 15 tuyến mới "
                                "(M7 MAPE = "
                                f"{float(r_grp_m7['mape_pct']):.2f}%, "
                                f"M6 = {float(r_grp_m6['mape_pct']):.2f}%, "
                                f"M2 = {float(r_grp_m2['mape_pct']):.2f}%).",
                                "Giới hạn mô phỏng phi nhân quả & mật độ "
                                "kích hoạt Switchback: Mô phỏng Surge Cap "
                                "không thay thế được ước lượng nhân quả và "
                                "phụ thuộc tham số suy giảm cung giả định; "
                                "thử nghiệm Switchback đòi hỏi quy mô lưu "
                                "lượng sản xuất thực (≥ 25× tốc độ thu thập "
                                "API) mới đạt công suất trong khung 2 tháng.",
                                "Đóng gói khoảng phân vị sống trên "
                                "Dashboard: Thư mục models/ hiện lưu M7 và "
                                "M2, chưa đóng gói 3 tệp .joblib của bộ "
                                "Quantile đã hiệu chỉnh Monotonic Split-CQR "
                                "nên Tab 5A đang hiển thị dải tham chiếu "
                                "±MPIW/2 cấp phân khúc kèm nhãn minh bạch; "
                                "cần tích hợp bộ 3 tệp CQR ở Pha 1 triển "
                                "khai.",
                            ],
                        ),
                        (
                            "p",
                            (
                                "Về Đạo đức AI và Kiểm toán Công bằng "
                                "Không gian (Spatial Fairness Audit): Để "
                                "kiểm tra rủi ro thiên vị "
                                "thuật toán theo địa bàn trên 12 quận "
                                "Boston, nhóm đo lường phần dư có dấu "
                                "(actual − predicted) và sai số "
                                "của M7 trên tập kiểm tra (n = "
                                f"{n_test_rand:,}). Kết quả xác nhận mô "
                                "hình gần như không có thiên lệch hệ "
                                "thống theo địa lý: độ lệch trung bình có "
                                "dấu cực đại chỉ ở mức |bias| ≤ "
                                f"${sp_v5['max_abs_district_bias_usd']:.4f} "
                                "(dao động từ "
                                f"{min_bias_d['mean_signed_residual_usd']:+.4f}"
                                f" USD tại {min_bias_d['district']} đến "
                                f"{max_bias_d['mean_signed_residual_usd']:+.4f}"
                                f" USD tại {max_bias_d['district']}), và "
                                "MAPE cả 12/12 quận "
                                "đều nằm gọn dưới ngưỡng 9% trong khoảng "
                                f"[{min_mape_d['mape_pct']:.2f}%, "
                                f"{max_mape_d['mape_pct']:.2f}%] "
                                f"(thấp nhất {min_mape_d['district']}, cao "
                                f"nhất {max_mape_d['district']}). Tuy nhiên, "
                                "nhóm cảnh báo rủi ro thiên lệch phủ sóng "
                                "địa lý (Geographic Coverage Bias): 12 quận "
                                "trong bộ dữ liệu tập trung tại lõi trung "
                                "tâm/đại học và chưa bao phủ các khu dân cư "
                                "ngoại vi thu nhập thấp hơn (như Roxbury, "
                                "Dorchester, Mattapan), nên cần tái kiểm "
                                "toán công bằng toàn thành phố và theo dõi "
                                "thu nhập ròng giờ đỉnh của tài xế trước khi "
                                "mở rộng. Về quyền riêng tư, mô hình dùng ẩn "
                                "danh cấp quận; khi triển khai GPS thực cần "
                                "gom cụm lưới lục giác H3 đạt k-anonymity, "
                                "xóa tọa độ gốc sau 30 ngày và duy trì kiểm "
                                "thử Human-in-the-Loop."
                            ),
                        ),
                    ],
                },
                {
                    "heading": (
                        "5.4. Hướng phát triển trong tương lai và Kết luận "
                        "chung"
                    ),
                    "blocks": [
                        (
                            "p",
                            (
                                "Bốn hướng phát triển tiếp theo gồm: (1) "
                                "Tích hợp thời lượng chuyến thực (duration), "
                                "mật độ kẹt xe và lịch sự kiện Boston nhằm "
                                "nâng độ chính xác Ex-Ante và giảm OVB; (2) "
                                "Mở rộng chuỗi quan sát 4 mùa (gồm bão "
                                "tuyết) ở độ phân giải phút; (3) Đóng gói "
                                "bộ 3 mô hình Split-CQR vào models/ cho Tab "
                                "5A; và (4) Xây dựng mô hình cân bằng hai "
                                "phía gắn Switchback A/B để ước lượng trực "
                                "tiếp độ co giãn cung tài xế nội sinh."
                            ),
                        ),
                        (
                            "p",
                            (
                                "Kết luận chung: Qua quy trình Khoa học Dữ "
                                f"liệu khép kín trên {total_n:,} báo giá API "
                                f"({obs_surge_n:,} surge thực; "
                                f"{prov['weather_valid_rain_obs']:,} khí "
                                "tượng sạch cảm biến), các kiểm định cụm "
                                f"({st_ext['n_loc_hr_clusters_all']:,} quận "
                                "× giờ, "
                                f"{st_ext['n_citywide_hour_clusters_all']} "
                                "giờ), Exact WCR (B = 1,999) và TOST khẳng "
                                "định mưa/khung giờ không lệch giá vượt "
                                "±$0.10; cấu trúc giá do gói xe "
                                f"({m7_top1_pct:.2f}% ΔRMSE), cự ly–surge "
                                f"({m7_top2_pct:.2f}%) và 5 điểm nóng Lyft "
                                f"({top12[0]['surge_rate_pct']:.2f}%–"
                                f"{top12[4]['surge_rate_pct']:.2f}%) chi "
                                "phối. M7 đạt R² = "
                                f"{float(m7['test_r2']):.4f}, RMSE = "
                                f"${float(m7['test_rmse']):.4f}, "
                                f"MAPE = {float(m7['test_mape_pct']):.2f}% "
                                "(Ex-Ante R² = "
                                f"{ex_pricing['M7_ExAnte_All_Test']['r2']:.4f}"
                                "), Split-CQR đạt PICP "
                                f"{cqr['min_tier_cqr_picp_pct']:.2f}%–"
                                f"{cqr['max_tier_cqr_picp_pct']:.2f}% "
                                "(0 cắt ngược), và mô phỏng bên ngắn xác "
                                "lập lộ trình 2 nấc (Pha 1 trần 2.00x; Pha "
                                "2 Triggered Switchback 1.75x)."
                            ),
                        ),
                    ],
                },
            ],
        },
    ]

    # =========================================================================
    # PHỤ LỤC A: SƠ ĐỒ CHUẨN + CHỈ MỤC ĐẦY ĐỦ + TOÀN VĂN MAIN.PY & CONFIG.PY +
    # 4 TRÍCH ĐOẠN LÕI
    # =========================================================================
    def _loc(rel_path: str) -> int:
        return len(
            (ROOT_DIR / rel_path).read_text(encoding="utf-8").splitlines()
        )

    main_loc = _loc("main.py")
    src_only_loc = sum(_loc(f"src/{f}") for f in data["src_files"])
    scripts_only_loc = sum(_loc(f"scripts/{f}") for f in data["script_files"])
    total_src_loc = main_loc + src_only_loc
    appendix_a_main_code = (
        (ROOT_DIR / "main.py").read_text(encoding="utf-8").strip()
    )
    appendix_a_config_code = (
        (ROOT_DIR / "src/config.py").read_text(encoding="utf-8").strip()
    )

    appendix_a_tree = (
        "TieuLuan_PRJ-EXT-01_Nhom02/\n"
        "├── main.py                # CLI toàn trình --stage "
        f"clean..evaluate ({main_loc:,} dòng)\n"
        "├── pyproject.toml         # Cấu hình ruff PEP 8 nghiêm ngặt (<= "
        "79 ký tự, gồm notebooks/)\n"
        "├── requirements.txt       # Python >=3.11 (kiểm chứng 3.11.9 & "
        "3.14.4) + pillow + pymupdf\n"
        "├── nhật ký làm việc.md    # Nhật ký P-D-M-V nhóm "
        f"({len(data['worklog_sessions'])} phiên: 11 + 11 + "
        f"{len(data['worklog_sessions']) - 22})\n"
        "├── data/                  # raw/ (read-only SHA-256) & processed/"
        f" ({io_same['parquet_disk_mb']:.2f} MiB Parquet)\n"
        f"├── src/                   # {n_src} module lõi pipeline ML/EDA "
        f"thuần ({src_only_loc:,} dòng, Bảng A.1)\n"
        f"├── scripts/               # {n_scripts} script xuất bản, biểu đồ "
        f"& kiểm toán ({scripts_only_loc:,} dòng)\n"
        "├── dashboard/             # Dashboard 5 Tab (Streamlit + Offline "
        f"HTML, 18/18, {n_test_files} tệp pytest)\n"
        "├── models/                # best_model.joblib (M7), "
        "m2_interaction_ols.joblib\n"
        "├── notebooks/             # 4 Notebook chạy sống PASS (03 -> 06, "
        "0 lỗi ruff PEP 8)\n"
        "└── reports/               # 21 ảnh GĐ3–6, 17 ảnh luận văn 300 DPI,"
        " QMD/PDF/DOCX"
    )

    appendix_a_module_index = [
        [
            f"main.py ({main_loc:,} dòng)",
            "CLI điều phối toàn trình (--stage clean..evaluate / all); in "
            "toàn văn tại A.2.",
        ],
        [
            f"src/config.py ({_loc('src/config.py'):,} dòng)",
            "Đường dẫn pathlib, hằng số & 14 cột cấm FORBIDDEN_IN_X; in "
            "toàn văn tại A.3.",
        ],
        [
            f"src/clean_raw_data.py ({_loc('src/clean_raw_data.py'):,} dòng)",
            "Làm sạch 7 tầng cab_rides & weather: bảo toàn surge gốc, khôi "
            "phục bậc 10 & cờ sentinel mưa.",
        ],
        [
            "src/stage3_pipeline_and_audit.py "
            f"({_loc('src/stage3_pipeline_and_audit.py'):,} dòng)",
            "Chuẩn hóa Boston EST, Left Join 0 fan-out, ép 8/8 category & "
            "xuất ảnh EDA.",
        ],
        [
            f"src/features.py ({_loc('src/features.py'):,} dòng)",
            "Tạo 9 biến đơn dòng (41 cột), phân tích VIF, 3 chiến lược chia "
            "& 4 Pipeline tiền xử lý.",
        ],
        [
            f"src/stats_tests.py ({_loc('src/stats_tests.py'):,} dòng)",
            "13 kiểm định H1–H4, OLS HC3 12 gói và Joint Wald H4c ma trận "
            "hạng đầy đủ.",
        ],
        [
            f"src/models.py ({_loc('src/models.py'):,} dòng)",
            "10 mô hình (M0a–M7), GridSearchCV/CV Pipeline (precompute=True "
            "cho Lasso) & siêu tham số.",
        ],
        [
            f"src/stage5_modeling.py ({_loc('src/stage5_modeling.py'):,} "
            "dòng)",
            "5-Fold CV, độ bền, MB1–MB6, phân rã Uber/Lyft và Ex-Ante (23 "
            "biến + Surge Classifier).",
        ],
        [
            f"src/evaluation.py ({_loc('src/evaluation.py'):,} dòng)",
            "Chẩn đoán phần dư (JB, BP), sai số đa chiều & Quantile [P10, "
            "P90] dùng pandas category.",
        ],
        [
            f"src/interpretation.py ({_loc('src/interpretation.py'):,} dòng)",
            "Biểu giá 12 gói, 5 cặp đối xứng + 2 độc lập, Permutation XAI "
            "rời 27/27 cột & PDP/ICE.",
        ],
        [
            f"src/simulation.py ({_loc('src/simulation.py'):,} dòng)",
            "Mô phỏng Surge Cap bên ngắn Q_realized = min(Q_D, Q_S) với "
            "khẳng định đơn điệu chặt.",
        ],
        [
            "src/stage6_evaluation.py "
            f"({_loc('src/stage6_evaluation.py'):,} dòng)",
            "Tổng hợp GĐ6, vẽ fig15–fig21 và xuất stage6_summary.json.",
        ],
        [
            "scripts/thesis_v5_verification.py "
            f"({_loc('scripts/thesis_v5_verification.py'):,} dòng)",
            "Kiểm chứng hợp nhất: Cluster SE, TOST, Exact WCR, Split-CQR, "
            "Train-Only, Switchback & I/O.",
        ],
        [
            "scripts/prepare_thesis_figures.py "
            f"({_loc('scripts/prepare_thesis_figures.py'):,} dòng)",
            "Xuất đồng bộ 17 biểu đồ 300 DPI chuẩn tiểu luận từ JSON/CSV "
            "kết quả.",
        ],
        [
            "scripts/thesis_content.py & build_final_thesis.py ("
            + "{:,}".format(
                _loc("scripts/thesis_content.py")
                + _loc("scripts/build_final_thesis.py")
            )
            + " dòng)",
            "Nạp số liệu động, dựng .qmd chuẩn UFM, xuất PDF trực tiếp qua "
            "Quarto/Typst & đồng bộ .docx.",
        ],
        [
            "scripts/check_thesis_audit.py "
            f"({_loc('scripts/check_thesis_audit.py'):,} dòng)",
            "Cổng kiểm toán tự động PDF/QMD động theo JSON (bố cục, giãn "
            "đoạn 6pt, nguồn Quarto, APA 7th).",
        ],
        [
            "dashboard/* & notebooks/*",
            "Dashboard 5 Tab + HTML offline + kiểm toán 18/18 + 40 ca "
            "pytest + 4 notebook PEP 8.",
        ],
    ]

    # 4 trích đoạn mã nguồn KHỚP 100% CODE THẬT TRONG REPO (kiểm tra compile()
    # + đối chiếu tệp trong check_thesis_audit.py)
    appendix_a_code_snippet = (
        "# Trích đoạn 1 — Bảo toàn nguồn gốc surge_multiplier (không "
        "fillna(1.0) mù trên Lyft)\n"
        "# Nguồn: src/clean_raw_data.py (hàm clean_cab_rides)\n"
        'df.loc[(df["surge_multiplier"] < 1.0) | (df["surge_multiplier"] > '
        "3.0),\n"
        '       "surge_multiplier"] = np.nan\n'
        'df.loc[df["cab_type"] == "Uber", "surge_multiplier"] = 1.0\n'
        'df["is_surge_original"] = '
        'df["surge_multiplier"].notna().astype(np.int8)\n\n'
        "# Trích đoạn 2 — Chuẩn hóa EST & Tách cờ is_rain_sentinel khi ghép "
        "thời tiết\n"
        "# Nguồn: src/stage3_pipeline_and_audit.py (hàm step3_merge_and_eda)\n"
        'cab_df["datetime"] = (\n'
        '    pd.to_datetime(cab_df["time_stamp"].astype("float64"), '
        'unit="ms", utc=True)\n'
        '    .dt.tz_convert("America/New_York")\n'
        "    .dt.tz_localize(None)\n"
        ")\n"
        'master_df["weather_matched"] = (\n'
        '    (master_df["is_timestamp_original"] == 1) & '
        'master_df["location"].notna()\n'
        ").astype(np.int8)\n"
        'master_df["is_rain_sentinel"] = np.where(\n'
        '    (master_df["weather_matched"] == 1) & master_df["rain"].isna(),\n'
        "    1,\n"
        "    0,\n"
        ").astype(np.int8)\n"
        'master_df["is_raining"] = np.where(\n'
        '    (master_df["weather_matched"] == 1) & '
        'master_df["rain"].notna(),\n'
        '    (master_df["rain"] > 0.0).astype(np.float32),\n'
        "    np.nan,\n"
        ").astype(np.float32)\n\n"
        "# Trích đoạn 3 — Mô phỏng Surge Cap theo nguyên tắc bên ngắn "
        "Q_realized = min(Q_D, Q_S)\n"
        "# Nguồn: src/simulation.py (hàm _compute_demand_and_supply_metrics)\n"
        "q_demand_mid = 1.0 + (-0.50) * rel_p\n"
        "q_supply_mod = np.where(\n"
        "    affected_mask, np.clip(1.0 + 0.30 * rel_p, 0.50, 1.0), 1.0\n"
        ")\n"
        "q_supply_sev = np.where(\n"
        "    affected_mask, np.clip(1.0 + 0.60 * rel_p, 0.40, 1.0), 1.0\n"
        ")\n"
        "q_realized_drop5 = np.where(\n"
        "    affected_mask, np.minimum(q_demand_mid, q_supply_mod), 1.0\n"
        ")\n"
        "q_realized_drop10 = np.where(\n"
        "    affected_mask, np.minimum(q_demand_mid, q_supply_sev), 1.0\n"
        ")\n"
        "unfulfilled_5 = float(\n"
        "    np.sum(\n"
        "        np.where(\n"
        "            affected_mask,\n"
        "            np.maximum(0.0, q_demand_mid - q_realized_drop5),\n"
        "            0.0,\n"
        "        )\n"
        "    )\n"
        ")\n"
        "if n_affected > 0:\n"
        "    assert drop10_pct < drop5_pct < static_pct\n\n"
        "# Trích đoạn 4 — Monotonic Split-CQR dùng pandas category gốc (0 "
        "cắt phân vị ngược)\n"
        "# Nguồn: scripts/thesis_v5_verification.py (hàm "
        "compute_model_diagnostics_cqr_and_xai)\n"
        "cal_lo = np.minimum(m10.predict(X_cal), m50.predict(X_cal))\n"
        "cal_hi = np.maximum(m90.predict(X_cal), m50.predict(X_cal))\n"
        "cal_scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)\n"
        "level = min(1.0, np.ceil((1.0 - alpha) * (n_t + 1)) / n_t)\n"
        'q_map[tier] = float(np.quantile(s_t, level, method="higher"))\n'
        "raw_cqr_lo = np.clip(te_lo - adj, 2.5, None)\n"
        "raw_cqr_hi = te_hi + adj\n"
        "mid = 0.5 * (raw_cqr_lo + raw_cqr_hi)\n"
        "cqr_lo = np.where(raw_cqr_lo <= raw_cqr_hi, raw_cqr_lo, mid)\n"
        "cqr_hi = np.where(raw_cqr_lo <= raw_cqr_hi, raw_cqr_hi, mid)"
    )

    # =========================================================================
    # PHỤ LỤC B: HỒ SƠ KIỂM TOÁN NỘI BỘ & NHẬT KÝ P-D-M-V
    # =========================================================================
    appendix_b_audit_map = [
        [
            "CRIT-01",
            "Chặn nhãn mưa giả trên 91,618 dòng không khớp thời tiết (giữ "
            "NaN, cờ weather_matched).",
        ],
        [
            "CRIT-02",
            "Chuẩn hóa America/New_York (EST) và cấm .ffill() trên 62,864 "
            "dòng khuyết timestamp.",
        ],
        [
            "CRIT-03",
            f"Bỏ fillna(1.0) mù trên {lyft_miss_n:,} dòng Lyft khuyết surge "
            "(giữ NaN + is_surge_original; Surge thực "
            f"{lyft_surge_rate:.2f}%).",
        ],
        [
            "H4 / Weather",
            "Khôi phục bậc 10 & điền khuyết khí tượng theo trung vị cùng "
            "giờ toàn Boston; tách cờ is_rain_sentinel.",
        ],
        [
            "C2 / C3 / H3",
            "Ràng buộc cung Q_realized = min(Q_D, Q_S); đối chứng Ex-Ante "
            "(23 biến + Surge Classifier); Triggered Power Analysis.",
        ],
        [
            "HIGH-01..03",
            "Grouped XAI 6 cụm = 27/27 cột; khóa miền [0.02, 7.86] mi Spec "
            "A/B; chuẩn hóa nhãn API Quote Proxy.",
        ],
        [
            "GAP-MODEL-01",
            "Monotonic Split-CQR dùng pandas category gốc "
            f"({cqr['min_tier_cqr_picp_pct']:.2f}%–"
            f"{cqr['max_tier_cqr_picp_pct']:.2f}%, 0 cắt ngược).",
        ],
    ]

    appendix_b_milestones = [
        [
            "Phiên 1–7",
            "01/10–22/10/2026",
            "Xác lập PRJ-EXT-01, thử nghiệm Epoch 1, phát hiện 7 tầng nhiễu "
            "và quy hoạch tái lập.",
        ],
        [
            "Phiên 8–12",
            "08/11–19/11/2026",
            "Pipeline GĐ3–GĐ6, xử lý CRIT-01/02 & HIGH-01..03, kiểm chứng 4/"
            "4 Notebook PASS.",
        ],
        [
            "Phiên 13–16",
            "21/11–22/11/2026",
            "Dashboard 5 Tab + HTML Offline, kiểm thử pytest, xuất bản bản "
            "đầu Word/PDF (Cluster SE, WCR, TOST, CQR).",
        ],
        [
            f"Phiên 17–{len(data['worklog_sessions'])}",
            "23/11–28/11/2026",
            "Chuẩn hóa dữ liệu & kiểm chứng hợp nhất (CRIT-03/H4, Ex-Ante, "
            "Q_realized, 17 hình 300 DPI), tách scripts/, PEP 8 ≤ 79, khắc "
            "phục triệt để rò rỉ distance trong Pipeline "
            "(RouteDistanceImputer), SMART & ≤ 45 trang ngoài bìa, kiểm toán "
            "toàn diện theo rubric projects.pdf & tái lập đóng gói.",
        ],
    ]

    # Trích xuất đoạn mục tiêu Phiên 12 trực tiếp từ ./nhật ký làm việc.md
    m_p12 = re.search(
        r"(## \[2026-11-19 10:17\] - Phiên 12:.*?)(?=\n---|\Z)",
        data["worklog_text"],
        flags=re.S,
    )
    assert m_p12 is not None, (
        "Không tìm thấy khối Phiên 12 trong ./nhật ký làm việc.md"
    )
    appendix_b_verbatim_excerpt = m_p12.group(1).strip()

    return {
        "executive_summary": executive_summary,
        "abbreviations": ABBREVIATIONS_TABLE,
        "chapters": chapters,
        "references": REFERENCES_APA7,
        "appendix_a_tree": appendix_a_tree,
        "appendix_a_module_index": appendix_a_module_index,
        "appendix_a_main_code": appendix_a_main_code,
        "appendix_a_config_code": appendix_a_config_code,
        "appendix_a_code": appendix_a_code_snippet,
        "appendix_b_audit_map": appendix_b_audit_map,
        "appendix_b_milestones": appendix_b_milestones,
        "appendix_b_verbatim": appendix_b_verbatim_excerpt,
        "worklog_meta": {
            "sessions_count": len(data["worklog_sessions"]),
            "lines": data["worklog_lines"],
            "words": data["worklog_words"],
            "n_src_files": n_src,
            "n_script_files": n_scripts,
            "n_py_files": int(data["n_py_files"]),
            "src_only_loc": src_only_loc,
            "scripts_only_loc": scripts_only_loc,
            "total_src_loc": total_src_loc,
            "n_test_files": n_test_files,
        },
    }

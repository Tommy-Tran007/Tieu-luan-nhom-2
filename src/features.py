"""
Module: features.py
Mục đích: Thực thi Giai đoạn 4 (Feature Engineering, Data Leakage Audit,
Multicollinearity VIF & Pipeline Packaging):
1. Kiểm toán toàn bộ 32 cột của bảng master_rideshare.parquet và phân loại vai
trò (Feature / Target / Leakage / Key / Redundant / Audit Flag).
2. Thiết kế các đặc trưng kinh tế lượng & chu kỳ thời gian không chạm vào biến
mục tiêu price.
3. Kiểm chứng thực nghiệm bẫy rò rỉ dữ liệu (price_per_mile) và kiểm toán tỷ lệ
bản ghi gốc không qua hiệu chỉnh quy mô 10^m (R3).
4. Phân tích Đa cộng tuyến (VIF) trước và sau khi trực giao hóa biến tương tác
tăng giá (lyft_surge_excess).
5. Thiết lập 3 chiến lược phân chia dữ liệu (Ngẫu nhiên 80/20, Theo mốc thời
gian, Theo nhóm 72 tuyến đường).
6. Đóng gói Scikit-Learn ColumnTransformer & Pipeline chống rò rỉ dữ liệu cho
cả 4 nhánh mô hình (M1, M2-M4, M5-M6, M7).

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import (
    BASE_CAT_FEATURES,
    BASE_NUM_FEATURES,
    ENGINEERED_NUM_FEATURES,
    FIG_STAGE4_DIR,
    FORBIDDEN_IN_X,
    MASTER_PARQUET_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_CAB_PATH,
    RESULTS_DIR,
    TABLES_DIR,
    TARGET_COL,
    TEST_SIZE,
    ensure_directories,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. NẠP DỮ LIỆU CHUẨN HÓA & TẠO ĐẶC TRƯNG KINH TẾ HỌC (STAGE 4 FEATURE
# CREATION)
# =============================================================================
def load_master_dataset(
    parquet_path: Path = MASTER_PARQUET_PATH,
) -> pd.DataFrame:
    """
    Nạp bảng dữ liệu sạch master_rideshare.parquet và kiểm tra tính toàn vẹn sơ
    bộ.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu sạch tại: {parquet_path}"
        )
    df = pd.read_parquet(parquet_path)
    logger.info(
        f"Đã nạp master_rideshare.parquet: {df.shape[0]:,} dòng x "
        f"{df.shape[1]} cột."
    )
    return df


def engineer_features_stage4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bộ đặc trưng mới phục vụ mô hình hóa mà KHÔNG sử dụng biến mục tiêu
    (price)
    và KHÔNG tính toán thống kê xuyên dòng (đảm bảo 0% rò rỉ giữa Train và
    Test).
    """
    out = df.copy()

    # 1. Biến tương tác cự ly x hệ số nhân giá tổng quát
    out["distance_x_surge"] = (
        out["distance"].astype(np.float32)
        * out["surge_multiplier"].astype(np.float32)
    ).astype(np.float32)

    # 2. Biến phi tuyến logarit quãng đường: nắm bắt độ cong cước tối thiểu
    # (Minimum Fare) ở cuốc xe cực ngắn
    out["log_distance"] = np.log1p(out["distance"].astype(np.float32)).astype(
        np.float32
    )

    # 3. Đặc trưng cấu trúc thị trường hai phía: chỉ Lyft mới có surge > 1.0
    # trong dữ liệu thực nghiệm
    out["is_lyft"] = (out["cab_type"].astype(str) == "Lyft").astype(np.int8)
    out["lyft_surge_excess"] = (
        (out["surge_multiplier"].astype(np.float32) - 1.0)
        * out["is_lyft"].astype(np.float32)
    ).astype(np.float32)
    out["distance_x_lyft_surge_excess"] = (
        out["distance"].astype(np.float32) * out["lyft_surge_excess"]
    ).astype(np.float32)

    # 4. Mã hóa lượng giác chu kỳ thời gian (Cyclical Encoding) cho Giờ (24h)
    # và Thứ trong tuần (7 ngày) Boston EST
    # Giữ nguyên NaN trên các dòng thiếu timestamp gốc (is_timestamp_original
    # == 0) để không suy đoán sai thời gian
    hour_rad = 2.0 * np.pi * out["hour"].astype(np.float32) / 24.0
    out["hour_sin"] = np.sin(hour_rad).astype(np.float32)
    out["hour_cos"] = np.cos(hour_rad).astype(np.float32)

    dow_rad = 2.0 * np.pi * out["day_of_week"].astype(np.float32) / 7.0
    out["dow_sin"] = np.sin(dow_rad).astype(np.float32)
    out["dow_cos"] = np.cos(dow_rad).astype(np.float32)

    # 5. Định danh tuyến đường (72 tuyến thực tế) phục vụ GroupSplit & mô hình
    # mốc M0b
    route_str = (
        out["source"].astype(str) + " -> " + out["destination"].astype(str)
    )
    out["route_id"] = route_str.astype("category")

    # Kiểm tra nghiêm ngặt: không có Inf trên bất kỳ đặc trưng nào; không có
    # NaN trên đặc trưng luôn đủ
    # (log_distance, is_lyft), không có NaN trên đặc trưng surge khi
    # is_surge_original == 1,
    # và không có NaN trên biến chu kỳ thời gian khi is_timestamp_original ==
    # 1.
    always_non_null = ["is_lyft"]
    surge_dependent_cols = [
        "distance_x_surge",
        "lyft_surge_excess",
        "distance_x_lyft_surge_excess",
    ]
    for col in ENGINEERED_NUM_FEATURES:
        n_inf = int(np.isinf(out[col].fillna(0.0)).sum())
        if n_inf > 0:
            raise ValueError(
                f"Đặc trưng {col} phát sinh {n_inf} giá trị Inf không hợp lệ!"
            )
        if col == "log_distance":
            n_na_dist = int(
                out.loc[out["distance"].notna(), "log_distance"].isna().sum()
            )
            if n_na_dist > 0:
                raise ValueError(
                    f"Đặc trưng {col} phát sinh {n_na_dist} NaN trên các "
                    "dòng có distance hợp lệ!"
                )
        elif col in always_non_null:
            n_na = int(out[col].isna().sum())
            if n_na > 0:
                raise ValueError(
                    f"Đặc trưng cốt lõi {col} phát sinh {n_na} giá trị NaN "
                    "không hợp lệ!"
                )
        elif col == "lyft_surge_excess" and "is_surge_original" in out.columns:
            n_na_surge = int(
                out.loc[out["is_surge_original"] == 1, col].isna().sum()
            )
            if n_na_surge > 0:
                raise ValueError(
                    f"Đặc trưng {col} phát sinh {n_na_surge} NaN trên các "
                    "dòng có is_surge_original == 1!"
                )
        elif (
            col in surge_dependent_cols and "is_surge_original" in out.columns
        ):
            n_na_surge = int(
                out.loc[
                    (out["is_surge_original"] == 1) & out["distance"].notna(),
                    col,
                ]
                .isna()
                .sum()
            )
            if n_na_surge > 0:
                raise ValueError(
                    f"Đặc trưng surge {col} phát sinh {n_na_surge} NaN trên "
                    "tập surge quan sát!"
                )
        elif "is_timestamp_original" in out.columns:
            n_na_valid = int(
                out.loc[out["is_timestamp_original"] == 1, col].isna().sum()
            )
            if n_na_valid > 0:
                raise ValueError(
                    f"Đặc trưng chu kỳ {col} phát sinh {n_na_valid} NaN "
                    "trên tập timestamp gốc!"
                )

    return out


# =============================================================================
# 2. KIỂM TOÁN ĐẶC TRƯNG & CHỐNG RÒ RỈ DỮ LIỆU (FEATURE & LEAKAGE AUDIT)
# =============================================================================
FEATURE_METADATA_SPECS: dict[str, tuple[str, bool, str, str, str]] = {
    # col: (source_stage, known_at_booking, role, decision, rationale)
    "price": (
        "Stage 3 Master",
        False,
        "Target (y)",
        "Target y",
        "Biến mục tiêu cần dự báo (USD). Chỉ dùng làm nhãn y, cấm đưa vào X.",
    ),
    "id": (
        "Stage 3 Master",
        True,
        "Identifier/Key",
        "Drop - Key/ID",
        "Mã định danh duy nhất của chuyến đi (UUID), không mang thông tin "
        "quy luật kinh tế.",
    ),
    "time_stamp": (
        "Stage 3 Master",
        True,
        "Identifier/Key",
        "Drop - Key/ID",
        "Dấu thời gian nguyên thủy (epoch ms); thông tin chu kỳ đã trích "
        "xuất qua hour, day_of_week.",
    ),
    "datetime": (
        "Stage 3 Master",
        True,
        "Identifier/Key",
        "Drop - Key/ID",
        "Mốc thời gian chi tiết dùng để sắp xếp Temporal Split; không đưa "
        "trực tiếp vào ma trận X.",
    ),
    "date_hour": (
        "Stage 3 Master",
        True,
        "Identifier/Key",
        "Drop - Key/ID",
        "Khóa ghép nối thời gian theo giờ với bảng thời tiết; đã đại diện "
        "bởi các biến thời tiết & chu kỳ.",
    ),
    "location": (
        "Stage 3 Master",
        True,
        "Identifier/Key",
        "Drop - Key/ID",
        "Bản sao địa danh trạm thời tiết sau phép Left Join (trùng 100% với "
        "cột source).",
    ),
    "product_id": (
        "Stage 3 Master",
        True,
        "Redundant",
        "Drop - Redundant",
        "Mã nội bộ ánh xạ tất định 1-1 với cặp (cab_type, name); giữ lại sẽ "
        "gây suy biến đa cộng tuyến hoàn hảo.",
    ),
    "day_name": (
        "Stage 3 Master",
        True,
        "Redundant",
        "Drop - Redundant",
        "Tên thứ dạng chuỗi ánh xạ 1-1 với day_of_week; loại bỏ để tránh "
        "trùng lặp thông tin.",
    ),
    "price_per_mile": (
        "Stage 3 Master",
        False,
        "Data Leakage",
        "Drop - Leakage",
        "RÒ RỈ MỤC TIÊU TRỰC TIẾP: được tính bằng price / distance. Đưa vào "
        "X sẽ làm mô hình gian lận.",
    ),
    "distance": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Quãng đường lộ trình (dặm) biết trước khi đặt xe; nhân tố vật lý "
        "cốt lõi quyết định chi phí vận hành.",
    ),
    "surge_multiplier": (
        "Stage 3 Master",
        True,
        "Conditional Pricing Input",
        "Keep in X (Conditional)",
        "Hệ số nhân giá động công bố đồng thời trên API (1.0x–3.0x với Lyft "
        "quan sát; cố định 1.0x trên API Uber; NaN nếu Lyft khuyết gốc). "
        "Biến đầu vào của nhánh Mô hình Giá có Điều kiện; được rút bỏ ở "
        "nhánh Ex-Ante.",
    ),
    "temp": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Nhiệt độ môi trường (°F) tại điểm đón; kiểm định tác động thời "
        "tiết đến giá báo API (RQ1).",
    ),
    "clouds": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Độ che phủ mây (0.0 - 1.0) tại khu vực đón khách.",
    ),
    "pressure": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Áp suất khí quyển (mb/hPa) tại khu vực đón khách.",
    ),
    "rain": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Lượng mưa (inch/giờ); biến khí tượng trọng tâm kiểm chứng giả "
        "thuyết mưa làm tăng giá.",
    ),
    "humidity": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Độ ẩm tương đối (0.0 - 1.0) tại khu vực đón khách.",
    ),
    "wind": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Tốc độ gió (mph) tại khu vực đón khách.",
    ),
    "hour": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Giờ khởi hành trong ngày (0 - 23); dùng cho mô hình cây và làm cơ "
        "sở tạo biến chu kỳ sin/cos.",
    ),
    "day_of_week": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Thứ trong tuần (0=Thứ Hai ... 6=Chủ Nhật).",
    ),
    "is_weekend": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Biến giả cuối tuần (1 nếu Thứ Bảy/Chủ Nhật, 0 nếu ngày thường).",
    ),
    "is_rush_hour": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Biến giả khung giờ cao điểm ngày thường (07:00-09:00 và 16:00-19:"
        "00).",
    ),
    "is_raining": (
        "Stage 3 Master",
        True,
        "Core Numerical",
        "Keep in X",
        "Biến chỉ báo trạng thái có mưa (rain > 0).",
    ),
    "cab_type": (
        "Stage 3 Master",
        True,
        "Core Categorical",
        "Keep in X",
        "Nền tảng gọi xe (Uber vs Lyft); phân biệt cấu trúc biểu giá và cơ "
        "chế surge giữa hai hãng.",
    ),
    "name": (
        "Stage 3 Master",
        True,
        "Core Categorical",
        "Keep in X",
        "Phân khúc hạng xe (12 gói dịch vụ); quyết định trực tiếp giá mở "
        "cửa và đơn giá mỗi dặm (RQ2).",
    ),
    "source": (
        "Stage 3 Master",
        True,
        "Core Categorical",
        "Keep in X",
        "Quận đón khách (12 địa bàn Boston); kiểm soát hiệu ứng cố định "
        "không gian điểm đi.",
    ),
    "destination": (
        "Stage 3 Master",
        True,
        "Core Categorical",
        "Keep in X",
        "Quận trả khách (12 địa bàn Boston); kiểm soát hiệu ứng cố định "
        "không gian điểm đến.",
    ),
    "time_of_day": (
        "Stage 3 Master",
        True,
        "Core Categorical",
        "Keep in X",
        "Buổi trong ngày (Morning, Afternoon, Evening, Night).",
    ),
    "distance_x_surge": (
        "Stage 3 & 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Tích số distance * surge_multiplier; phản ánh quãng đường quy đổi "
        "theo hệ số nhân giá.",
    ),
    "log_distance": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Biến đổi ln(1 + distance); nắm bắt độ cong cước sàn tối thiểu ở "
        "các cuốc xe dưới 1 dặm.",
    ),
    "is_lyft": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Chỉ báo nền tảng Lyft (1=Lyft, 0=Uber); tách biệt hiệu ứng nền "
        "tảng có áp dụng surge động.",
    ),
    "lyft_surge_excess": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Phần phụ trội (surge_multiplier - 1.0) * is_lyft; khử đa cộng "
        "tuyến cấu trúc với hằng số 1.0 của Uber.",
    ),
    "distance_x_lyft_surge_excess": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Tương tác distance * (surge - 1.0) * is_lyft; đo phần tăng thêm "
        "trên mỗi dặm khi Lyft kích hoạt surge.",
    ),
    "hour_sin": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Thành phần sin(2*pi*hour/24); đảm bảo tính liên tục giữa 23:00 đêm "
        "và 00:00 sáng cho mô hình tuyến tính.",
    ),
    "hour_cos": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Thành phần cos(2*pi*hour/24); cặp trực giao với hour_sin biểu diễn "
        "vị trí trên vòng tròn 24 giờ.",
    ),
    "dow_sin": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Thành phần sin(2*pi*day_of_week/7); biểu diễn chu kỳ tuần liên tục.",
    ),
    "dow_cos": (
        "Stage 4 Engineered",
        True,
        "Engineered Feature",
        "Keep in X",
        "Thành phần cos(2*pi*day_of_week/7); cặp trực giao với dow_sin.",
    ),
    "route_id": (
        "Stage 4 Engineered",
        True,
        "Auxiliary Group Key",
        "Aux Group Only",
        "Chuỗi định danh 72 tuyến 'source -> destination' dùng để chia "
        "GroupShuffleSplit và lập bảng tra M0b.",
    ),
    "is_timestamp_original": (
        "Stage 3 Provenance Flag",
        True,
        "Provenance Flag",
        "Drop - Audit Flag",
        "Cờ nguồn gốc thời gian (1 = timestamp gốc từ API, 0 = khuyết trong "
        "dữ liệu thô, không suy đoán ffill).",
    ),
    "weather_matched": (
        "Stage 3 Provenance Flag",
        True,
        "Provenance Flag",
        "Drop - Audit Flag",
        "Cờ khớp thời tiết thực (1 = khớp bản tin trạm giờ trên timestamp "
        "gốc, 0 = không khớp/thiếu timestamp).",
    ),
    "is_surge_original": (
        "Stage 3 Provenance Flag",
        True,
        "Provenance Flag",
        "Drop - Audit Flag",
        "Cờ nguồn gốc hệ số Surge (1 = Lyft quan sát thực trong [1.0, 3.0] "
        "hoặc Uber API cố định 1.0x; 0 = Lyft khuyết/sentinel giữ NaN, "
        "không fillna(1.0)).",
    ),
    "is_rain_sentinel": (
        "Stage 3 Provenance Flag",
        True,
        "Provenance Flag",
        "Drop - Audit Flag",
        "Cờ mã lính canh lượng mưa (1 = toàn bộ bản tin mưa trong ô quận × "
        "giờ khớp bị sentinel hỏng nên giữ rain/is_raining = NaN; 0 = bình "
        "thường).",
    ),
}

# [MED-03 FIX] Từ điển đơn vị đo đầy đủ cho toàn bộ 41 cột (Data Dictionary hợp
# nhất)
FEATURE_UNITS: dict[str, str] = {
    "id": "Chuỗi UUID (Khóa chính)",
    "cab_type": "Danh mục (Uber / Lyft)",
    "name": "Danh mục (12 gói dịch vụ)",
    "product_id": "Mã chuỗi (Khóa ánh xạ 1-1)",
    "source": "Danh mục (12 quận Boston)",
    "destination": "Danh mục (12 quận Boston)",
    "distance": "Dặm (miles)",
    "surge_multiplier": "Hệ số nhân (1.0x - 3.0x; NaN nếu Lyft khuyết)",
    "is_surge_original": "Nhị phân {0, 1}",
    "price": "USD (Đô la Mỹ)",
    "time_stamp": "Unix Epoch (milliseconds)",
    "is_timestamp_original": "Nhị phân {0, 1}",
    "datetime": "Thời gian Boston EST (YYYY-MM-DD HH:MM:SS)",
    "date_hour": "Khung giờ Boston EST (làm tròn giờ)",
    "location": "Danh mục (12 trạm khí tượng Boston)",
    "temp": "Độ Fahrenheit (°F)",
    "clouds": "Tỷ lệ che phủ [0.0 - 1.0]",
    "pressure": "Millibar / hPa",
    "rain": "Inch / giờ (in/h; NaN nếu sentinel mưa)",
    "humidity": "Tỷ lệ độ ẩm [0.0 - 1.0]",
    "wind": "Dặm / giờ (mph)",
    "weather_matched": "Nhị phân {0, 1}",
    "is_rain_sentinel": "Nhị phân {0, 1}",
    "hour": "Giờ trong ngày EST (0 - 23)",
    "day_of_week": "Thứ trong tuần EST (0=T2 ... 6=CN)",
    "day_name": "Danh mục (Thứ trong tuần / Unknown)",
    "is_weekend": "Nhị phân {0, 1} (NaN nếu thiếu TS)",
    "is_rush_hour": "Nhị phân {0, 1} (NaN nếu thiếu TS)",
    "time_of_day": "Danh mục (Morning/Afternoon/Evening/Night/Unknown)",
    "is_raining": "Nhị phân {0, 1} (NaN nếu không khớp thời tiết)",
    "distance_x_surge": "Dặm quy đổi (miles × multiplier)",
    "price_per_mile": "USD / dặm (Cấm đưa vào X - Leakage)",
    "log_distance": "ln(1 + dặm) (không thứ nguyên)",
    "is_lyft": "Nhị phân {0, 1}",
    "lyft_surge_excess": "Phần phụ trội hệ số nhân (>= 0.0)",
    "distance_x_lyft_surge_excess": "Dặm phụ trội surge (miles × excess)",
    "hour_sin": "Chu kỳ sin [-1, 1]",
    "hour_cos": "Chu kỳ cos [-1, 1]",
    "dow_sin": "Chu kỳ sin [-1, 1]",
    "dow_cos": "Chu kỳ cos [-1, 1]",
    "route_id": "Danh mục (72 tuyến source -> destination)",
}


def audit_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng bảng kiểm toán & Từ điển Dữ liệu (Data Dictionary) đầy đủ đơn vị
    cho toàn bộ các cột.
    """
    rows = []
    for col in df_feat.columns:
        spec = FEATURE_METADATA_SPECS.get(
            col,
            (
                "Unknown",
                False,
                "Unclassified",
                "Drop",
                "Chưa được khai báo trong danh mục kiểm soát.",
            ),
        )
        rows.append(
            {
                "column_name": col,
                "source_stage": spec[0],
                "unit": FEATURE_UNITS.get(col, "Không thứ nguyên"),
                "dtype": str(df_feat[col].dtype),
                "n_missing": int(df_feat[col].isna().sum()),
                "n_unique": int(df_feat[col].nunique(dropna=True)),
                "known_at_booking": bool(spec[1]),
                "role": spec[2],
                "decision": spec[3],
                "rationale": spec[4],
            }
        )
    audit_df = pd.DataFrame(rows)
    return audit_df


def _parse_raw_abs(series: pd.Series) -> pd.Series:
    """Chuẩn hóa chuỗi số thô (bỏ hậu tố đơn vị, sentinel) và lấy |x|."""
    s = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s*(?:USD|VND|đ|kg|%)\s*$", "", case=False, regex=True)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    nums = pd.to_numeric(s, errors="coerce")
    nums = nums.where(
        ~nums.isin([-1, -1.0, -100, 1000, 9999, 10000, 100000]), np.nan
    )
    return nums.abs()


def audit_raw_unmodified_flags(
    df_feat: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Kiểm toán Rủi ro R3: Đối chiếu từng bản ghi trong master_rideshare.parquet
    với data/raw/cab_rides.csv để xác định các dòng có price và distance hoàn
    toàn NGUYÊN BẢN (không chia tỷ lệ 10^m hay điền khuyết trung vị tuyến).
    """
    logger.info(
        "Đang đối chiếu với data/raw/cab_rides.csv để kiểm toán tỷ lệ bản "
        "ghi nguyên bản (R3 Audit)..."
    )
    raw_cab = pd.read_csv(
        RAW_CAB_PATH,
        usecols=["id", "price", "distance"],
        dtype=str,
        low_memory=False,
    )
    raw_cab["id"] = raw_cab["id"].str.strip()
    raw_cab["raw_price_abs"] = _parse_raw_abs(raw_cab["price"])
    raw_cab["raw_dist_abs"] = _parse_raw_abs(raw_cab["distance"])
    raw_dedup = raw_cab.drop_duplicates(subset=["id"]).set_index("id")

    clean_ids = df_feat["id"].astype(str).str.strip()
    matched_raw_price = clean_ids.map(raw_dedup["raw_price_abs"])
    matched_raw_dist = clean_ids.map(raw_dedup["raw_dist_abs"])

    price_unmodified = matched_raw_price.notna() & (
        np.abs(
            matched_raw_price.values
            - df_feat["price"].astype(np.float64).values
        )
        < 1e-2
    )
    dist_unmodified = matched_raw_dist.notna() & (
        np.abs(
            matched_raw_dist.values
            - df_feat["distance"].astype(np.float64).values
        )
        < 1e-2
    )
    strictly_unmodified = price_unmodified & dist_unmodified

    flags_df = pd.DataFrame(
        {
            "id": clean_ids.values,
            "is_raw_unmodified_price": price_unmodified.values,
            "is_raw_unmodified_distance": dist_unmodified.values,
            "is_strictly_unmodified_raw": strictly_unmodified.values,
        }
    )

    n_total = len(df_feat)
    n_price_unmod = int(price_unmodified.sum())
    n_dist_unmod = int(dist_unmodified.sum())
    n_strict_unmod = int(strictly_unmodified.sum())

    r3_summary = {
        "total_clean_rows": n_total,
        "price_unmodified_count": n_price_unmod,
        "price_unmodified_pct": round(n_price_unmod / n_total * 100, 2),
        "price_scale_recovered_count": int(n_total - n_price_unmod),
        "price_scale_recovered_pct": round(
            (n_total - n_price_unmod) / n_total * 100, 2
        ),
        "distance_unmodified_count": n_dist_unmod,
        "distance_unmodified_pct": round(n_dist_unmod / n_total * 100, 2),
        "distance_recovered_or_imputed_count": int(n_total - n_dist_unmod),
        "distance_recovered_or_imputed_pct": round(
            (n_total - n_dist_unmod) / n_total * 100, 2
        ),
        "strictly_unmodified_both_count": n_strict_unmod,
        "strictly_unmodified_both_pct": round(
            n_strict_unmod / n_total * 100, 2
        ),
    }
    logger.info(
        f"Hoàn tất kiểm toán R3: {n_strict_unmod:,}/{n_total:,} dòng "
        f"({r3_summary['strictly_unmodified_both_pct']}%) "
        f"hoàn toàn không qua khôi phục quy mô 10^m hay điền khuyết distance."
    )
    return flags_df, r3_summary


# =============================================================================
# 3. KIỂM ĐỊNH ĐA CỘNG TUYẾN (VARIANCE INFLATION FACTOR - VIF)
# =============================================================================
def compute_vif_table(df_num: pd.DataFrame, config_label: str) -> pd.DataFrame:
    """
    Tính toán hệ số phóng đại phương sai VIF (kèm hằng số chặn const) trên tập
    huấn luyện.
    """
    X_mat = df_num.dropna().astype(np.float64).copy()
    X_mat = X_mat.assign(const=1.0)
    cols = list(X_mat.columns)
    vif_rows = []
    values = X_mat.values
    for idx, col in enumerate(cols):
        if col == "const":
            continue
        vif_val = float(variance_inflation_factor(values, idx))
        if vif_val >= 10.0:
            status = "Cao (VIF >= 10 - Đa cộng tuyến cấu trúc)"
        elif vif_val >= 5.0:
            status = "Trung bình (5 <= VIF < 10)"
        else:
            status = "An toàn (VIF < 5)"
        vif_rows.append(
            {
                "config": config_label,
                "feature": col,
                "vif": round(vif_val, 4),
                "tolerance_1_over_vif": round(1.0 / vif_val, 4)
                if vif_val > 0
                else 0.0,
                "assessment": status,
            }
        )
    return (
        pd.DataFrame(vif_rows)
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def run_vif_analysis(
    df_train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    So sánh VIF trên tập Train giữa 2 cấu hình:
    - Cấu hình A (Chưa trực giao hóa): dùng đồng thời distance,
      surge_multiplier, distance_x_surge.
    - Cấu hình B (Đã trực giao hóa theo cơ chế kinh tế): thay surge_multiplier
      bằng lyft_surge_excess = (surge - 1)*is_lyft
      và dùng biến chu kỳ sin/cos cho thời gian.
    """
    logger.info("Đang tính toán VIF trên tập huấn luyện (X_train)...")
    cols_uncentered = [
        "distance",
        "surge_multiplier",
        "distance_x_surge",
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_rush_hour",
    ]
    cols_orthogonal = [
        "distance",
        "lyft_surge_excess",
        "distance_x_lyft_surge_excess",
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_rush_hour",
    ]

    # [CRIT-01/02/03 & H4 FIX] Tính VIF strictly trên các bản ghi tập Train
    # khớp thời tiết thực,
    # không bị sentinel mưa và có surge quan sát thực (0 NaN, 0 imputation
    # giả).
    vif_mask = pd.Series(True, index=df_train.index)
    if "weather_matched" in df_train.columns:
        vif_mask &= df_train["weather_matched"] == 1
    if "is_surge_original" in df_train.columns:
        vif_mask &= df_train["is_surge_original"] == 1
    vif_mask &= df_train["rain"].notna()
    vif_base = (
        df_train.loc[vif_mask]
        .dropna(subset=cols_uncentered + cols_orthogonal)
        .copy()
    )
    vif_a = compute_vif_table(
        vif_base[cols_uncentered], "A_Uncentered_Raw_Interaction"
    )
    vif_b = compute_vif_table(
        vif_base[cols_orthogonal], "B_Centered_Excess_Interaction"
    )
    vif_all = pd.concat([vif_a, vif_b], ignore_index=True)

    max_weather_vif = float(
        vif_b.loc[
            vif_b["feature"].isin(
                ["temp", "clouds", "pressure", "rain", "humidity", "wind"]
            ),
            "vif",
        ].max()
    )
    dist_vif_before = float(
        vif_a.loc[vif_a["feature"] == "distance", "vif"].iloc[0]
    )
    dist_vif_after = float(
        vif_b.loc[vif_b["feature"] == "distance", "vif"].iloc[0]
    )

    summary = {
        "uncentered_max_vif": float(vif_a["vif"].max()),
        "uncentered_distance_vif": dist_vif_before,
        "orthogonal_max_vif": float(vif_b["vif"].max()),
        "orthogonal_distance_vif": dist_vif_after,
        "weather_max_vif": round(max_weather_vif, 4),
        "table_uncentered": vif_a.to_dict(orient="records"),
        "table_orthogonal": vif_b.to_dict(orient="records"),
    }
    logger.info(
        f"VIF của 'distance' giảm từ {dist_vif_before:.2f} (chưa tâm hóa) "
        f"xuống {dist_vif_after:.2f} "
        "(sau khi chuyển sang lyft_surge_excess); VIF tối đa của nhóm thời "
        f"tiết = {max_weather_vif:.2f}."
    )
    return vif_all, summary


# =============================================================================
# 4. BA CHIẾN LƯỢC CHIA TẬP DỮ LIỆU (RANDOM, TEMPORAL, ROUTE GROUP SPLIT)
# =============================================================================
def _summarize_split(
    df_feat: pd.DataFrame,
    split_name: str,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
) -> dict[str, Any]:
    """Tính phân phối giá, KS, SMD và phạm vi ngày/tuyến cho 1 split."""
    y_tr = df_feat[TARGET_COL].iloc[tr_idx].astype(np.float64).values
    y_te = df_feat[TARGET_COL].iloc[te_idx].astype(np.float64).values

    ks_stat, ks_pval = stats.ks_2samp(y_tr, y_te)
    pooled_sd = np.sqrt((np.var(y_tr, ddof=1) + np.var(y_te, ddof=1)) / 2.0)
    smd = (
        abs(np.mean(y_tr) - np.mean(y_te)) / pooled_sd
        if pooled_sd > 0
        else 0.0
    )

    tr_dates = df_feat["datetime"].iloc[tr_idx]
    te_dates = df_feat["datetime"].iloc[te_idx]
    split_total = len(tr_idx) + len(te_idx)
    return {
        "split_strategy": split_name,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "train_pct": round(len(tr_idx) / split_total * 100, 2),
        "test_pct": round(len(te_idx) / split_total * 100, 2),
        "train_price_mean": round(float(np.mean(y_tr)), 4),
        "test_price_mean": round(float(np.mean(y_te)), 4),
        "train_price_median": round(float(np.median(y_tr)), 4),
        "test_price_median": round(float(np.median(y_te)), 4),
        "train_price_std": round(float(np.std(y_tr, ddof=1)), 4),
        "test_price_std": round(float(np.std(y_te, ddof=1)), 4),
        "ks_statistic": round(float(ks_stat), 5),
        "ks_pvalue": round(float(ks_pval), 5),
        "standardized_mean_diff": round(float(smd), 5),
        "train_routes": int(df_feat["route_id"].iloc[tr_idx].nunique()),
        "test_routes": int(df_feat["route_id"].iloc[te_idx].nunique()),
        "train_start": str(tr_dates.min())[:19],
        "train_end": str(tr_dates.max())[:19],
        "test_start": str(te_dates.min())[:19],
        "test_end": str(te_dates.max())[:19],
    }


def create_data_splits(
    df_feat: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Khởi tạo 3 chiến lược chia dữ liệu độc lập (random_80_20, temporal_split,
    route_group_split) trên tập quan sát thực surge_multiplier.
    """
    logger.info(
        f"Đang thiết lập 3 chiến lược chia dữ liệu (test_size={test_size}, "
        f"random_state={random_state})..."
    )
    if "is_surge_original" in df_feat.columns:
        obs_indices = np.where(df_feat["is_surge_original"].values == 1)[0]
    else:
        obs_indices = np.where(df_feat["surge_multiplier"].notna().values)[0]

    rand_train_idx, rand_test_idx = train_test_split(
        obs_indices, test_size=test_size, random_state=random_state
    )

    ts_cond = (
        (df_feat["is_timestamp_original"].values == 1)
        if "is_timestamp_original" in df_feat.columns
        else df_feat["datetime"].notna().values
    )
    surge_cond = (
        (df_feat["is_surge_original"].values == 1)
        if "is_surge_original" in df_feat.columns
        else df_feat["surge_multiplier"].notna().values
    )
    valid_ts_indices = np.where(ts_cond & surge_cond)[0]
    sorted_valid_order = valid_ts_indices[
        np.argsort(df_feat["datetime"].iloc[valid_ts_indices].values)
    ]
    cutoff_pos = int(round(len(sorted_valid_order) * (1.0 - test_size)))
    temp_train_idx = sorted_valid_order[:cutoff_pos]
    temp_test_idx = sorted_valid_order[cutoff_pos:]

    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    grp_tr_pos, grp_te_pos = next(
        gss.split(
            obs_indices,
            groups=df_feat["route_id"].iloc[obs_indices].astype(str).values,
        )
    )
    splits = {
        "random_80_20": (rand_train_idx, rand_test_idx),
        "temporal_split": (temp_train_idx, temp_test_idx),
        "route_group_split": (
            obs_indices[grp_tr_pos],
            obs_indices[grp_te_pos],
        ),
    }
    summary_rows = [
        _summarize_split(df_feat, name, tr_idx, te_idx)
        for name, (tr_idx, te_idx) in splits.items()
    ]
    return splits, pd.DataFrame(summary_rows)


# =============================================================================
# 5. ĐÓNG GÓI PIPELINE & BỘ BIẾN ĐỔI TƯƠNG TÁC KINH TẾ LƯỢNG (M1 -> M7)
# =============================================================================
class RouteDistanceImputer(BaseEstimator, TransformerMixin):
    """Nội suy `distance` theo median tuyến strictly-on-train trong Pipeline.

    Bảo đảm 0% rò rỉ dữ liệu giữa train và test/validation bằng cách chỉ học
    trung vị cự ly từng cặp (source, destination) từ tập huấn luyện, đồng thời
    tính lại các đặc trưng dẫn xuất (`log_distance`, `distance_x_surge`,
    `distance_x_lyft_surge_excess`) và bảo toàn nguyên vẹn dtype `category`.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> RouteDistanceImputer:
        """Học bảng median `distance` theo tuyến từ tập huấn luyện."""
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = int(X.shape[1])

        dist_s = pd.to_numeric(X["distance"], errors="coerce").astype(
            np.float64
        )
        valid_mask = np.isfinite(dist_s)
        if valid_mask.any():
            self.global_median_ = float(np.nanmedian(dist_s[valid_mask]))
        else:
            self.global_median_ = 2.15

        if {"source", "destination"}.issubset(X.columns) and valid_mask.any():
            tmp = pd.DataFrame(
                {
                    "source": X.loc[valid_mask, "source"].astype(str),
                    "destination": X.loc[valid_mask, "destination"].astype(
                        str
                    ),
                    "distance": dist_s.loc[valid_mask],
                }
            )
            grp = tmp.groupby(["source", "destination"], observed=True)[
                "distance"
            ].median()
            self.route_medians_: dict[tuple[str, str], float] = {
                (str(k[0]), str(k[1])): float(v) for k, v in grp.items()
            }
        else:
            self.route_medians_ = {}
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Bảo toàn tên cột đầu vào cho Pipeline.get_feature_names_out()."""
        if input_features is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray(input_features, dtype=object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Điền khuyết `distance` và cập nhật đặc trưng dẫn xuất liên quan."""
        out = X.copy()
        dist_num = pd.to_numeric(out["distance"], errors="coerce").astype(
            np.float64
        )
        miss_mask = ~np.isfinite(dist_num)

        if miss_mask.any():
            fill_vals = pd.Series(
                self.global_median_, index=out.index, dtype=np.float64
            )
            if {"source", "destination"}.issubset(out.columns):
                keys = list(
                    zip(
                        out.loc[miss_mask, "source"].astype(str),
                        out.loc[miss_mask, "destination"].astype(str),
                        strict=False,
                    )
                )
                mapped = [
                    self.route_medians_.get(k, self.global_median_)
                    for k in keys
                ]
                fill_vals.loc[miss_mask] = mapped
            dist_num.loc[miss_mask] = fill_vals.loc[miss_mask]
            out["distance"] = dist_num.astype(np.float32)

        d32 = pd.to_numeric(out["distance"], errors="coerce").astype(
            np.float32
        )
        if "log_distance" in out.columns:
            ld_num = pd.to_numeric(
                out["log_distance"], errors="coerce"
            ).astype(np.float32)
            ld_mask = (miss_mask | ~np.isfinite(ld_num)) & np.isfinite(d32)
            if ld_mask.any():
                out.loc[ld_mask, "log_distance"] = np.log1p(
                    d32.loc[ld_mask]
                ).astype(np.float32)
        if (
            "distance_x_surge" in out.columns
            and "surge_multiplier" in out.columns
        ):
            s32 = pd.to_numeric(
                out["surge_multiplier"], errors="coerce"
            ).astype(np.float32)
            dxs_num = pd.to_numeric(
                out["distance_x_surge"], errors="coerce"
            ).astype(np.float32)
            dxs_mask = (
                (miss_mask | ~np.isfinite(dxs_num))
                & np.isfinite(d32)
                & np.isfinite(s32)
            )
            if dxs_mask.any():
                out.loc[dxs_mask, "distance_x_surge"] = (
                    d32.loc[dxs_mask] * s32.loc[dxs_mask]
                ).astype(np.float32)
        if (
            "distance_x_lyft_surge_excess" in out.columns
            and "lyft_surge_excess" in out.columns
        ):
            ex32 = pd.to_numeric(
                out["lyft_surge_excess"], errors="coerce"
            ).astype(np.float32)
            dxex_num = pd.to_numeric(
                out["distance_x_lyft_surge_excess"], errors="coerce"
            ).astype(np.float32)
            dxex_mask = (
                (miss_mask | ~np.isfinite(dxex_num))
                & np.isfinite(d32)
                & np.isfinite(ex32)
            )
            if dxex_mask.any():
                out.loc[dxex_mask, "distance_x_lyft_surge_excess"] = (
                    d32.loc[dxex_mask] * ex32.loc[dxex_mask]
                ).astype(np.float32)

        return out


class EconomicInteractionPreprocessor(BaseEstimator, TransformerMixin):
    """
    Bộ biến đổi Scikit-Learn tùy chỉnh (Custom Transformer) xây dựng ma trận
    thiết kế
    có tương tác kinh tế lượng cho nhóm mô hình tuyến tính nâng cao (M2: OLS
    tương tác, M3: Ridge, M4: Lasso):
      price ~ C(name) + C(name):distance + C(name):log_distance
            + C(name_lyft):lyft_surge_excess +
            C(name_lyft):distance_x_lyft_surge_excess
            + C(source) + C(destination) + C(time_of_day)
            + is_rush_hour + is_weekend + hour_sin + hour_cos + dow_sin +
            dow_cos
            + temp + rain + humidity + wind + clouds + pressure

    Đặc điểm kỹ thuật:
    - Lưu danh sách các phân khúc (tiers), điểm đón (sources), điểm đến
      (destinations) strictly từ tập Train trong .fit().
    - Chuẩn hóa Z-score (StandardScaler) cho các biến liên tục ngoại sinh dựa
      trên mean/std của tập Train.
    - Với các gói Uber (surge luôn bằng 1.0 -> lyft_surge_excess = 0), tự động
      loại bỏ các cột tương tác surge bằng 0
      để ma trận thiết kế luôn khả nghịch hoàn toàn (Full Column Rank), không
      bị suy biến (Singular Matrix).
    """

    def __init__(self, scale_continuous: bool = True):
        self.scale_continuous = scale_continuous

    def fit(
        self, X: pd.DataFrame, y: Any = None
    ) -> EconomicInteractionPreprocessor:
        self.route_dist_imputer_ = RouteDistanceImputer().fit(X, y)
        self.feature_names_in_ = self.route_dist_imputer_.feature_names_in_
        self.n_features_in_ = self.route_dist_imputer_.n_features_in_
        self.tiers_ = sorted(X["name"].astype(str).unique().tolist())
        self.reference_tier_ = self.tiers_[0]
        all_lyft_tiers = sorted(
            X.loc[X["cab_type"].astype(str) == "Lyft", "name"]
            .astype(str)
            .unique()
            .tolist()
        )
        if "lyft_surge_excess" in X.columns:
            surge_vals_all = X["lyft_surge_excess"].astype(np.float64)
        else:
            surge_vals_all = (
                X["surge_multiplier"].astype(np.float64) - 1.0
            ).fillna(0.0)
        name_vals_str = X["name"].astype(str)
        # Chỉ giữ các phân khúc Lyft thực sự phát sinh biến động surge > 0
        # (tự động loại Shared vốn luôn 1.0x để tránh cột toàn 0)
        self.lyft_tiers_ = [
            t
            for t in all_lyft_tiers
            if (surge_vals_all[name_vals_str == t].abs() > 1e-9).any()
        ]
        all_sources = sorted(X["source"].astype(str).unique().tolist())
        all_dests = sorted(X["destination"].astype(str).unique().tolist())
        self.sources_ = all_sources[1:]
        # Đồ thị 12 quận là K_{6,6} hai phía: ngoài điểm đón tham chiếu,
        # cần loại 1 điểm đến ở mỗi phía phân hoạch để khử đa cộng tuyến tổng.
        ref_src = all_sources[0]
        reach_from_ref = sorted(
            X.loc[X["source"].astype(str) == ref_src, "destination"]
            .astype(str)
            .unique()
            .tolist()
        )
        dropped_dests = {all_dests[0]}
        for rd in reach_from_ref:
            if rd not in dropped_dests:
                dropped_dests.add(rd)
                break
        self.destinations_ = [d for d in all_dests if d not in dropped_dests]
        self.times_of_day_ = sorted(
            X["time_of_day"].astype(str).unique().tolist()
        )[1:]

        self.exog_num_cols_ = [
            "is_rush_hour",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "temp",
            "rain",
            "humidity",
            "wind",
            "clouds",
            "pressure",
        ]
        exog_vals = X[self.exog_num_cols_].astype(np.float64).values
        # Fit trung vị khuyết thiếu CHỈ trên tập Train (0% Train-Test Leakage)
        self.exog_medians_ = np.nanmedian(exog_vals, axis=0)
        self.exog_medians_ = np.where(
            np.isnan(self.exog_medians_), 0.0, self.exog_medians_
        )
        exog_filled = np.where(
            np.isnan(exog_vals), self.exog_medians_, exog_vals
        )
        self.exog_means_ = np.mean(exog_filled, axis=0)
        exog_std_raw = np.std(exog_filled, axis=0)
        self.exog_stds_ = np.where(exog_std_raw > 1e-8, exog_std_raw, 1.0)

        # Xây dựng danh sách tên cột đầu ra cố định
        names = []
        # 1. Cước mở cửa theo phân khúc (bỏ phân khúc tham chiếu đầu tiên vì mô
        # hình có hệ số chặn intercept)
        for t in self.tiers_[1:]:
            names.append(f"tier_{t}")
        # 2. Đơn giá mỗi dặm riêng cho cả 12 phân khúc (C(name):distance)
        for t in self.tiers_:
            names.append(f"tier_{t}_x_distance")
        # 3. Độ cong cự ly ngắn riêng cho từng phân khúc (C(name):log_distance)
        for t in self.tiers_:
            names.append(f"tier_{t}_x_log_distance")
        # 4. Phụ trội surge mở cửa & theo dặm riêng cho 6 phân khúc Lyft
        for t in self.lyft_tiers_:
            names.append(f"lyft_{t}_x_surge_excess")
            names.append(f"lyft_{t}_x_dist_surge_excess")
        # 5. Hiệu ứng cố định địa bàn đón & trả khách + buổi trong ngày
        for s in self.sources_:
            names.append(f"source_{s}")
        for d in self.destinations_:
            names.append(f"dest_{d}")
        for tod in self.times_of_day_:
            names.append(f"tod_{tod}")
        # 6. Các biến thời gian & khí tượng ngoại sinh
        names.extend(self.exog_num_cols_)
        self.feature_names_out_ = np.array(names, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self, "route_dist_imputer_"):
            X = self.route_dist_imputer_.transform(X)
        n = len(X)
        name_str = X["name"].astype(str).values
        src_str = X["source"].astype(str).values
        dst_str = X["destination"].astype(str).values
        tod_str = X["time_of_day"].astype(str).values

        dist = X["distance"].astype(np.float64).values
        log_dist = (
            X["log_distance"].astype(np.float64).values
            if "log_distance" in X.columns
            else np.log1p(dist)
        )
        surge_ex = (
            X["lyft_surge_excess"].astype(np.float64).values
            if "lyft_surge_excess" in X.columns
            else np.where(
                X["cab_type"].astype(str).values == "Lyft",
                X["surge_multiplier"].astype(np.float64).values - 1.0,
                0.0,
            )
        )
        dist_surge_ex = (
            X["distance_x_lyft_surge_excess"].astype(np.float64).values
            if "distance_x_lyft_surge_excess" in X.columns
            else dist * surge_ex
        )

        cols_list = []

        # 1. Dummy phân khúc
        for t in self.tiers_[1:]:
            cols_list.append((name_str == t).astype(np.float64))
        # 2. Tương tác phân khúc x distance
        for t in self.tiers_:
            cols_list.append(((name_str == t).astype(np.float64)) * dist)
        # 3. Tương tác phân khúc x log_distance
        for t in self.tiers_:
            cols_list.append(((name_str == t).astype(np.float64)) * log_dist)
        # 4. Tương tác phân khúc Lyft x surge_excess và distance_x_surge_excess
        for t in self.lyft_tiers_:
            mask = (name_str == t).astype(np.float64)
            cols_list.append(mask * surge_ex)
            cols_list.append(mask * dist_surge_ex)
        # 5. Dummy điểm đi, điểm đến, buổi trong ngày
        for s in self.sources_:
            cols_list.append((src_str == s).astype(np.float64))
        for d in self.destinations_:
            cols_list.append((dst_str == d).astype(np.float64))
        for tod in self.times_of_day_:
            cols_list.append((tod_str == tod).astype(np.float64))

        mat_struct = (
            np.column_stack(cols_list)
            if cols_list
            else np.empty((n, 0), dtype=np.float64)
        )

        exog_vals = X[self.exog_num_cols_].astype(np.float64).values
        exog_vals = np.where(
            np.isnan(exog_vals), self.exog_medians_, exog_vals
        )
        if self.scale_continuous:
            exog_vals = (exog_vals - self.exog_means_) / self.exog_stds_

        return np.hstack([mat_struct, exog_vals])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_out_


def get_feature_columns_for_modeling() -> list[str]:
    """
    Trả về danh sách các cột đặc trưng hợp lệ trong X (đảm bảo 100% không chứa
    bất kỳ cột cấm nào).
    """
    feature_cols = (
        BASE_NUM_FEATURES + BASE_CAT_FEATURES + ENGINEERED_NUM_FEATURES
    )
    # Loại bỏ trùng lặp nhưng giữ nguyên thứ tự
    seen = set()
    ordered_cols = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            ordered_cols.append(c)
    overlap = set(ordered_cols).intersection(FORBIDDEN_IN_X)
    if overlap:
        raise RuntimeError(
            f"PHÁT HIỆN VI PHẠM RÒ RỈ DỮ LIỆU: Các cột cấm {overlap} xuất "
            "hiện trong X!"
        )
    return ordered_cols


class BipartiteCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Mã hóa One-Hot đầy đủ hạng cho đặc trưng phân loại của M1, loại bỏ
    cab_type (phụ thuộc tuyến tính 1-1 với name) và khử bẫy đồ thị hai phía
    K_{6,6} trên cặp (source, destination).
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> BipartiteCategoryEncoder:
        self.tiers_ = sorted(X["name"].astype(str).unique().tolist())[1:]
        all_sources = sorted(X["source"].astype(str).unique().tolist())
        all_dests = sorted(X["destination"].astype(str).unique().tolist())
        self.sources_ = all_sources[1:]
        ref_src = all_sources[0]
        reach_from_ref = sorted(
            X.loc[X["source"].astype(str) == ref_src, "destination"]
            .astype(str)
            .unique()
            .tolist()
        )
        dropped_dests = {all_dests[0]}
        for rd in reach_from_ref:
            if rd not in dropped_dests:
                dropped_dests.add(rd)
                break
        self.destinations_ = [d for d in all_dests if d not in dropped_dests]
        self.times_of_day_ = sorted(
            X["time_of_day"].astype(str).unique().tolist()
        )[1:]
        self.feature_names_out_ = np.array(
            [f"tier_{t}" for t in self.tiers_]
            + [f"source_{s}" for s in self.sources_]
            + [f"dest_{d}" for d in self.destinations_]
            + [f"tod_{tod}" for tod in self.times_of_day_],
            dtype=object,
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        name_s = X["name"].astype(str).values
        src_s = X["source"].astype(str).values
        dst_s = X["destination"].astype(str).values
        tod_s = X["time_of_day"].astype(str).values
        cols = (
            [(name_s == t).astype(np.float64) for t in self.tiers_]
            + [(src_s == s).astype(np.float64) for s in self.sources_]
            + [(dst_s == d).astype(np.float64) for d in self.destinations_]
            + [(tod_s == tod).astype(np.float64) for tod in self.times_of_day_]
        )
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_out_


def build_stage4_preprocessors() -> dict[str, Any]:
    """
    Khởi tạo 4 bộ tiền xử lý (Preprocessors) chuyên biệt cho từng họ thuật
    toán:
    1. m1_basic_linear: StandardScaler + BipartiteCategoryEncoder trên đặc
    trưng cơ sở (đầy đủ hạng, không tương tác gói xe x dặm).
    2. m2_m4_interaction_linear: EconomicInteractionPreprocessor (biểu giá
    riêng từng gói xe + chuẩn hóa biến ngoại sinh).
    3. m5_m6_tree_ohe: Passthrough biến số + OneHotEncoder cho Decision Tree &
    Random Forest.
    4. m7_hgb_native: Chuẩn hóa kiểu category nguyên bản cho
    HistGradientBoostingRegressor.
    """
    basic_num = [
        "distance",
        "surge_multiplier",
        "distance_x_surge",
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_rush_hour",
    ]
    basic_cat = ["cab_type", "name", "source", "destination", "time_of_day"]
    m1_cat = ["name", "source", "destination", "time_of_day"]

    m1_basic_linear_ct = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                basic_num,
            ),
            ("cat", BipartiteCategoryEncoder(), m1_cat),
        ],
        remainder="drop",
    )
    m1_basic_linear = Pipeline(
        [
            ("route_dist_imp", RouteDistanceImputer()),
            ("ct", m1_basic_linear_ct),
        ]
    )

    m2_m4_interaction_linear = EconomicInteractionPreprocessor(
        scale_continuous=True
    )

    tree_num = BASE_NUM_FEATURES + ENGINEERED_NUM_FEATURES
    # Khử trùng lặp tên cột
    tree_num = list(dict.fromkeys(tree_num))

    m5_m6_tree_ohe_ct = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), tree_num),
            (
                "cat",
                OneHotEncoder(
                    drop=None, sparse_output=False, handle_unknown="ignore"
                ),
                basic_cat,
            ),
        ],
        remainder="drop",
    )
    m5_m6_tree_ohe = Pipeline(
        [
            ("route_dist_imp", RouteDistanceImputer()),
            ("ct", m5_m6_tree_ohe_ct),
        ]
    )

    m7_hgb_native_ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", tree_num),
            ("cat", "passthrough", basic_cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    m7_hgb_native = Pipeline(
        [
            ("route_dist_imp", RouteDistanceImputer()),
            ("ct", m7_hgb_native_ct),
        ]
    )

    return {
        "m1_basic_linear": m1_basic_linear,
        "m2_m4_interaction_linear": m2_m4_interaction_linear,
        "m5_m6_tree_ohe": m5_m6_tree_ohe,
        "m7_hgb_native": m7_hgb_native,
    }


# =============================================================================
# 6. KIỂM CHỨNG PIPELINE & THÍ NGHIỆM MUTATE & BREAK (R4 & R5 VERIFICATION)
# =============================================================================
def _eval_regression_preds(
    y_true: np.ndarray, y_hat: np.ndarray
) -> dict[str, float]:
    """Tính 4 thước đo hồi quy (R2, RMSE, MAE, MAPE%)."""
    return {
        "r2": round(float(r2_score(y_true, y_hat)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_hat)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_hat)), 4),
        "mape_pct": round(
            float(mean_absolute_percentage_error(y_true, y_hat) * 100), 2
        ),
    }


def _run_r4_leakage_experiment(
    df_feat: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Chạy đối chứng R4 bằng cách cố tình đưa price_per_mile vào ma trận X."""
    leak_cols = ["distance", "surge_multiplier", "price_per_mile"]
    X_tr_leak = df_feat[leak_cols].iloc[train_idx].copy()
    X_te_leak = df_feat[leak_cols].iloc[test_idx].copy()
    dist_med = float(X_tr_leak["distance"].median())
    X_tr_leak["distance"] = X_tr_leak["distance"].fillna(dist_med)
    X_te_leak["distance"] = X_te_leak["distance"].fillna(dist_med)
    ppm_med = float(X_tr_leak["price_per_mile"].median())
    X_tr_leak["price_per_mile"] = X_tr_leak["price_per_mile"].fillna(ppm_med)
    X_te_leak["price_per_mile"] = X_te_leak["price_per_mile"].fillna(ppm_med)

    leak_hgb = HistGradientBoostingRegressor(
        max_iter=150, random_state=RANDOM_STATE
    )
    leak_hgb.fit(X_tr_leak, y_train)
    leak_hgb_metrics = _eval_regression_preds(
        y_test, leak_hgb.predict(X_te_leak)
    )

    leak_ols = LinearRegression()
    leak_ols.fit(X_tr_leak[["distance", "price_per_mile"]], y_train)
    leak_ols_metrics = _eval_regression_preds(
        y_test, leak_ols.predict(X_te_leak[["distance", "price_per_mile"]])
    )
    ppm_corr = float(
        df_feat[["price", "price_per_mile"]].dropna().corr().iloc[0, 1]
    )
    return {
        "price_per_mile_pearson_r_with_price": round(ppm_corr, 4),
        "corrupted_ols_additive_with_ppm": leak_ols_metrics,
        "corrupted_hgb_with_ppm": leak_hgb_metrics,
    }


def verify_pipelines_and_leakage_trap(
    df_feat: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """
    Kiểm chứng 4 bộ tiền xử lý và chạy đối chứng R4 (Leakage) & R5
    (Interaction) trên tập Train/Test 80/20.
    """
    logger.info(
        "Đang kiểm chứng các Pipeline và chạy thực nghiệm đối chứng R4 "
        "(Leakage) & R5 (Interaction)..."
    )
    valid_cols = get_feature_columns_for_modeling()
    X_train = df_feat[valid_cols].iloc[train_idx].copy()
    X_test = df_feat[valid_cols].iloc[test_idx].copy()
    y_train = df_feat[TARGET_COL].iloc[train_idx].astype(np.float64).values
    y_test = df_feat[TARGET_COL].iloc[test_idx].astype(np.float64).values

    for c in BASE_CAT_FEATURES:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    preprocessors = build_stage4_preprocessors()
    shapes_info = {}
    for name, prep in preprocessors.items():
        Xt_tr = prep.fit_transform(X_train, y_train)
        Xt_te = prep.transform(X_test)
        feat_names = prep.get_feature_names_out()
        shapes_info[name] = {
            "train_shape": [int(Xt_tr.shape[0]), int(Xt_tr.shape[1])],
            "test_shape": [int(Xt_te.shape[0]), int(Xt_te.shape[1])],
            "n_features_out": int(len(feat_names)),
        }

    pipe_m1 = Pipeline(
        [
            ("prep", preprocessors["m1_basic_linear"]),
            ("reg", LinearRegression()),
        ]
    )
    pipe_m1.fit(X_train, y_train)
    m1_metrics = _eval_regression_preds(y_test, pipe_m1.predict(X_test))

    pipe_m2 = Pipeline(
        [
            ("prep", EconomicInteractionPreprocessor(scale_continuous=True)),
            ("reg", LinearRegression()),
        ]
    )
    pipe_m2.fit(X_train, y_train)
    m2_metrics = _eval_regression_preds(y_test, pipe_m2.predict(X_test))

    pipe_m7 = Pipeline(
        [
            ("prep", preprocessors["m7_hgb_native"]),
            (
                "reg",
                HistGradientBoostingRegressor(
                    categorical_features="from_dtype",
                    max_iter=150,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe_m7.fit(X_train, y_train)
    m7_metrics = _eval_regression_preds(y_test, pipe_m7.predict(X_test))

    leakage_res = _run_r4_leakage_experiment(
        df_feat, train_idx, test_idx, y_train, y_test
    )
    leak_hgb = leakage_res["corrupted_hgb_with_ppm"]
    logger.info(
        f"Kết quả kiểm chứng R5: OLS cơ bản (M1) R2={m1_metrics['r2']}, "
        f"MAPE={m1_metrics['mape_pct']}% --> OLS tương tác kinh tế (M2) "
        f"R2={m2_metrics['r2']}, MAPE={m2_metrics['mape_pct']}%."
    )
    logger.info(
        "Kết quả Mutate & Break R4 (bẫy rò rỉ price_per_mile): HistGB với "
        f"price_per_mile đạt R2={leak_hgb['r2']}, MAPE={leak_hgb['mape_pct']}%"
    )
    return {
        "preprocessor_shapes": shapes_info,
        "m1_basic_ols": m1_metrics,
        "m2_interaction_ols": m2_metrics,
        "m7_stage4_hgb_preview": m7_metrics,
        "leakage_trap_experiment": leakage_res,
    }


# =============================================================================
# 7. TRỰC QUAN HÓA CHUYÊN SÂU GIAI ĐOẠN 4 (FIG08, FIG09, FIG10 - 300 DPI)
# =============================================================================
def _plot_fig08_corr_panel(ax: Any, df_feat: pd.DataFrame) -> None:
    """Vẽ Panel A của Fig 08: Tương quan Pearson với price."""
    corr_cols = [
        "price",
        "distance_x_surge",
        "distance",
        "log_distance",
        "lyft_surge_excess",
        "distance_x_lyft_surge_excess",
        "surge_multiplier",
        "price_per_mile",
        "temp",
        "rain",
        "humidity",
        "is_rush_hour",
    ]
    corr_with_price = (
        df_feat[corr_cols]
        .corr()["price"]
        .drop("price")
        .sort_values(ascending=True)
    )
    colors_corr = [
        "#dc2626"
        if idx == "price_per_mile"
        else ("#2563eb" if abs(val) >= 0.15 else "#64748b")
        for idx, val in corr_with_price.items()
    ]
    bars = ax.barh(
        corr_with_price.index,
        corr_with_price.values,
        color=colors_corr,
        edgecolor="#1e293b",
        height=0.68,
    )
    ax.axvline(0, color="#0f172a", linewidth=1.0)
    ax.set_xlim(-0.18, 0.72)
    ax.set_title(
        "A. Tương quan Pearson (r) với Giá báo API (price)\n[Đỏ = Biến rò "
        "rỉ bị cấm | Xanh = Đặc trưng kinh tế hợp lệ]",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(
        "Hệ số tương quan Pearson (r) với biến mục tiêu price (USD)",
        fontsize=10,
    )
    ax.set_ylabel("Đặc trưng (Features)", fontsize=10)
    for bar, val in zip(bars, corr_with_price.values, strict=False):
        offset = 0.012 if val >= 0 else -0.012
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
        )


def _plot_fig08_vif_panel(ax: Any, vif_summary: dict[str, Any]) -> None:
    """Vẽ Panel B của Fig 08: So sánh VIF trước và sau trực giao hóa."""
    vif_a = {r["feature"]: r["vif"] for r in vif_summary["table_uncentered"]}
    vif_b = {r["feature"]: r["vif"] for r in vif_summary["table_orthogonal"]}
    compare_items = [
        ("distance\n(Quãng đường)", vif_a["distance"], vif_b["distance"]),
        (
            "surge / lyft_excess\n(Hệ số nhân giá)",
            vif_a["surge_multiplier"],
            vif_b["lyft_surge_excess"],
        ),
        (
            "dist x surge\n(Biến tương tác)",
            vif_a["distance_x_surge"],
            vif_b["distance_x_lyft_surge_excess"],
        ),
        ("temp\n(Nhiệt độ)", vif_a["temp"], vif_b["temp"]),
        ("humidity\n(Độ ẩm)", vif_a["humidity"], vif_b["humidity"]),
        ("rain\n(Lượng mưa)", vif_a["rain"], vif_b["rain"]),
    ]
    labels_v = [x[0] for x in compare_items]
    y_pos = np.arange(len(labels_v))
    h = 0.36
    b1 = ax.barh(
        y_pos + h / 2,
        [x[1] for x in compare_items],
        height=h,
        color="#f97316",
        edgecolor="#1e293b",
        label="Trước tâm hóa (Dùng surge gốc >= 1.0)",
    )
    b2 = ax.barh(
        y_pos - h / 2,
        [x[2] for x in compare_items],
        height=h,
        color="#0d9488",
        edgecolor="#1e293b",
        label="Sau trực giao hóa (Dùng lyft_surge_excess)",
    )
    ax.axvline(
        5.0,
        color="#eab308",
        linestyle="--",
        linewidth=1.3,
        label="Ngưỡng cảnh báo (VIF = 5)",
    )
    ax.axvline(
        10.0,
        color="#dc2626",
        linestyle=":",
        linewidth=1.5,
        label="Ngưỡng đa cộng tuyến cao (VIF = 10)",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_v, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 450.0)
    ax.set_xlabel(
        "Hệ số phóng đại phương sai VIF (Thang đo Log10)", fontsize=10
    )
    ax.set_title(
        "B. Kiểm định Đa cộng tuyến (VIF) trên Tập Train\n[Trực giao hóa "
        "lyft_surge_excess khử hoàn toàn VIF của distance]",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    for bars_grp, color in ((b1, "#c2410c"), (b2, "#0f766e")):
        for bar in bars_grp:
            w = bar.get_width()
            ax.text(
                w * 1.06,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}",
                va="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )


def _save_fig08_corr_and_vif(
    df_feat: pd.DataFrame, vif_summary: dict[str, Any]
) -> str:
    """Xuất Fig 08 (tương quan đặc trưng & VIF) tại 300 DPI."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=300)
    _plot_fig08_corr_panel(axes[0], df_feat)
    _plot_fig08_vif_panel(axes[1], vif_summary)
    fig.suptitle(
        "HÌNH 4.1 (FIG 08): KIỂM TOÁN TƯƠNG QUAN ĐẶC TRƯNG & XỬ LÝ ĐA CỘNG "
        "TUYẾN CẤU TRÚC (N = 270,898)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig08_path = FIG_STAGE4_DIR / "fig08_feature_correlation_after_fe.png"
    fig.savefig(fig08_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig08_path.relative_to(PROCESSED_DIR.parent.parent).as_posix()


def _save_fig09_split_distribution(
    df_feat: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    split_df: pd.DataFrame,
) -> str:
    """Xuất Fig 09 (phân phối price Train/Test qua 3 chiến lược chia)."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8), dpi=300)
    n_rand = len(splits["random_80_20"][0]) + len(splits["random_80_20"][1])
    n_temp = len(splits["temporal_split"][0]) + len(
        splits["temporal_split"][1]
    )
    titles = {
        "random_80_20": (
            "A. Chia Ngẫu Nhiên 80/20\n(Random Split trên n = "
            f"{n_rand:,} surge quan sát thực)"
        ),
        "temporal_split": (
            "B. Chia Theo Mốc Thời Gian Boston EST\n(Temporal Split trên n "
            f"= {n_temp:,} TS & surge gốc: 80% đầu vs 20% cuối)"
        ),
        "route_group_split": (
            "C. Chia Theo Nhóm Tuyến Đường\n(GroupShuffleSplit: 57 tuyến "
            "Train vs 15 tuyến Test)"
        ),
    }
    for ax, (sp_key, (tr_idx, te_idx)) in zip(
        axes, splits.items(), strict=False
    ):
        y_tr = df_feat[TARGET_COL].iloc[tr_idx].values
        y_te = df_feat[TARGET_COL].iloc[te_idx].values
        row = split_df.loc[split_df["split_strategy"] == sp_key].iloc[0]
        sns.kdeplot(
            y_tr,
            ax=ax,
            color="#1d4ed8",
            fill=True,
            alpha=0.25,
            linewidth=2.0,
            label=(
                f"Train (n={len(y_tr):,}, Mean=${row['train_price_mean']:.2f})"
            ),
        )
        sns.kdeplot(
            y_te,
            ax=ax,
            color="#e11d48",
            fill=True,
            alpha=0.20,
            linewidth=2.0,
            linestyle="--",
            label=(
                f"Test (n={len(y_te):,}, Mean=${row['test_price_mean']:.2f})"
            ),
        )
        ax.axvline(np.median(y_tr), color="#1d4ed8", linestyle=":", lw=1.4)
        ax.axvline(np.median(y_te), color="#e11d48", linestyle=":", lw=1.4)
        ax.set_xlim(0, 65)
        ax.set_title(titles[sp_key], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Giá báo API — price (USD)", fontsize=10)
        ax.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.text(
            0.97,
            0.66,
            f"KS stat = {row['ks_statistic']:.4f}\nSMD = "
            f"{row['standardized_mean_diff']:.4f}\n"
            f"Train median = ${row['train_price_median']:.2f}\nTest "
            f"median = ${row['test_price_median']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f8fafc",
                "edgecolor": "#94a3b8",
            },
        )
    fig.suptitle(
        "HÌNH 4.2 (FIG 09): ĐỐI SOÁT PHÂN PHỐI GIÁ BÁO API GIỮA TẬP TRAIN "
        "VÀ TEST QUA 3 CHIẾN LƯỢC PHÂN CHIA",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig09_path = FIG_STAGE4_DIR / "fig09_split_distribution.png"
    fig.savefig(fig09_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig09_path.relative_to(PROCESSED_DIR.parent.parent).as_posix()


def _draw_pipeline_box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    bg_color: str,
    border_color: str = "#1e293b",
    text_color: str = "#0f172a",
    fontsize: float = 8.8,
) -> None:
    """Vẽ một khối chữ nhật bo góc kèm chữ căn giữa cho sơ đồ pipeline."""
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.18,rounding_size=0.2",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.4,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
    )


def _fig10_pipeline_boxes(
    n_cols: int, p_shapes: dict[str, Any]
) -> list[tuple[Any, ...]]:
    """Tạo danh sách 7 khối mô tả kiến trúc Anti-Leakage Pipeline."""
    n_tr = int(p_shapes["m1_basic_linear"]["train_shape"][0])
    n_te = int(p_shapes["m1_basic_linear"]["test_shape"][0])
    tr_pct = round(n_tr / max(n_tr + n_te, 1) * 100)
    d1 = p_shapes["m1_basic_linear"]["n_features_out"]
    d2 = p_shapes["m2_m4_interaction_linear"]["n_features_out"]
    d5 = p_shapes["m5_m6_tree_ohe"]["n_features_out"]
    d7 = p_shapes["m7_hgb_native"]["n_features_out"]
    return [
        (
            0.4,
            8.2,
            9.2,
            1.25,
            "BẢNG DỮ LIỆU TỔNG THỂ: master_rideshare.parquet\n(270,898 dòng "
            f"× 32 cột master + 9 cột FE = {n_cols} cột)",
            "#dbeafe",
            "#1d4ed8",
            "#0f172a",
            8.8,
        ),
        (
            0.4,
            6.1,
            4.3,
            1.45,
            f"BỘ LỌC CHỐNG RÒ RỈ (DROP {len(FORBIDDEN_IN_X) - 1} CỘT + "
            "TARGET)\n- Rò rỉ: price_per_mile | Cờ: TS/Weather/Surge/Rain\n"
            "- Định danh/Khóa: id, time_stamp, datetime...\n- Trùng lặp "
            "1-1: product_id, day_name",
            "#fee2e2",
            "#dc2626",
            "#7f1d1d",
            8.0,
        ),
        (
            5.3,
            6.1,
            4.3,
            1.45,
            "MA TRẬN ĐẶC TRƯNG HỢP LỆ X (27 CỘT)\n- 13 biến số cơ sở + 5 "
            "biến phân loại\n- 9 đặc trưng kinh tế & chu kỳ thời gian\n"
            "(100% biết trước tại thời điểm đặt xe)",
            "#dcfce7",
            "#15803d",
            "#14532d",
            8.3,
        ),
        (
            0.4,
            3.9,
            9.2,
            1.45,
            "PHÂN TÁCH TẬP DỮ LIỆU ĐỘC LẬP (TRAIN / TEST SPLIT — SURGE QUAN "
            f"SÁT THỰC)\n- Tập Train ({tr_pct}% = {n_tr:,} dòng): Dùng để "
            ".fit() bộ chuẩn hóa, mã hóa và 5-Fold CV\n- Tập Test "
            f"({100 - tr_pct}% = {n_te:,} dòng): KHÓA CHẶT, chỉ gọi "
            ".transform() và .predict() đúng 1 lần",
            "#fef3c7",
            "#b45309",
            "#78350f",
            8.5,
        ),
        (
            0.4,
            0.7,
            2.9,
            2.4,
            "Nhánh 1 (M1)\nBasic Linear\n---\nStandardScaler\n+ "
            f"OneHot(drop=1st)\n({d1} chiều)",
            "#f1f5f9",
            "#475569",
            "#0f172a",
            8.2,
        ),
        (
            3.55,
            0.7,
            2.9,
            2.4,
            "Nhánh 2 (M2-M4)\nEconomic Interaction\n---\nC(name)*distance\n"
            f"+ Lyft Surge Excess\n({d2} chiều)",
            "#e0e7ff",
            "#4338ca",
            "#0f172a",
            8.2,
        ),
        (
            6.7,
            0.7,
            2.9,
            2.4,
            f"Nhánh 3 & 4 (M5-M7)\nTree / Boosting\n---\nOneHot ({d5} "
            f"chiều)\nhoặc Native Category\n({d7} chiều)",
            "#ccfbf1",
            "#0f766e",
            "#0f172a",
            8.2,
        ),
    ]


def _plot_fig10_pipeline_panel(
    ax_left: Any, df_feat: pd.DataFrame, verification_res: dict[str, Any]
) -> None:
    """Vẽ Panel A của Fig 10: Sơ đồ kiến trúc Anti-Leakage Pipeline."""
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.axis("off")
    ax_left.set_title(
        "A. Kiến trúc Scikit-Learn Pipeline Chống Rò Rỉ Dữ Liệu\n(Anti-"
        "Leakage Feature Governance & ColumnTransformer)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    boxes = _fig10_pipeline_boxes(
        df_feat.shape[1], verification_res["preprocessor_shapes"]
    )
    for x, y, w, h, txt, bg, bd, tc, fs in boxes:
        _draw_pipeline_box(ax_left, x, y, w, h, txt, bg, bd, tc, fs)

    arrow_kw = {
        "arrowstyle": "->",
        "color": "#1e293b",
        "lw": 1.6,
        "mutation_scale": 14,
    }
    arrows = [
        ((2.55, 7.6), (3.5, 8.2)),
        ((7.45, 7.6), (6.5, 8.2)),
        ((7.45, 5.4), (7.45, 6.1)),
        ((1.85, 3.15), (3.5, 3.9)),
        ((5.0, 3.15), (5.0, 3.9)),
        ((8.15, 3.15), (6.5, 3.9)),
    ]
    for xy, xytext in arrows:
        ax_left.annotate("", xy=xy, xytext=xytext, arrowprops=arrow_kw)


def _annotate_fig10_bars(
    ax_r: Any, ax_twin: Any, b_r2: Any, b_mape: Any, r2_v: list, mape_v: list
) -> None:
    """Gắn nhãn số trị R² và MAPE lên 2 dãy cột của Fig 10 Panel B."""
    for bar, v in zip(b_r2, r2_v, strict=False):
        ax_r.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.1,
            f"R²={v:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#0f172a",
        )
    for bar, v in zip(b_mape, mape_v, strict=False):
        y_off = 0.65 if abs(v - 10.0) < 1.0 else 0.45
        ax_twin.text(
            bar.get_x() + bar.get_width() / 2,
            v + y_off,
            f"MAPE={v:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#881337",
        )


def _plot_fig10_benchmark_panel(
    ax_right: Any, verification_res: dict[str, Any]
) -> None:
    """Vẽ Panel B của Fig 10: Đối chứng hiệu quả sửa R5 và bẫy rò rỉ R4."""
    m1 = verification_res["m1_basic_ols"]
    m2 = verification_res["m2_interaction_ols"]
    m7 = verification_res["m7_stage4_hgb_preview"]
    lk = verification_res["leakage_trap_experiment"]["corrupted_hgb_with_ppm"]
    labels = [
        "M1: OLS Cơ bản\n(1 hệ số dặm chung\ncho cả 12 gói xe)",
        "M2: OLS Tương tác\n(Mỗi gói xe có đơn giá\ndặm & surge riêng)",
        "M7: HistGradientBoosting\n(Cây quyết định\nphi tuyến hợp lệ)",
        "BẪY RÒ RỈ (R4):\nHistGB + price_per_mile\n(Gian lận mục tiêu!)",
    ]
    r2_vals = [m1["r2"] * 100, m2["r2"] * 100, m7["r2"] * 100, lk["r2"] * 100]
    mape_vals = [
        m1["mape_pct"],
        m2["mape_pct"],
        m7["mape_pct"],
        lk["mape_pct"],
    ]
    x_idx = np.arange(len(labels))
    w_bar = 0.36
    b_r2 = ax_right.bar(
        x_idx - w_bar / 2,
        r2_vals,
        width=w_bar,
        color=["#64748b", "#2563eb", "#0d9488", "#dc2626"],
        edgecolor="#0f172a",
        label="R² Score (%) - Cao là tốt",
    )
    ax_twin = ax_right.twinx()
    b_mape = ax_twin.bar(
        x_idx + w_bar / 2,
        mape_vals,
        width=w_bar,
        color=["#cbd5e1", "#93c5fd", "#5eead4", "#fca5a5"],
        edgecolor="#0f172a",
        hatch="//",
        label="MAPE (%) - Thấp là tốt",
    )
    ax_right.axhline(
        90.0,
        color="#1d4ed8",
        ls="--",
        lw=1.2,
        label="Mục tiêu SMART: R² >= 90%",
    )
    ax_twin.axhline(
        10.0,
        color="#e11d48",
        ls=":",
        lw=1.5,
        label="Mục tiêu SMART: MAPE <= 10%",
    )
    ax_right.set_xticks(x_idx)
    ax_right.set_xticklabels(labels, fontsize=8.8)
    ax_right.set_ylim(75, 108)
    ax_twin.set_ylim(0, 19.5)
    ax_right.set_ylabel(
        "Hệ số xác định R² (%)",
        fontsize=10,
        color="#1e3a8a",
        fontweight="bold",
    )
    ax_twin.set_ylabel(
        "Sai số phần trăm tuyệt đối trung bình MAPE (%)",
        fontsize=10,
        color="#9f1239",
        fontweight="bold",
    )
    ax_right.grid(False)
    ax_twin.grid(False)
    ax_right.set_title(
        "B. Kiểm chứng Thực nghiệm: Tác động của Đặc trưng Tương tác "
        "(R5)\nvà Bằng chứng Bẫy Rò rỉ Dữ liệu (Mutate & Break R4)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    _annotate_fig10_bars(ax_right, ax_twin, b_r2, b_mape, r2_vals, mape_vals)
    l1, lb1 = ax_right.get_legend_handles_labels()
    l2, lb2 = ax_twin.get_legend_handles_labels()
    ax_right.legend(
        l1 + l2, lb1 + lb2, loc="upper left", fontsize=8.0, frameon=True
    )


def _save_fig10_pipeline_and_r4(
    df_feat: pd.DataFrame, verification_res: dict[str, Any]
) -> str:
    """Xuất Fig 10 (sơ đồ Anti-Leakage Pipeline & kết quả R4/R5)."""
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(16.5, 6.5),
        dpi=300,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    _plot_fig10_pipeline_panel(ax_left, df_feat, verification_res)
    _plot_fig10_benchmark_panel(ax_right, verification_res)
    fig.suptitle(
        "HÌNH 4.3 (FIG 10): KIẾN TRÚC PIPELINE CHỐNG RÒ RỈ & ĐỐI CHỨNG HIỆU "
        "NĂNG KỸ THUẬT TẠO ĐẶC TRƯNG",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig10_path = FIG_STAGE4_DIR / "fig10_pipeline_diagram.png"
    fig.savefig(fig10_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig10_path.relative_to(PROCESSED_DIR.parent.parent).as_posix()


def generate_stage4_figures(
    df_feat: pd.DataFrame,
    vif_summary: dict[str, Any],
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    split_df: pd.DataFrame,
    verification_res: dict[str, Any],
) -> list[str]:
    """
    Xuất bản 3 biểu đồ kỹ thuật Giai đoạn 4 đạt chuẩn 300 DPI tại
    reports/figures/stage4/.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.8
    return [
        _save_fig08_corr_and_vif(df_feat, vif_summary),
        _save_fig09_split_distribution(df_feat, splits, split_df),
        _save_fig10_pipeline_and_r4(df_feat, verification_res),
    ]


# =============================================================================
# 8. HÀM ĐIỀU PHỐI CHÍNH CỦA GIAI ĐOẠN 4 (RUN_STAGE4)
# =============================================================================
def run_stage4(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Thực thi trọn vẹn Giai đoạn 4 và xuất toàn bộ bảng CSV, JSON kết quả và
    hình ảnh 300 DPI.
    """
    ensure_directories()
    logger.info(
        "=== BẮT ĐẦU GIAI ĐOẠN 4: FEATURE ENGINEERING & PIPELINE "
        "PREPARATION ==="
    )

    df_master = load_master_dataset()
    df_feat = engineer_features_stage4(df_master)

    # 1. Xuất bảng kiểm toán đặc trưng (Feature Audit) & Từ điển Dữ liệu hợp
    # nhất có đơn vị (MED-03)
    audit_df = audit_features(df_feat)
    audit_csv_path = TABLES_DIR / "stage4_feature_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False, encoding="utf-8-sig")
    dict_csv_path = TABLES_DIR / "data_dictionary_complete.csv"
    audit_df.to_csv(dict_csv_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Đã xuất bảng kiểm toán đặc trưng & Data Dictionary đầy đủ đơn vị: "
        f"{audit_csv_path}, {dict_csv_path}"
    )

    # 2. Kiểm toán cờ nguyên bản (R3 Audit Flags) và lưu phục vụ Giai đoạn 5
    flags_df, r3_summary = audit_raw_unmodified_flags(df_feat)
    flags_path = PROCESSED_DIR / "stage4_audit_flags.parquet"
    flags_df.to_parquet(flags_path, index=False)

    # 3. Tạo 3 chiến lược phân chia dữ liệu
    splits, split_df = create_data_splits(
        df_feat, test_size=test_size, random_state=random_state
    )
    split_csv_path = TABLES_DIR / "stage4_split_summary.csv"
    split_df.to_csv(split_csv_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"Đã xuất bảng đối soát 3 chiến lược chia dữ liệu: {split_csv_path}"
    )

    # 4. Tính toán VIF trên tập Train của chia ngẫu nhiên 80/20
    rand_tr_idx, rand_te_idx = splits["random_80_20"]
    vif_df, vif_summary = run_vif_analysis(df_feat.iloc[rand_tr_idx])
    vif_csv_path = TABLES_DIR / "stage4_vif.csv"
    vif_df.to_csv(vif_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã xuất bảng phân tích đa cộng tuyến VIF: {vif_csv_path}")

    # 5. Kiểm chứng các Pipeline & chạy thí nghiệm đối chứng R4 / R5
    verification_res = verify_pipelines_and_leakage_trap(
        df_feat, rand_tr_idx, rand_te_idx
    )

    # 6. Vẽ và lưu bộ 3 biểu đồ kỹ thuật Giai đoạn 4 (fig08, fig09, fig10)
    saved_figs = generate_stage4_figures(
        df_feat, vif_summary, splits, split_df, verification_res
    )

    # 7. Tổng hợp kết quả vào tệp JSON duy nhất cho Giai đoạn 4
    valid_feature_cols = get_feature_columns_for_modeling()
    summary = {
        "stage": "Stage 4 - Feature Engineering & Pipeline Preparation",
        "dataset_overview": {
            "rows": int(df_feat.shape[0]),
            "master_cols": int(df_master.shape[1]),
            "total_cols_after_fe": int(df_feat.shape[1]),
            "valid_modeling_features_count": int(len(valid_feature_cols)),
            "valid_modeling_features": valid_feature_cols,
            "forbidden_cols_dropped": sorted(FORBIDDEN_IN_X),
            "date_min": str(df_feat["datetime"].min())[:19],
            "date_max": str(df_feat["datetime"].max())[:19],
            "observed_days": int(df_feat["datetime"].dt.date.nunique()),
            "observed_routes": int(df_feat["route_id"].nunique()),
            "observed_tiers": int(df_feat["name"].nunique()),
        },
        "feature_audit_counts": audit_df["decision"].value_counts().to_dict(),
        "r3_scale_recovery_audit": r3_summary,
        "vif_analysis": vif_summary,
        "splits_summary": split_df.to_dict(orient="records"),
        "pipeline_and_leakage_verification": verification_res,
        "figures": saved_figs,
    }

    summary_json_path = RESULTS_DIR / "stage4_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu trọn bộ kết quả Giai đoạn 4 tại: {summary_json_path}")
    return summary


if __name__ == "__main__":
    run_stage4()

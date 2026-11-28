"""
Module: config.py
Mục đích: Quản lý tập trung toàn bộ đường dẫn tương đối, siêu tham số toàn cục
(RANDOM_STATE, TEST_SIZE)
và phân loại vai trò của từng cột dữ liệu nhằm ngăn chặn triệt để rò rỉ dữ liệu
(Data Leakage)
và lỗi đường dẫn tuyệt đối cứng (Hardcoded Paths).

Đề tài: PRJ-EXT-01 — Phân tích Biểu hiện Định giá Động và Dự báo Giá báo API
(Uber & Lyft Boston)
Nhóm thực hiện: Nhóm 02 — Trần Thanh Huy (2521001060) & Lê Huy Thắng
(2521001092) — Lớp 25DDS01
"""

from pathlib import Path

# =============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TƯƠNG ĐỐI CHUẨN (TUYỆT ĐỐI KHÔNG DÙNG HARDCODED PATH)
# =============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = BASE_DIR / "reports"
TABLES_DIR: Path = REPORTS_DIR / "tables"
RESULTS_DIR: Path = REPORTS_DIR / "results"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
FIG_STAGE3_DIR: Path = FIGURES_DIR / "stage3"
FIG_STAGE4_DIR: Path = FIGURES_DIR / "stage4"
FIG_STAGE5_DIR: Path = FIGURES_DIR / "stage5"
FIG_STAGE6_DIR: Path = FIGURES_DIR / "stage6"

MODELS_DIR: Path = BASE_DIR / "models"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

# Đường dẫn các tệp dữ liệu chính
RAW_CAB_PATH: Path = RAW_DIR / "cab_rides.csv"
MASTER_PARQUET_PATH: Path = PROCESSED_DIR / "master_rideshare.parquet"

# =============================================================================
# 2. THAM SỐ TÁI LẬP THỰC NGHIỆM (REPRODUCIBILITY CONSTANTS)
# =============================================================================
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
TUNING_SUBSAMPLE_SIZE: int = 60_000

# =============================================================================
# 3. PHÂN LOẠI VAI TRÒ CỘT DỮ LIỆU & KIỂM SOÁT RÒ RỈ (FEATURE GOVERNANCE)
# =============================================================================
TARGET_COL: str = "price"

# Các cột định danh & khóa ghép nối thời gian/không gian (Không đưa vào X)
ID_AND_KEY_COLS: list[str] = [
    "id",
    "time_stamp",
    "datetime",
    "date_hour",
    "location",
]

# Các cột dư thừa hoàn toàn (Ánh xạ 1-1 với cột khác đã có trong X)
REDUNDANT_COLS: list[str] = [
    "product_id",  # Ánh xạ tất định 1-1 với (cab_type, name)
    "day_name",  # Ánh xạ tất định 1-1 với day_of_week
]

# Các cột RÒ RỈ DỮ LIỆU MỤC TIÊU (Target Leakage - CẤM TUYỆT ĐỐI đưa vào X)
LEAKAGE_COLS: list[str] = [
    "price_per_mile",  # Tính trực tiếp từ biến mục tiêu: price / distance
]

# Các cột cờ kiểm toán nguồn gốc dữ liệu (Provenance Flags - dùng để lọc kiểm
# định/mô hình, không đưa vào X)
PROVENANCE_FLAG_COLS: list[str] = [
    # 1 = mốc thời gian gốc hợp lệ, 0 = thiếu trong dữ liệu thô (không ffill)
    "is_timestamp_original",
    # 1 = khớp bản tin thời tiết thực tại (source, date_hour) trên timestamp
    # gốc
    "weather_matched",
    # 1 = Lyft quan sát thực surge hoặc Uber API cố định 1.0x, 0 = Lyft khuyết
    # surge (giữ NaN)
    "is_surge_original",
    # 1 = toàn bộ bản tin mưa tại ô (source, date_hour) khớp là mã lính canh
    # (giữ rain=NaN)
    "is_rain_sentinel",
]

# Tập hợp toàn bộ các cột bị cấm xuất hiện trong ma trận đặc trưng X
FORBIDDEN_IN_X: set[str] = set(
    [TARGET_COL]
    + ID_AND_KEY_COLS
    + REDUNDANT_COLS
    + LEAKAGE_COLS
    + PROVENANCE_FLAG_COLS
)

# Các đặc trưng định lượng cơ sở (có sẵn tại thời điểm đặt xe)
BASE_NUM_FEATURES: list[str] = [
    "distance",
    "surge_multiplier",
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
    "is_raining",
]

# Các đặc trưng phân loại cơ sở (có sẵn tại thời điểm đặt xe)
BASE_CAT_FEATURES: list[str] = [
    "cab_type",
    "name",
    "source",
    "destination",
    "time_of_day",
]

# Các đặc trưng kỹ thuật tạo mới ở Giai đoạn 4 (Không sử dụng biến mục tiêu
# price)
ENGINEERED_NUM_FEATURES: list[str] = [
    # Tương tác vật lý - định giá động: distance * surge_multiplier
    "distance_x_surge",
    "log_distance",  # Đường cong phi tuyến cự ly ngắn: ln(1 + distance)
    "is_lyft",  # Chỉ báo nền tảng có surge động: 1 nếu Lyft, 0 nếu Uber
    # Phần phụ trội hệ số nhân giá của Lyft: (surge_multiplier - 1.0) * is_lyft
    "lyft_surge_excess",
    # Quãng đường phụ trội do surge trên nền tảng Lyft
    "distance_x_lyft_surge_excess",
    "hour_sin",  # Mã hóa chu kỳ 24 giờ: sin(2*pi*hour/24)
    "hour_cos",  # Mã hóa chu kỳ 24 giờ: cos(2*pi*hour/24)
    "dow_sin",  # Mã hóa chu kỳ 7 ngày: sin(2*pi*day_of_week/7)
    "dow_cos",  # Mã hóa chu kỳ 7 ngày: cos(2*pi*day_of_week/7)
]


def ensure_directories() -> None:
    """
    Đảm bảo toàn bộ cấu trúc thư mục đầu ra tồn tại trước khi chạy pipeline.
    """
    for folder in [
        PROCESSED_DIR,
        TABLES_DIR,
        RESULTS_DIR,
        FIG_STAGE3_DIR,
        FIG_STAGE4_DIR,
        FIG_STAGE5_DIR,
        FIG_STAGE6_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

"""
Script: main.py
Mục đích: Điểm vào CLI duy nhất (Unified Pipeline Entrypoint) điều phối toàn bộ
quy trình Khoa học Dữ liệu
theo chuẩn CRISP-DM của đề tài PRJ-EXT-01 (Uber & Lyft Boston Dynamic Pricing):
  - --stage clean    : Làm sạch dữ liệu thô (cab_rides.csv, weather.csv).
  - --stage eda      : Ghép nối đa bảng không fan-out, kiểm toán & trực quan
    hóa EDA (Giai đoạn 3).
  - --stage features : Kỹ thuật tạo đặc trưng, kiểm toán rò rỉ, VIF & đóng gói
    Pipeline (Giai đoạn 4).
  - --stage model    : Kiểm định giả thuyết thống kê & huấn luyện đa mô hình
    học máy (Giai đoạn 5).
  - --stage evaluate : Chẩn đoán phần dư, diễn giải mô hình & mô phỏng chính
    sách Surge Cap (Giai đoạn 6).
  - --stage all      : Thực thi thông suốt từ đầu đến cuối.

Hỗ trợ trực tiếp thử nghiệm phản chứng Mutate & Break qua tham số:
  python main.py --stage features --test-size 0.30

Tác giả: Trần Thanh Huy (MSSV: 2521001060) & Lê Huy Thắng (MSSV: 2521001092) —
Nhóm 02 — Lớp 25DDS01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, ensure_directories  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main_pipeline")


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho pipeline PRJ-EXT-01."""
    parser = argparse.ArgumentParser(
        description=(
            "CLI điều phối toàn trình dự án PRJ-EXT-01 (Uber & Lyft Dynamic "
            "Pricing)."
        )
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="features",
        choices=["clean", "eda", "features", "model", "evaluate", "all"],
        help="Giai đoạn cần thực thi (mặc định: features).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=(
            "Tỷ lệ tập kiểm tra Test Set (mặc định: 0.20, dùng cho thử "
            "nghiệm Mutate & Break)."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Hạt giống ngẫu nhiên đảm bảo tính tái lập (mặc định: 42).",
    )
    return parser.parse_args()


def main() -> int:
    """Điều phối chạy từng giai đoạn hoặc toàn trình CRISP-DM."""
    args = parse_args()
    ensure_directories()

    stage = args.stage
    logger.info(
        f"Khởi chạy Pipeline PRJ-EXT-01 | stage='{stage}' | "
        f"test_size={args.test_size} | random_state={args.random_state}"
    )

    if stage == "clean":
        from clean_raw_data import clean_cab_rides, clean_weather

        clean_cab = clean_cab_rides()
        clean_wea = clean_weather()
        clean_cab.to_parquet(
            PROCESSED_DIR / "cab_rides_cleaned.parquet", index=False
        )
        clean_wea.to_csv(PROCESSED_DIR / "weather_cleaned.csv", index=False)
        logger.info("Hoàn tất giai đoạn CLEAN.")

    if stage in ("eda", "all"):
        import subprocess

        # stage3_pipeline_and_audit.py gọi trực tiếp clean_cab_rides() &
        # clean_weather() nên --stage all chỉ làm sạch đúng 1 lần.
        res = subprocess.run(
            [sys.executable, str(SRC_DIR / "stage3_pipeline_and_audit.py")],
            cwd=str(BASE_DIR),
            check=True,
        )
        if res.returncode == 0:
            logger.info("Hoàn tất giai đoạn CLEAN + EDA (Stage 3).")

    if stage in ("features", "all"):
        from features import run_stage4

        summary_s4 = run_stage4(
            test_size=args.test_size, random_state=args.random_state
        )
        m1 = summary_s4["pipeline_and_leakage_verification"]["m1_basic_ols"]
        m2 = summary_s4["pipeline_and_leakage_verification"][
            "m2_interaction_ols"
        ]
        logger.info(
            f"Hoàn tất giai đoạn FEATURES (Stage 4) | "
            f"OLS cơ bản (M1): R2={m1['r2']}, MAPE={m1['mape_pct']}% -> "
            f"OLS tương tác (M2): R2={m2['r2']}, MAPE={m2['mape_pct']}%"
        )

    if stage in ("model", "all"):
        from stage5_modeling import run_stage5

        run_stage5(test_size=args.test_size, random_state=args.random_state)

    if stage in ("evaluate", "all"):
        from stage6_evaluation import run_stage6

        run_stage6(test_size=args.test_size, random_state=args.random_state)

    logger.info("Pipeline kết thúc thành công (Exit code = 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

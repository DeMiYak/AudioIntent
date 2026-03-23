from __future__ import annotations

import argparse
from pathlib import Path

from .validation_io import build_gold_dataframe_for_evaluation, load_validation_windows, save_dataframe_to_excel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Экспортирует gold.xlsx из листа валидации для evaluation.ipynb."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gold/data_val.xlsx",
        help="Путь к Excel-файлу с листом валидации.",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="Вал - Статус свободен",
        help="Имя validation-листа.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/validation/gold.xlsx",
        help="Куда сохранить gold.xlsx.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = load_validation_windows(args.input, sheet_name=args.sheet_name)
    gold_df = build_gold_dataframe_for_evaluation(windows)
    output_path = save_dataframe_to_excel(gold_df, args.output)
    print(f"Gold для evaluation.ipynb сохранён в: {Path(output_path)}")


if __name__ == "__main__":
    main()

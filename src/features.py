import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH = Path(__file__).parent.parent / "data/financials.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data/features.csv"

REQUIRED_COLUMNS = {
    "ticker",
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "operating_cash_flow",
    "rd_expense",
    "stockholders_equity",
}


def safe_divide(num: pd.Series, denom: pd.Series, index: pd.Index) -> pd.Series:
    """
    Divide two Series element-wise, replacing infinite results with NaN.

    Handles division by zero gracefully — ``x / 0`` produces ``NaN``
    rather than ``inf``, and ``0 / 0`` remains ``NaN``.

    Args:
        num: Numerator Series.
        denom: Denominator Series.
        index: Index to assign to the resulting Series.

    Returns:
        Series of the same length as the inputs with ``inf`` replaced by ``NaN``.
    """
    result = num.values / denom.values
    return pd.Series(np.where(np.isinf(result), np.nan, result), index=index)


def engineer_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 8 size-normalized financial ratios from raw financial metrics.

    Fills missing ``rd_expense`` values with 0 before computing
    ``rd_intensity``. Division by zero produces ``NaN``, not ``inf``.

    Args:
        df: Raw financials DataFrame. Must contain the following columns:
            ``ticker``, ``revenue``, ``net_income``, ``total_assets``,
            ``total_liabilities``, ``operating_cash_flow``, ``rd_expense``,
            ``stockholders_equity``.

    Returns:
        DataFrame with ``ticker`` and 8 ratio columns: ``profit_margin``,
        ``debt_ratio``, ``roe``, ``asset_turnover``, ``ocf_margin``,
        ``rd_intensity``, ``roa``, ``equity_multiplier``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If ``df`` is empty or missing any required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame, got %s." % type(df).__name__
        )
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input DataFrame is missing required columns: %s." % sorted(missing)
        )

    logger.debug("Engineering ratios for %d rows.", len(df))

    df = df.copy()
    df["rd_expense"] = df["rd_expense"].fillna(0)

    ratios = pd.DataFrame(index=df.index)
    ratios["ticker"]           = df["ticker"]
    ratios["profit_margin"]    = safe_divide(df["net_income"],          df["revenue"],               df.index)
    ratios["debt_ratio"]       = safe_divide(df["total_liabilities"],   df["total_assets"],          df.index)
    ratios["roe"]              = safe_divide(df["net_income"],          df["stockholders_equity"],   df.index)
    ratios["asset_turnover"]   = safe_divide(df["revenue"],             df["total_assets"],          df.index)
    ratios["ocf_margin"]       = safe_divide(df["operating_cash_flow"], df["revenue"],               df.index)
    ratios["rd_intensity"]     = safe_divide(df["rd_expense"],          df["revenue"],               df.index)
    ratios["roa"]              = safe_divide(df["net_income"],          df["total_assets"],          df.index)
    ratios["equity_multiplier"]= safe_divide(df["total_assets"],        df["stockholders_equity"],   df.index)

    logger.info(
        "Engineered %d ratio(s) for %d row(s).",
        len(ratios.columns) - 1, len(ratios),
    )

    return ratios


def load_and_run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load raw financials, engineer ratios, and save the result to CSV.

    Args:
        input_path: Path to the raw financials CSV file.
        output_path: Destination path for the engineered features CSV.

    Returns:
        DataFrame containing the engineered ratio columns plus ``ticker``.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist on disk.
        ValueError: If the loaded DataFrame is empty or cannot be parsed.
        OSError: If the output file cannot be written to ``output_path``.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            "Input file does not exist: %s." % input_path
        )

    logger.info("Loading raw financials from %s.", input_path.name)

    try:
        df = pd.read_csv(input_path)
    except pd.errors.ParserError as e:
        raise ValueError("Failed to parse %s: %s." % (input_path.name, e))

    if df.empty:
        raise ValueError("Input file is empty: %s." % input_path.name)

    logger.info("Loaded %d rows from %s.", len(df), input_path.name)

    df_ratios = engineer_ratios(df)

    try:
        df_ratios.to_csv(output_path, index=False)
    except OSError as e:
        logger.error(
            "Failed to save output to %s: %s.", output_path, e
        )
        raise

    logger.info(
        "Saved %d rows to %s.", len(df_ratios), output_path.name
    )

    return df_ratios


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    load_and_run()
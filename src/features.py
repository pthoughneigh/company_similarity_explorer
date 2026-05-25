from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

INPUT_PATH = Path(__file__).parent.parent / "data/financials.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data/features.csv"

def engineer_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 8 size-normalized financial ratios from raw metrics.

    Fills null rd_expense with 0 before computing rd_intensity.w
    Division by zero (e.g. negative equity) produces NaN, not inf.

    Args:
        df: Raw financials DataFrame with columns: ticker, revenue,
            net_income, total_assets, total_liabilities,
            operating_cash_flow, rd_expense, stockholders_equity.

    Returns:
        DataFrame with 8 ratio columns plus ticker.
    """
    df = df.copy()
    df["rd_expense"] = df["rd_expense"].fillna(0)

    ratios = pd.DataFrame()

    def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
        """Divide two series, replacing inf with NaN."""
        result = num.values / denom.values  # numpy division: 0/0 = nan, x/0 = inf
        return pd.Series(np.where(np.isinf(result), np.nan, result), index=df.index)

    ratios['ticker'] = df['ticker']
    ratios["profit_margin"] = safe_divide(df["net_income"], df["revenue"])
    ratios["debt_ratio"] = safe_divide(df["total_liabilities"], df["total_assets"])
    ratios["roe"] = safe_divide(df["net_income"], df["stockholders_equity"])
    ratios["asset_turnover"] = safe_divide(df["revenue"], df["total_assets"])
    ratios["ocf_margin"] = safe_divide(df["operating_cash_flow"], df["revenue"])
    ratios["rd_intensity"] = safe_divide(df["rd_expense"], df["revenue"])
    ratios["roa"] = safe_divide(df["net_income"], df["total_assets"])
    ratios["equity_multiplier"] = safe_divide(df["total_assets"], df["stockholders_equity"])

    return ratios

def load_and_run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load raw financials, engineer ratios, and save to CSV.

    Args:
        input_path: Path to financials.csv.
        output_path: Path to write features.csv.
    Returns:
        The engineered ratios DataFrame.
    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the loaded DataFrame is empty.
    """
    if not Path(input_path).exists():
        logger.error(f"{input_path} does not exist.")
        raise FileNotFoundError(f"{input_path} does not exist.")

    logger.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    if df.empty:
        logger.error(f"Could not load {input_path}.")
        raise ValueError(f"Could not load {input_path}.")
    df_ratios = engineer_ratios(df)
    logger.info(f"Processed {df_ratios.shape[0]} rows.")

    try:
        df_ratios.to_csv(output_path, index=False)
    except Exception as e:
        logger.error(f"Could not save in {output_path} - {e}")
        raise

    logger.info(f"Ratios made, data saved in {output_path}.")
    return df_ratios

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    load_and_run()
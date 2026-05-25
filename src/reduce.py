from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

INPUT_PATH = Path(__file__).parent.parent / "data/features.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data/features_reduced_PCA.csv"

PROFIT_MARGIN_CLIP_VALUE = 1.0
OCF_MARGIN_CLIP_VALUE = 5.0
RD_INTENSITY_CLIP_VALUE = 1.0
EQUITY_MULTIPLIER_CLIP_VALUE = 20.0

def load_features(input_path: Path) -> pd.DataFrame:
    """
    It loads features.csv, drops rows with any NaN values (the pre-revenue companies),
    and returns the DataFrame.

    Args:
        input_path: Path to features.csv
    Returns:
        Dataframe with dropped rows where all ratio columns are NaN.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse {input_path.name}: {e}")

    if df.empty:
        raise ValueError(f"{input_path.name} is empty.")

    num_rows = df.shape[0]
    df.dropna(axis=0, how='any', inplace=True)
    dropped = num_rows - df.shape[0]
    logger.info(f"Loaded {num_rows} rows from {input_path.name}. Dropped {dropped} NaN rows, {df.shape[0]} remaining.")
    return df


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the DataFrame, clips the extreme outlier values, and returns the clipped DataFrame.
    Args:
        df: Dataframe to clip.
    Returns:
        Dataframe with clipped values.
    """
    df = df.copy()
    clip_config = {
        'profit_margin': PROFIT_MARGIN_CLIP_VALUE,
        'ocf_margin': OCF_MARGIN_CLIP_VALUE,
        'rd_intensity': RD_INTENSITY_CLIP_VALUE,
        'equity_multiplier': EQUITY_MULTIPLIER_CLIP_VALUE,
    }

    missing = [col for col in clip_config if col not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found in DataFrame: {missing}")

    for col, upper in clip_config.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric, got {df[col].dtype}.")
        n_clipped = (df[col] > upper).sum()
        df[col] = df[col].clip(upper=upper)
        if n_clipped > 0:
            logger.info(f"Clipped {n_clipped} values in '{col}' at upper={upper}.")
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric features prior to PCA.
    
    Args:
        df: Input dataframe with raw numeric features. Should be free
            of NaNs before calling. Categorical columns are ignored.
    
    Returns:
        New dataframe of the same shape with numeric columns standardized.
        Original dataframe is not modified.
    
    Raises:
        ValueError: If df is empty or contains no numeric columns.
    """
    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty.")

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns to scale.")

    if df[numeric_cols].isnull().any().any():
        raise ValueError("NaN values detected in numeric columns. Handle them before scaling.")

    constant_cols = [col for col in numeric_cols if df[col].std() == 0]
    if constant_cols:
        logger.warning(f"Constant columns detected (zero variance), dropping before scaling: {constant_cols}.")
        numeric_cols = numeric_cols.drop(constant_cols)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"Scaled {len(numeric_cols)} numeric columns: {list(numeric_cols)}.")
    return df

def run_pca(df: pd.DataFrame, n_components: int=2) -> pd.DataFrame:
    """
    Run PCA on scaled numeric features and return the principal components.

    Args:
        df: Scaled input dataframe. Should have been passed through
            scale_features() before calling this.
        n_components: Number of principal components to retain.

    Returns:
        Dataframe of shape (n_rows, n_components) with columns
        named PC1, PC2, ... and the ticker column preserved as index.

    Raises:
        ValueError: If n_components exceeds the number of numeric columns.
    """
    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty.")

    numeric_cols = df.select_dtypes(include='number').columns

    if n_components > len(numeric_cols):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed "
            f"number of numeric columns ({len(numeric_cols)})."
        )

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df[numeric_cols])

    explained = pca.explained_variance_ratio_
    logger.info(
        f"PCA retained {n_components} components explaining "
        f"{explained.sum():.1%} of variance. "
        f"Per component: {[f'{v:.1%}' for v in explained]}."
    )

    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    result = pd.DataFrame(components, columns=pc_cols, index=df.index)

    if 'ticker' in df.columns:
        result.insert(0, 'ticker', df['ticker'].values)

    return result

def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the PCA output DataFrame to a CSV file.

    Args:
        df: DataFrame to save, typically the output of run_pca().
        output_path: Destination path for the CSV file.

    Raises:
        OSError: If the output directory does not exist or is not writable.
    """
    if not output_path.parent.exists():
        raise OSError(f"Output directory does not exist: {output_path.parent}")

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {df.shape[0]} rows to {output_path.name}.")

def load_and_run() -> pd.DataFrame:
    """
    Execute the full preprocessing and PCA pipeline.
    Loads raw features from disk, clips outliers, standardizes numeric
    columns, and reduces dimensionality via PCA. Each step is handled
    by a dedicated function; see those for individual error handling.

    Returns:
        DataFrame of shape (n_rows, n_components + 1) with principal
        component scores and ticker column preserved.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the data is empty or malformed at any stage.
    """
    logger.info("Starting pipeline.")
    try:
        df = load_features(INPUT_PATH)
        df = clip_features(df)
        df = scale_features(df)
        result = run_pca(df)
        save_features(result, OUTPUT_PATH)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        logger.error(f"Pipeline failed: {type(e).__name__}: {e}")
        raise

    logger.info(f"Pipeline complete. Output shape: {result.shape}.")
    return result

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    pca_df = load_and_run()
    print(pca_df)
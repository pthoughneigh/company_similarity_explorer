import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from config import (
    SECTOR_MAP,
    INPUT_PATH,
    PCA_OUTPUT_PATH,
    PCA_EXPLAINED_OUTPUT_PATH,
    TSNE_OUTPUT_PATH,
    MDS_OUTPUT_PATH,
    LDA_OUTPUT_PATH
)

logger = logging.getLogger(__name__)

PROFIT_MARGIN_CLIP_VALUE: float = 1.0
OCF_MARGIN_CLIP_VALUE: float = 5.0
RD_INTENSITY_CLIP_VALUE: float = 1.0
EQUITY_MULTIPLIER_CLIP_VALUE: float = 20.0


def load_features(input_path: Path) -> pd.DataFrame:
    """
    Load features from a CSV file, dropping rows with any NaN values.

    Pre-revenue companies typically have all ratio columns missing and are
    removed at this stage via row-wise NaN dropping.

    Args:
        input_path: Path to the CSV file to load.

    Returns:
        DataFrame with rows containing any NaN values removed.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist on disk.
        ValueError: If the file cannot be parsed as CSV, or if the resulting
            DataFrame is empty after loading.
    """
    if not input_path.exists():
        raise FileNotFoundError("Input file does not exist: %s" % input_path)

    try:
        df = pd.read_csv(input_path)
    except pd.errors.ParserError as e:
        raise ValueError("Failed to parse %s: %s" % (input_path.name, e))

    if df.empty:
        raise ValueError("Input file is empty: %s" % input_path.name)

    num_rows = df.shape[0]
    df.dropna(axis=0, how="any", inplace=True)
    dropped = num_rows - df.shape[0]

    logger.info(
        "Loaded %d rows from %s. Dropped %d NaN rows, %d remaining.",
        num_rows, input_path.name, dropped, df.shape[0],
    )

    return df


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip extreme outlier values in known ratio columns to fixed upper bounds.

    Only the columns defined in ``clip_config`` are affected. All other
    columns are left unchanged. The original DataFrame is not modified.

    Args:
        df: Input DataFrame containing the ratio columns to clip. Must
            include all columns defined in the internal clip configuration.

    Returns:
        Copy of the input DataFrame with outlier values clipped.

    Raises:
        KeyError: If any expected ratio column is missing from ``df``.
        TypeError: If any ratio column is not of a numeric dtype.
    """
    df = df.copy()

    clip_config = {
        "profit_margin": PROFIT_MARGIN_CLIP_VALUE,
        "ocf_margin": OCF_MARGIN_CLIP_VALUE,
        "rd_intensity": RD_INTENSITY_CLIP_VALUE,
        "equity_multiplier": EQUITY_MULTIPLIER_CLIP_VALUE,
    }

    missing = [col for col in clip_config if col not in df.columns]
    if missing:
        raise KeyError("Expected columns not found in DataFrame: %s" % missing)

    for col, upper in clip_config.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                "Column '%s' must be numeric, got %s." % (col, df[col].dtype)
            )
        n_clipped = (df[col] > upper).sum()
        df[col] = df[col].clip(upper=upper)
        if n_clipped > 0:
            logger.info(
                "Clipped %d value(s) in '%s' at upper=%.2f.",
                n_clipped, col, upper,
            )

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric features using zero mean and unit variance scaling.

    Non-numeric columns are left unchanged. Constant columns (zero variance)
    are dropped before scaling with a warning logged. The original DataFrame
    is not modified.

    Args:
        df: Input DataFrame with numeric features to standardize. Should be
            free of NaN values before calling this function.

    Returns:
        Copy of the input DataFrame with numeric columns standardized.

    Raises:
        ValueError: If ``df`` is empty, contains no numeric columns, or
            contains NaN values in any numeric column.
    """
    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty.")

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns to scale.")

    if df[numeric_cols].isnull().any().any():
        raise ValueError(
            "NaN values detected in numeric columns. "
            "Handle them before scaling."
        )

    constant_cols = [col for col in numeric_cols if df[col].std() == 0]
    if constant_cols:
        logger.warning(
            "Constant columns detected (zero variance), dropping before scaling: %s.",
            constant_cols,
        )
        numeric_cols = numeric_cols.drop(constant_cols)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logger.info(
        "Scaled %d numeric column(s): %s.",
        len(numeric_cols), list(numeric_cols),
    )

    return df


def add_sector_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map tickers to their corresponding sector labels and add as a new column.

    Looks up each ticker in ``SECTOR_MAP`` and appends the result as a
    ``sector`` column. Tickers not found in ``SECTOR_MAP`` will produce
    ``NaN`` in the ``sector`` column — a warning is logged in that case.

    Args:
        df: Input DataFrame containing a ``ticker`` column with stock
            ticker symbols. Must not be empty.

    Returns:
        Copy of the input DataFrame with an additional ``sector`` column
        containing the sector label for each ticker.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If ``df`` is empty or does not contain a ``ticker`` column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame, got %s." % type(df).__name__
        )
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if "ticker" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'ticker' column.")

    logger.debug("Adding sector labels to %d rows.", len(df))

    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP)

    unmapped = df["ticker"][df["sector"].isna()].unique().tolist()
    if unmapped:
        logger.warning(
            "%d ticker(s) not found in SECTOR_MAP and will be NaN: %s.",
            len(unmapped), unmapped,
        )

    logger.debug(
        "Sector labels added. Unique sectors found: %d.",
        df["sector"].nunique(),
    )

    return df


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a DataFrame to a CSV file at the specified path.

    Args:
        df: DataFrame to save.
        output_path: Destination path for the CSV file. The parent
            directory must already exist.

    Raises:
        OSError: If the output directory does not exist or is not writable.
    """
    if not output_path.parent.exists():
        raise OSError(
            "Output directory does not exist: %s" % output_path.parent
        )

    df.to_csv(output_path, index=False)

    logger.info("Saved %d rows to %s.", df.shape[0], output_path.name)


def run_pca(df: pd.DataFrame, numeric_cols: pd.Index, n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run PCA on scaled numeric features and return the principal components.

    Args:
        df: Scaled input DataFrame. Should have been passed through
            scale_features() before calling this. Must not contain
            NaN or infinite values.
        n_components: Number of principal components to retain. Typically, 2
            for visualization. Cannot exceed the number of numeric columns.
        numeric_cols: Index of numeric column names to use as model input.
            Typically obtained from ``df.select_dtypes(include="number").columns``.

    Returns:
        Tuple of two DataFrames:
            - Components DataFrame of shape ``(n_rows, n_components)`` with columns
              ``PC1``, ``PC2``, etc., with the ticker column preserved.
            - Explained variance DataFrame with columns ``component`` and
              ``explained_variance_ratio``, one row per component.

    Raises:
        ValueError: If ``df`` is empty, contains no numeric columns, or if
            ``n_components`` exceeds the number of available numeric columns.
    """
    logger.debug(
        "run_pca called | rows=%d, n_components=%d",
        len(df), n_components,
    )

    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns.")

    if n_components > len(numeric_cols):
        raise ValueError(
            "n_components (%d) cannot exceed number of numeric columns (%d)."
            % (n_components, len(numeric_cols))
        )

    logger.debug("Fitting PCA on %d columns: %s.", len(numeric_cols), list(numeric_cols))

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df[numeric_cols])

    explained = pca.explained_variance_ratio_

    pc_cols = ["PC%d" % (i + 1) for i in range(n_components)]
    explained_df = pd.DataFrame({
        "component": pc_cols,
        "explained_variance_ratio": explained,
    })

    logger.info("Per-component variance: %s.", [("%.1f%%" % (v * 100)) for v in explained])

    result = pd.DataFrame(components, columns=pc_cols, index=df.index)

    if "ticker" in df.columns:
        result.insert(0, "ticker", df["ticker"].values)

    logger.info(
        "run_pca complete | output shape=%s, columns=%s.",
        result.shape, list(result.columns),
    )

    return result, explained_df


def run_tsne(
    df: pd.DataFrame,
    numeric_cols: pd.Index,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run t-SNE on scaled numeric features and return embeddings in reduced space.

    Args:
        df: Scaled input DataFrame. Should have been passed through
            scale_features() before calling this. Must not contain
            NaN or infinite values.
        n_components: Dimensionality of the embedding space. Typically, 2
            for visualization. Values above 3 are rarely useful.
        perplexity: Controls the effective number of neighbors considered
            per point. Must be less than ``len(df)``. Typical range is
            [5, 50]; sklearn's default is 30; larger datasets can tolerate higher values.
        random_state: Seed for the random number generator. Note that
            results may still differ across sklearn versions or platforms.
        numeric_cols: Index of numeric column names to use as model input.
            Typically obtained from ``df.select_dtypes(include="number").columns``.

    Returns:
        DataFrame of shape ``(n_rows, n_components)`` with columns
        ``TSNE1``, ``TSNE2`` (and ``TSNE3`` if ``n_components=3``),
        with the ticker column preserved.

    Raises:
        ValueError: If ``perplexity >= len(df)`` — t-SNE requires at least
            ``perplexity + 1`` samples.
        ValueError: If ``df`` is empty or contains no numeric columns.
    """
    logger.debug(
        "run_tsne called | rows=%d, n_components=%d, perplexity=%d, random_state=%d.",
        len(df), n_components, perplexity, random_state,
    )

    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if perplexity <= 0 or perplexity >= len(df):
        raise ValueError(
            "perplexity (%d) must be greater than 0 and less than n_samples (%d)."
            % (perplexity, len(df))
        )

    logger.debug(
        "Fitting t-SNE on %d columns: %s.", len(numeric_cols), list(numeric_cols)
    )

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
    )
    components = tsne.fit_transform(df[numeric_cols])

    tsne_cols = ["TSNE%d" % (i + 1) for i in range(n_components)]
    result = pd.DataFrame(components, columns=tsne_cols, index=df.index)

    if "ticker" in df.columns:
        result.insert(0, "ticker", df["ticker"].values)

    logger.info(
        "run_tsne complete | output shape=%s, columns=%s.",
        result.shape, list(result.columns),
    )

    return result


def run_mds(
    df: pd.DataFrame,
    numeric_cols: pd.Index,
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run MDS on scaled numeric features and return embeddings in reduced space.

    Args:
        df: Scaled input DataFrame. Should have been passed through
            scale_features() before calling this. Must not contain
            NaN or infinite values.
        n_components: Dimensionality of the embedding space. Typically, 2
            for visualization. Values above 3 are rarely useful.
        random_state: Seed for the random number generator. Note that
            results may still differ across sklearn versions or platforms.
        numeric_cols: Index of numeric column names to use as model input.
            Typically obtained from ``df.select_dtypes(include="number").columns``.

    Returns:
        DataFrame of shape ``(n_rows, n_components)`` with columns
        ``MDS1``, ``MDS2`` (and ``MDS3`` if ``n_components=3``),
        with the ticker column preserved.

    Raises:
        ValueError: If ``df`` is empty or contains no numeric columns.
        ValueError: If ``n_components >= len(df)`` — MDS requires more
            samples than output dimensions.
    """
    logger.debug(
        "run_mds called | rows=%d, n_components=%d, random_state=%d.",
        len(df), n_components, random_state,
    )

    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if n_components >= len(df):
        raise ValueError(
            "n_components (%d) must be less than n_samples (%d)."
            % (n_components, len(df))
        )

    logger.debug(
        "Fitting MDS on %d columns: %s.", len(numeric_cols), list(numeric_cols)
    )

    mds = MDS(n_components=n_components, random_state=random_state)
    components = mds.fit_transform(df[numeric_cols])

    mds_cols = ["MDS%d" % (i + 1) for i in range(n_components)]
    result = pd.DataFrame(components, columns=mds_cols, index=df.index)

    if "ticker" in df.columns:
        result.insert(0, "ticker", df["ticker"].values)

    logger.info(
        "run_mds complete | output shape=%s, columns=%s.",
        result.shape, list(result.columns),
    )

    return result

def run_lda(df: pd.DataFrame, numeric_cols: pd.Index, n_components: int = 2) -> pd.DataFrame:
    """
    Run LDA on scaled numeric features and return embeddings in reduced space.

    Unlike PCA, LDA uses class labels (``sector``) to find the directions
    that maximally separate the classes, rather than directions of maximum
    variance. The number of meaningful components is bounded by ``k - 1``
    where ``k`` is the number of unique sectors.

    Args:
        df: Scaled input DataFrame containing a ``sector`` column for class
            labels and a ``ticker`` column for identification. Should have
            been passed through ``scale_features()`` and
            ``add_sector_labels()`` before calling this. Must not contain
            NaN or infinite values.
        n_components: Dimensionality of the embedding space. Cannot exceed
            ``k - 1`` where ``k`` is the number of unique classes in the
            ``sector`` column. Typically, 2 for visualization.
        numeric_cols: Index of numeric column names to use as model input.
            Typically obtained from ``df.select_dtypes(include="number").columns``.

    Returns:
        DataFrame of shape ``(n_rows, n_components)`` with columns
        ``LDA1``, ``LDA2``, etc., with the ``ticker`` column preserved.

    Raises:
        ValueError: If ``df`` is empty or contains no numeric columns.
        ValueError: If the ``sector`` column is missing or contains
            fewer than 2 unique classes.
        ValueError: If ``n_components`` exceeds ``k - 1`` where ``k``
            is the number of unique classes in ``sector``.
    """
    logger.debug(
        "run_lda called | rows=%d, n_components=%d.",
        len(df), n_components,
    )

    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if "sector" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'sector' column. "
            "Call add_sector_labels() before run_lda()."
        )

    n_classes = df["sector"].nunique()
    if n_classes < 2:
        raise ValueError(
            "LDA requires at least 2 unique classes in 'sector', found %d."
            % n_classes
        )

    if n_components > n_classes - 1:
        raise ValueError(
            "n_components (%d) cannot exceed k - 1 = %d "
            "where k=%d is the number of unique sector."
            % (n_components, n_classes - 1, n_classes)
        )

    x = df[numeric_cols]
    y = df["sector"]

    logger.debug(
        "Fitting LDA on %d columns across %d classes: %s.",
        len(numeric_cols), n_classes, list(numeric_cols),
    )

    lda = LDA(n_components=n_components)
    components = lda.fit_transform(x, y)

    explained = lda.explained_variance_ratio_
    logger.info(
        "LDA complete | %d components explain %.1f%% of variance. "
        "Per component: %s.",
        n_components,
        explained.sum() * 100,
        [("%.1f%%" % (v * 100)) for v in explained],
    )

    lda_cols = ["LDA%d" % (i + 1) for i in range(n_components)]
    result = pd.DataFrame(components, columns=lda_cols, index=df.index)

    if "ticker" in df.columns:
        result.insert(0, "ticker", df["ticker"].values)

    logger.info(
        "run_lda complete | output shape=%s, columns=%s.",
        result.shape, list(result.columns),
    )

    return result

def load_and_run() -> tuple[pd.DataFrame, ...]:
    """
    Execute the full preprocessing, PCA, t-SNE, MDS and LDA pipeline.

    Loads raw features from disk, clips outliers, standardizes numeric
    columns, adds sector labels, and reduces dimensionality via PCA, t-SNE, MDS and LDA.
    Each step is handled by a dedicated function; see those for individual
    error handling.

    Returns:
        Tuple of four DataFrames containing PCA components, t-SNE
        embeddings, MDS embeddings, and LDA embeddings respectively,
        each with the ticker column preserved.
        - result_pca: Full PCA DataFrame with all ``n_components`` columns.
        Only ``PC1`` and ``PC2`` are written to disk; the full DataFrame
        is returned to the caller

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the data is empty or malformed at any pipeline stage.
        KeyError: If expected columns are missing from the input data.
        TypeError: If column types are incompatible with any pipeline step.
    """
    logger.info("Pipeline starting.")

    try:
        df = load_features(INPUT_PATH)
        df = clip_features(df)
        df = scale_features(df)

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("DataFrame contains no numeric columns.")

        result_pca, result_explained = run_pca(df, numeric_cols, n_components=len(numeric_cols))
        result_pca_saved = result_pca[["ticker", "PC1", "PC2"]]
        result_tsne = run_tsne(df, numeric_cols)
        result_mds = run_mds(df, numeric_cols)

        df = add_sector_labels(df)
        results_lda = run_lda(df, numeric_cols)

        save_features(result_pca_saved, PCA_OUTPUT_PATH)
        save_features(result_explained, PCA_EXPLAINED_OUTPUT_PATH)
        save_features(result_tsne, TSNE_OUTPUT_PATH)
        save_features(result_mds, MDS_OUTPUT_PATH)
        save_features(results_lda, LDA_OUTPUT_PATH)

    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        logger.error("Pipeline failed: %s: %s.", type(e).__name__, e)
        raise

    logger.info(
        "PCA pipeline complete | full shape=%s, saved shape=%s.",
        result_pca.shape, result_pca_saved.shape,
    )
    logger.info("t-SNE pipeline complete | output shape=%s.", result_tsne.shape)
    logger.info("MDS pipeline complete | output shape=%s.", result_mds.shape)
    logger.info("LDA pipeline complete | output shape=%s.", results_lda.shape)
    return result_pca, result_tsne, result_mds, results_lda

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    pca_df, tsne_df, mds_df, lda_df = load_and_run()

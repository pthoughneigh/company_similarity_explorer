from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging
from config import SECTOR_MAP

logger = logging.getLogger(__name__)

def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    method_name: str,
) -> go.Figure:
    """
    Create a 2-D scatter plot of dimensionality-reduction embeddings coloured by sector.

    Maps each ticker to its sector via ``SECTOR_MAP`` and passes the result
    to Plotly Express as the colour dimension. Tickers absent from
    ``SECTOR_MAP`` will appear as ``NaN`` in the legend — a warning is
    logged in that case.

    Args:
        df: Input DataFrame containing at least a ``ticker`` column and the
            two embedding columns identified by ``x_col`` and ``y_col``.
            Must not be empty.
        x_col: Name of the column to plot on the x-axis (e.g. ``"PC1"``).
        y_col: Name of the column to plot on the y-axis (e.g. ``"PC2"``).
        method_name: Human-readable name of the reduction method used as
            the figure title (e.g. ``"PCA"``, ``"t-SNE"``).

    Returns:
        Plotly ``Figure`` object containing the scatter plot, ready for
        ``fig.show()`` or ``fig.write_html()``.

    Raises:
        ValueError: If ``df`` is empty or does not contain a ``ticker``
            column, ``x_col``, or ``y_col``.
    """
    logger.debug(
        "plot_scatter called | rows=%d, x_col=%s, y_col=%s, method_name=%s.",
        len(df), x_col, y_col, method_name,
    )

    if df.empty:
        raise ValueError("DataFrame is empty.")
    if "ticker" not in df.columns:
        raise ValueError("DataFrame must contain a 'ticker' column.")
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError("Expected column '%s' not found in DataFrame." % col)

    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP)

    unmapped = df["ticker"][df["sector"].isna()].unique().tolist()
    if unmapped:
        logger.warning(
            "%d ticker(s) not found in SECTOR_MAP and will appear as NaN: %s.",
            len(unmapped), unmapped,
        )

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="sector",
        title="%s projection" % method_name,
    )
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)

    logger.info(
        "plot_scatter complete | method=%s, n_points=%d, n_sectors=%d.",
        method_name, len(df), df["sector"].nunique(),
    )

    return fig


def plot_scree(explained_variance_path: Path) -> go.Figure:
    """
    Create a scree bar chart showing the variance contribution of each PCA component.

    Reads the explained variance CSV produced by ``run_pca()``, converts the
    ratios to percentages, and plots them as a bar chart ordered by component.

    Args:
        explained_variance_path: Path to the CSV file containing PCA explained
        variance data. Must contain ``component`` and ``explained_variance_ratio`` columns.

    Returns:
        Plotly ``Figure`` object containing the scree bar chart, ready for
        ``fig.show()`` or ``fig.write_html()``.

    Raises:
        FileNotFoundError: If ``explained_variance_path`` does not exist on disk.
        ValueError: If the file cannot be parsed as CSV, is empty, or is missing
        the required ``component`` or ``explained_variance_ratio`` columns.
    """

    logger.debug("plot_scree called | path=%s.", explained_variance_path)

    if not explained_variance_path.exists():
        raise FileNotFoundError(
            "Explained variance file does not exist: %s" % explained_variance_path
        )

    try:
        pca_variance = pd.read_csv(explained_variance_path, header=0)
    except pd.errors.ParserError as e:
        raise ValueError(
            "Failed to parse %s: %s" % (explained_variance_path.name, e)
        )

    if pca_variance.empty:
        raise ValueError(
            "Explained variance file is empty: %s" % explained_variance_path.name
        )

    required_cols = {"component", "explained_variance_ratio"}
    missing = required_cols - set(pca_variance.columns)
    if missing:
        raise ValueError(
            "Expected columns not found in %s: %s"
            % (explained_variance_path.name, sorted(missing))
        )

    pca_variance["explained_variance_ratio"] = pca_variance["explained_variance_ratio"] * 100

    fig = px.bar(
        data_frame=pca_variance,
        x="component",
        y="explained_variance_ratio",
        title="PCA Explained Variance"
    )

    fig.update_layout(yaxis_title="Variance Contribution [%]")

    logger.info(
        "plot_scree complete | n_components=%d, total_variance=%.1f%%.",
        len(pca_variance),
        pca_variance["explained_variance_ratio"].sum(),
    )

    return fig

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent / 'data/features_reduced_PCA.csv')
    fig = plot_scatter(df, "PC1", "PC2", "PCA")
    fig.show()

    fig2 = plot_scree(Path(__file__).parent.parent / 'data/features_reduced_explained_PCA.csv')
    fig2.show()
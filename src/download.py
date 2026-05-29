import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sec_edgar_downloader import Downloader

from config import TICKERS


logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data/filings"
DOWNLOADER_COMPANY = "MyResearch"
DOWNLOADER_EMAIL = "me@example.com"
MAX_WORKERS = 5


def download_ticker(ticker: str, output_dir: Path) -> tuple[str, bool]:
    """
    Download the most recent 10-K filing for a single ticker.

    Args:
        ticker: Stock ticker symbol to download the filing for.
        output_dir: Directory where the filing will be saved.

    Returns:
        Tuple of ``(ticker, success)`` where ``success`` is ``True`` if the
        download completed without error, ``False`` otherwise.
    """
    try:
        dl = Downloader(DOWNLOADER_COMPANY, DOWNLOADER_EMAIL, output_dir)
        dl.get("10-K", ticker, limit=1)
        logger.info("%s — OK.", ticker)
        return ticker, True
    except Exception as e:
        logger.error("%s — failed: %s: %s.", ticker, type(e).__name__, e)
        return ticker, False


def download_filings(
    tickers: list[str],
    output_dir: Path = OUTPUT_DIR,
    max_workers: int = MAX_WORKERS,
) -> dict[str, bool]:
    """
    Download the most recent 10-K filing for each ticker in parallel.

    Args:
        tickers: List of stock ticker symbols to download filings for.
        output_dir: Directory where all filings will be saved.
        max_workers: Maximum number of parallel download threads.

    Returns:
        Dictionary mapping each ticker to a boolean indicating whether
        its download succeeded.
    """
    results: dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_ticker, ticker, output_dir): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker, success = future.result()
            results[ticker] = success

    failed = [ticker for ticker, success in results.items() if not success]
    if failed:
        logger.warning(
            "%d ticker(s) failed to download: %s.", len(failed), failed
        )
    else:
        logger.info("All %d ticker(s) downloaded successfully.", len(tickers))

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    download_filings(TICKERS, OUTPUT_DIR)
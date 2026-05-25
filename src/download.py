import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TICKERS
from sec_edgar_downloader import Downloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def download_ticker(ticker: str, output_dir: str) -> tuple[str, bool]:
    """
    Download the most recent 10-K for a single ticker.
    Returns (ticker, success).
    """
    try:
        dl = Downloader("MyResearch", "me@example.com", output_dir)
        dl.get("10-K", ticker, limit=1)
        logger.info(f"{ticker} — OK")
        return ticker, True
    except Exception as e:
        logger.error(f"{ticker} — FAILED: {e}")
        return ticker, False


def download_filings(tickers: list[str], output_dir: str, max_workers: int = 5) -> dict[str, bool]:
    """
    Download the most recent 10-K for each ticker in parallel.
    Returns a dict of ticker -> success (True/False).
    """
    results: dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_ticker, ticker, output_dir): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker, success = future.result()
            results[ticker] = success

    failed = [t for t, ok in results.items() if not ok]
    if failed:
        logger.warning(f"Failed tickers: {failed}")
    else:
        logger.info("All tickers downloaded successfully.")

    return results


if __name__ == "__main__":
    download_filings(TICKERS, "../data/filings")
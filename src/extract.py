import csv
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup

from config import TICKERS


logger = logging.getLogger(__name__)

FILING_BASE = Path(__file__).parent.parent / "data/filings/sec-edgar-filings"
OUTPUT_PATH = Path(__file__).parent.parent / "data/financials.csv"

XBRL_TAGS = {
    "revenue": [
        "us-gaap:regulatedandunregulatedoperatingrevenue",
        "us-gaap:RegulatedAndUnregulatedOperatingRevenue",
        "us-gaap:revenues",
        "us-gaap:Revenues",
        "us-gaap:revenuefromcontractwithcustomerexcludingassessedtax",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:revenuefromcontractwithcustomerincludingassessedtax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:salesrevenuenet",
        "us-gaap:SalesRevenueNet",
        "us-gaap:interestincomeexpensenet",
        "us-gaap:InterestIncomeExpenseNet",
        "us-gaap:oilandgasrevenue",
        "us-gaap:OilAndGasRevenue",
        "us-gaap:naturalgasproductionrevenue",
        "us-gaap:NaturalGasProductionRevenue",
        "us-gaap:oilandcondensaterevenue",
        "us-gaap:OilAndCondensateRevenue",
        "us-gaap:revenuesexcludinginterestanddividends",
        "us-gaap:RevenuesExcludingInterestAndDividends",
        "us-gaap:resultsofoperationsrevenuefromoilandgasproducingactivities",
        "us-gaap:ResultsOfOperationsRevenueFromOilAndGasProducingActivities",
        "us-gaap:operatingleaseleaseIncome",
        "us-gaap:OperatingLeaseLeaseIncome",
        "apa:RevenuesAndOther",
        "apa:revenuesandother",
    ],
    "net_income": [
        "us-gaap:netincomeloss",
        "us-gaap:NetIncomeLoss",
        "us-gaap:profitloss",
        "us-gaap:ProfitLoss",
    ],
    "total_assets": [
        "us-gaap:assets",
        "us-gaap:Assets",
    ],
    "total_liabilities": [
        "us-gaap:liabilities",
        "us-gaap:Liabilities",
    ],
    "operating_cash_flow": [
        "us-gaap:netcashprovidedbyusedinoperatingactivities",
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:netcashprovidedbyusedinoperatingactivitiescontinuingoperations",
        "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "rd_expense": [
        "us-gaap:researchanddevelopmentexpense",
        "us-gaap:ResearchAndDevelopmentExpense",
        "us-gaap:researchanddevelopmentexpenseexcludingacquiredinprocesscost",
        "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "stockholders_equity": [
        "us-gaap:stockholdersequity",
        "us-gaap:StockholdersEquity",
        "us-gaap:stockholdersequityincludingportionattributabletononcontrollinginterest",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}


def _scale_inline_value(value: float, decimals: str) -> float:
    """
    Normalize an inline XBRL value to millions using its ``decimals`` attribute.

    XBRL inline values are reported at varying scales. This function converts
    them to a consistent millions scale based on the ``decimals`` attribute
    found on the ``ix:nonFraction`` tag.

    Args:
        value: Raw numeric value extracted from the tag.
        decimals: The ``decimals`` attribute string from the tag (e.g. ``"-6"``).

    Returns:
        Value scaled to millions.
    """
    if decimals in ("0", "1", "2"):
        return value / 1_000_000
    if decimals in ("-3", "-4", "-5"):
        return value / 1_000
    if decimals == "-9":
        return value * 1_000
    return value


def load_filing(ticker: str) -> BeautifulSoup | None:
    """
    Load and parse the 10-K full-submission file for a given ticker.

    Searches for ``full-submission.txt`` under the ticker's 10-K filing
    directory. If multiple filings exist, the first match is used.

    Args:
        ticker: Stock ticker symbol used to locate the filing directory.

    Returns:
        Parsed ``BeautifulSoup`` object for the filing, or ``None`` if the
        file is not found or cannot be parsed.
    """
    pattern = FILING_BASE / ticker / "10-K"
    matches = list(pattern.glob("*/full-submission.txt"))

    if not matches:
        logger.error("%s — no full-submission.txt found.", ticker)
        return None

    filing_path = matches[0]
    logger.info("%s — loading from %s.", ticker, filing_path)

    try:
        with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return BeautifulSoup(content, "lxml-xml")
    except Exception as e:
        logger.error("%s — failed to parse filing: %s.", ticker, e)
        return None


def extract_xbrl_value(
    soup: BeautifulSoup,
    tag_candidates: list[str],
) -> float | None:
    """
    Extract the first matching consolidated XBRL value from a filing.

    Tries each candidate tag in order, checking both plain XBRL and inline
    XBRL (``ix:nonFraction``) formats. Skips values attached to segment
    contexts (i.e. non-consolidated figures).

    Args:
        soup: Parsed BeautifulSoup object of the full filing.
        tag_candidates: Ordered list of XBRL tag names to try. The first
            tag that yields a valid numeric value is used.

    Returns:
        First numeric value found in millions, or ``None`` if no tag matched.
    """
    for tag_name in tag_candidates:
        # plain XBRL
        for tag in soup.find_all(tag_name):
            context_id = tag.get("contextref", "")
            context_el = soup.find("xbrli:context", {"id": context_id})
            if context_el and context_el.find("xbrli:segment"):
                continue
            try:
                return float(tag.text.strip().replace(",", ""))
            except ValueError:
                continue

        # inline XBRL
        for tag in soup.find_all("ix:nonFraction", {"name": tag_name}):
            context_id = tag.get("contextRef", "")
            context_el = soup.find("xbrli:context", {"id": context_id})
            if context_el and context_el.find("xbrli:segment"):
                continue
            try:
                value = float(tag.text.strip().replace(",", ""))
                decimals = tag.get("decimals", "-6")
                return _scale_inline_value(value, decimals)
            except ValueError:
                continue

    return None


def extract_xbrl_value_sum_segments(
    soup: BeautifulSoup,
    tag_candidates: list[str],
) -> float | None:
    """
    Fallback extractor that sums segment-level XBRL values into a consolidated total.

    Used when no consolidated (non-segment) value exists for a tag. Groups
    inline ``ix:nonFraction`` values by ``contextRef``, then sums one value
    per context to approximate the consolidated figure.

    Args:
        soup: Parsed BeautifulSoup object of the full filing.
        tag_candidates: Ordered list of XBRL tag names to try.

    Returns:
        Summed value in millions across segments, or ``None`` if no usable
        segment values were found.
    """
    for tag_name in tag_candidates:
        inline_tags = soup.find_all("ix:nonFraction", {"name": tag_name})
        if not inline_tags:
            continue

        groups: dict[str, list[float]] = defaultdict(list)

        for tag in inline_tags:
            context_id = tag.get("contextRef", "")
            context_el = soup.find("xbrli:context", {"id": context_id})
            if not context_el or not context_el.find("xbrli:segment"):
                continue
            try:
                value = float(tag.text.strip().replace(",", ""))
                decimals = tag.get("decimals", "-6")
                groups[context_id].append(_scale_inline_value(value, decimals))
            except ValueError:
                continue

        if not groups:
            continue

        total = sum(values[0] for values in groups.values() if values)
        if total != 0:
            return total

    return None


def extract_financials(ticker: str) -> dict[str, float | str | None]:
    """
    Extract core financial figures for a ticker from its 10-K filing.

    Tries the primary extractor first, then falls back to segment summation
    if no consolidated value is found. If ``total_liabilities`` cannot be
    extracted by either method, it is derived from ``total_assets`` minus
    ``stockholders_equity``.

    Args:
        ticker: Stock ticker symbol to extract financials for.

    Returns:
        Dictionary mapping metric names to their extracted values, with
        ``ticker`` included as the first key. Returns an empty dict if
        the filing cannot be loaded.
    """
    soup = load_filing(ticker)
    if soup is None:
        return {}

    results: dict[str, float | str | None] = {"ticker": ticker}

    for metric, tag_candidates in XBRL_TAGS.items():
        value = extract_xbrl_value(soup, tag_candidates)
        if value is None:
            value = extract_xbrl_value_sum_segments(soup, tag_candidates)
        if value is None:
            logger.warning("%s — could not extract '%s'.", ticker, metric)
        results[metric] = value

    if results.get("total_liabilities") is None:
        assets = results.get("total_assets")
        equity = results.get("stockholders_equity")
        if assets is not None and equity is not None:
            results["total_liabilities"] = assets - equity
            logger.info(
                "%s — total_liabilities derived from total_assets - stockholders_equity.",
                ticker,
            )

    return results


def save_to_csv(records: list[dict], output_path: Path) -> None:
    """
    Save a list of financial record dictionaries to a CSV file.

    Args:
        records: List of dictionaries where each dict represents one ticker's
            extracted financials. All dicts must share the same keys.
        output_path: Destination path for the CSV file.

    Raises:
        OSError: If the file cannot be written to ``output_path``.
    """
    if not records:
        logger.warning("No records to save — output file will not be written.")
        return

    fieldnames = list(records[0].keys())

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info("Saved %d records to %s.", len(records), output_path)
    except OSError as e:
        logger.error("Failed to save CSV to %s: %s.", output_path, e)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    records = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(extract_financials, ticker): ticker
            for ticker in TICKERS
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)

    for record in records:
        print(record)

    save_to_csv(records, OUTPUT_PATH)
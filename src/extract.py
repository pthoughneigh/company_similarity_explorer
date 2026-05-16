import logging
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TICKERS

logger = logging.getLogger(__name__)

FILING_BASE = Path("data/filings/sec-edgar-filings")

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


def load_filing(ticker: str) -> BeautifulSoup | None:
    """
    Load and parse the full-submission.txt for a given ticker.
    Returns a BeautifulSoup object or None if file not found.
    """
    pattern = FILING_BASE / ticker / "10-K"
    matches = list(pattern.glob("*/full-submission.txt"))

    if not matches:
        logger.error(f"{ticker} — no full-submission.txt found")
        return None

    filing_path = matches[0]
    logger.info(f"{ticker} — loading from {filing_path}")

    try:
        with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return BeautifulSoup(content, "lxml-xml")
    except Exception as e:
        logger.error(f"{ticker} — failed to parse filing: {e}")
        return None


def extract_xbrl_value(
    soup: BeautifulSoup,
    tag_candidates: list[str]
) -> float | None:
    """
    Try each candidate XBRL tag in order, handling both plain and inline XBRL.
    Returns the first numeric value found, or None if no tag matched.
    """
    for tag_name in tag_candidates:
        # plain XBRL
        tags = soup.find_all(tag_name)
        for tag in tags:
            context_id = tag.get("contextref", "")
            context_el = soup.find("xbrli:context", {"id": context_id})
            if context_el and context_el.find("xbrli:segment"):
                continue
            try:
                return float(tag.text.strip().replace(",", ""))
            except ValueError:
                continue

        # inline XBRL
        inline_tags = soup.find_all("ix:nonFraction", {"name": tag_name})
        for tag in inline_tags:
            context_id = tag.get("contextRef", "")
            context_el = soup.find("xbrli:context", {"id": context_id})
            if context_el and context_el.find("xbrli:segment"):
                continue
            try:
                value = float(tag.text.strip().replace(",", ""))
                decimals = tag.get("decimals", "-6")
                if decimals in ("0", "1", "2"):
                    value = value / 1_000_000
                elif decimals in ("-3", "-4", "-5"):
                    value = value / 1_000
                elif decimals == "-9":
                    value = value * 1_000
                return value
            except ValueError:
                continue

    return None

def extract_xbrl_value_sum_segments(
    soup: BeautifulSoup,
    tag_candidates: list[str]
) -> float | None:
    """
    Fallback: sum segment values for a tag when no consolidated value exists.
    Takes the first context period found and sums all segments for that period.
    """
    for tag_name in tag_candidates:
        inline_tags = soup.find_all("ix:nonFraction", {"name": tag_name})
        if not inline_tags:
            continue

        # group by period (contextRef), pick the group with most entries
        from collections import defaultdict
        groups: dict[str, list[float]] = defaultdict(list)

        for tag in inline_tags:
            context_id = tag.get("contextRef", "")
            context_el = soup.find("xbrli:context", {"id": context_id})
            if not context_el:
                continue
            # only sum leaf segments (has segment but no nested segment of segment)
            if not context_el.find("xbrli:segment"):
                continue
            try:
                value = float(tag.text.strip().replace(",", ""))
                decimals = tag.get("decimals", "-6")
                if decimals in ("0", "1", "2"):
                    value = value / 1_000_000
                elif decimals == "-3":
                    value = value / 1_000
                elif decimals == "-9":
                    value = value * 1_000
                groups[context_id].append(value)
            except ValueError:
                continue

        if not groups:
            continue

        # pick the period context that appears most (likely the annual segments)
        # but we want unique context_ids summed, not duplicates
        # find all unique context periods by stripping segment dimension
        # simplest: return sum of all unique-context values for largest group count
        total = sum(v for values in groups.values() for v in values[:1])
        if total != 0:
            return total

    return None

def extract_financials(ticker: str) -> dict[str, float | str | None]:
    """
    Extract core financial figures for a ticker from its 10-K filing.
    Returns a dict of metric name -> value (or None if not found).
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
            logger.warning(f"{ticker} — could not extract '{metric}'")
        results[metric] = value

    # fallback: compute total_liabilities from total_assets - stockholders_equity
    if results.get("total_liabilities") is None:
        assets = results.get("total_assets")
        equity = results.get("stockholders_equity")
        if assets is not None and equity is not None:
            results["total_liabilities"] = assets - equity
            logger.info(f"{ticker} — total_liabilities computed from assets - equity")

    return results

def save_to_csv(records: list[dict], output_path: str) -> None:
    """
    Save a list of financial records to a CSV file.
    """
    if not records:
        logger.warning("No records to save.")
        return

    fieldnames = list(records[0].keys())

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info(f"Saved {len(records)} records to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    records = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(extract_financials, ticker): ticker for ticker in TICKERS}
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)

    for record in records:
        print(record)

    save_to_csv(records, "data/financials.csv")
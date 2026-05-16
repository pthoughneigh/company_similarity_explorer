import logging
import pandas as pd
from src.extract import extract_financials
from concurrent.futures import ThreadPoolExecutor, as_completed

PATCH_TICKERS = [
    "APA"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # load existing CSV
    df = pd.read_csv("data/financials.csv")

    # re-extract patch tickers
    records = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_financials, t): t for t in PATCH_TICKERS}
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)

    # upsert: replace rows for patched tickers
    patch_df = pd.DataFrame(records)
    df = df[~df["ticker"].isin(PATCH_TICKERS)]
    df = pd.concat([df, patch_df], ignore_index=True)
    df = df.sort_values("ticker").reset_index(drop=True)

    df.to_csv("data/financials.csv", index=False)
    print(f"Patched {len(records)} tickers. CSV now has {len(df)} rows.")
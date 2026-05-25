# Feature Engineering Notes
## Company Similarity Explorer — `src/features.py`

Raw financial numbers (revenue, assets, net income) are not comparable across companies
because they scale with company size. A $1B profit means something different for Apple
than for a small-cap. These 8 ratios normalize everything so PCA and clustering compare
**business models**, not company size.

--- 

## Ratios

### 1. Profit Margin
**Question:** Out of every $1 of revenue, how much is left as profit?

```
profit_margin = net_income / revenue
```

- AAPL ≈ 0.27 → keeps 27 cents of every dollar
- Below 0: company is losing money
- Above 1.0: red flag — likely a data artifact (see AUR)

---

### 2. Debt Ratio
**Question:** What fraction of the company's assets are funded by debt?

```
debt_ratio = total_liabilities / total_assets
```

- 0.5 → half the company is borrowed money
- Above 1.0 → liabilities exceed assets (e.g. airlines, heavily leveraged firms)
- Banks naturally run high (0.9+) — that's their business model

---

### 3. Return on Equity (ROE)
**Question:** For every $1 shareholders have invested, how much profit did the company generate?

```
roe = net_income / stockholders_equity
```

- Higher is better for shareholders
- Can be artificially inflated by high debt (less equity in denominator)
- Negative equity companies produce meaningless values → stored as NaN

---

### 4. Asset Turnover
**Question:** For every $1 of assets the company owns, how much revenue does it generate?

```
asset_turnover = revenue / total_assets
```

- Retailers (WMT, COST) are high — assets work hard, turn over fast
- Utilities and REITs are low — lots of infrastructure, slow revenue
- Pre-revenue companies (IONQ ≈ 0.02) are near zero

---

### 5. Operating Cash Flow Margin (OCF Margin)
**Question:** Out of every $1 of revenue, how much becomes real cash from operations?

```
ocf_margin = operating_cash_flow / revenue
```

- Different from profit margin — ignores non-cash accounting items
- A company can be profitable on paper but cash-flow negative
- More reliable signal of business health than net income alone

---

### 6. R&D Intensity
**Question:** Out of every $1 of revenue, how much is spent on R&D?

```
rd_intensity = rd_expense / revenue
```

- `rd_expense` null → filled with 0 before computing (financials/retail/energy don't report R&D)
- Pre-revenue companies (IONQ ≈ 2.35) spend more on R&D than they earn
- Drug companies and semiconductors tend to be high

---

### 7. Return on Assets (ROA)
**Question:** For every $1 of assets, how much profit does the company generate?

```
roa = net_income / total_assets
```

- Similar to asset_turnover but measures profit, not revenue
- Asset-light businesses (software) tend to be high
- Capital-intensive businesses (utilities, industrials) tend to be low

---

### 8. Equity Multiplier
**Question:** How many dollars of assets does the company control per $1 of shareholder equity?

```
equity_multiplier = total_assets / stockholders_equity
```

- Higher = more leverage (more debt relative to equity)
- Banks are extreme (10x+) — that's their model
- Equity multiplier × ROA = ROE (this is the DuPont identity)

---

## Data Quality Notes

| Issue | How handled |
|---|---|
| `rd_expense` null (244 companies) | Filled with 0 — these companies don't do R&D |
| 6 pre-revenue companies (NNE, LTBR, QS, OKLO, PCVX, LAC) | Revenue-denominator ratios are NaN — decide in `reduce.py` whether to drop or impute |
| Division by zero | `safe_divide()` replaces `inf` with `NaN` |
| Extreme outliers (AUR, IONQ, SERV) | Expected for pre-revenue or micro-revenue companies — clip or drop before PCA |

---

## DuPont Identity (bonus)
ROE can be decomposed into three drivers:

```
ROE = profit_margin × asset_turnover × equity_multiplier
```

This means a company can have high ROE by:
- Making fat margins (luxury brands)
- Turning assets over fast (retailers)
- Using lots of leverage (banks)

Useful for explaining cluster differences in the agent layer.
from pathlib import Path

# ---------------------------------------------------------------------------
# Sector definitions and ticker registry
# ---------------------------------------------------------------------------

SECTORS: dict[str, list[str]] = {
    "technology": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "AMD", "CRM", "ORCL", "IBM",
        "ADBE", "QCOM", "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC", "NOW", "SNOW",
        "PANW", "CRWD", "ZS", "DDOG", "MDB", "NET", "TEAM", "OKTA", "UBER", "LYFT",
        "ABNB", "DASH", "RBLX", "U", "TTWO", "EA", "NTAP", "WDC", "STX", "HPQ",
        "HPE", "DELL", "CSCO", "FFIV", "AKAM", "CDW", "CTSH",
        "EPAM", "FLUT", "PAYC", "PCTY", "ADP", "PAYX", "FIS", "FISV", "GPN",
        "MA", "V", "PYPL", "AFRM", "BILL", "DOCU", "ZM", "RNG",
        "IONQ", "QBTS", "RGTI",
    ],
    "finance": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "PNC", "TFC",
        "COF", "SCHW", "ICE", "CME", "CB", "AON", "TRV", "MET", "PRU",
        "ALL", "AFL", "MTB", "CFG", "HBAN", "RF", "KEY", "FITB", "STT", "BK",
        "NTRS", "ALLY", "SYF", "NDAQ", "CBOE", "MKTX", "LPLA", "RJF", "EVR",
        "LAZ", "HIG", "LNC", "UNM", "PFG", "VOYA", "FNF", "FAF", "RLI", "WRB",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "HAL",
        "OXY", "DVN", "APA", "BKR", "CTRA", "EQT", "RRC", "AR",
        "NOV", "FTI", "SM", "MTDR", "WHD", "GTLS", "DNOW",
        "OKLO", "NNE", "LTBR", "SMR", "UUUU", "UEC", "URG", "USEG",
    ],
    "healthcare": [
        "JNJ", "PFE", "UNH", "ABT", "MRK", "TMO", "DHR", "BMY", "AMGN", "GILD",
        "CVS", "CI", "HCA", "ISRG", "BSX", "EW", "ZBH", "BAX", "BDX", "VRTX",
        "REGN", "BIIB", "ILMN", "IQV", "CRL", "IDXX", "MTD", "WAT", "A",
        "STE", "TFX", "PODD", "DXCM", "ALGN", "RMD", "COO", "SYK", "ZTS",
        "MRNA", "INCY", "ALNY", "NTRA", "PCVX", "RARE", "MCK", "CAH", "HSIC", "PRGO", "JAZZ",
    ],
    "consumer_discretionary": [
        "AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "TJX",
        "EBAY", "ROST", "DLTR", "DG", "YUM", "QSR", "CMG", "HLT", "MAR", "MGM",
        "LVS", "WYNN", "RCL", "CCL", "NCLH", "F", "GM", "TSLA", "BWA", "LEA",
        "PHM", "LEN", "DHI", "TOL", "NVR", "POOL", "WSM", "RH", "BBWI", "TPR",
        "PVH", "RL", "VFC", "G", "CPRI", "SIG", "KSS", "M", "ANF", "AEO",
        "URBN", "BOOT", "DECK", "WWW", "CROX",
    ],
    "consumer_staples": [
        "PG", "KO", "PEP", "PM", "MO", "CL", "KMB", "GIS", "HSY",
        "MDLZ", "CAG", "CPB", "HRL", "SJM", "MKC", "CHD", "CLX", "EL", "COTY",
        "SPB", "KR", "SFM", "GO", "CASY", "BJ", "WDFC", "FLO", "JJSF",
    ],
    "industrials": [
        "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "EMR",
        "FDX", "NSC", "CSX", "UNP", "WM", "RSG", "PCAR", "CMI", "ETN", "PH",
        "ROK", "AME", "XYL", "GNRC", "GWW", "FAST", "SNA", "SWK", "IR", "TT",
        "OTIS", "CARR", "ROP", "IEX", "ITW", "DOV", "FTV", "HUBB", "LII", "MAS",
        "AXON", "TRMB", "MHK", "OC", "AWI", "BLDR", "IBP", "SITE", "TREX",
    ],
    "telecom": [
        "T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN", "ATNI", "SHEN", "GSAT",
    ],
    "utilities": [
        "NEE", "DUK", "AEP", "ETR", "EXC", "PCG", "XEL", "WEC", "ES", "CMS",
    ],
    "materials": [
        "LIN", "APD", "ECL", "DD", "NEM",
        "FCX", "NUE", "STLD", "ALB", "PPG", "SHW", "RPM", "CE", "EMN", "HUN",
        "PKG", "IP", "BALL", "AVY", "IFF", "FMC", "MOS", "CF",
        "MP", "LAC", "NOVT", "CRUS", "SITM", "DIOD", "ONTO",
        "SLDP", "AMPX", "QS", "AUR",
    ],
    "real_estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA",
        "SPG", "O", "WELL", "VTR", "DLR", "EXR", "AVB", "EQR", "MAA", "UDR",
        "WY", "IRM", "HST", "RHP", "SUI", "ELS", "NNN", "EPRT",
        "CBRE", "JLL", "CWK", "MMI", "OPEN", "Z", "EXPI", "RKT",
    ],
    "media_and_entertainment": [
        "DIS", "NFLX", "WBD", "FOX", "NYT", "LYV", "SIRI",
        "AMC", "CNK", "IMAX", "NCMI", "TKO", "DKNG", "PENN", "CZR",
    ],
    "space_and_defense": [
        "NOC", "GD", "HII", "KTOS", "RKLB", "SPCE", "ASTS", "BWXT",
        "LDOS", "SAIC", "BAH", "CACI", "VSAT", "RDW", "MNTS",
        "LUNR", "PL", "SERV", "ASPI",
    ],
}

# Maps each ticker to its sector name.
SECTOR_MAP: dict[str, str] = {
    ticker: sector
    for sector, tickers in SECTORS.items()
    for ticker in tickers
}

# Flat list of all tickers across all sectors.
TICKERS: list[str] = [
    ticker
    for tickers in SECTORS.values()
    for ticker in tickers
]

INPUT_PATH = Path(__file__).parent / "data/features.csv"
PCA_OUTPUT_PATH = Path(__file__).parent / "data/features_reduced_PCA.csv"
PCA_EXPLAINED_OUTPUT_PATH = Path(__file__).parent / "data/features_reduced_explained_PCA.csv"
TSNE_OUTPUT_PATH = Path(__file__).parent / "data/features_reduced_TSNE.csv"
MDS_OUTPUT_PATH = Path(__file__).parent / "data/features_reduced_MDS.csv"
LDA_OUTPUT_PATH = Path(__file__).parent / "data/features_reduced_LDA.csv"
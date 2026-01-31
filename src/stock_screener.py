"""
Stock Screener - Find low-value, recession-proof stocks with long-term trends
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Recession-proof sector categories
SECTORS = {
    # Defensive Utilities
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "XEL", "ES", "WEC", "ED", "EIX"],
    "water_utilities": ["AWK", "WTR", "WTRG", "SJW", "AWR", "CWT", "MSEX", "YORW"],

    # Consumer Defensive
    "consumer_staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "GIS", "K", "HSY"],
    "food_beverage": ["KO", "PEP", "MDLZ", "KHC", "GIS", "CPB", "SJM", "CAG", "HRL", "TSN"],
    "household_products": ["PG", "CL", "CHD", "CLX", "KMB", "SPB", "EPC", "CENT"],

    # Retail
    "discount_retail": ["WMT", "COST", "DG", "DLTR", "TJX", "ROST", "BJ", "OLLI", "FIVE", "PSMT"],
    "grocery": ["KR", "ACI", "SFM", "GO", "NGVC", "WMK", "VLGEA"],

    # Healthcare
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "CVS"],
    "pharmaceuticals": ["PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "VTRS", "TAK", "NVO"],
    "health_insurance": ["UNH", "ELV", "CI", "HUM", "CNC", "MOH"],

    # Telecom & Communication
    "telecom": ["T", "VZ", "TMUS", "CMCSA", "CHTR"],

    # Real Estate (Defensive REITs)
    "reits_healthcare": ["WELL", "VTR", "OHI", "PEAK", "HR", "DOC", "SBRA", "LTC"],
    "reits_residential": ["AVB", "EQR", "ESS", "MAA", "UDR", "CPT", "INVH", "AMH"],
    "reits_essential": ["O", "WPC", "NNN", "ADC", "STOR", "FCPT", "EPRT"],

    # Financial (Defensive)
    "insurance": ["BRK-B", "PGR", "ALL", "TRV", "CB", "MET", "PRU", "AFL", "AIG", "HIG"],
    "regional_banks": ["USB", "PNC", "TFC", "FITB", "RF", "KEY", "CFG", "MTB", "HBAN"],

    # Infrastructure & Industrial
    "waste_management": ["WM", "RSG", "WCN", "CWST", "GFL", "CLH", "SRCL"],
    "defense_aerospace": ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII", "TXT", "LDOS"],
    "railroads": ["UNP", "CSX", "NSC", "CP", "CNI"],
    "infrastructure": ["AMT", "CCI", "SBAC", "NEE", "AEP", "PCG", "SRE", "WMB", "KMI"],

    # Energy (Defensive - Pipelines/Storage)
    "midstream_energy": ["EPD", "ET", "MPLX", "WMB", "KMI", "OKE", "TRGP", "PAA"],

    # Precious Metals (Recession Hedge)
    "gold_miners": ["NEM", "GOLD", "AEM", "FNV", "WPM", "RGLD", "KGC", "AGI"],

    # Dividend Aristocrats (25+ years of dividend growth)
    "dividend_aristocrats": ["JNJ", "PG", "KO", "PEP", "MMM", "ABT", "ABBV", "MCD",
                            "WMT", "XOM", "CVX", "CL", "EMR", "GPC", "ITW", "SWK"],

    # Sin Stocks (Recession-resistant vices)
    "sin_stocks": ["MO", "PM", "BTI", "STZ", "BF-B", "DEO", "TAP", "SAM", "WYNN", "LVS"],

    # Small/Mid-Cap Defense Contractors (competing for US contracts)
    "defense_smallcap": [
        "KTOS",   # Kratos Defense - drones, unmanned systems, missile defense
        "MRCY",   # Mercury Systems - defense electronics, processing
        "AVAV",   # AeroVironment - tactical drones, loitering munitions
        "BWXT",   # BWX Technologies - nuclear components, reactors
        "PSN",    # Parsons Corp - defense engineering, cybersecurity
        "CACI",   # CACI International - defense IT, intel services
        "SAIC",   # Science Applications - defense IT services
        "BAH",    # Booz Allen Hamilton - defense consulting
        "AJRD",   # Aerojet Rocketdyne - rocket propulsion (being acquired)
        "TDG",    # TransDigm - aerospace components
        "HEI",    # HEICO - aircraft replacement parts
        "CW",     # Curtiss-Wright - defense components
        "MOG-A",  # Moog Inc - flight controls, defense systems
        "PLTR",   # Palantir - defense data analytics, AI
        "RKLB",   # Rocket Lab - small launch vehicles, space
        "RDW",    # Redwire - space infrastructure
        "IRDM",   # Iridium - satellite communications
        "GILT",   # Gilat Satellite - satellite networking
        "VSAT",   # Viasat - defense communications
        "AXON",   # Axon - law enforcement tech (adjacent)
        "SWBI",   # Smith & Wesson - firearms
        "RGR",    # Sturm Ruger - firearms
        "POWW",   # AMMO Inc - ammunition
        "AAXN",   # Axon Enterprise
    ],

    # Space & Hypersonics (emerging defense tech)
    "defense_space": [
        "RKLB",   # Rocket Lab
        "LUNR",   # Intuitive Machines - lunar landers
        "RDW",    # Redwire - space manufacturing
        "MNTS",   # Momentus - space transport
        "BKSY",   # BlackSky - satellite imagery intel
        "PL",     # Planet Labs - earth imaging
        "ASTS",   # AST SpaceMobile - space-based cellular
        "IRDM",   # Iridium
        "SPIR",   # Spire Global - satellite data
    ],
}


def get_stock_data(symbol: str) -> dict | None:
    """Fetch stock data and key metrics."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get historical data for trend analysis (2 years)
        hist = ticker.history(period="2y")
        if hist.empty:
            return None

        # Calculate trend metrics
        current_price = hist["Close"].iloc[-1]
        price_1y_ago = hist["Close"].iloc[len(hist)//2] if len(hist) > 250 else hist["Close"].iloc[0]
        price_2y_ago = hist["Close"].iloc[0]

        return {
            "symbol": symbol,
            "name": info.get("shortName", symbol),
            "sector": info.get("sector", "Unknown"),
            "price": current_price,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "market_cap": info.get("marketCap", 0),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "1y_return": ((current_price - price_1y_ago) / price_1y_ago) * 100,
            "2y_return": ((current_price - price_2y_ago) / price_2y_ago) * 100,
            "avg_volume": info.get("averageVolume", 0),
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def calculate_value_score(stock: dict) -> float:
    """Calculate a value score (lower = better value)."""
    score = 0

    # P/E ratio scoring (lower is better)
    pe = stock.get("pe_ratio")
    if pe and pe > 0:
        if pe < 15:
            score += 3
        elif pe < 20:
            score += 2
        elif pe < 25:
            score += 1

    # P/B ratio scoring (lower is better)
    pb = stock.get("pb_ratio")
    if pb and pb > 0:
        if pb < 1.5:
            score += 3
        elif pb < 3:
            score += 2
        elif pb < 5:
            score += 1

    # Dividend yield scoring (higher is better for stability)
    div_yield = stock.get("dividend_yield", 0)
    if div_yield > 0.04:
        score += 3
    elif div_yield > 0.02:
        score += 2
    elif div_yield > 0.01:
        score += 1

    # Beta scoring (lower beta = more stable/recession-proof)
    beta = stock.get("beta")
    if beta:
        if beta < 0.8:
            score += 3
        elif beta < 1.0:
            score += 2
        elif beta < 1.2:
            score += 1

    return score


def calculate_trend_score(stock: dict) -> float:
    """Calculate trend score (positive = uptrend)."""
    score = 0

    # 1-year return
    ret_1y = stock.get("1y_return", 0)
    if ret_1y > 20:
        score += 3
    elif ret_1y > 10:
        score += 2
    elif ret_1y > 0:
        score += 1
    elif ret_1y < -20:
        score -= 2

    # 2-year return
    ret_2y = stock.get("2y_return", 0)
    if ret_2y > 30:
        score += 3
    elif ret_2y > 15:
        score += 2
    elif ret_2y > 0:
        score += 1

    return score


def screen_stocks(category: str = None, min_value_score: int = 5, min_trend_score: int = 2) -> pd.DataFrame:
    """
    Screen stocks based on category and scoring criteria.

    Args:
        category: Sector category (utilities, consumer_staples, healthcare, etc.)
                 If None, screens all categories
        min_value_score: Minimum value score to include (default 5)
        min_trend_score: Minimum trend score to include (default 2)

    Returns:
        DataFrame of qualifying stocks sorted by combined score
    """
    if category:
        symbols = SECTORS.get(category, [])
        if not symbols:
            print(f"Unknown category: {category}")
            print(f"Available categories: {list(SECTORS.keys())}")
            return pd.DataFrame()
    else:
        # Get all unique symbols from all categories
        symbols = list(set(sym for syms in SECTORS.values() for sym in syms))

    print(f"Screening {len(symbols)} stocks...")

    results = []
    for i, symbol in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] Analyzing {symbol}...", end="\r")
        data = get_stock_data(symbol)
        if data:
            data["value_score"] = calculate_value_score(data)
            data["trend_score"] = calculate_trend_score(data)
            data["combined_score"] = data["value_score"] + data["trend_score"]
            results.append(data)

    print(" " * 50, end="\r")  # Clear progress line

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Filter by minimum scores
    df = df[(df["value_score"] >= min_value_score) & (df["trend_score"] >= min_trend_score)]

    # Sort by combined score (descending)
    df = df.sort_values("combined_score", ascending=False)

    return df


def display_results(df: pd.DataFrame):
    """Display screening results in a formatted table."""
    if df.empty:
        print("No stocks matched the criteria.")
        return

    display_cols = [
        "symbol", "name", "price", "pe_ratio", "pb_ratio",
        "dividend_yield", "beta", "1y_return", "value_score",
        "trend_score", "combined_score"
    ]

    # Format for display
    display_df = df[display_cols].copy()
    display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")
    display_df["pe_ratio"] = display_df["pe_ratio"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    display_df["pb_ratio"] = display_df["pb_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df["dividend_yield"] = display_df["dividend_yield"].apply(lambda x: f"{x*100:.2f}%" if x else "0%")
    display_df["beta"] = display_df["beta"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df["1y_return"] = display_df["1y_return"].apply(lambda x: f"{x:+.1f}%")

    print("\n" + "="*100)
    print("RECESSION-PROOF VALUE STOCKS - SCREENING RESULTS")
    print("="*100)
    print(display_df.to_string(index=False))
    print("="*100)
    print(f"Total matches: {len(df)}")


def main():
    """Main entry point with interactive menu."""
    print("\n" + "="*50)
    print("  RECESSION-PROOF STOCK SCREENER")
    print("="*50)
    print("\nCategories:")
    for i, cat in enumerate(SECTORS.keys(), 1):
        print(f"  {i}. {cat.replace('_', ' ').title()}")
    print(f"  {len(SECTORS)+1}. All Categories")
    print("  0. Exit")

    while True:
        try:
            choice = input(f"\nSelect category (0-{len(SECTORS)+1}): ").strip()
            if choice == "0":
                print("Goodbye!")
                break

            choice = int(choice)
            categories = list(SECTORS.keys())

            if choice == len(SECTORS) + 1:
                category = None
                print("\nScreening all categories...")
            elif 1 <= choice <= len(SECTORS):
                category = categories[choice - 1]
                print(f"\nScreening {category.replace('_', ' ').title()}...")
            else:
                print("Invalid choice. Try again.")
                continue

            # Run screening
            results = screen_stocks(category=category)
            display_results(results)

        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

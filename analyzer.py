"""
S&P 500 10-Q Sentiment Analyzer
================================
Extracts and analyzes thematic sentiment from the most recent 10-Q filings
of a random sample of S&P 500 companies using the SEC EDGAR API.

Scoring logic (applied to all themes):
    +2 = strong concern signal
    +1 = mild concern signal
    -1 = mild positive / resilience signal
    -2 = strong positive / resilience signal

Negation detection inverts the sign when a negation appears within
60 characters before the keyword.

Themes analyzed:
    - Macroeconomic: inflation, recession, interest rates, credit
    - Operational:   supply chain, consumer spending
    - Geopolitical:  conflicts, tariffs, sanctions
    - Technology:    AI bubble vs. AI solidity

Sector breakdown:
    Companies are grouped by GICS Sector (11 macro sectors) for
    aggregate peer-group analysis.

DISCLAIMER
----------
This is a personal, non-commercial project for educational purposes only.
It does NOT constitute financial, investment, or legal advice.
The author makes no warranties about the accuracy or completeness of the
analysis and assumes no liability for any use of this code or its outputs.
See DISCLAIMER.md for the full statement.

Author : Francesco Giordano
Date   : 2026
"""

import re
import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from io import StringIO
from bs4 import BeautifulSoup
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# CONFIGURATION

HEADERS_WIKI = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
HEADERS_SEC  = {"User-Agent": "your.email@example.com"}

SAMPLE_SIZE   = 50
RANDOM_STATE  = 42
SLEEP_BETWEEN = 0.2
MIN_TEXT_LEN  = 500

# STEP 1 — S&P 500 COMPANY LIST WITH GICS SECTOR

def load_sp500(sample_size: int, random_state: int) -> pd.DataFrame:
    """Load S&P 500 constituents with GICS sector info and sample n companies."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r   = requests.get(url, headers=HEADERS_WIKI)
    df  = pd.read_html(StringIO(r.text))[0][["Symbol", "Security", "GICS Sector"]]
    df.columns = ["Symbol", "Security", "gics_sector"]
    return df.sample(sample_size, random_state=random_state).reset_index(drop=True)

# STEP 2 — TICKER TO CIK MAPPING

def load_cik_map() -> dict:
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS_SEC)
    return {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in r.json().values()
    }

# STEP 3 — RETRIEVE LATEST 10-Q FILING METADATA

def get_latest_10q(cik: str):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r   = requests.get(url, headers=HEADERS_SEC)
    f   = r.json()["filings"]["recent"]
    df  = pd.DataFrame({
        "form":            f["form"],
        "filingDate":      f["filingDate"],
        "accessionNumber": f["accessionNumber"],
    })
    df10 = df[df["form"] == "10-Q"].sort_values("filingDate", ascending=False)
    return df10.iloc[0] if not df10.empty else None

# STEP 4 — DOWNLOAD AND CLEAN FILING TEXT

def get_full_text(ticker: str, accession_number: str, cik_map: dict) -> str:
    cik_raw   = cik_map.get(ticker, "").lstrip("0")
    acc       = accession_number.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_raw}/{acc}/{accession_number}-index.htm"
    )

    r    = requests.get(index_url, headers=HEADERS_SEC)
    soup = BeautifulSoup(r.text, "html.parser")

    SKIP = ["ex", "exhibit", "xsd", "def", "lab", "pre", "cal", "r1", "r2", "r3"]

    best_url = None
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/ix?doc=" in href:
            href = href.split("/ix?doc=")[-1]
        if not href.endswith(".htm"):
            continue
        if any(x in href.lower() for x in SKIP):
            continue
        best_url = ("https://www.sec.gov" + href
                    if href.startswith("/") else href)
        break

    if not best_url:
        return ""

    r2    = requests.get(best_url, headers=HEADERS_SEC)
    soup2 = BeautifulSoup(r2.text, "html.parser")
    for tag in soup2.find_all(["table", "script", "style"]):
        tag.decompose()
    return soup2.get_text(separator=" ", strip=True)

# STEP 5 — WEIGHTED KEYWORD DICTIONARIES

# +2 = strong concern   |  +1 = mild concern
# -1 = mild positive    |  -2 = strong positive

TOPIC_KEYWORDS = {

    "inflation": {
        "runaway inflation":           +2,
        "persistent inflation":        +2,
        "hyperinflation":              +2,
        "severe price pressure":       +2,
        "unprecedented cost increase": +2,
        "inflation":                   +1,
        "inflationary":                +1,
        "price pressure":              +1,
        "pricing pressure":            +1,
        "rising costs":                +1,
        "cost increases":              +1,
        "cost inflation":              +1,
        "higher input costs":          +1,
        "wage pressure":               +1,
        "commodity price":             +1,
        "moderating inflation":        -1,
        "easing price pressure":       -1,
        "cost discipline":             -1,
        "pricing power":               -1,
        "stable input costs":          -1,
        "deflation":                   -2,
        "disinflation":                -2,
        "declining costs":             -2,
    },

    "recession": {
        "recession":                   +2,
        "recessionary":                +2,
        "economic contraction":        +2,
        "gdp contraction":             +2,
        "hard landing":                +2,
        "severe downturn":             +2,
        "economic slowdown":           +1,
        "downturn":                    +1,
        "economic weakness":           +1,
        "macro headwind":              +1,
        "slowing economy":             +1,
        "soft landing":                -1,
        "economic stabilization":      -1,
        "resilient economy":           -1,
        "economic expansion":          -2,
        "strong economic growth":      -2,
        "economic recovery":           -2,
    },

    "geopolitical": {
        "armed conflict":              +2,
        "military conflict":           +2,
        "trade war":                   +2,
        "escalating sanctions":        +2,
        "war":                         +2,
        "geopolitical":                +1,
        "ukraine":                     +1,
        "russia":                      +1,
        "sanctions":                   +1,
        "middle east":                 +1,
        "israel":                      +1,
        "tariff":                      +1,
        "trade tension":               +1,
        "export control":              +1,
        "import restriction":          +1,
        "trade agreement":             -1,
        "easing tensions":             -1,
        "sanctions relief":            -1,
        "ceasefire":                   -2,
        "peace agreement":             -2,
        "trade normalization":         -2,
    },

    "interest_rates": {
        "aggressive tightening":       +2,
        "sharp rate increase":         +2,
        "restrictive monetary policy": +2,
        "rate hike":                   +1,
        "higher rates":                +1,
        "higher borrowing cost":       +1,
        "tightening cycle":            +1,
        "elevated interest rate":      +1,
        "rising cost of capital":      +1,
        "rate cut":                    -1,
        "easing cycle":                -1,
        "accommodative policy":        -1,
        "lower borrowing cost":        -1,
        "substantial rate cuts":       -2,
        "aggressive easing":           -2,
    },

    "credit": {
        "credit crunch":               +2,
        "default":                     +2,
        "credit default":              +2,
        "liquidity crisis":            +2,
        "tightening credit":           +2,
        "credit risk":                 +1,
        "default risk":                +1,
        "liquidity risk":              +1,
        "credit spread":               +1,
        "widening spread":             +1,
        "tighter lending standard":    +1,
        "ample liquidity":             -1,
        "strong balance sheet":        -1,
        "investment grade":            -1,
        "stable credit":               -1,
        "credit upgrade":              -2,
        "improved credit rating":      -2,
    },

    "supply_chain": {
        "supply chain disruption":     +2,
        "severe shortage":             +2,
        "supplier failure":            +2,
        "semiconductor shortage":      +2,
        "chip shortage":               +2,
        "port congestion":             +2,
        "supply constraint":           +1,
        "supply shortage":             +1,
        "supply risk":                 +1,
        "component shortage":          +1,
        "shipping delay":              +1,
        "delivery delay":              +1,
        "extended lead time":          +1,
        "rising freight cost":         +1,
        "rising shipping cost":        +1,
        "supplier concentration":      +1,
        "single source":               +1,
        "sole source":                 +1,
        "china dependency":            +1,
        "raw material cost":           +1,
        "excess inventory":            +1,
        "inventory buildup":           +1,
        "backlog":                     +1,
        "supply chain resilience":     -1,
        "diversified sourcing":        -1,
        "dual sourcing":               -1,
        "alternative supplier":        -1,
        "reshoring":                   -1,
        "nearshoring":                 -1,
        "normalized inventory":        -1,
        "improving lead time":         -1,
        "stable supply":               -1,
        "supply chain optimization":   -2,
        "fully diversified supply":    -2,
    },

    "consumer": {
        "demand destruction":          +2,
        "demand collapse":             +2,
        "financially stressed":        +2,
        "stretched consumer":          +2,
        "consumer recession":          +2,
        "significant delinquency":     +2,
        "rising charge off":           +2,
        "demand weakness":             +1,
        "demand softness":             +1,
        "demand slowdown":             +1,
        "lower demand":                +1,
        "weaker demand":               +1,
        "sluggish demand":             +1,
        "demand headwind":             +1,
        "volume decline":              +1,
        "volume pressure":             +1,
        "trading down":                +1,
        "down trading":                +1,
        "price sensitive":             +1,
        "consumer hesitancy":          +1,
        "cautious consumer":           +1,
        "wait and see":                +1,
        "affordability":               +1,
        "cost of living":              +1,
        "declining foot traffic":      +1,
        "lower discretionary":         +1,
        "strong demand":               -1,
        "resilient consumer":          -1,
        "healthy consumer":            -1,
        "consumer strength":           -1,
        "robust spending":             -1,
        "premiumization":              -1,
        "luxury demand":               -1,
        "wage growth":                 -1,
        "rising disposable income":    -1,
        "improving consumer":          -1,
        "record demand":               -2,
        "exceptional consumer":        -2,
        "accelerating demand":         -2,
    },
}

# AI BUBBLE DICTIONARY

AI_KEYWORDS = {
    "ai investment":              +2,
    "ai spending":                +2,
    "ai capital expenditure":     +2,
    "data center":                +2,
    "gpu":                        +2,
    "nvidia":                     +2,
    "transformative":             +2,
    "revolutionary":              +2,
    "game changer":               +2,
    "unprecedented opportunity":  +2,
    "ai-first":                   +2,
    "ai race":                    +2,
    "ai arms race":               +2,
    "first mover":                +2,
    "exploring ai":               +2,
    "piloting ai":                +2,
    "testing ai":                 +2,
    "ai roadmap":                 +2,
    "ai monetization":            +2,
    "artificial intelligence":    +1,
    "machine learning":           +1,
    "generative ai":              +1,
    "large language model":       +1,
    "llm":                        +1,
    "foundation model":           +1,
    "copilot":                    +1,
    "chatbot":                    +1,
    "ai strategy":                +1,
    "ai initiative":              +1,
    "ai platform":                +1,
    "ai solution":                +1,
    "ai integration":             +1,
    "ai model":                   +1,
    "ai tool":                    +1,
    "openai":                     +1,
    "anthropic":                  +1,
    "google gemini":              +1,
    "microsoft ai":               +1,
    "digital transformation":     +1,
    "automation":                 +1,
    "ai capabilities":            +1,
    "ai-driven":                  +1,
    "ai productivity":            -1,
    "cost savings":               -1,
    "efficiency gains":           -1,
    "measurable impact":          -1,
    "ai deployed":                -1,
    "ai in production":           -1,
    "ai revenue contribution":    -1,
    "ai-powered product":         -1,
    "ai reduced":                 -1,
    "ai improved":                -1,
    "responsible ai":             -1,
    "ai governance":              -1,
    "ai compliance":              -1,
    "ai ethics":                  -1,
    "ai regulation":              -1,
    "ai limitation":              -1,
    "ai challenge":               -1,
    "ai risk":                    -1,
}

# NEGATIONS

NEGATIONS = [
    "no ", "not ", "without ", "unlike ", "absence of ",
    "no signs of ", "do not expect", "did not see",
    "no evidence of", "not anticipate", "not experiencing",
    "not expect", "not observed", "not material",
]

# STEP 6 — SCORING FUNCTIONS

def score_weighted_topic(text: str, keywords: dict) -> int:
    """Score a topic by summing weighted keyword matches with negation detection."""
    t     = text.lower()
    total = 0
    for kw, weight in keywords.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        for match in re.finditer(pattern, t):
            start         = match.start()
            window_before = t[max(0, start - 60): start]
            effective     = (-weight
                             if any(neg in window_before for neg in NEGATIONS)
                             else weight)
            total += effective
    return total


def count_all_topics(text: str) -> dict:
    return {topic: score_weighted_topic(text, kw_dict)
            for topic, kw_dict in TOPIC_KEYWORDS.items()}


def count_ai_bubble(text: str) -> dict:
    t             = text.lower()
    bubble_score  = 0
    solid_score   = 0
    generic_score = 0

    for kw, weight in AI_KEYWORDS.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        for match in re.finditer(pattern, t):
            start         = match.start()
            window_before = t[max(0, start - 60): start]
            effective     = (-weight
                             if any(neg in window_before for neg in NEGATIONS)
                             else weight)

            if effective == +2:
                bubble_score  += 1
            elif effective == +1:
                generic_score += 1
            elif effective <= -1:
                solid_score   += abs(effective)

    return {
        "ai_bubble_score": bubble_score - solid_score,
        "ai_bubble_hits":  bubble_score,
        "ai_solid_hits":   solid_score,
        "ai_generic_hits": generic_score,
    }

# STEP 7 — MAIN LOOP

def run_analysis(sp500: pd.DataFrame, cik_map: dict) -> pd.DataFrame:
    results = []

    for _, row in sp500.iterrows():
        ticker  = row["Symbol"]
        company = row["Security"]
        sector  = row["gics_sector"]
        cik     = cik_map.get(ticker)

        if not cik:
            print(f"[SKIP]  {ticker}: CIK not found")
            continue

        try:
            latest = get_latest_10q(cik)
            if latest is None:
                print(f"[SKIP]  {ticker}: no 10-Q found")
                continue

            text = get_full_text(ticker, latest["accessionNumber"], cik_map)

            if len(text) < MIN_TEXT_LEN:
                print(f"[SKIP]  {ticker}: text too short ({len(text)} chars)")
                continue

            scores = count_all_topics(text)
            scores.update(count_ai_bubble(text))
            scores["ticker"]      = ticker
            scores["company"]     = company
            scores["gics_sector"] = sector
            scores["filingDate"]  = latest["filingDate"]
            scores["text_len"]    = len(text)
            results.append(scores)

            print(
                f"[OK]    {ticker:<6} | {sector[:15]:<15} | "
                f"{latest['filingDate']} | inflation: {scores['inflation']:+d} "
                f"| recession: {scores['recession']:+d} "
                f"| geo: {scores['geopolitical']:+d} "
                f"| ai_net: {scores['ai_bubble_score']:+d}"
            )

        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

        time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(results).set_index("ticker")
    df.to_csv("sp500_sentiment_topics.csv")
    return df

# STEP 8 — INDIVIDUAL VISUALIZATIONS

def plot_results(df: pd.DataFrame) -> None:
    """Generate and save the core individual visualizations."""
    topic_cols = list(TOPIC_KEYWORDS.keys())
    ai_cols    = ["ai_bubble_hits", "ai_solid_hits", "ai_generic_hits"]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
    TITLE_PAD = 14

    # FIGURE 1 — Per-company heatmap
    fig1, ax1 = plt.subplots(figsize=(13, max(6, len(df) * 0.35)))
    sns.heatmap(
        df[topic_cols].astype(float),
        ax=ax1, cmap="RdYlGn_r", center=0,
        linewidths=0.4, linecolor="white",
        annot=True, fmt=".0f",
        cbar_kws={"label": "Weighted Score  (+ = concern / - = positive)"},
    )
    ax1.set_title("Weighted Thematic Sentiment per Company  |  S&P 500 10-Q Sample",
                  fontsize=13, fontweight="bold", pad=TITLE_PAD)
    ax1.set_xlabel("")
    ax1.set_ylabel("Ticker")
    ax1.tick_params(axis="x", rotation=30)
    ax1.tick_params(axis="y", rotation=0)
    fig1.tight_layout()
    fig1.savefig("fig1_heatmap_weighted.png", dpi=150)

    # FIGURE 2 — Aggregate bar chart
    totals = df[topic_cols].sum().sort_values(ascending=True)
    colors = ["#d73027" if v >= 0 else "#1a9850" for v in totals.values]

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    bars = ax2.barh(totals.index, totals.values, color=colors, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.bar_label(bars, fmt="%+.0f", padding=4, fontsize=8)
    ax2.set_title("Aggregate Weighted Score by Theme  |  S&P 500 10-Q Sample",
                  fontsize=13, fontweight="bold", pad=TITLE_PAD)
    ax2.set_xlabel("Net Weighted Score  (positive = concern  |  negative = positive)")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig2.tight_layout()
    fig2.savefig("fig2_bar_themes.png", dpi=150)

    # FIGURE 3 — AI bubble scatter
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    sc = ax3.scatter(df["ai_bubble_hits"], df["ai_solid_hits"],
                     c=df["ai_bubble_score"], cmap="RdYlGn_r",
                     s=80, edgecolors="grey", linewidths=0.5, zorder=3)
    for ticker, row in df.iterrows():
        ax3.annotate(ticker, (row["ai_bubble_hits"], row["ai_solid_hits"]),
                     fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax3, label="Net AI Bubble Score")
    ax3.set_xlabel("Bubble-Signal Hits  (hype / unquantified capex)")
    ax3.set_ylabel("Solidity-Signal Hits  (ROI / governance / risk acknowledgment)")
    ax3.set_title("AI Bubble vs. Solidity  |  S&P 500 10-Q Sample",
                  fontsize=13, fontweight="bold", pad=TITLE_PAD)
    ax3.grid(True, linestyle="--", alpha=0.5)
    fig3.tight_layout()
    fig3.savefig("fig3_ai_bubble_scatter.png", dpi=150)

    # FIGURE 4 — Top concern companies
    df_local = df.copy()
    df_local["total_concern"] = df_local[topic_cols].sum(axis=1)
    top_concern = df_local.nlargest(15, "total_concern")["total_concern"].sort_values()

    fig4, ax4 = plt.subplots(figsize=(9, 6))
    ax4.barh(top_concern.index, top_concern.values, color="#d73027", edgecolor="white")
    ax4.set_title("Top 15 Companies by Aggregate Concern Score  |  S&P 500 10-Q Sample",
                  fontsize=13, fontweight="bold", pad=TITLE_PAD)
    ax4.set_xlabel("Sum of All Weighted Theme Scores")
    ax4.bar_label(ax4.containers[0], fmt="%+.0f", padding=4, fontsize=8)
    fig4.tight_layout()
    fig4.savefig("fig4_top_concern.png", dpi=150)

    plt.show()
    plt.close("all")
    print("Individual figures saved  (fig1 - fig4).")


# STEP 9 — SECTOR BREAKDOWN VISUALIZATIONS

def plot_sector_analysis(df: pd.DataFrame) -> None:
    """
    Generate sector-level aggregate analysis:
        - fig5: sector x theme heatmap (total scores)
        - fig6: stacked bar per sector
        - fig7: radar chart per sector
    """
    topic_cols = list(TOPIC_KEYWORDS.keys())

    # Aggregate totals per sector
    sector_totals = df.groupby("gics_sector")[topic_cols].sum()
    sector_totals["total"] = sector_totals.sum(axis=1)
    sector_totals = sector_totals.sort_values("total", ascending=False)

    # Also count how many companies per sector (useful context)
    sector_counts = df.groupby("gics_sector").size().rename("n_companies")
    sector_totals = sector_totals.join(sector_counts)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
    TITLE_PAD = 14

    # FIGURE 5 — Sector x theme heatmap
    fig5, ax5 = plt.subplots(figsize=(11, max(5, len(sector_totals) * 0.5)))
    sns.heatmap(
        sector_totals[topic_cols].astype(float),
        ax=ax5, cmap="RdYlGn_r", center=0,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".0f",
        cbar_kws={"label": "Total Weighted Score per Sector"},
    )
    ax5.set_title(
        "Thematic Concern by GICS Sector  |  S&P 500 10-Q Sample",
        fontsize=13, fontweight="bold", pad=TITLE_PAD,
    )

    # Add company count next to sector names
    y_labels = [f"{s} (n={int(sector_totals.loc[s, 'n_companies'])})"
                for s in sector_totals.index]
    ax5.set_yticklabels(y_labels, rotation=0)
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    ax5.tick_params(axis="x", rotation=30)
    fig5.tight_layout()
    fig5.savefig("fig5_sector_heatmap.png", dpi=150)

    # FIGURE 6 — Stacked bar per sector (composition of concern)
    theme_palette = {
        "inflation":      "#fbbf24",
        "recession":      "#fb923c",
        "geopolitical":   "#ef4444",
        "interest_rates": "#60a5fa",
        "credit":         "#a78bfa",
        "supply_chain":   "#10b981",
        "consumer":       "#ec4899",
    }

    # Clip negatives to 0 for stacking (only show concern contributions)
    stack_data = sector_totals[topic_cols].clip(lower=0)
    # Sort sectors by total concern ascending so largest appears on top
    stack_data = stack_data.loc[stack_data.sum(axis=1).sort_values().index]

    fig6, ax6 = plt.subplots(figsize=(11, max(5, len(stack_data) * 0.5)))
    bottom = np.zeros(len(stack_data))
    for theme in topic_cols:
        vals = stack_data[theme].values
        ax6.barh(stack_data.index, vals, left=bottom,
                 color=theme_palette.get(theme, "#888"),
                 label=theme.replace("_", " "),
                 edgecolor="white", linewidth=0.5)
        bottom += vals

    # Total label at end of each bar
    totals_sorted = stack_data.sum(axis=1)
    for i, total in enumerate(totals_sorted):
        ax6.text(total + max(totals_sorted) * 0.01, i, f"+{int(total)}",
                 va="center", fontsize=9, fontweight="bold")

    ax6.set_title(
        "Concern Composition by GICS Sector  |  S&P 500 10-Q Sample",
        fontsize=13, fontweight="bold", pad=TITLE_PAD,
    )
    ax6.set_xlabel("Total Concern Score  (stacked by theme)")
    ax6.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax6.tick_params(axis="y", labelsize=9)
    fig6.tight_layout()
    fig6.savefig("fig6_sector_stacked.png", dpi=150)

    # FIGURE 7 — Radar chart per sector (top sectors only to keep it readable)
    # Normalize each theme 0-1 across sectors for fair comparison
    radar_data = sector_totals[topic_cols].copy().astype(float)
    for theme in topic_cols:
        col = radar_data[theme]
        rng = col.max() - col.min()
        radar_data[theme] = (col - col.min()) / rng if rng > 0 else 0

    # Show top 6 sectors by total to keep radar readable
    top_sectors = sector_totals.nlargest(min(6, len(sector_totals)),
                                         "total").index.tolist()

    angles = np.linspace(0, 2 * np.pi, len(topic_cols), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    n_plots = len(top_sectors)
    ncols   = 3
    nrows   = int(np.ceil(n_plots / ncols))

    fig7, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.2, nrows * 4.2),
                              subplot_kw={"projection": "polar"})
    axes = np.array(axes).flatten()

    for i, sector in enumerate(top_sectors):
        ax = axes[i]
        values = radar_data.loc[sector, topic_cols].tolist()
        values += values[:1]

        ax.plot(angles, values, color="#ef4444", linewidth=2)
        ax.fill(angles, values, color="#ef4444", alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([t.replace("_", "\n") for t in topic_cols],
                           fontsize=8)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["", "", ""], fontsize=7)
        ax.set_ylim(0, 1)
        n_co = int(sector_totals.loc[sector, "n_companies"])
        ax.set_title(f"{sector}\n(n={n_co})",
                     fontsize=10, fontweight="bold", pad=16)

    # Hide unused subplots
    for j in range(len(top_sectors), len(axes)):
        axes[j].set_visible(False)

    fig7.suptitle(
        "Concern Profile by GICS Sector  |  S&P 500 10-Q Sample",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig7.tight_layout()
    fig7.savefig("fig7_sector_radar.png", dpi=150, bbox_inches="tight")

    plt.show()
    plt.close("all")
    print("Sector figures saved  (fig5 - fig7).")


# STEP 10 — COMPOSITE DASHBOARD

def plot_dashboard(df: pd.DataFrame) -> None:
    """
    Composite dashboard with three panels:
        - Top:          heatmap of normalized scores per theme x company
        - Bottom-left:  aggregate theme ranking
        - Bottom-right: sector x theme heatmap (replaces AI top 20)
    Dark professional theme.
    """
    topic_cols = list(TOPIC_KEYWORDS.keys())

    DARK_BG    = "#0b1220"
    PANEL_BG   = "#111a2b"
    TEXT_COLOR = "#e6e9ef"
    GRID_COLOR = "#1f2a3d"

    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "savefig.facecolor": DARK_BG,
    })

    fig = plt.figure(figsize=(20, 12))
    gs  = GridSpec(2, 2, figure=fig,
                   height_ratios=[1.4, 1],
                   hspace=0.35, wspace=0.25)

    fig.suptitle("S&P 500  ·  10-Q Sentiment Analysis",
                 fontsize=18, fontweight="bold", y=0.98)

    # Panel 1: Heatmap (top, full width)
    ax_heat = fig.add_subplot(gs[0, :])
    heat_data = df[topic_cols].T.astype(float)
    heat_norm = heat_data.copy()
    for theme in heat_norm.index:
        row = heat_norm.loc[theme]
        rng = row.max() - row.min()
        heat_norm.loc[theme] = (row - row.min()) / rng if rng > 0 else 0

    sns.heatmap(
        heat_norm, ax=ax_heat, cmap="RdYlGn_r",
        cbar_kws={"label": "", "shrink": 0.7},
        linewidths=0.3, linecolor=PANEL_BG,
        xticklabels=True,
        yticklabels=[t.replace("_", " ") for t in heat_norm.index],
    )
    ax_heat.set_title("Heatmap: score per theme x company  (green = low, red = high)",
                      fontsize=11, pad=10)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", rotation=60, labelsize=7)
    ax_heat.tick_params(axis="y", rotation=0, labelsize=9)

    # Panel 2: Theme ranking (bottom-left)
    ax_rank = fig.add_subplot(gs[1, 0])
    totals = df[topic_cols].sum().sort_values(ascending=True)
    theme_colors = {
        "credit":         "#a78bfa",
        "recession":      "#fb923c",
        "interest_rates": "#60a5fa",
        "inflation":      "#fbbf24",
        "consumer":       "#ec4899",
        "geopolitical":   "#ef4444",
        "supply_chain":   "#10b981",
    }
    bar_colors = [theme_colors.get(t, "#94a3b8") for t in totals.index]
    bars = ax_rank.barh(
        [t.replace("_", " ") for t in totals.index],
        totals.values,
        color=bar_colors, edgecolor="none",
    )
    ax_rank.bar_label(bars, fmt="%+.0f", padding=6,
                      fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax_rank.set_title("Theme ranking — total concern",
                      fontsize=11, pad=10)
    ax_rank.set_xlabel("Total mentions in sample", fontsize=9)
    ax_rank.grid(axis="x", linestyle="--", alpha=0.3)
    ax_rank.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_rank.spines[spine].set_visible(False)

    # Panel 3: Sector x theme heatmap (bottom-right)
    ax_sect = fig.add_subplot(gs[1, 1])

    sector_totals = df.groupby("gics_sector")[topic_cols].sum()
    sector_totals = sector_totals.loc[
        sector_totals.sum(axis=1).sort_values(ascending=False).index
    ]

    sns.heatmap(
        sector_totals.astype(float),
        ax=ax_sect,
        cmap="RdYlGn_r", center=0,
        linewidths=0.4, linecolor=PANEL_BG,
        annot=True, fmt=".0f",
        annot_kws={"fontsize": 7, "color": TEXT_COLOR},
        cbar_kws={"label": "", "shrink": 0.7},
    )
    ax_sect.set_title("Sector x theme — total concern",
                      fontsize=11, pad=10)
    ax_sect.set_xlabel("")
    ax_sect.set_ylabel("")
    ax_sect.tick_params(axis="x", rotation=30, labelsize=7)
    ax_sect.tick_params(axis="y", rotation=0, labelsize=8)

    fig.savefig("dashboard_sentiment.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    plt.close("all")
    print("Dashboard saved to  dashboard_sentiment.png")

    plt.rcParams.update(plt.rcParamsDefault)


# ENTRY POINT

if __name__ == "__main__":

    print("Loading S&P 500 company list ...")
    sp500 = load_sp500(SAMPLE_SIZE, RANDOM_STATE)
    print(f"{len(sp500)} companies loaded across "
          f"{sp500['gics_sector'].nunique()} sectors.\n")

    print("Loading SEC CIK map ...")
    cik_map = load_cik_map()
    print("CIK map loaded.\n")

    print("Running filing analysis ...\n")
    df_results = run_analysis(sp500, cik_map)

    print(f"\nAnalysis complete. {len(df_results)} companies processed.")
    print("Results saved to  sp500_sentiment_topics.csv\n")

    print("Generating individual visualizations ...")
    plot_results(df_results)

    print("Generating sector breakdown ...")
    plot_sector_analysis(df_results)

    print("Generating composite dashboard ...")
    plot_dashboard(df_results)

    print("\nAll outputs generated successfully.")

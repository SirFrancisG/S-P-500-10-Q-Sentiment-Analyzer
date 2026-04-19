# S&P 500 10-Q Sentiment Analyzer

Quantitative sentiment analysis of the most recent 10-Q filings from a random sample of S&P 500 companies, extracted directly from the SEC EDGAR API.

The goal is to answer a simple question: **what are large U.S. corporations actually worried about this quarter?** Rather than reading fifty filings by hand, this tool scores corporate language across eight macroeconomic and strategic themes and produces publication-ready visualizations.

---

## Disclaimer

**This is a personal, non-commercial project for educational purposes only. It does not constitute financial, investment, or legal advice. The author makes no warranties about the accuracy or completeness of the analysis and assumes no liability for any use of this code or its outputs. See [DISCLAIMER.md](DISCLAIMER.md) for the full statement.**

---

## What it does

For each company in the sample, the script:

1. Retrieves the latest 10-Q filing metadata from the SEC EDGAR API
2. Downloads and cleans the main filing document (handling the XBRL inline viewer wrapper)
3. Scores the narrative text across eight themes using a weighted keyword dictionary
4. Detects negations to avoid false positives (e.g. "no signs of recession")
5. Produces a CSV with per-company scores and five visualizations

---

## Themes analyzed

| Theme | Description |
|---|---|
| `inflation` | Pricing pressure, cost inflation, wage pressure |
| `recession` | Economic contraction, slowdown, soft/hard landing |
| `geopolitical` | Conflicts, tariffs, sanctions, trade tensions |
| `interest_rates` | Monetary policy, rate hikes/cuts, borrowing costs |
| `credit` | Credit risk, default, liquidity, lending standards |
| `supply_chain` | Disruptions, shortages, logistics, reshoring |
| `consumer` | Demand, spending, trading down, affordability |
| `ai_bubble` | AI hype vs. AI solidity (ROI, governance, risk) |

---

## Methodology

### Weighted scoring

Each keyword is assigned a directional weight:

| Weight | Meaning | Example |
|---|---|---|
| `+2` | Strong concern signal | `credit crunch`, `demand destruction` |
| `+1` | Mild concern signal | `rising costs`, `demand softness` |
| `-1` | Mild positive signal | `resilient consumer`, `pricing power` |
| `-2` | Strong positive signal | `record demand`, `economic expansion` |

The score for each theme is the sum of all weighted keyword matches found in the filing text.

### Negation detection

When a negation appears within 60 characters before a keyword, the sign is inverted. For example, `"we do not expect a recession"` scores `-2` instead of `+2`, correctly capturing that management is denying the concern rather than raising it.

### AI bubble vs. solidity

AI language is analyzed separately using three buckets:

- **Bubble signals** (hype, unquantified capex, AI race rhetoric)
- **Solidity signals** (measurable ROI, governance, risk acknowledgment)
- **Generic mentions** (neutral AI references)

The net `ai_bubble_score` is `bubble_hits - solid_hits`. A highly positive score suggests hype-driven language; a negative score suggests concrete, measurable AI deployment.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sp500-10q-sentiment-analyzer.git
cd sp500-10q-sentiment-analyzer
pip install -r requirements.txt
```

---

## Configuration

Before running, open `src/analyzer.py` and replace the placeholder email in the `HEADERS_SEC` constant with your own:

```python
HEADERS_SEC = {"User-Agent": "your.email@example.com"}
```

The SEC requires a valid contact email in the `User-Agent` header for all API requests. See [SEC EDGAR fair access policy](https://www.sec.gov/os/webmaster-faq#code-support) for details.

---

## Usage

```bash
python src/analyzer.py
```

The script will:

- Print progress for each company processed
- Save `data/sp500_sentiment_topics.csv` with raw scores
- Save five PNG visualizations to `output/`

Expected runtime: about 3-5 minutes for 50 companies (respects SEC rate limit of 10 requests per second).

---

## Project structure

```
sp500-10q-sentiment-analyzer/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   └── analyzer.py
├── data/
│   └── sp500_sentiment_topics.csv
└── output/
    ├── fig1_heatmap_weighted.png
    ├── fig2_bar_themes.png
    ├── fig3_ai_bubble_scatter.png
    ├── fig4_ai_stacked_bar.png
    └── fig5_top_concern.png
```

---

## Limitations

This project is intended as an exploratory tool, not a production-grade sentiment engine. Known limitations:

- **Keyword-based approach** cannot capture complex semantic nuance, sarcasm, or conditional statements
- **Sample size of 50 companies** is not statistically representative of the full S&P 500
- **Negation window of 60 characters** may miss long-range negations or complex clause structures
- **English only** — non-English filings or translated content are not supported
- **No sector normalization** — a bank and a semiconductor company will naturally emphasize different themes
- **Dictionary bias** — keyword selection reflects the author's interpretation of what constitutes a "concern signal"

Results should be interpreted as directional and illustrative, not absolute.

---

## Possible improvements

- Replace keyword matching with a finance-tuned LLM such as **FinBERT**
- Expand to the full S&P 500 using async HTTP requests (`aiohttp`)
- Add time-series analysis across multiple quarters to detect shifts in sentiment
- Aggregate results by GICS sector for peer comparison
- Build a Streamlit dashboard for interactive exploration
- Include the Risk Factors section (Item 1A) in addition to MD&A

---

## Data sources

- **SEC EDGAR API** — https://www.sec.gov/edgar
- **Wikipedia S&P 500 constituents** — https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

All data used is publicly available. No proprietary data, no API keys required.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built as a personal exploration of NLP techniques applied to corporate disclosures. Contributions, suggestions, and pull requests are welcome.

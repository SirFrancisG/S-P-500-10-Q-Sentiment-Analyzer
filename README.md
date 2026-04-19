# S&P 500 10-Q Sentiment Analyzer

Quantitative sentiment analysis of the most recent 10-Q filings from a random sample of S&P 500 companies, extracted directly from the SEC EDGAR API and broken down by GICS sector.

The goal is to answer two questions: **what are large U.S. corporations actually worried about this quarter, and how does that vary by industry?** Rather than reading fifty filings by hand, this tool scores corporate language across seven macroeconomic and strategic themes, adds a dedicated AI bubble analysis, and aggregates results by sector for peer-group comparison.

---

## Disclaimer

**This is a personal, non-commercial project for educational purposes only. It does not constitute financial, investment, or legal advice. The author makes no warranties about the accuracy or completeness of the analysis and assumes no liability for any use of this code or its outputs. See [DISCLAIMER.md](DISCLAIMER.md) for the full statement.**

---

## What it does

For each company in the sample, the script:

1. Retrieves the latest 10-Q filing metadata from the SEC EDGAR API
2. Downloads and cleans the main filing document (handling the XBRL inline viewer wrapper)
3. Scores the narrative text across seven themes using weighted keyword dictionaries
4. Detects negations to avoid false positives (e.g. "no signs of recession")
5. Computes a dedicated AI bubble vs. solidity score
6. Aggregates all results by GICS sector for peer-group comparison
7. Produces a CSV with per-company scores and eight visualizations, including a composite dashboard and a dedicated sector breakdown

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

Companies are also grouped by GICS Sector — one of the eleven macro sectors defined by the Global Industry Classification Standard: Information Technology, Financials, Health Care, Consumer Discretionary, Consumer Staples, Communication Services, Industrials, Energy, Utilities, Real Estate, Materials.

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

The score for each theme is the sum of all weighted keyword matches found in the filing text. A company with score `+20` on `inflation` is heavily focused on inflationary pressure; a company with score `-3` is actively denying or framing it positively.

### Negation detection

When a negation appears within 60 characters before a keyword, the sign is inverted. For example, `"we do not expect a recession"` scores `-2` instead of `+2`, correctly capturing that management is denying the concern rather than raising it.

### AI bubble vs. solidity

AI language is analyzed separately using three buckets:

- **Bubble signals** (hype, unquantified capex, AI race rhetoric)
- **Solidity signals** (measurable ROI, governance, risk acknowledgment)
- **Generic mentions** (neutral AI references)

The net `ai_bubble_score` is `bubble_hits - solid_hits`. A highly positive score suggests hype-driven language; a negative score suggests concrete, measurable AI deployment.

### Sector aggregation

After scoring each company individually, results are grouped by GICS sector and summed. This allows for three types of comparison:

- **Cross-sector** — which sector is most worried about geopolitical risk?
- **Intra-sector composition** — what makes up Financials' total concern: credit, interest rates, or something else?
- **Sector fingerprint** — a radar-chart profile showing each sector's unique concern mix

Sector labels come directly from the Wikipedia S&P 500 constituent table, which follows the official GICS classification.

---


## Project structure

```
sp500-10q-sentiment-analyzer/
├── README.md
├── DISCLAIMER.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   └── analyzer.py
├── notebooks/
│   └── exploration.ipynb
└── output/
    ├── dashboard_sentiment.png
    ├── fig1_heatmap_weighted.png
    ├── fig2_bar_themes.png
    ├── fig3_ai_bubble_scatter.png
    ├── fig4_top_concern.png
    ├── fig5_sector_heatmap.png
    ├── fig6_sector_stacked.png
    ├── fig7_sector_radar.png
    └── sp500_sentiment_topics.csv
```

---

## Limitations

This project is intended as an exploratory tool, not a production-grade sentiment engine. Known limitations:

- **Keyword-based approach** cannot capture complex semantic nuance, sarcasm, or conditional statements
- **Sample size** may not be statistically representative of the full S&P 500 or of individual sectors, especially at small sample sizes where some sectors are represented by only one or two companies
- **Negation window of 60 characters** may miss long-range negations or complex clause structures
- **English only** — non-English filings or translated content are not supported
- **No within-sector normalization** — a bank and a semiconductor company will naturally emphasize different themes, and sector aggregation helps with this, but within a sector the analysis still does not account for company size, revenue, or market cap
- **Dictionary bias** — keyword selection reflects the author's interpretation of what constitutes a "concern signal"

Results should be interpreted as directional and illustrative, not absolute.

---

## Possible improvements

- Replace keyword matching with a finance-tuned LLM such as **FinBERT** for better semantic understanding
- Expand to the full S&P 500 using async HTTP requests (`aiohttp`)
- Add time-series analysis across multiple quarters to detect shifts in sentiment
- Weight company-level scores by market cap before sector aggregation
- Use GICS Industry (58 groups) or Sub-Industry (163 groups) for more granular breakdowns
- Build a Streamlit dashboard for interactive exploration
- Include the Risk Factors section (Item 1A) in addition to MD&A

---

## Data sources

- **SEC EDGAR API** — https://www.sec.gov/edgar
- **Wikipedia S&P 500 constituents** — https://en.wikipedia.org/wiki/List_of_S%26P_500_companies (includes GICS Sector labels)

All data used is publicly available. No proprietary data, no API keys required.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built by Francesco Giordano as a personal exploration of NLP techniques applied to corporate disclosures. Contributions, suggestions, and pull requests are welcome.

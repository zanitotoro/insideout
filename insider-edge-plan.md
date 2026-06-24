# Insider-Buying Edge — Self-Contained Research & Backtest Plan

> **For the implementing agent.** This document is the *sole* input. It contains
> the goal, the decision logic, the exact data sources and schema, the hardware
> strategy for an Apple **M5** MacBook, runnable code skeletons, and every
> external link you need. Read it top to bottom before writing code. Where a
> field name or earliest-available date is marked *(verify)*, confirm it against
> the linked primary source rather than guessing.

---

## 0. What this project is (and is not)

**The deliverable is a *truthful yes/no answer* to: "Does a tradeable edge survive
in insider open-market purchases, for a retail-sized account, after realistic
costs and out-of-sample?"** It is **not** a mandate to build a live trading bot.

The single most likely honest outcome is **"no edge survives at this size."**
That is a *successful* result — it stops the user from funding a mirage. The
dangerous outcome is a **beautiful backtest you believe**; if results look great,
treat that as a signal to hunt for a bias bug (below), not to celebrate.

Legality context (not legal advice): trading on Form 4 data is legal because the
data is **public** once filed on EDGAR. Insider-trading law concerns *material
non-public* information. This project uses only published filings.

---

## 1. The spine — decision gates (do not reorder)

The project is staged so the cheapest, most decisive test runs first. Each phase
ends in a **gate**. **If a gate fails, stop.** Do not skip ahead to per-individual
track-record mining — that step can only *manufacture* an edge through
overfitting; it cannot reveal one the simpler tests missed.

1. **Population horizon curve (Phase 2 gate).** Replicate the documented effect on
   a *fixed horizon* at the *whole-population* level first. The shape of the
   curve (where abnormal return accumulates) *empirically determines* the holding
   period — you do not assume "6–12 months," you measure it.
2. **Segment, don't identify (Phase 3).** Ask "is *this kind of buy* good"
   (role, size, cluster, opportunistic vs. routine, market-cap band), not "is
   *this person* good." Each segment has thousands of observations → reliable and
   generalizable.
3. **Dumb basket backtest with real costs, out-of-sample (Phase 4 gate).** Only if
   a simple basket survives do you proceed.
4. **Refinements & exits (Phase 5, only past the gate).** Per-individual records
   via Bayesian *shrinkage*; and exit overlays (the user's trailing-stop idea)
   tested *against* the fixed-hold baseline.

### The four biases that fabricate edges — checked by tests in every phase
- **Survivorship.** If your price data lacks delisted names, you have silently
  deleted the insider-buy stocks that went to zero. This alone turns a losing
  strategy into a gorgeous backtest. Delisting must be handled as −100% (or
  documented recovery value).
- **Look-ahead.** You may only act on the **filing date** (next session's open),
  never the **transaction date**. A unit test must assert `entry_date >= filing_date`.
- **Costs.** The residual edge lives in illiquid small caps where spreads and
  market impact are large. Midpoint fills are fiction.
- **Overfitting.** Use strict out-of-sample / walk-forward splits. Pre-register
  rules before looking at the test set. Count your degrees of freedom.

---

## 2. Hardware strategy — exploiting the Apple M5

**Chip facts (base M5, Oct 2025):** 10-core CPU (4 performance + 6 efficiency),
10-core GPU with a **Neural Accelerator in each GPU core** (matrix-mul units,
>4× M4 GPU AI compute), 16-core Neural Engine, **unified memory** up to 32 GB at
~153 GB/s. M5 Pro/Max scale to 14–18 CPU cores, 16–40 GPU cores, 64–128 GB at
307–614 GB/s. Unified memory means **CPU and GPU share one pool with zero-copy** —
no host↔device transfers.

**The honest mapping of this workload to the hardware** (do not over-GPU it):

| Stage | Nature | Best tool on M5 |
|---|---|---|
| Download filings + prices | I/O / network-bound | **async** (httpx + rate limiter). M5 irrelevant; network is the bottleneck. |
| Unzip + parse 8 TSVs/quarter | CPU, parallel | **polars** (auto-multithreaded) or **DuckDB** (`SET threads`). Saturates all cores for free. |
| Join / filter / group-by | CPU, columnar | **polars / DuckDB**. Keep this on CPU — columnar engines beat naive GPU for tabular ops. |
| Forward & abnormal returns over millions of (event × horizon) | dense numeric, vectorized | **polars** for moderate size; **MLX** on GPU if the return/benchmark matrices are large. Unified memory = no copy. |
| Bootstrap confidence intervals (thousands of resamples) | embarrassingly parallel | **MLX** (draw a `B×N` index matrix on GPU, gather, mean — extremely fast) *or* `ProcessPoolExecutor`. |
| Walk-forward folds + parameter sweeps | embarrassingly parallel | `concurrent.futures.ProcessPoolExecutor` — one fold/combo per core. |
| Phase-5 hierarchical/Bayesian shrinkage; any ML classifier | matrix / gradient-heavy | **MLX** (autodiff, GPU matmul via M5 Neural Accelerators) or NumPyro/PyMC (CPU). Gradient-boosted trees → XGBoost multithreaded on CPU. |

**Rules of thumb**
- **MLX** (`pip install mlx`) is Apple's NumPy-like array framework; arrays live in
  unified memory and run on CPU or GPU without copies. Use it for *dense array
  math*: return matrices, vectorized bootstrap, model training. Docs:
  https://ml-explore.github.io/mlx/ · GitHub: https://github.com/ml-explore/mlx ·
  M5 Neural-Accelerator notes: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- **Do not** push joins/group-bys to the GPU. polars and DuckDB already use every
  CPU core and are typically faster for tabular work.
- **Ignore the Neural Engine** here — it is for Core ML inference, not general
  numerics. No benefit to this project.
- **Memory:** the full Form 4 history (~ low-tens-of-millions of transaction rows)
  plus daily prices fits in 32 GB unified memory — prefer **in-memory** (polars
  eager or DuckDB `:memory:`). On a 16 GB unit, use **DuckDB out-of-core** or
  **polars streaming** (`collect(streaming=True)`).
- Let polars use all cores (default; tune with `POLARS_MAX_THREADS`). For numpy,
  prefer a build linked against Apple **Accelerate** BLAS, or just do heavy linear
  algebra in MLX.

---

## 3. Environment & dependencies

Use **uv** (https://docs.astral.sh/uv/) or Poetry. Target Python 3.12+, macOS ≥ 13.3 (MLX requirement).

```bash
uv init insider-edge && cd insider-edge
uv add httpx aiolimiter tenacity polars duckdb pyarrow pydantic numpy scipy \
       statsmodels python-dateutil
uv add mlx                 # Apple-silicon GPU array math (Phases 2/5)
uv add --dev pytest ruff
# Phase-5 options (install when you reach it):
#   uv add numpyro     # or pymc, for Bayesian shrinkage
#   uv add xgboost     # gradient-boosted classifier (CPU multithreaded on macOS)
```

Library docs: httpx https://www.python-httpx.org/ · aiolimiter
https://github.com/mjpieters/aiolimiter · tenacity https://tenacity.readthedocs.io/ ·
polars https://docs.pola.rs/ · DuckDB https://duckdb.org/docs/ · pydantic
https://docs.pydantic.dev/ · PyArrow/Parquet https://arrow.apache.org/docs/python/ ·
SciPy https://docs.scipy.org/ · statsmodels https://www.statsmodels.org/ ·
NumPyro https://num.pyro.ai/ · XGBoost https://xgboost.readthedocs.io/

---

## 4. Data sources (exact URLs, schema, gotchas)

### 4.1 Historical Form 4 dump — SEC Insider Transactions Data Sets (your backtest universe)

This is the **gold-standard bulk source**: the SEC pre-flattens the Form 3/4/5 XML
into tab-delimited tables, updated quarterly, and it **includes filings from
delisted companies** (it is all filings as-filed).

- Landing / docs: https://www.sec.gov/dera/data/form-345
- **README (authoritative schema — read this):** https://www.sec.gov/files/insider_transactions_readme.pdf
- Quarterly file URL pattern (confirmed for 2025 Q1):
  `https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{Q}_form345.zip`
  e.g. `.../2025q1_form345.zip`. Iterate quarters back to the earliest available
  *(verify earliest quarter on the landing page; electronic ownership XML became
  mandatory ~2003 post-SOX)*.

Each quarterly zip contains **8 tab-delimited, UTF-8 tables**, all keyed on
`ACCESSION_NUMBER`:

| Table | Holds | Notes |
|---|---|---|
| `SUBMISSION` | one row per filing | `ACCESSION_NUMBER`, `FILING_DATE`, `PERIOD_OF_REPORT`, `DOCUMENT_TYPE` (3/4/5, plus `/A` amendments), `ISSUERCIK`, `ISSUERNAME`, `ISSUERTRADINGSYMBOL` *(verify exact names in README)* |
| `REPORTINGOWNER` | the insider(s) | `RPTOWNERCIK` (stable person id), `RPTOWNERNAME`, relationship flags (isDirector / isOfficer / isTenPercentOwner / isOther) and `RPTOWNER_TITLE` *(verify)* |
| `NONDERIV_TRANS` | **Table I transactions — your signal lives here** | key `ACCESSION_NUMBER` + `NONDERIV_TRANS_SK`; fields `SECURITY_TITLE`, `TRANS_DATE`, `TRANS_CODE`, `TRANS_SHARES`, `TRANS_PRICEPERSHARE`, `TRANS_ACQUIRED_DISP_CD` (A/D), `SHRS_OWND_FOLWNG_TRANS`, `DIRECT_INDIRECT_OWNERSHIP` *(verify)* |
| `NONDERIV_HOLDING` | Table I holdings (no transaction) | usually ignore |
| `DERIV_TRANS` | Table II derivative transactions | options etc.; mostly excluded |
| `DERIV_HOLDING` | Table II derivative holdings | ignore |
| `FOOTNOTES` | footnote text | key `ACCESSION_NUMBER` + `FOOTNOTE_ID`; **needed for pre-2023 10b5-1 detection** |
| `OWNER_SIGNATURE` | signatures | ignore |

The minimal join for an event table: `SUBMISSION` (filing/issuer/dates) ⋈
`REPORTINGOWNER` (who + role) ⋈ `NONDERIV_TRANS` (what), all on `ACCESSION_NUMBER`.

### 4.2 Recent gap + ongoing signal — EDGAR live (async)

Quarterly sets lag, so fetch the period since the last published quarter from
EDGAR directly.

- **Fair-access policy (must obey):** max **10 requests/second per IP**; cap
  yourself at ~8/s. You **must** send a descriptive `User-Agent` with a contact
  email or you get HTTP 403 and a temporary IP block. Official:
  https://www.sec.gov/about/webmaster-frequently-asked-questions and
  https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- **Ticker ↔ CIK map:** https://www.sec.gov/files/company_tickers.json — and note
  **CIK must be zero-padded to 10 digits** for `data.sec.gov` endpoints
  (320193 → `CIK0000320193`); un-padded → 404.
- **A filer's filing history (JSON):** `https://data.sec.gov/submissions/CIK##########.json`
- **Daily index of all filings** (filter `Form Type == 4`):
  `https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{q}/` (also
  `full-index/` for quarter-level `form.idx`).
- **Dissemination timing:** ownership forms 3/4/5 filed after ~10:00 p.m. ET
  appear in the *next* business day's index. Form 4 itself is due within **2
  business days** of the transaction.

### 4.3 Transaction codes — keep only genuine open-market buys

Filter `NONDERIV_TRANS.TRANS_CODE`:

| Code | Meaning | Use |
|---|---|---|
| **`P`** | Open-market or private **purchase** | **KEEP — this is the signal** |
| `S` | Open-market or private sale | drop (noisy; taxes/diversification/liquidity) |
| `A` | Grant/award/acquisition from the company (comp) | **exclude** (not a conviction buy) |
| `M` | Exercise/conversion of derivative (16b-3 exempt) | **exclude** |
| `F` | Shares withheld to pay exercise price / tax | **exclude** |
| `G` | Bona fide gift | exclude |
| `X` | Exercise of in-the-money derivative | exclude |
| `C`,`D`,`J`,`W`,`Z`,`L`,`U`,`E`,`H`,`O`,`I`,`K` | conversions/dispositions/other | exclude |

Also require `TRANS_ACQUIRED_DISP_CD == 'A'` (acquired). Full code list is in the
README appendix (4.1) and the SEC Form 3/4/5 guide: https://www.sec.gov/files/forms-3-4-5.pdf

### 4.4 Exclude 10b5-1 scheduled trades (low signal — they are pre-planned)

- **Post-2023:** the SEC's Rule 10b5-1 amendments (Release No. 33-11138) added an
  explicit affirmation checkbox to Form 4/5 ("made pursuant to a Rule 10b5-1(c)
  plan"), surfaced as a boolean flag in the structured data. Check whether this
  column is present in recent quarters of the flat-file set; if so, drop where true.
- **Pre-2023:** no flag exists. Infer by scanning `FOOTNOTES.text` for `10b5-1`
  (case-insensitive) and drop matches. Be conservative when ambiguous.

### 4.5 Prices — the #1 risk in the whole project

You need **survivorship-bias-free** daily prices **including delisted tickers**,
plus corporate actions (splits/dividends) for adjusted returns.

- `yfinance` (free) **lacks delisted names** → upward bias. **Smoke test only,
  never the verdict.**
- Correct sources (put the price layer behind a swappable interface so you can
  upgrade): **CRSP via WRDS** (academic gold standard, includes delisting
  returns), **Norgate Data** (retail, survivorship-bias-free), **Nasdaq Data Link
  / Sharadar SEP + SF1 + ACTIONS**, **Polygon.io**.
- Map ticker→security carefully across time (tickers get reused). Prefer a
  permanent id (CRSP `PERMNO`, or issuer CIK + figi) over raw ticker.
- **Treat a delisting as −100%** (or the documented delisting recovery value),
  never as a silent gap or a forward-filled last price.

### 4.6 Academic grounding (for parameter choices and sanity checks)
- Lakonishok & Lee (2001), *Are Insider Trades Informative?*, Rev. Fin. Studies — the base anomaly.
- Cohen, Malloy & Pomorski (2012), *Decoding Inside Information*, J. Finance — **routine vs. opportunistic** split (the stronger sub-signal).
- Jeng, Metrick & Zeckhauser (2003), *Estimating the Returns to Insider Trading*.
- "Insider filings as trading signals — *Does it pay to be fast?*" (2024): https://www.sciencedirect.com/science/article/pii/S1544612324015435 — finds the *fast/intraday* version dies after costs & liquidity. (Confirms: hold for the drift, don't day-trade the filing.)
- Microcap gradient-boosting detection (2026): https://arxiv.org/pdf/2602.06198 — features that matter; signal strongest but least liquid in micro-caps.
- McLean & Pontiff (2016), *Does Academic Research Destroy Stock Return Predictability?*, J. Finance — **post-publication anomaly decay (~50%)**; expect a smaller live edge than papers report.

---

## 5. Project layout

```
insider-edge/
  pyproject.toml
  README.md
  data/                      # raw zips, parquet store (gitignored)
  src/insider_edge/
    config.py                # paths, USER_AGENT (with contact email), thread caps
    sources/
      edgar_bulk.py          # async download + extract quarterly Form 345 zips
      edgar_live.py          # async fetch recent Form 4 from daily index
      prices.py              # swappable price interface (yfinance | sharadar | crsp)
    ingest/
      schema.py              # pydantic InsiderBuy model
      parse_form345.py       # join + flatten the 8 tables -> events
      filters.py             # TRANS_CODE / acquired / 10b5-1 filtering
    store/db.py              # duckdb/parquet helpers
    analysis/
      returns.py             # no-lookahead forward & abnormal returns (MLX-ready)
      horizon_curve.py       # population + segment curves with bootstrap CIs
      segments.py            # role/size/cluster/opportunistic/mcap features
      backtest.py            # basket, costs, OOS split, walk-forward (ProcessPool)
      shrinkage.py           # PHASE 5 ONLY: hierarchical per-insider (MLX/NumPyro)
    cli.py
  tests/
    test_filters.py
    test_no_lookahead.py     # the look-ahead gate
    test_survivorship.py     # the survivorship gate
```

---

## 6. Phases

### Phase 0 — Scaffold & guardrails  *(½–1 day)*
Set up the repo; **write the bias-check tests first** so gates are enforced by
code. Define the event schema.

```python
# ingest/schema.py
from datetime import date
from pydantic import BaseModel

class InsiderBuy(BaseModel):
    accession: str
    cik_issuer: str
    ticker: str
    insider_id: str          # RPTOWNERCIK — stable person identifier
    role: str                # CEO / CFO / Director / TenPercentOwner / Officer / Other
    txn_date: date           # when the trade happened (DO NOT trade on this)
    filing_date: date        # when it hit EDGAR — the ONLY actionable date
    shares: float
    price: float
    value: float             # shares * price
    is_10b5_1: bool          # scheduled -> usually excluded
    txn_code: str            # kept == "P"
```

### Phase 1 — Ingestion  *(2–4 days)*

**1a. Async bulk download** of the quarterly zips (small number of files → modest
concurrency is plenty; the politeness rules still apply):

```python
# sources/edgar_bulk.py
import asyncio, httpx
from aiolimiter import AsyncLimiter

# REQUIRED by SEC fair-access. Put a real contact email here.
HEADERS = {"User-Agent": "insider-edge research <you@example.com>"}
limiter = AsyncLimiter(max_rate=8, time_period=1.0)     # < 10 req/s
sem = asyncio.Semaphore(8)

def quarter_url(year: int, q: int) -> str:
    return ("https://www.sec.gov/files/structureddata/data/"
            f"insider-transactions-data-sets/{year}q{q}_form345.zip")

async def get(client, url, attempts=5) -> bytes:
    for i in range(attempts):
        async with limiter, sem:
            r = await client.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.content
        if r.status_code in (403, 429, 500, 502, 503, 504):
            await asyncio.sleep(2 ** i + 0.1 * i)        # backoff + jitter
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed: {url}")

async def download_quarters(quarters: list[tuple[int, int]], outdir: str):
    async with httpx.AsyncClient(http2=True) as c:
        blobs = await asyncio.gather(*(get(c, quarter_url(y, q)) for y, q in quarters))
    for (y, q), b in zip(quarters, blobs):
        (open(f"{outdir}/{y}q{q}.zip", "wb")).write(b)
```

Then **parse with DuckDB or polars** (multithreaded — uses all M5 cores). DuckDB
reads the TSVs and joins in SQL without loading everything into Python:

```python
# ingest/parse_form345.py  (DuckDB reads tab-delimited tables straight from disk)
import duckdb
con = duckdb.connect()
con.execute("SET threads TO 10")          # match M5 core count
events = con.execute("""
  SELECT s.ACCESSION_NUMBER, s.FILING_DATE, s.PERIOD_OF_REPORT,
         s.ISSUERCIK, s.ISSUERTRADINGSYMBOL AS ticker,
         o.RPTOWNERCIK AS insider_id, o.RPTOWNERNAME, o.RPTOWNER_RELATIONSHIP AS role,
         t.TRANS_DATE, t.TRANS_CODE, t.TRANS_SHARES AS shares,
         t.TRANS_PRICEPERSHARE AS price, t.TRANS_ACQUIRED_DISP_CD AS ad
  FROM read_csv_auto('data/raw/*SUBMISSION.tsv', delim='\t', header=true) s
  JOIN read_csv_auto('data/raw/*REPORTINGOWNER.tsv', delim='\t', header=true) o USING (ACCESSION_NUMBER)
  JOIN read_csv_auto('data/raw/*NONDERIV_TRANS.tsv', delim='\t', header=true) t USING (ACCESSION_NUMBER)
  WHERE s.DOCUMENT_TYPE = '4' AND t.TRANS_CODE = 'P' AND t.TRANS_ACQUIRED_DISP_CD = 'A'
""").pl()                                  # -> polars DataFrame
# (verify exact column names against the README PDF; they are the source of truth)
```

**1b. Filters** (`ingest/filters.py`): enforce `TRANS_CODE == 'P'` and acquired;
exclude 10b5-1 (post-2023 flag, else footnote scan per §4.4); normalize to the
`InsiderBuy` schema; compute `value = shares * price`. **Deliverable:**
`events.parquet`, one row per qualifying buy.
**Gate:** sanity counts — buy/sell ratio, distribution by year, and hand-verify
5–10 rows against the live filing on EDGAR.

### Phase 2 — Prices & population horizon curve  *(3–5 days)* — **FIRST DECISION GATE**

Compute forward returns **indexed off the filing date** (first tradable open
≥ `filing_date`) at **1d / 1w / 1m / 3m / 6m / 12m**. Abnormal return = stock
return − benchmark (market; ideally size/sector-matched). **Pool all buys**;
report mean abnormal return per horizon with **bootstrap CIs**.

```python
# tests/test_no_lookahead.py  — the look-ahead gate, as code
def test_entry_never_before_filing(trades):
    assert (trades["entry_date"] >= trades["filing_date"]).all()
```

GPU-accelerated bootstrap on the M5 with MLX (draw all resamples at once):

```python
# analysis/horizon_curve.py  (vectorized bootstrap on GPU via unified memory)
import mlx.core as mx

def bootstrap_mean_ci(abn_returns, n_boot=10_000, lo=2.5, hi=97.5):
    x = mx.array(abn_returns)                       # lives in unified memory
    n = x.shape[0]
    idx = mx.random.randint(0, n, shape=(n_boot, n))   # B x N on GPU
    means = mx.take(x, idx).mean(axis=1)               # B bootstrap means
    means = mx.sort(means)
    mx.eval(means)                                      # force materialization
    return float(x.mean()), (float(means[int(lo/100*n_boot)]),
                             float(means[int(hi/100*n_boot)]))
```

**Gate:** if the population curve shows **no positive, statistically meaningful
drift** (even before costs) → **STOP**. If it drifts, its **shape sets the
holding period**: drift realized by ~1m → hold ~1m; builds to 6m → hold 6m; pops
at 1w then partially reverts → first evidence an exit overlay has a job (Phase 5).

### Phase 3 — Segmentation  *(2–3 days)*
Build features and plot a horizon curve **per segment** (each segment = thousands
of obs → reliable, generalizable):
- **Role**: CEO / CFO / Director / 10% owner.
- **Size**: trade value vs. the insider's own median trade / vs. company size.
- **Cluster**: count of distinct `insider_id` buying the same issuer within window
  W (e.g., 30 days) — the strongest classic signal.
- **Opportunistic vs. routine** (Cohen-Malloy-Pomorski): routine = recurring buys
  in the same calendar month across years; opportunistic = irregular timing.
- **Market-cap band** and **distance from 52-week high**.

**Gate:** identify which *structural* segments carry the signal. **No
per-individual selection yet.**

### Phase 4 — Dumb basket backtest  *(3–5 days)* — **GO/NO-GO GATE**
Equal-weight basket of the signal (or best segments), **fixed hold = Phase-2
horizon**, monthly rebalance. Frictions are mandatory: **fills at next open** (not
midpoint), **bid-ask spread** by liquidity bucket, **market impact/slippage**
scaled to order size vs. ADV, commissions. Validate with **time-split OOS** and/or
**walk-forward**; never tune on the test set; pre-register rules.

Run the folds/parameter grid in parallel across M5 cores:

```python
# analysis/backtest.py
from concurrent.futures import ProcessPoolExecutor
def run_walk_forward(folds):              # each fold independent
    with ProcessPoolExecutor(max_workers=10) as ex:   # ~ M5 core count
        return list(ex.map(backtest_one_fold, folds))
```

Report net CAGR, Sharpe, max drawdown, hit rate, turnover, and a **capacity
estimate at the user's size**.
**Gate:** no edge net of costs OOS → **stop** (the money-saving result).

### Phase 5 — Refinements & exits  *(only past Phase 4; open-ended)*
- **Per-individual records, done honestly:** hierarchical / Bayesian **shrinkage**
  (partial pooling). Start each insider at their segment mean; let the estimate
  deviate only in proportion to trade count (4 trades → pulled back to the group;
  60 trades → trusted). Real test: does adding individual identity improve
  **out-of-sample** prediction *over the segment model*? Expect little lift beyond
  role/size/cluster for all but a handful of prolific insiders. Implement in MLX
  (autodiff on GPU) or NumPyro/PyMC.
- **Exit overlays (the user's original idea, finally fair-tested):** baseline
  **fixed-hold** vs. **trailing stop** (activate at +X%, trail width W) vs.
  **scale-out** (sell half on a double, ride rest under a wide trail). Each
  measured OOS *against* the fixed-hold baseline. **Model gap risk explicitly:** in
  illiquid names, big up-moves unwind through gap-downs, so a stop fills at the
  next print, often far below the trail level. Mind the trail-width dilemma — wide
  enough to ride a real run gives back a lot; tight enough to lock gains chops you
  out of healthy pullbacks. A long position floors at −100% (cannot go below zero),
  so EV leans on the rare large winners — do not truncate the right tail casually.

---

## 7. Bias checklist (enforce as tests; revisit every phase)
- [ ] **Survivorship** — delisted names present; delist = −100% (or recovery value)
- [ ] **Look-ahead** — `entry_date >= filing_date`; unit test asserts it
- [ ] **Costs** — spread + impact + commission; next-open fills, never midpoint
- [ ] **Overfit** — OOS / walk-forward; rules pre-registered; degrees of freedom counted

## 8. Order of attack
Phase 0 → 1 → **2 (gate)** → 3 → **4 (gate)** → 5. Do not jump to Phase 5; the
per-individual mining is the one step that can only *invent* an edge.

## 9. Expectation setting (state this in the final report)
Even if everything works, the realistic prize is a **modest, high-variance tilt**,
not a money printer — and live edges run smaller than published ones (anomaly
decay). The most probable result is "little/no edge survives at this size," which
is a valid, valuable answer. The failure mode to fear is a gorgeous backtest built
on survivorship-biased prices and curve-fit exits.
```

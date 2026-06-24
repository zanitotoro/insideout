# insideout

Research & backtest: **does a tradeable edge survive in insider open-market
purchases, for a retail-sized account, after realistic costs and out-of-sample?**

The full plan — goal, decision gates, data sources, schema, and hardware strategy
— is in [`insider-edge-plan.md`](insider-edge-plan.md). Read it before changing code.

The honest most-likely answer is **"no edge survives at this size,"** which is a
*successful* result. The failure mode to fear is a gorgeous backtest built on
survivorship-biased prices and curve-fit exits.

## Setup

```bash
uv sync                 # create the venv and install deps (incl. MLX on Apple silicon)
uv run pytest           # run the bias-gate tests
uv run ruff check .     # lint
```

Set your SEC contact email (required by EDGAR fair-access) if it differs from the
default in `src/insider_edge/config.py`:

```bash
export INSIDER_EDGE_CONTACT="you@example.com"
```

## Status

**Phase 1 complete** — ingestion verified against the real 2025q1 SEC data set.
`insider-edge download` pulls + extracts quarterly Form 345 sets; `insider-edge
parse` flattens, normalizes, dedupes co-filers, drops 10b5-1 trades, and writes
`data/parquet/events.parquet` with a sanity report. Schema corrections found in
the real data (vs. the plan's *(verify)* guesses) are documented in
`src/insider_edge/ingest/parse_form345.py`: `AFF10B5ONE` lives in `SUBMISSION` as
a messy string; `RPTOWNER_RELATIONSHIP` is comma-joined multi-role; dates are
`DD-MON-YYYY`; the accession join fans out co-filers (deduped); ~4% of prices are
footnoted placeholders (flagged, not trusted).

**Phase 2 machinery complete (gate decision pending real prices).** No-lookahead
forward/abnormal returns (`analysis/returns.py`, `join_asof` forward off the
filing date), bootstrap CIs on the M5 GPU, and the population curve
(`analysis/horizon_curve.py`). `insider-edge horizon` runs it.

The Phase 2 GATE is **not yet decided**: it currently runs on the
`YFinanceProvider` SMOKE TEST, which lacks delisted names (coverage ~86%) and so
is survivorship-biased upward — and the benchmark is plain SPY, not size/sector
matched. A trustworthy go/no-go needs a survivorship-free source (Sharadar / CRSP
/ Polygon) dropped in behind `PriceProvider`. The smoke-test curve on 2025q1 shows
significant short-horizon drift (1d–1m) decaying by ~3m — suggestive only.

Order of attack: Phase 0 → 1 → **2 (gate)** → 3 → **4 (gate)** → 5.

```bash
insider-edge download --start-year 2025 --start-q 1 --end-year 2025 --end-q 1
insider-edge parse                 # -> events.parquet + sanity report
insider-edge horizon --sample 600  # Phase 2 smoke-test curve (NOT the verdict)
```

## The four biases (enforced as tests)

- **Survivorship** — delisted names present; a delisting is booked as −100%.
- **Look-ahead** — entry on the filing date, never the transaction date.
- **Costs** — spread + impact + commission; next-open fills, never midpoint.
- **Overfit** — out-of-sample / walk-forward; rules pre-registered.

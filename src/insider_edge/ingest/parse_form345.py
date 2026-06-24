"""Join + flatten the quarterly Form 345 TSVs into a raw buy table (plan §4.1, §1a).

Column names were verified against the 2025q1 data set (and the bundled
FORM_345_readme.htm). Notable realities that differ from first guesses:
  * The 10b5-1 affirmation column AFF10B5ONE lives in SUBMISSION, not the
    transaction table, and is a *string* ('0'/'false'/'1'/'true'/null).
  * RPTOWNER_RELATIONSHIP is a comma-joined multi-role string
    ("Director,Officer,TenPercentOwner"), not a single role.
  * Dates are DD-MON-YYYY strings (e.g. 31-MAR-2025) — parsed downstream.
  * Joining on ACCESSION_NUMBER fans out transactions across co-filers; the
    normalize step dedupes to one row per (accession, NONDERIV_TRANS_SK).

This step keeps only TRANS_CODE='P' acquisitions (the buy signal) to keep the
join cheap; everything else (date parsing, role flags, dedup) is in normalize.py.
"""

from __future__ import annotations

import duckdb
import polars as pl

from insider_edge import config


def parse_buys(
    raw_root: str | None = None, *, document_types: tuple[str, ...] = ("4",)
) -> pl.DataFrame:
    """Read the extracted TSVs and return raw open-market buy rows (pre-normalize).

    `raw_root` holds the per-quarter subfolders; defaults to config.RAW_DIR.
    `document_types` defaults to originals only ("4"); pass ("4", "4/A") to include
    amendments (which risk double-counting restated transactions).
    """
    root = raw_root if raw_root is not None else str(config.RAW_DIR)
    con = duckdb.connect()
    con.execute(f"SET threads TO {config.N_THREADS}")
    doc_list = ", ".join(f"'{d}'" for d in document_types)

    def tsv(name: str) -> str:
        # union_by_name tolerates column-order/schema drift across quarters.
        return f"read_csv_auto('{root}/**/{name}.tsv', delim='\t', header=true, union_by_name=true)"

    sql = f"""
      SELECT s.ACCESSION_NUMBER                       AS accession,
             s.FILING_DATE                            AS filing_date_raw,
             s.PERIOD_OF_REPORT                       AS period_raw,
             s.DOCUMENT_TYPE                          AS document_type,
             CAST(s.ISSUERCIK AS VARCHAR)             AS issuer_cik,
             s.ISSUERNAME                             AS issuer_name,
             s.ISSUERTRADINGSYMBOL                    AS ticker,
             s.AFF10B5ONE                             AS aff10b5one,
             CAST(o.RPTOWNERCIK AS VARCHAR)           AS insider_id,
             o.RPTOWNERNAME                           AS insider_name,
             o.RPTOWNER_RELATIONSHIP                  AS relationship,
             o.RPTOWNER_TITLE                         AS title,
             t.NONDERIV_TRANS_SK                      AS txn_sk,
             t.TRANS_DATE                             AS txn_date_raw,
             t.TRANS_CODE                             AS txn_code,
             t.TRANS_SHARES                           AS shares,
             t.TRANS_PRICEPERSHARE                    AS price,
             t.TRANS_PRICEPERSHARE_FN                 AS price_fn,
             t.TRANS_ACQUIRED_DISP_CD                 AS ad,
             t.SHRS_OWND_FOLWNG_TRANS                 AS shares_owned_following,
             t.DIRECT_INDIRECT_OWNERSHIP             AS direct_indirect
      FROM {tsv("SUBMISSION")} s
      JOIN {tsv("REPORTINGOWNER")} o USING (ACCESSION_NUMBER)
      JOIN {tsv("NONDERIV_TRANS")} t USING (ACCESSION_NUMBER)
      WHERE s.DOCUMENT_TYPE IN ({doc_list})
        AND t.TRANS_CODE = 'P'
        AND t.TRANS_ACQUIRED_DISP_CD = 'A'
    """
    return con.execute(sql).pl()


def code_distribution(raw_root: str | None = None) -> pl.DataFrame:
    """Buy/sell/other transaction-code counts (for the Phase 1 sanity gate)."""
    root = raw_root if raw_root is not None else str(config.RAW_DIR)
    con = duckdb.connect()
    con.execute(f"SET threads TO {config.N_THREADS}")
    return con.execute(
        f"""
        SELECT TRANS_CODE, TRANS_ACQUIRED_DISP_CD AS ad, count(*) AS n
        FROM read_csv_auto('{root}/**/NONDERIV_TRANS.tsv', delim='\t',
                           header=true, union_by_name=true)
        GROUP BY 1, 2 ORDER BY n DESC
        """
    ).pl()


def load_footnotes(raw_root: str | None = None) -> pl.DataFrame:
    """Footnote text per accession (for pre-2023 10b5-1 detection, plan §4.4)."""
    root = raw_root if raw_root is not None else str(config.RAW_DIR)
    con = duckdb.connect()
    con.execute(f"SET threads TO {config.N_THREADS}")
    return con.execute(
        f"""
        SELECT ACCESSION_NUMBER, FOOTNOTE_TXT
        FROM read_csv_auto('{root}/**/FOOTNOTES.tsv', delim='\t',
                           header=true, union_by_name=true)
        """
    ).pl()

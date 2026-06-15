"""
download_rates.py
=================
Download daily Refinitiv/LSEG data for the EUR duration-overlay project
(Bund / OAT / BTP tactical duration, à la the Vontobel FINCA mandate).

Mirrors the structure of download_reuters.py: dual eikon / refinitiv-data
backend, retry-with-backoff, one CSV per instrument in outputs/. A separate
build_panel.py (to be written) joins these into the analysis panel and
constructs the roll-adjusted continuous futures and the excess-return series.

WHY THESE INSTRUMENTS (project rationale)
-----------------------------------------
A duration overlay's daily P&L is, to first order,
        P&L_i  ≈  active_DV01_i · (−Δy_i)  ≈  notional_i · futures_return_i
so the whole job is (a) a one-step-ahead return/yield forecast per market and
(b) risk-controlled sizing. That means we need two parallel data sets:

  1. CASH BENCHMARK YIELDS  (the signal layer)
     The Carry / Mean-Reversion / Momentum models all operate on the yield
     CURVE, not on futures prices. Yields give clean, roll-free continuous
     series. We pull the full DE/FR/IT curve (2/5/10/30Y) plus the wider
     global markets the FINCA research model uses (US/UK/CA/AU) so the
     "global rates factor" feature set is available even though the fund
     only TRADES the EUR three.

  2. FUTURES  (the execution layer)
     What the fund actually holds. Needed for realistic roll cost, financing,
     and to cross-check the yield-space P&L. Eurex bond futures are quarterly
     (H/M/U/Z = Mar/Jun/Sep/Dec); the generic ...c1/...c2 continuations carry
     a basis jump at each roll, so we ALSO pull the individual quarterly
     contracts to let build_panel construct a back-adjusted series.

  3. RISK / REGIME PROXIES  (the conditioning layer)
     Bond vol (MOVE if entitled), equity vol (VSTOXX), EUR/USD, and a
     commodity/inflation proxy (Brent). These feed regime gating and the
     vol-targeting overlay — the single biggest lever given the Mar-2026
     drawdown was a vol/regime event, not a signal-accuracy event.

OUTPUT (outputs/)
-----------------
    yields_de.csv, yields_fr.csv, yields_it.csv   wide: date × {2y,5y,10y,30y}
    yields_global.csv                             wide: 10Y for US/GB/CA/AU
    fut_bund.csv  fut_bobl.csv  fut_schatz.csv  fut_buxl.csv
    fut_oat.csv   fut_btp.csv                     generic c1 (+ c2) close
    fut_chain_<root>.csv                          individual quarterly contracts
    short_rates.csv                               3M Euribor, €STR
    risk_proxies.csv                              VSTOXX, EUR=, Brent, (MOVE)
    fut_roll_dates.csv                            approx Eurex roll-day flags

USAGE
-----
    python download_rates.py
    python download_rates.py --start 1999-01-01 --end 2026-05-01
    python download_rates.py --skip-existing
    python download_rates.py --only yields_de.csv fut_bund.csv

REQUIREMENTS
------------
    pip install eikon pandas numpy        # legacy
    # or: pip install refinitiv-data      # newer LSEG Data Library
    Workspace / Eikon Desktop must be running and logged in.
        export REFINITIV_APP_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

RIC / FIELD CAVEATS  (verify a couple in Workspace; entitlements vary)
---------------------------------------------------------------------
  * Benchmark yield instruments use the "=RR" (Reuters Reference) chain and
    quote IN YIELD, so the CLOSE / TRDPRC_1 field returns the yield directly.
    If a given tenor comes back empty, try field "TR.MIDYIELD" (ADC) — the
    per-instrument `field` slot below makes that a one-line change.
  * Eurex quarterly RICs here are <ROOT><MONTH><YY> (e.g. FGBLZ25). Some
    older feed vintages use a single-digit year (FGBLZ5). If the chain pull
    is empty, flip `YEAR_DIGITS` to 1.
  * The mandate benchmark (Bloomberg Euro Treasury 1-10Y TR) is Bloomberg IP
    and usually NOT on Refinitiv. Pull it from Bloomberg, or reconstruct an
    iBoxx/own 1-10Y total-return basket. Flagged, not silently faked.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ─── Instrument maps ────────────────────────────────────────────────────────
# Each entry: out_file -> list of (RIC, column_name, field_or_None)
# field=None  -> use the backend's default close field (CLOSE / TRDPRC_1).
# For "=RR" yield instruments the default field already returns the yield.

YIELD_CURVES: dict[str, list[tuple[str, str, Optional[str]]]] = {
    "yields_de.csv": [
        ("DE2YT=RR",  "de_2y",  None),
        ("DE5YT=RR",  "de_5y",  None),
        ("DE10YT=RR", "de_10y", None),
        ("DE30YT=RR", "de_30y", None),
    ],
    "yields_fr.csv": [
        ("FR2YT=RR",  "fr_2y",  None),
        ("FR5YT=RR",  "fr_5y",  None),
        ("FR10YT=RR", "fr_10y", None),
        ("FR30YT=RR", "fr_30y", None),
    ],
    "yields_it.csv": [
        ("IT2YT=RR",  "it_2y",  None),
        ("IT5YT=RR",  "it_5y",  None),
        ("IT10YT=RR", "it_10y", None),
        ("IT30YT=RR", "it_30y", None),
    ],
    # Wider markets used by the FINCA research model for the global rates
    # factor (not traded in the fund). 10Y is enough for the factor; add
    # 2Y here if you want curve features for these too.
    "yields_global.csv": [
        ("US10YT=RR", "us_10y", None),
        ("GB10YT=RR", "gb_10y", None),
        ("CA10YT=RR", "ca_10y", None),
        ("AU10YT=RR", "au_10y", None),
    ],
}

# Generic-continuation futures. We grab c1 and c2 so build_panel can compute
# the calendar spread / implied roll yield and back-adjust.
FUTURES_CONT: dict[str, list[tuple[str, str, Optional[str]]]] = {
    "fut_bund.csv":   [("FGBLc1", "bund_c1", None),   ("FGBLc2", "bund_c2", None)],
    "fut_bobl.csv":   [("FGBMc1", "bobl_c1", None),   ("FGBMc2", "bobl_c2", None)],
    "fut_schatz.csv": [("FGBSc1", "schatz_c1", None), ("FGBSc2", "schatz_c2", None)],
    "fut_buxl.csv":   [("FGBXc1", "buxl_c1", None),   ("FGBXc2", "buxl_c2", None)],
    "fut_oat.csv":    [("FOATc1", "oat_c1", None),    ("FOATc2", "oat_c2", None)],
    "fut_btp.csv":    [("FBTPc1", "btp_c1", None),    ("FBTPc2", "btp_c2", None)],
}

SHORT_RATES: dict[str, list[tuple[str, str, Optional[str]]]] = {
    "short_rates.csv": [
        ("EURIBOR3MD=", "euribor_3m", None),  # legacy fixing; verify if empty
        ("EUROSTR=",    "estr",       None),  # €STR; some feeds use ESTRON=
    ],
}

RISK_PROXIES: dict[str, list[tuple[str, str, Optional[str]]]] = {
    "risk_proxies.csv": [
        (".VDAXNEW", "vdax",  None),  # DAX implied vol — EU vol regime proxy.
                                      # Replaces .V2TX (VSTOXX): only ~24 obs on
                                      # this feed per the carbon pull probe.
        ("EUR=",   "eurusd", None),   # USD control — may be entitlement-gated
                                      # here; if empty, pull EURUSD=X via yfinance
        ("LCOc1",  "brent",  None),   # commodity / inflation proxy
        (".MOVE",  "move",   None),   # US bond vol — often not entitled
    ],
}

# ── Credit (optional; SEPARATE ENTITLEMENT — VERIFY EVERY RIC IN WORKSPACE) ──
# WHY: *sovereign* CDS (IT/FR) maps ~1:1 to the BTP-Bund / OAT-Bund SPREAD —
# the relative-value leg of the overlay — and can LEAD the cash spread in
# stress because it prices redenomination risk directly. The CDS-bond basis is
# itself a signal. *Broad* credit indices (iTraxx) are REGIME proxies (flight-
# to-quality co-movers), NOT yield drivers. A single corporate name (e.g.
# Oracle) or private debt does NOT drive sovereign yields — excluded by design.
#
# CDS/credit RICs vary by vintage and entitlement; the strings below are
# ILLUSTRATIVE. Confirm via Workspace CDS/credit search before trusting them.
CDS_SOVEREIGN: dict[str, list[tuple[str, str, Optional[str]]]] = {
    "cds_sovereign.csv": [
        ("ITGV5YEUAC=R", "it_5y", None),  # Italy 5Y sovereign CDS (bp) — VERIFY
        ("FRGV5YEUAC=R", "fr_5y", None),  # France 5Y sovereign CDS (bp) — VERIFY
        ("DEGV5YEUAC=R", "de_5y", None),  # Germany 5Y sovereign CDS (bp) — VERIFY
    ],
}
CREDIT_INDICES: dict[str, list[tuple[str, str, Optional[str]]]] = {
    "credit_indices.csv": [
        ("ITRXEBE5Y=R", "itraxx_main",  None),  # iTraxx Europe Main 5Y — VERIFY
        ("ITRXEXE5Y=R", "itraxx_xover", None),  # iTraxx Crossover 5Y   — VERIFY
    ],
}

# All wide-format pulls in one mapping for the driver loop. Credit pulls are
# included but degrade gracefully (empty -> skipped) if you lack entitlements.
WIDE_PULLS: dict[str, list[tuple[str, str, Optional[str]]]] = {
    **YIELD_CURVES, **FUTURES_CONT, **SHORT_RATES, **RISK_PROXIES,
    **CDS_SOVEREIGN, **CREDIT_INDICES,
}

# ─── Quarterly futures chains (for roll-adjusted continuous series) ──────────
QUARTERLY_MONTHS = {3: "H", 6: "M", 9: "U", 12: "Z"}
YEAR_DIGITS = 2  # flip to 1 if your feed uses single-digit-year RICs

# root -> (out_file, first_year_available). BTP future launched Sep-2009.
# NOTE: the carbon pull found dated-contract RICs (CFI2Z<yy>) returned NO DATA
# on this feed. The same dated style is used here (FGBLZ25, FOATH26, …), so
# these chain pulls may also come back empty. If so: (1) try YEAR_DIGITS = 1,
# (2) verify one contract RIC in Workspace, or (3) fall back to the generic
# continuations (fut_*.csv) + fut_roll_dates.csv, which build_panel handles.
FUT_CHAIN_ROOTS: dict[str, tuple[str, int]] = {
    "FGBL": ("fut_chain_bund.csv", 1999),
    "FOAT": ("fut_chain_oat.csv",  1999),
    "FBTP": ("fut_chain_btp.csv",  2009),
}

def eurex_ric(root: str, year: int, month: int) -> str:
    """Eurex quarterly future RIC, e.g. eurex_ric('FGBL', 2025, 12) -> FGBLZ25."""
    code = QUARTERLY_MONTHS[month]
    yy = year % 100
    return f"{root}{code}{yy:0{YEAR_DIGITS}d}" if YEAR_DIGITS == 2 else f"{root}{code}{yy % 10}"


# ─── Backend abstraction (identical contract to download_reuters.py) ─────────
# Connection-check instruments. EUR= is intentionally NOT first: on some feeds
# (confirmed on this user's entitlements) EUR= is restricted. We try a small
# list and pass if ANY returns data, so a working session is never rejected.
SMOKE_RICS = [".GDAXI", "DE10YT=RR", "LCOc1", "EUR="]


class Backend:
    name: str
    def open(self, app_key: str) -> None: raise NotImplementedError
    def close(self) -> None: pass
    def get_timeseries(self, ric, start, end, field=None) -> pd.DataFrame:
        raise NotImplementedError

    def smoke_test(self) -> None:
        """Force a tiny REAL request so a half-open session fails HERE, at
        connect time, instead of on the first data pull. Passes if ANY of the
        candidate RICs returns data; raises only if the session genuinely
        can't fetch anything (so a working session is never wrongly rejected)."""
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=10)
        last_err = None
        for ric in SMOKE_RICS:
            try:
                df = self.get_timeseries(ric, start.strftime("%Y-%m-%d"),
                                         end.strftime("%Y-%m-%d"))
                if df is not None and not df.empty:
                    return  # session works
            except Exception as e:
                last_err = e
        raise RuntimeError(
            f"{self.name}: session opened but no test RIC returned data "
            f"({', '.join(SMOKE_RICS)}) — session not truly connected"
            + (f"; last error: {last_err}" if last_err else ""))


class EikonBackend(Backend):
    name = "eikon"
    def open(self, app_key: str) -> None:
        import eikon as ek
        self._ek = ek
        ek.set_app_key(app_key)

    def get_timeseries(self, ric, start, end, field=None):
        f = field or "CLOSE"
        df = self._ek.get_timeseries(
            ric, start_date=start, end_date=end, interval="daily", fields=f,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        if f in df.columns:
            df = df[[f]].rename(columns={f: "value"})
        else:
            df = df.iloc[:, [0]]
            df.columns = ["value"]
        return df


class RDBackend(Backend):
    """refinitiv-data / LSEG Data Library backend.

    'Session is not opened. Can't send any request' almost always means the
    session was created but never CONNECTED. The key lesson from the working
    carbon pull: with Workspace running, rd.open_session() with NO ARGUMENTS
    connects via Workspace's own auth. Passing app_key= can instead resolve a
    session that never connects (or feeds it a stale key), which opens without
    error and only fails on the first request. So we open NO-ARG FIRST and only
    fall back to an explicit desktop Definition if that fails.
    """
    name = "refinitiv-data"
    # Price fields: futures/indices/FX (futures often have empty TRDPRC_1, so
    # SETTLE rescues them). Yield fields: =RR benchmark yields have NO traded
    # price — the value lives in a yield/ADC field. We try price fields first
    # (with a daily interval), then yield fields (ADC, no interval), then the
    # instrument's DEFAULT fields (no fields arg at all). First non-empty wins.
    PRICE_FIELDS = ["TRDPRC_1", "SETTLE", "CF_LAST"]
    YIELD_FIELDS = ["TR.MIDYIELD", "TR.YIELDTOMATURITY", "MID_YLD_1", "YIELD"]

    def open(self, app_key: str) -> None:
        import refinitiv.data as rd
        self._rd = rd
        # (a) no-arg open — proven to work with a running Workspace (carbon).
        try:
            rd.open_session()
            return
        except Exception as e:
            print(f"      rd.open_session() failed: {e}")
        # (b) explicit desktop session with app key (only if no-arg failed,
        #     e.g. no config file present)
        if app_key:
            sess = rd.session.desktop.Definition(app_key=app_key).get_session()
            sess.open()
            rd.session.set_default(sess)
        else:
            raise RuntimeError("rd.open_session() failed and no app_key for a "
                               "fallback desktop session")

    def close(self):
        try:
            self._rd.close_session()
        except Exception:
            pass

    def _hist(self, ric, start, end, fields, use_interval):
        kw = dict(universe=ric, start=start, end=end)
        if fields is not None:
            kw["fields"] = fields
        if use_interval:
            kw["interval"] = "1D"
        return self._rd.get_history(**kw)

    @staticmethod
    def _norm(df) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        # coerce a date column to the index if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            for c in list(df.columns):
                if str(c).lower() in ("date", "timestamp"):
                    df = df.set_index(c)
                    break
        try:
            df.index = pd.to_datetime(df.index).normalize()
        except Exception:
            return pd.DataFrame()
        df.index.name = "date"
        df = df.iloc[:, [0]]
        df.columns = ["value"]
        return df.dropna()

    def get_timeseries(self, ric, start, end, field=None):
        if field:
            attempts = [([field], True), ([field], False)]
        else:
            attempts = ([([f], True) for f in self.PRICE_FIELDS]
                        + [([f], False) for f in self.YIELD_FIELDS]
                        + [(None, False)])  # instrument default fields
        for fields, use_int in attempts:
            try:
                df = self._hist(ric, start, end, fields, use_int)
            except Exception:
                continue
            out = self._norm(df)
            if not out.empty:
                return out
        return pd.DataFrame()


def open_backend(app_key: str, prefer: str = "auto") -> Backend:
    """Open and VERIFY a data session. 'auto' tries refinitiv-data FIRST,
    because its no-arg open authenticates via a running Workspace and needs no
    app key — the most reliable path and it avoids spurious eikon app-key
    errors. Falls back to eikon. Each candidate must pass smoke_test(): a
    session that opens but can't fetch is rejected so the next one is tried."""
    order = {
        "auto":            [RDBackend, EikonBackend],
        "eikon":           [EikonBackend],
        "refinitiv-data":  [RDBackend],
    }.get(prefer, [RDBackend, EikonBackend])

    last_err = None
    for cls in order:
        be = cls()
        try:
            be.open(app_key)
            be.smoke_test()  # ← verify the session actually works
            print(f"  ✓ Connected via {be.name} (session verified)")
            return be
        except ImportError as e:
            last_err = e
            print(f"  ⚠ {cls.name}: library not installed ({e})")
        except Exception as e:
            last_err = e
            print(f"  ⚠ {cls.name}: not usable — {e}")
            try:
                be.close()
            except Exception:
                pass
    raise SystemExit(
        "\nNo working Refinitiv session.\n"
        "  Checklist:\n"
        "   1. Is the Workspace / Eikon DESKTOP app running and logged in?\n"
        "      (desktop sessions talk to the local proxy on port 9000)\n"
        "   2. Is REFINITIV_APP_KEY a valid *Eikon Data API* key from the\n"
        "      'App Key Generator' app inside Workspace?\n"
        "   3. Try forcing a backend:  --backend eikon   (or  refinitiv-data)\n"
        "   4. Headless server with no Workspace? You need a PLATFORM session\n"
        "      with machine credentials, not a desktop session — tell me and\n"
        "      I'll wire that path.\n"
        f"  Last error: {last_err}"
    )


# ─── Download helpers ───────────────────────────────────────────────────────
def fetch_with_retry(backend, ric, start, end, field=None,
                     max_retries=3, backoff=2.0) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            df = backend.get_timeseries(ric, start, end, field=field)
            if not df.empty:
                return df
            print(f"      empty result, retry {attempt}/{max_retries}")
        except Exception as e:
            print(f"      error on attempt {attempt}: {e}")
        time.sleep(backoff ** attempt)
    return pd.DataFrame()


def download_wide(backend, mapping, start, end, skip_existing=False,
                  only=None) -> None:
    """Pull each (multi-RIC) file as a single wide DataFrame on a date index."""
    for fname, specs in mapping.items():
        if only is not None and fname not in only:
            continue
        out = OUT_DIR / fname
        if skip_existing and out.exists():
            print(f"  skip {fname} (exists)"); continue
        print(f"\n  {fname}")
        cols = {}
        for ric, colname, field in specs:
            print(f"    {colname:12s} [{ric}]")
            df = fetch_with_retry(backend, ric, start, end, field=field)
            if df.empty:
                print(f"      ⚠ no data for {ric}")
                continue
            cols[colname] = df["value"]
            print(f"      ✓ {len(df)} obs, {df.index[0].date()} → {df.index[-1].date()}")
        if not cols:
            print(f"      ⚠ nothing retrieved for {fname}")
            continue
        wide = pd.concat(cols, axis=1).sort_index()
        wide.index.name = "date"
        wide.to_csv(out)
        print(f"      → {fname}: {wide.shape[0]} rows × {wide.shape[1]} cols")


def download_fut_chains(backend, start, end, skip_existing=False,
                        only=None) -> None:
    """Pull every quarterly contract per root into a long panel so build_panel
    can construct a back-adjusted continuous series.

    Output schema (long): date, root, contract_code, year, month, ric, close
    """
    print("\n──── Quarterly futures chains ────")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    for root, (fname, first_year) in FUT_CHAIN_ROOTS.items():
        if only is not None and fname not in only:
            continue
        out = OUT_DIR / fname
        if skip_existing and out.exists():
            print(f"  skip {fname} (exists)"); continue
        print(f"\n  {root} → {fname}")
        # CANARY: test a recent contract first. If dated RICs aren't carried on
        # this feed (as with CFI2Z previously), skip the whole root in seconds
        # instead of grinding through hundreds of empty contracts.
        canary_ok = False
        for cy in (end_ts.year, end_ts.year - 1):
            cric = eurex_ric(root, cy, 12)
            ctest = fetch_with_retry(backend, cric,
                                     f"{cy - 1}-06-01", f"{cy}-12-28",
                                     max_retries=1)
            if not ctest.empty:
                canary_ok = True
                break
        if not canary_ok:
            print(f"  ⚠ dated contracts for {root} not on this feed "
                  f"(canary {eurex_ric(root, end_ts.year, 12)} empty). "
                  f"Skipping — build_panel uses the {root} continuation + roll "
                  f"flags. (Try YEAR_DIGITS=1 if you expect them to exist.)")
            continue
        rows: list[pd.DataFrame] = []
        for yr in range(max(first_year, start_ts.year), end_ts.year + 2):
            for mo in QUARTERLY_MONTHS:
                ric = eurex_ric(root, yr, mo)
                # A contract trades for ~9 months before delivery; pad the
                # window so we capture its full active life inside the sample.
                s = max(start_ts, pd.Timestamp(f"{yr - 1}-06-01"))
                e = min(end_ts, pd.Timestamp(f"{yr}-{mo:02d}-28"))
                if s >= e:
                    continue
                df = fetch_with_retry(backend, ric,
                                      s.strftime("%Y-%m-%d"),
                                      e.strftime("%Y-%m-%d"),
                                      max_retries=1)
                if df.empty:
                    continue
                df = (df.rename(columns={"value": "close"})
                        .assign(root=root, contract_code=QUARTERLY_MONTHS[mo],
                                year=yr, month=mo, ric=ric)
                        .reset_index())
                rows.append(df[["date", "root", "contract_code",
                                "year", "month", "ric", "close"]])
                print(f"    ✓ {ric:10s} {len(df):4d} obs")
        if not rows:
            print(f"  ⚠ no contracts retrieved for {root}")
            continue
        chain = pd.concat(rows, ignore_index=True)
        chain.to_csv(out, index=False)
        print(f"  → {fname}: {len(chain)} rows")


def build_roll_flags() -> None:
    """Approximate Eurex bond-future roll-day flags on the Bund c1 index.

    Eurex fixed-income delivery is the 10th calendar day of the contract month
    (next business day if non-trading); liquidity rolls ~5 business days before.
    We flag the roll window [delivery-7bd, delivery] so build_panel can drop
    returns that span the roll if using the generic continuation.
    """
    bund = OUT_DIR / "fut_bund.csv"
    if not bund.exists():
        print("  ⚠ fut_bund.csv missing; skipping roll-flag build")
        return
    df = pd.read_csv(bund, index_col=0, parse_dates=True)
    idx = df.index
    flags = pd.Series(False, index=idx, name="roll_window_c1")
    for yr in range(idx.year.min(), idx.year.max() + 1):
        for mo in QUARTERLY_MONTHS:
            delivery = pd.Timestamp(f"{yr}-{mo:02d}-10")
            # snap to first trading day on/after the 10th present in the index
            pos = idx.searchsorted(delivery)
            if pos >= len(idx):
                continue
            d_pos = pos
            start_pos = max(0, d_pos - 7)
            flags.iloc[start_pos:d_pos + 1] = True
    out = OUT_DIR / "fut_roll_dates.csv"
    flags.to_frame().to_csv(out)
    print(f"  ✓ {out.name}: {int(flags.sum())} roll-window days flagged")


# ─── Main ───────────────────────────────────────────────────────────────────
def probe_rics(backend, rics) -> None:
    """For each RIC, report which fields return data on THIS feed (and how
    many obs), trying both with and without a daily interval. Use this to
    confirm the right field/RIC before a full run — e.g. yields usually need
    TR.MIDYIELD, futures need TRDPRC_1 or SETTLE, dated contracts may be empty."""
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=30)
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    candidate_fields = ["TRDPRC_1", "SETTLE", "CF_LAST", "B_YLD_1", "MID_YLD_1",
                        "YIELD", "TR.MIDYIELD", "TR.YIELDTOMATURITY", "TR.MIDPRICE"]
    rd = getattr(backend, "_rd", None)
    print("\n──── Field probe ────")
    for ric in rics:
        print(f"\n  {ric}")
        hits = []
        if rd is not None:  # RD backend: probe each field directly, both modes
            for f in candidate_fields:
                for use_int in (True, False):
                    try:
                        kw = dict(universe=ric, fields=[f], start=s, end=e)
                        if use_int:
                            kw["interval"] = "1D"
                        df = rd.get_history(**kw)
                        if df is not None and len(df) and not df.dropna().empty:
                            hits.append((f, "1D" if use_int else "—", len(df.dropna())))
                            break
                    except Exception:
                        pass
            try:  # instrument default fields (no fields arg)
                df = rd.get_history(universe=ric, start=s, end=e)
                if df is not None and len(df):
                    cols = ",".join(str(c) for c in df.columns[:4])
                    hits.append((f"(default: {cols})", "—", len(df)))
            except Exception:
                pass
        else:  # generic: just use the backend's own cascade
            df = backend.get_timeseries(ric, s, e)
            if not df.empty:
                hits.append(("(auto-cascade)", "—", len(df)))
        if hits:
            for f, iv, n in hits:
                print(f"    ✓ {f:30s} interval={iv:3s} {n:4d} obs")
        else:
            print("    ⚠ nothing returned — RIC likely wrong or unentitled here")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download EUR rates data for the duration overlay")
    p.add_argument("--start", default="1999-01-01",
                   help="ISO start date (default: 1999-01-01, EUR era)")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--app-key", default=os.environ.get("REFINITIV_APP_KEY"))
    p.add_argument("--backend", default="auto",
                   choices=["auto", "eikon", "refinitiv-data"],
                   help="force a data backend (default: auto = refinitiv-data then eikon)")
    p.add_argument("--selftest", action="store_true",
                   help="just open+verify the session and pull one series, then exit")
    p.add_argument("--probe", nargs="*", default=None,
                   help="diagnostic: for each RIC, report which fields return "
                        "data on your feed, then exit. "
                        "E.g. --probe DE10YT=RR FGBLc1 FGBLZ25")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--only", nargs="*", default=None,
                   help="Only these output files (e.g. --only yields_de.csv fut_bund.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.app_key:
        # Not fatal: with Workspace running, the no-arg rd.open_session() path
        # authenticates via Workspace and needs no key. Only the explicit
        # desktop-Definition fallback and the eikon backend require one.
        print("  · note: no REFINITIV_APP_KEY set — relying on the running "
              "Workspace session (no-arg open). Pass --app-key to enable the "
              "explicit fallback / eikon backend.")

    print("=" * 72)
    print(" Download Refinitiv data — EUR duration overlay (Bund/OAT/BTP)")
    print("=" * 72)
    print(f"  Sample: {args.start} → {args.end}")
    print(f"  Output: {OUT_DIR}")

    backend = open_backend(args.app_key, prefer=args.backend)

    if args.probe is not None:
        probe_rics(backend, args.probe or ["DE10YT=RR", "FGBLc1",
                                           eurex_ric("FGBL", pd.Timestamp.today().year, 12)])
        backend.close()
        return

    if args.selftest:
        print("\n──── Self-test: pulling DE10YT=RR (1 month) ────")
        df = backend.get_timeseries("DE10YT=RR",
                                    (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                                    pd.Timestamp.today().strftime("%Y-%m-%d"))
        if df.empty:
            print("  ⚠ empty — DE10YT=RR may need field 'TR.MIDYIELD' on your feed")
        else:
            print(df.tail(3).to_string())
            print(f"  ✓ self-test OK: {len(df)} obs via {backend.name}")
        backend.close()
        return

    only = set(args.only) if args.only else None
    try:
        print("\n──── Wide-format pulls (yields / futures / rates / proxies) ────")
        download_wide(backend, WIDE_PULLS, args.start, args.end,
                      args.skip_existing, only)
        # Chains are slow (many contracts); skip unless requested or full run.
        want_chains = only is None or any(f.startswith("fut_chain_") for f in only)
        if want_chains:
            download_fut_chains(backend, args.start, args.end,
                                args.skip_existing, only)
        print("\n──── Roll-day flags ────")
        build_roll_flags()
    finally:
        backend.close()

    print("\n" + "=" * 72)
    print(" Done. Files in outputs/:")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(f"   {p.name:28s}  {p.stat().st_size / 1024:8.1f} KB")
    print("=" * 72)


if __name__ == "__main__":
    main()

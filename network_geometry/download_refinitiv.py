"""
Refinitiv / LSEG Workspace data download for the Bund Duration Overlay project.
==============================================================================

Run this ON THE MACHINE WITH LSEG WORKSPACE (or Eikon) RUNNING and logged in.

Install once:
    pip install lseg-data          # new library (recommended)
    #  or:  pip install refinitiv-data   (older name, same API surface)
    #  or:  pip install eikon            (legacy; see fallback at bottom)

Then:
    python download_refinitiv.py

Output: ./rft_data/ -> one wide CSV per block + rft_daily_all.csv (date x RIC)
Attach the folder (or just rft_daily_all.csv) to the Claude session.

NOTE on RICs: benchmark yield RICs ("DE10YT=RR") and FX vol RICs can differ by
entitlement package. If a RIC errors, check the correct code in Workspace
(search the instrument, copy the RIC) and edit the BLOCKS dict below.
"""

import os
import datetime as dt
import pandas as pd

try:
    import lseg.data as ld  # pip install lseg-data
    _LIB = "lseg"
except ImportError:
    try:
        import refinitiv.data as ld  # pip install refinitiv-data
        _LIB = "refinitiv"
    except ImportError:
        ld = None
        _LIB = None

START = "2004-01-01"
END = dt.date.today().isoformat()
OUTDIR = "rft_data"
os.makedirs(OUTDIR, exist_ok=True)

BLOCKS = {
    # Benchmark government yields (=RR RICs quote the yield in the close)
    "yields": [
        "DE2YT=RR", "DE5YT=RR", "DE10YT=RR", "DE30YT=RR",
        "US2YT=RR", "US10YT=RR",
        "GB10YT=RR", "JP10YT=RR", "IT10YT=RR", "FR10YT=RR",
        "ES10YT=RR", "CH10YT=RR",
    ],
    # Continuous futures
    "futures": [
        "FGBLc1",   # Bund
        "FGBMc1",   # Bobl
        "FGBSc1",   # Schatz
        "TYc1",     # US 10y note
        "FLGc1",    # Gilt
        "JGBc1",    # JGB
    ],
    # Equity & volatility nodes
    "equity_vol": [
        ".SPX", ".GDAXI", ".STOXX50E",
        ".VIX", ".V2TX", ".MOVE",
    ],
    # FX spot (EUR crosses)
    "fx_spot": [
        "EUR=", "EURJPY=R", "EURGBP=R", "EURCHF=R",
        "EURSEK=R", "EURNOK=R", "EURAUD=R", "EURCAD=R",
    ],
    # FX 3m ATM implied vols (check exact RICs in Workspace if these error)
    "fx_ivol": [
        "EUR3MO=R", "EURJPY3MO=R", "EURGBP3MO=R", "EURCHF3MO=R",
        "EURSEK3MO=R", "EURNOK3MO=R", "EURAUD3MO=R", "EURCAD3MO=R",
    ],
    # Short rate
    "money_market": [
        "EURIBOR3MD=", "EUROSTR=",
    ],
}


def fetch_block(rics):
    """Daily close history for a list of RICs -> wide DataFrame."""
    df = ld.get_history(universe=rics, start=START, end=END, interval="daily")
    # get_history returns MultiIndex columns (ric, field) for multi-universe;
    # keep the close-like field per RIC.
    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        for ric in df.columns.get_level_values(0).unique():
            sub = df[ric]
            for f in ("TRDPRC_1", "CLOSE", "MID_PRICE", "OFFICIAL_CLOSE", "B_YLD_1", "RY"):
                if f in sub.columns and sub[f].notna().sum() > 0:
                    out[ric] = pd.to_numeric(sub[f], errors="coerce")
                    break
            else:
                num = sub.apply(pd.to_numeric, errors="coerce")
                good = num.notna().sum()
                if good.max() > 0:
                    out[ric] = num[good.idxmax()]
        df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def main():
    if ld is None:
        raise SystemExit("Install the API first:  pip install lseg-data")
    ld.open_session()   # desktop session via running Workspace

    all_frames = []
    for name, rics in BLOCKS.items():
        print(f"Fetching block: {name} ({len(rics)} RICs)")
        try:
            df = fetch_block(rics)
        except Exception as e:
            print(f"  block failed en masse ({e}); retrying one by one")
            cols = {}
            for r in rics:
                try:
                    cols[r] = fetch_block([r]).iloc[:, 0]
                except Exception as e2:
                    print(f"  !! {r}: {e2}")
            df = pd.DataFrame(cols)
        df.to_csv(os.path.join(OUTDIR, f"{name}.csv"))
        print(f"  -> {df.shape[0]} rows, {df.shape[1]} series")
        all_frames.append(df)

    allw = pd.concat(all_frames, axis=1).sort_index()
    allw.to_csv(os.path.join(OUTDIR, "rft_daily_all.csv"))
    print("\nCoverage summary (non-null obs per series):")
    print(allw.notna().sum().sort_values().to_string())
    print(f"\nSaved -> {OUTDIR}/rft_daily_all.csv "
          f"({allw.shape[0]} rows x {allw.shape[1]} cols)")
    ld.close_session()


# ---------------------------------------------------------------------------
# Legacy Eikon fallback (uncomment if you only have the old eikon library):
#
# import eikon as ek
# ek.set_app_key("YOUR_APP_KEY")   # generate in Workspace: APPKEY
# df = ek.get_timeseries(["DE10YT=RR"], fields="CLOSE",
#                        start_date=START, end_date=END, interval="daily")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

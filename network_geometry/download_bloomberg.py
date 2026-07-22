"""
Bloomberg data download for the Bund Duration Overlay project.
=============================================================

Run this ON THE MACHINE WITH A BLOOMBERG TERMINAL (logged in), using the
Desktop API (localhost:8194 — enabled by default with a terminal session).

Install the API package once:
    pip install blpapi --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/

Then:
    python download_bloomberg.py

Output: ./bbg_data/  -> one wide CSV per block + bbg_daily_all.csv (date x ticker)
Attach the whole bbg_data folder (or just bbg_daily_all.csv) to the Claude session.
"""

import os
import datetime as dt
import pandas as pd

try:
    import blpapi
except ImportError:
    raise SystemExit("blpapi not installed. Run:\n  pip install blpapi --index-url "
                     "https://blpapi.bloomberg.com/repository/releases/python/simple/")

START = "20040101"
END = dt.date.today().strftime("%Y%m%d")
OUTDIR = "bbg_data"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
BLOCKS = {
    # German curve + global 10y benchmark yields (generic Bloomberg yield indices)
    "yields": [
        "GDBR2 Index", "GDBR5 Index", "GDBR10 Index", "GDBR30 Index",
        "USGG2YR Index", "USGG10YR Index",
        "GUKG10 Index", "GJGB10 Index", "GBTPGR10 Index", "GFRN10 Index",
        "GSPG10YR Index", "GSWISS10 Index",
    ],
    # Futures (continuous front contracts, for validating synthetic returns)
    "futures": [
        "RX1 Comdty",   # Bund
        "OE1 Comdty",   # Bobl
        "DU1 Comdty",   # Schatz
        "TY1 Comdty",   # US 10y note
        "G 1 Comdty",   # Gilt
        "JB1 Comdty",   # JGB
    ],
    # Equity & volatility nodes for the cross-asset network
    "equity_vol": [
        "SPX Index", "DAX Index", "SX5E Index",
        "VIX Index", "V2X Index", "MOVE Index",
    ],
    # FX spot (EUR crosses)
    "fx_spot": [
        "EURUSD Curncy", "EURJPY Curncy", "EURGBP Curncy", "EURCHF Curncy",
        "EURSEK Curncy", "EURNOK Curncy", "EURAUD Curncy", "EURCAD Curncy",
    ],
    # FX 3m ATM implied vols (upgrade the gravity-radius factor to implied vol)
    "fx_ivol": [
        "EURUSDV3M Curncy", "EURJPYV3M Curncy", "EURGBPV3M Curncy",
        "EURCHFV3M Curncy", "EURSEKV3M Curncy", "EURNOKV3M Curncy",
        "EURAUDV3M Curncy", "EURCADV3M Curncy",
    ],
    # Short rate for carry (3m Euribor; ESTR compounded if you prefer post-2021)
    "money_market": [
        "EUR003M Index", "ESTRON Index",
    ],
}

FIELD = "PX_LAST"


def fetch_block(session, securities, field=FIELD, start=START, end=END):
    refsvc = session.getService("//blp/refdata")
    req = refsvc.createRequest("HistoricalDataRequest")
    for s in securities:
        req.getElement("securities").appendValue(s)
    req.getElement("fields").appendValue(field)
    req.set("startDate", start)
    req.set("endDate", end)
    req.set("periodicitySelection", "DAILY")
    req.set("nonTradingDayFillOption", "NON_TRADING_WEEKDAYS")
    req.set("nonTradingDayFillMethod", "NIL_VALUE")
    session.sendRequest(req)

    frames = {}
    while True:
        ev = session.nextEvent(500)
        for msg in ev:
            if not msg.hasElement("securityData"):
                continue
            sd = msg.getElement("securityData")
            tkr = sd.getElementAsString("security")
            if sd.hasElement("securityError"):
                print(f"  !! {tkr}: {sd.getElement('securityError')}")
                continue
            rows = []
            fdata = sd.getElement("fieldData")
            for i in range(fdata.numValues()):
                pt = fdata.getValueAsElement(i)
                if pt.hasElement(field):
                    rows.append((pt.getElementAsDatetime("date"),
                                 pt.getElementAsFloat(field)))
            if rows:
                s = pd.Series(dict(rows), name=tkr)
                s.index = pd.to_datetime(s.index)
                frames[tkr] = s
        if ev.eventType() == blpapi.Event.RESPONSE:
            break
    return pd.DataFrame(frames)


def main():
    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    session = blpapi.Session(opts)
    if not session.start():
        raise SystemExit("Failed to start Bloomberg session — is the terminal running?")
    if not session.openService("//blp/refdata"):
        raise SystemExit("Failed to open //blp/refdata")

    all_frames = []
    for name, secs in BLOCKS.items():
        print(f"Fetching block: {name} ({len(secs)} securities)")
        df = fetch_block(session, secs)
        df.to_csv(os.path.join(OUTDIR, f"{name}.csv"))
        print(f"  -> {df.shape[0]} rows, {df.shape[1]} series "
              f"({df.index.min().date()} .. {df.index.max().date()})")
        all_frames.append(df)
    session.stop()

    allw = pd.concat(all_frames, axis=1).sort_index()
    allw.to_csv(os.path.join(OUTDIR, "bbg_daily_all.csv"))
    print("\nCoverage summary (non-null obs per series):")
    print(allw.notna().sum().sort_values().to_string())
    print(f"\nSaved -> {OUTDIR}/bbg_daily_all.csv  "
          f"({allw.shape[0]} rows x {allw.shape[1]} cols)")


if __name__ == "__main__":
    main()

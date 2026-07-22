"""Download public data and build the FX/vol panels. Re-runnable."""
import os, urllib.request
import pandas as pd, numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
os.makedirs(D, exist_ok=True)

def get(url, dest):
    r = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=180)
    with open(dest, "wb") as f:
        f.write(r.read())

def main():
    # --- FX: Fed H.10 daily via datasets/exchange-rates (all quotes = units per USD)
    fxp = os.path.join(D, "fx_daily_raw.csv")
    if not os.path.exists(fxp):
        get("https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv", fxp)
    piv = (pd.read_csv(fxp).pivot(index="Date", columns="Country", values="Exchange rate")
           .apply(pd.to_numeric, errors="coerce"))
    piv.index = pd.to_datetime(piv.index)
    piv = piv.loc["2004":]
    eur_per_usd = piv["Euro"]
    crosses = {"USD": 1.0 / eur_per_usd}
    for c, code in {"Japan": "JPY", "United Kingdom": "GBP", "Switzerland": "CHF",
                    "Sweden": "SEK", "Norway": "NOK", "Australia": "AUD",
                    "Canada": "CAD", "Denmark": "DKK"}.items():
        crosses[code] = piv[c] / eur_per_usd
    lvl = pd.DataFrame(crosses)
    ret = np.log(lvl).diff().dropna(how="all")
    lvl.to_csv(os.path.join(D, "fx_eur_levels.csv"))
    ret.to_csv(os.path.join(D, "fx_eur_logret.csv"))
    print("FX:", ret.shape, ret.index.min().date(), "->", ret.index.max().date())

    # --- VIX daily
    vixp = os.path.join(D, "vix_daily.csv")
    if not os.path.exists(vixp):
        get("https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv", vixp)
    v = pd.read_csv(vixp)
    print("VIX:", len(v), v.iloc[-1, 0])

if __name__ == "__main__":
    main()

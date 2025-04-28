import yfinance as yf
import pandas as pd
from primary_sensitivity import primary_sensitivity as ps
from utils import *


def main():
    tickers = ["SPY", "XLF"]
    start_date = "2008-09-22"
    end_date = "2008-09-29"

    df = yf.download(tickers, start=start_date, end=end_date)
    spy = df["Close"]["SPY"]
    xlf = df["Close"]["XLF"]

    data = list(zip(spy, xlf))
    bounds = [(0.0, 0.0), (spy.max(), xlf.max())]

    r = compute_correlation(data)
    p = p_value_for_pcc(r, len(spy))
    sens_r = ps(data, bounds)
    brute = brute_force_search(data, bounds, grid_size=100)

    print("r_n and p_n:", r, p)
    print("Predicted:", sens_r)
    print("Brute-force:", brute)


if __name__ == "__main__":
    main()

"""RFM+ behavioral features for richer customer representation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_retail_transactions(
    root: Path,
    *,
    prefer: str = "ii",
) -> tuple[pd.DataFrame, dict]:
    """Load Online Retail II (local zip/xlsx) or Online Retail (ucimlrepo 352).

    Returns cleaned positive-line transactions + metadata.
    Does not alter baseline notebook loaders.
    """
    root = Path(root)
    data_dir = root / "data"
    meta: dict = {"dataset": prefer}

    if prefer == "ii":
        xlsx = data_dir / "online_retail_II.xlsx"
        zip_path = data_dir / "online_retail_ii.zip"
        if not xlsx.exists() or xlsx.stat().st_size < 1_000_000:
            import urllib.request
            import zipfile

            url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
        xl = pd.ExcelFile(xlsx)
        raw = pd.concat(
            [pd.read_excel(xlsx, sheet_name=s) for s in xl.sheet_names],
            ignore_index=True,
        )
        colmap = {
            "invoice": "Invoice",
            "customer": "Customer ID",
            "qty": "Quantity",
            "price": "Price",
            "date": "InvoiceDate",
            "stock": "StockCode",
            "country": "Country",
        }
        meta["source"] = f"UCI 502 xlsx sheets={xl.sheet_names}"
    else:
        from ucimlrepo import fetch_ucirepo

        retail = fetch_ucirepo(id=352)
        raw = (
            retail.data.original.copy()
            if retail.data.original is not None
            else retail.data.features.copy()
        )
        if "InvoiceNo" not in raw.columns and retail.data.ids is not None:
            raw = pd.concat(
                [
                    retail.data.ids.reset_index(drop=True),
                    retail.data.features.reset_index(drop=True),
                ],
                axis=1,
            )
        colmap = {
            "invoice": "InvoiceNo",
            "customer": "CustomerID",
            "qty": "Quantity",
            "price": "UnitPrice",
            "date": "InvoiceDate",
            "stock": "StockCode",
            "country": "Country",
        }
        meta["source"] = "ucimlrepo id=352"

    inv, cust, qty, price, date = (
        colmap["invoice"],
        colmap["customer"],
        colmap["qty"],
        colmap["price"],
        colmap["date"],
    )
    tx = raw.copy()
    tx[inv] = tx[inv].astype(str)
    tx[date] = pd.to_datetime(tx[date], errors="coerce")
    tx["is_cancel"] = tx[inv].str.startswith("C")
    # keep cancels for return-rate feature, but flag them
    tx["line_total"] = tx[qty] * tx[price]
    tx = tx.rename(
        columns={
            inv: "invoice",
            cust: "customer_id",
            qty: "quantity",
            price: "unit_price",
            date: "invoice_date",
            colmap.get("stock", "StockCode"): "stock_code",
            colmap.get("country", "Country"): "country",
        }
    )
    # normalize stock/country if missing after rename
    if "stock_code" not in tx.columns:
        tx["stock_code"] = np.nan
    if "country" not in tx.columns:
        tx["country"] = np.nan
    tx = tx.dropna(subset=["invoice_date"])
    tx["customer_id"] = pd.to_numeric(tx["customer_id"], errors="coerce")
    meta["colmap"] = colmap
    meta["n_raw"] = len(raw)
    return tx, meta


def build_rfm_plus(
    tx: pd.DataFrame,
    *,
    asof: pd.Timestamp | None = None,
    customer_col: str = "customer_id",
    min_positive: bool = True,
) -> pd.DataFrame:
    """Build classic RFM + behavioral features up to an as-of date (inclusive).

    Features
    --------
    Recency, Frequency, Monetary
    TenureDays, AvgOrderValue, StdOrderValue
    MeanInterPurchaseDays, StdInterPurchaseDays
    CancelRate (cancel lines / all lines with customer id)
    NUniqueStock, NCountries
    MonetaryLast90, MonetaryPrev90, MonetaryTrend90
    FreqLast90
    """
    df = tx.copy()
    if asof is None:
        asof = df["invoice_date"].max()
    asof = pd.Timestamp(asof)
    df = df.loc[df["invoice_date"] <= asof]
    df = df.loc[df[customer_col].notna()]

    pos = df.loc[(~df["is_cancel"]) & (df["quantity"] > 0) & (df["unit_price"] > 0)].copy()
    if min_positive and pos.empty:
        return pd.DataFrame()

    # per invoice aggregates for inter-purchase and AOV
    inv = (
        pos.groupby([customer_col, "invoice"], as_index=False)
        .agg(
            invoice_date=("invoice_date", "max"),
            order_value=("line_total", "sum"),
        )
        .sort_values([customer_col, "invoice_date"])
    )

    def _inter_stats(g: pd.DataFrame) -> pd.Series:
        d = g["invoice_date"].sort_values()
        if len(d) < 2:
            return pd.Series({"MeanInterPurchaseDays": np.nan, "StdInterPurchaseDays": np.nan})
        gaps = d.diff().dt.days.dropna()
        return pd.Series(
            {
                "MeanInterPurchaseDays": float(gaps.mean()),
                "StdInterPurchaseDays": float(gaps.std(ddof=0)) if len(gaps) else np.nan,
            }
        )

    inter = inv.groupby(customer_col).apply(_inter_stats, include_groups=False).reset_index()

    rfm = (
        pos.groupby(customer_col)
        .agg(
            Recency=("invoice_date", lambda s: (asof - s.max()).days),
            Frequency=("invoice", "nunique"),
            Monetary=("line_total", "sum"),
            FirstPurchase=("invoice_date", "min"),
            LastPurchase=("invoice_date", "max"),
            NUniqueStock=("stock_code", "nunique"),
            NCountries=("country", "nunique"),
        )
        .reset_index()
    )
    rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days.clip(lower=0)
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"].clip(lower=1)

    aov_std = inv.groupby(customer_col)["order_value"].std(ddof=0).rename("StdOrderValue")
    rfm = rfm.merge(aov_std, on=customer_col, how="left")
    rfm = rfm.merge(inter, on=customer_col, how="left")

    # cancel rate from all lines with customer id
    all_c = df.loc[df[customer_col].notna()]
    cancel_stats = all_c.groupby(customer_col).agg(
        n_lines=("invoice", "count"),
        n_cancel=("is_cancel", "sum"),
    )
    cancel_stats["CancelRate"] = cancel_stats["n_cancel"] / cancel_stats["n_lines"].clip(lower=1)
    rfm = rfm.merge(cancel_stats[["CancelRate"]], on=customer_col, how="left")
    rfm["CancelRate"] = rfm["CancelRate"].fillna(0.0)

    # 90-day windows
    start_last = asof - pd.Timedelta(days=90)
    start_prev = asof - pd.Timedelta(days=180)
    last90 = pos.loc[pos["invoice_date"] > start_last]
    prev90 = pos.loc[(pos["invoice_date"] > start_prev) & (pos["invoice_date"] <= start_last)]
    m_last = last90.groupby(customer_col)["line_total"].sum().rename("MonetaryLast90")
    m_prev = prev90.groupby(customer_col)["line_total"].sum().rename("MonetaryPrev90")
    f_last = last90.groupby(customer_col)["invoice"].nunique().rename("FreqLast90")
    rfm = rfm.merge(m_last, on=customer_col, how="left")
    rfm = rfm.merge(m_prev, on=customer_col, how="left")
    rfm = rfm.merge(f_last, on=customer_col, how="left")
    for c in ["MonetaryLast90", "MonetaryPrev90", "FreqLast90"]:
        rfm[c] = rfm[c].fillna(0.0)
    rfm["MonetaryTrend90"] = rfm["MonetaryLast90"] - rfm["MonetaryPrev90"]

    # fill inter-purchase for single-order customers with large sentinel (then robust-scale)
    rfm["MeanInterPurchaseDays"] = rfm["MeanInterPurchaseDays"].fillna(
        rfm["MeanInterPurchaseDays"].median() if rfm["MeanInterPurchaseDays"].notna().any() else 999
    )
    rfm["StdInterPurchaseDays"] = rfm["StdInterPurchaseDays"].fillna(0.0)
    rfm["StdOrderValue"] = rfm["StdOrderValue"].fillna(0.0)
    rfm["asof"] = asof
    return rfm


# Default feature columns for clustering (exclude ids/dates)
RFM_PLUS_CLUSTER_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "TenureDays",
    "AvgOrderValue",
    "StdOrderValue",
    "MeanInterPurchaseDays",
    "StdInterPurchaseDays",
    "CancelRate",
    "NUniqueStock",
    "NCountries",
    "MonetaryLast90",
    "MonetaryPrev90",
    "MonetaryTrend90",
    "FreqLast90",
]

RFM_BASIC_COLS = ["Recency", "Frequency", "Monetary"]

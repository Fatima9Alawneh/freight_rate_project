from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "posted_rate"
SEEDS = (11, 22, 33)


# Cleaning + imputation (statistics are learned on training data only)
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def normalize_equipment(value) -> object:
    if pd.isna(value):
        return np.nan
    key = re.sub(r"[^a-z]", "", str(value).lower())
    if key in {"dryvan", "van", "dv"}:
        return "Dry Van"
    if key in {"reefer", "refrigerated", "rf"}:
        return "Reefer"
    if key in {"flatbed", "fb"}:
        return "Flatbed"
    return str(value).strip().title()


class Cleaner:
    NUMERIC = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
               "distance", "weight", "market_index", "quote_signal"]

    def _basic(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for c in ("pickup", "delivery"):
            d[c] = d[c].astype("string").str.strip().str.title()
        d["equipment"] = d["equipment"].map(normalize_equipment)
        for c in self.NUMERIC:
            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        d["date"] = pd.to_datetime(d["date"], errors="coerce", format="mixed")
        # impossible values -> NaN (then imputed)
        d.loc[~d["distance"].between(1, 5000), "distance"] = np.nan
        d.loc[~d["weight"].between(1, 80_000), "weight"] = np.nan
        for c in ("pickup_lat", "delivery_lat"):
            d.loc[~d[c].between(-90, 90), c] = np.nan
        for c in ("pickup_lon", "delivery_lon"):
            d.loc[~d[c].between(-180, 180), c] = np.nan
        for c in ("market_index", "quote_signal"):
            d.loc[d[c] <= 0, c] = np.nan
        return d

    def fit(self, df: pd.DataFrame) -> "Cleaner":
        d = self._basic(df)
        pk = d[["pickup", "pickup_lat", "pickup_lon"]
               ].set_axis(["city", "lat", "lon"], axis=1)
        dl = d[["delivery", "delivery_lat", "delivery_lon"]
               ].set_axis(["city", "lat", "lon"], axis=1)
        self.city_xy = pd.concat([pk, dl]).groupby("city")[
            ["lat", "lon"]].median()
        hav = haversine_miles(d.pickup_lat, d.pickup_lon,
                              d.delivery_lat, d.delivery_lon)
        self.road_factor = float(np.nanmedian(
            d["distance"] / hav.replace(0, np.nan)))
        self.weight_by_equip = d.groupby("equipment")["weight"].median()
        self.weight_med = float(d["weight"].median())
        self.dist_med = float(d["distance"].median())
        self.lane_dist = d.groupby(["pickup", "delivery"])["distance"].median()
        self.lane_count = d.groupby(["pickup", "delivery"]).size()
        self.market_by_date = d.groupby("date")["market_index"].median()
        self.market_med = float(d["market_index"].median())
        cities = sorted(set(d["pickup"].dropna()) |
                        set(d["delivery"].dropna()))
        self.city_code = {c: i for i, c in enumerate(cities)}
        self.equip_code = {c: i for i, c in enumerate(
            sorted(d["equipment"].dropna().unique()))}
        years = range(int(d["date"].dt.year.min()) - 1,
                      int(d["date"].dt.year.max()) + 3)
        self.holidays = USFederalHolidayCalendar().holidays(
            f"{min(years)}-01-01", f"{max(years)}-12-31").values
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._basic(df)
        for c in ("distance", "weight", "market_index", "quote_signal"):
            d[f"{c}_missing"] = d[c].isna().astype(int)
        # coordinates from city medians
        for side in ("pickup", "delivery"):
            for c in ("lat", "lon"):
                col = f"{side}_{c}"
                d[col] = d[col].fillna(d[side].map(self.city_xy[c]))
        hav = haversine_miles(d.pickup_lat, d.pickup_lon,
                              d.delivery_lat, d.delivery_lon)
        lane_idx = pd.MultiIndex.from_frame(d[["pickup", "delivery"]])
        lane_d = pd.Series(self.lane_dist.reindex(
            lane_idx).values, index=d.index)
        d["haversine"] = hav
        d["distance"] = d["distance"].fillna(lane_d).fillna(
            hav * self.road_factor).fillna(self.dist_med)
        d["weight"] = d["weight"].fillna(d["equipment"].map(
            self.weight_by_equip)).fillna(self.weight_med)
        # market index: same-day median (features only, no target) -> train map -> global
        same_day = d.groupby("date")["market_index"].transform("median")
        d["market_index"] = (d["market_index"].fillna(same_day)
                             .fillna(d["date"].map(self.market_by_date)).fillna(self.market_med))
        d["lane_freq"] = self.lane_count.reindex(lane_idx).fillna(0).values
        return d

    # feature engineering
    def features(self, d: pd.DataFrame, use_market: bool) -> tuple[pd.DataFrame, list[bool]]:
        X = pd.DataFrame(index=d.index)
        X["distance"] = d["distance"]
        X["log_distance"] = np.log1p(d["distance"])
        X["haversine"] = d["haversine"]
        X["dist_ratio"] = d["distance"] / d["haversine"].replace(0, np.nan)
        X["short_haul"] = (d["distance"] < 250).astype(int)
        X["weight"] = d["weight"]
        X["weight_per_mile"] = d["weight"] / d["distance"]
        X["weight_missing"] = d["weight_missing"]
        X["distance_missing"] = d["distance_missing"]
        for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
            X[c] = d[c]
        X["dlat"] = d["delivery_lat"] - d["pickup_lat"]
        X["dlon"] = d["delivery_lon"] - d["pickup_lon"]
        X["lane_freq"] = d["lane_freq"]
        # calendar
        dt = d["date"]
        X["dow"] = dt.dt.dayofweek
        X["dom"] = dt.dt.day
        X["month"] = dt.dt.month
        X["doy"] = dt.dt.dayofyear
        X["week"] = dt.dt.isocalendar().week.astype(float)
        X["is_weekend"] = (X["dow"] >= 5).astype(float)
        X["days_to_month_end"] = dt.dt.days_in_month - dt.dt.day
        X["doy_sin"] = np.sin(2 * np.pi * X["doy"] / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * X["doy"] / 365.25)
        X["days_to_holiday"] = self._days_to_holiday(dt)
        xmas = pd.to_datetime(dict(year=dt.dt.year, month=12, day=25))
        X["days_to_xmas"] = (xmas - dt).dt.days
        if use_market:
            X["market_index"] = d["market_index"]
            X["market_missing"] = d["market_index_missing"]
            # NaN allowed (native HGB support)
            X["quote_signal"] = d["quote_signal"]
            X["quote_missing"] = d["quote_signal_missing"]
            X["quote_est"] = d["quote_signal"] * d["distance"]
            X["market_x_dist"] = d["market_index"] * d["distance"]
            X["market_x_weight"] = d["market_index"] * d["weight"]
        # categoricals (unknown -> NaN)
        X["pickup_id"] = d["pickup"].map(self.city_code).astype(float)
        X["delivery_id"] = d["delivery"].map(self.city_code).astype(float)
        X["equipment_id"] = d["equipment"].map(self.equip_code).astype(float)
        cat_cols = {"pickup_id", "delivery_id", "equipment_id"}
        return X, [c in cat_cols for c in X.columns]

    def _days_to_holiday(self, dt: pd.Series) -> np.ndarray:
        v = dt.values.astype("datetime64[D]")
        h = self.holidays.astype("datetime64[D]")
        pos = np.clip(np.searchsorted(h, v), 1, len(h) - 1)
        diff = np.minimum(np.abs(
            (v - h[pos]).astype(int)), np.abs((v - h[pos - 1]).astype(int))).astype(float)
        diff[np.isnat(v)] = np.nan
        return diff


# Model

def fit_predict(Xtr, ytr, Xte, cat_mask, seeds=SEEDS) -> np.ndarray:
    """Seed-averaged HistGradientBoosting on log(rate)."""
    preds = []
    for s in seeds:
        m = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=1200, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, categorical_features=cat_mask, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=40, random_state=s)
        m.fit(Xtr, np.log(ytr))
        preds.append(m.predict(Xte))
    return np.exp(np.mean(preds, axis=0))


def metrics(y, p) -> dict:
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE_%": float(np.mean(np.abs(y - p) / y) * 100),
            "R2": float(r2_score(y, p))}


def time_cv(cleaner, train, use_market, n_folds=4) -> dict:
    """Rolling-origin CV: train on all earlier months, test on the next month."""
    d = cleaner.transform(train)
    X, cat = cleaner.features(d, use_market)
    y = train[TARGET].values
    months = d["date"].dt.to_period("M")
    uniq = sorted(months.dropna().unique())
    folds = []
    for m in uniq[-n_folds:]:
        te, tr = (months == m).values, (months < m).values
        if tr.sum() < 500 or te.sum() < 50:
            continue
        r = metrics(y[te], fit_predict(
            X[tr], y[tr], X[te], cat, seeds=SEEDS[:1]))
        r["test_month"] = str(m)
        folds.append(r)
    if not folds:
        return {}
    avg = {k: float(np.mean([f[k] for f in folds]))
           for k in ("MAE", "RMSE", "MAPE_%", "R2")}
    return {"folds": folds, "mean": avg}


# Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data / "train_test.csv")
    val = pd.read_csv(data / "validation.csv")
    dec = pd.read_csv(data / "december_chart_inputs.csv")
    report: dict = {}

    #  training set data quality
    n0 = len(train)
    train = train.drop_duplicates(
        "load_id") if "load_id" in train else train.drop_duplicates()
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train[train[TARGET] > 0].reset_index(drop=True)
    cleaner = Cleaner().fit(train)
    d_tr = cleaner._basic(train)
    rpm = np.log(train[TARGET] / d_tr["distance"].fillna(cleaner.dist_med))
    z = (rpm - rpm.median()) / (1.4826 *
                                (rpm - rpm.median()).abs().median() + 1e-9)
    # drop extreme target outliers (train only)
    train = train[z.abs() <= 6].reset_index(drop=True)
    report["rows_raw"], report["rows_after_cleaning"] = n0, len(train)
    report["missing_share_train"] = (cleaner._basic(
        train).isna().mean().round(4)).to_dict()
    tr_dates, va_dates = cleaner._basic(
        train)["date"], cleaner._basic(val)["date"]
    report["train_date_range"] = [
        str(tr_dates.min().date()), str(tr_dates.max().date())]
    report["validation_date_range"] = [
        str(va_dates.min().date()), str(va_dates.max().date())]
    print("Train dates:", report["train_date_range"],
          "| Validation dates:", report["validation_date_range"])

    #  validation (time-based)
    if not args.skip_cv:
        report["cv_full_features"] = time_cv(cleaner, train, True)
        report["cv_core_features"] = time_cv(cleaner, train, False)
        print(json.dumps({k: v.get("mean")
              for k, v in report.items() if k.startswith("cv_")}, indent=2))

    #  final fit + validation predictions
    d_all = cleaner.transform(train)
    X_full, cat_full = cleaner.features(d_all, True)
    Xv, _ = cleaner.features(cleaner.transform(val), True)
    p_val = np.clip(fit_predict(
        X_full, train[TARGET].values, Xv, cat_full), 1.0, None)
    sub = pd.DataFrame(
        {"load_id": val["load_id"], "predicted_rate": p_val.round(2)})
    template = pd.read_csv(
        data / "validation_predictions_template.csv")[["load_id"]]
    sub = template.merge(sub, on="load_id", how="left")
    assert sub["predicted_rate"].notna().all() and len(sub) == len(template)
    sub.to_csv(out / "validation_predictions.csv", index=False)

    # December: only date/lane/equipment/weight are available
    X_core, cat_core = cleaner.features(d_all, False)
    dec_clean = cleaner.transform(
        dec.drop(columns=["predicted_rate"], errors="ignore"))
    Xd, _ = cleaner.features(dec_clean, False)
    dec["predicted_rate"] = np.clip(fit_predict(
        X_core, train[TARGET].values, Xd, cat_core), 1.0, None).round(2)
    dec.to_csv(out / "december_chart_inputs.csv", index=False)

    (out / "metrics_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("Wrote validation_predictions.csv, december_chart_inputs.csv, metrics_report.json")


if __name__ == "__main__":
    main()

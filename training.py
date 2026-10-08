"""Freight rate prediction.

Reads data/train_test.csv, validates the model with a time-based split,
then writes validation_predictions.csv and fills data/december_chart_inputs.csv.

Run:  python training.py --data-dir data --out-dir .
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET = "posted_rate"
SEEDS = (0, 1, 2)
CATEGORICAL = ["pickup_id", "delivery_id", "equipment_id"]
BASE = ["distance", "weight", "pickup_id", "delivery_id",
        "equipment_id", "dow", "dom", "month", "doy"]
QUOTE = ["quote_signal", "quote_est", "quote_ok"]


def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["weight"] = df["weight"].abs()  # a few weights are negative sign errors
    return df


def price_outliers(df, limit=6):
    """Rows whose price per mile is far from normal (robust z-score on log price per mile)."""
    log_rpm = np.log(df[TARGET] / df["distance"])
    mid = log_rpm.median()
    mad = 1.4826 * (log_rpm - mid).abs().median()
    return ((log_rpm - mid) / mad).abs() > limit


def quote_is_reliable(df, min_rows=30):
    """quote_signal only tracks the real price on some days. On those days it falls as
    distance grows (corr about -0.8); on the other days it does not. Uses features only."""
    day = pd.DataFrame(
        {"date": df["date"], "q": df["quote_signal"], "d": np.log(df["distance"])})
    by_day = day.groupby("date")
    corr = by_day[["q", "d"]].corr().xs("q", level=1)["d"]
    corr[by_day.size() < min_rows] = np.nan
    return df["date"].map(corr).lt(-0.5).astype(float)


class Encoder:
    """Turns raw rows into model features. Codes and medians come from the training data."""

    def fit(self, df):
        cities = pd.concat([df["pickup"], df["delivery"]]).unique()
        self.city = {c: i for i, c in enumerate(sorted(cities))}
        self.equipment = {e: i for i, e in enumerate(
            sorted(df["equipment"].unique()))}
        self.weight_median = df.groupby("equipment")["weight"].median()
        return self

    def transform(self, df):
        X = pd.DataFrame(index=df.index)
        X["distance"] = df["distance"]
        X["weight"] = df["weight"].fillna(
            df["equipment"].map(self.weight_median))
        X["pickup_id"] = df["pickup"].map(self.city)
        X["delivery_id"] = df["delivery"].map(self.city)
        X["equipment_id"] = df["equipment"].map(self.equipment)
        X["dow"] = df["date"].dt.dayofweek
        X["dom"] = df["date"].dt.day
        X["month"] = df["date"].dt.month
        X["doy"] = df["date"].dt.dayofyear
        if "quote_signal" in df:
            X["quote_signal"] = df["quote_signal"]
            X["quote_est"] = df["quote_signal"] * df["distance"]
            X["quote_ok"] = quote_is_reliable(df)
        return X


def fit_predict(X_train, y_train, X_test):
    """Gradient boosting on log(price), averaged over a few seeds."""
    is_cat = [c in CATEGORICAL for c in X_train.columns]
    preds = []
    for seed in SEEDS:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, categorical_features=is_cat,
            early_stopping=True, n_iter_no_change=40, random_state=seed)
        model.fit(X_train, np.log(y_train))
        preds.append(model.predict(X_test))
    return np.exp(np.mean(preds, axis=0))


def predict(X_train, y_train, X_test, use_quote):
    """use_quote=False: model with load/date features only.
    use_quote=True: also uses quote_signal, and on reliable days the price is taken as
    quote_signal * distance (it matches the real price within about 1% on those days)."""
    cols = BASE + QUOTE if use_quote else BASE
    pred = fit_predict(X_train[cols], y_train, X_test[cols])
    if use_quote:
        ok = (X_train["quote_ok"] == 1).values
        factor = np.median(y_train[ok] / X_train.loc[ok, "quote_est"])
        pred = np.where(X_test["quote_ok"] == 1,
                        factor * X_test["quote_est"], pred)
    return pred


def score(y, pred, clean):
    err = np.abs(pred - y)
    return {"MAE_clean": err[clean].mean(),
            "MAPE_clean_%": (err / y)[clean].mean() * 100,
            "MAE_all": err.mean(),
            "RMSE_all": np.sqrt(((pred - y) ** 2).mean())}


def time_cv(X, y, outlier, month, use_quote, n_folds=4):
    """For each of the last months: train on all earlier months, test on that month.
    'clean' scores skip the corrupted prices, 'all' scores keep every row."""
    rows = {}
    for m in sorted(month.unique())[-n_folds:]:
        train = (month < m).values & ~outlier
        test = (month == m).values
        pred = predict(X[train], y[train], X[test], use_quote)
        rows[str(m)] = score(y[test], pred, ~outlier[test])
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()
    data, out = Path(args.data_dir), Path(args.out_dir)

    train = load(data / "train_test.csv")
    outlier = price_outliers(train).values
    print(
        f"{len(train)} rows, {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"price outliers: {outlier.sum()} rows ({outlier.mean():.1%})")
    print("missing values:", train.isna().sum()[lambda s: s > 0].to_dict())

    enc = Encoder().fit(train[~outlier])
    X = enc.transform(train)
    y = train[TARGET].values

    if not args.no_cv:
        month = train["date"].dt.to_period("M")
        for name, use_quote in [("Without quote_signal", False), ("With quote_signal", True)]:
            res = time_cv(X, y, outlier, month, use_quote)
            print(
                f"\n{name}\n{res.round(2)}\nmean:\n{res.mean().round(2).to_string()}")

    # final model: all development data, outliers removed
    X_fit, y_fit = X[~outlier], y[~outlier]

    val = load(data / "validation.csv")
    X_val = enc.transform(val)
    # quote_signal is used only if validation.csv has days where it is reliable
    reliable_share = X_val["quote_ok"].mean() if "quote_ok" in X_val else 0.0
    use_quote = reliable_share > 0
    print(
        f"\nvalidation: quote_signal reliable on {reliable_share:.0%} of rows -> use quote_signal = {use_quote}")
    sub = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": predict(
        X_fit, y_fit, X_val, use_quote)})
    sub["predicted_rate"] = sub["predicted_rate"].clip(lower=1).round(2)
    template = pd.read_csv(
        data / "validation_predictions_template.csv", usecols=["load_id"])
    sub = template.merge(sub, on="load_id", how="left")
    assert sub["predicted_rate"].notna().all() and len(sub) == len(template)
    sub.to_csv(out / "validation_predictions.csv", index=False)

    # December file has no quote_signal, so it always uses the model without it
    december = pd.read_csv(data / "december_chart_inputs.csv")
    X_dec = enc.transform(load(data / "december_chart_inputs.csv"))
    december["predicted_rate"] = predict(
        X_fit, y_fit, X_dec, False).clip(min=1).round(2)
    december.to_csv(data / "december_chart_inputs.csv", index=False)
    print("\nwrote validation_predictions.csv and data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()

"""Flask app - Prediksi Curah Hujan Mingguan (LSTM).

Inferensi LSTM dijalankan dengan NumPy murni dari bobot hasil training (model/lstm_weights.npz),
karena TensorFlow melebihi batas ukuran fungsi serverless Vercel.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, abort

BASE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE / "templates"))

# ---------- Muat artefak sekali saat cold start ----------
META = json.loads((BASE / "model" / "meta.json").read_text())
W = np.load(BASE / "model" / "lstm_weights.npz")
FEATURES = META["features"]
T = META["timesteps"]
T_IDX = META["target_idx"]
S_MIN = np.array(META["scaler_min"])
S_MAX = np.array(META["scaler_max"])
S_RNG = np.where((S_MAX - S_MIN) == 0, 1.0, S_MAX - S_MIN)
SPLIT = pd.Timestamp(META["split_date"])

DF = pd.read_csv(BASE / "data" / "weather_clean.csv", parse_dates=["date"]).sort_values(["code", "date"])
DATES = sorted(DF["date"].unique())
STATIONS = (
    DF.drop_duplicates("code")[["code", "city", "state"]]
    .assign(label=lambda d: d["city"] + ", " + d["state"] + " (" + d["code"] + ")")
    .sort_values("label")
)
SERIES = {c: g.reset_index(drop=True) for c, g in DF.groupby("code")}


# ---------- Inferensi LSTM (NumPy) ----------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x, k, r, b, return_seq):
    n, steps, _ = x.shape
    u = r.shape[0]
    h = np.zeros((n, u)); c = np.zeros((n, u)); outs = []
    for t in range(steps):
        z = x[:, t, :] @ k + h @ r + b
        i, f, g, o = np.split(z, 4, axis=1)  # urutan gate Keras: i, f, c, o
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        outs.append(h)
    return np.stack(outs, 1) if return_seq else h


def predict_scaled(X):
    h = _lstm(X, W["lstm0_kernel"], W["lstm0_recurrent"], W["lstm0_bias"], True)
    h = _lstm(h, W["lstm1_kernel"], W["lstm1_recurrent"], W["lstm1_bias"], False)
    h = np.maximum(h @ W["dense0_kernel"] + W["dense0_bias"], 0)
    return (h @ W["dense1_kernel"] + W["dense1_bias"]).ravel()


def inv_target(v):
    return np.clip(np.asarray(v) * S_RNG[T_IDX] + S_MIN[T_IDX], 0, None)


def station_windows(g):
    """Semua window (T minggu -> minggu berikutnya) + window terakhir untuk peramalan."""
    arr = (g[FEATURES].to_numpy() - S_MIN) / S_RNG
    X = np.stack([arr[t - T:t] for t in range(T, len(arr) + 1)]).astype("float32")
    return X  # window ke-j menghasilkan prediksi untuk minggu index (j + T); window terakhir = peramalan


def build_response(g, label):
    X = station_windows(g)
    p = inv_target(predict_scaled(X))
    back_pred, next_pred = p[:-1], float(p[-1])
    dates = g["date"].dt.strftime("%Y-%m-%d").tolist()
    back_dates = dates[T:]
    actual = g["precip"].to_numpy()[T:]
    is_test = (g["date"].iloc[T:] >= SPLIT).to_numpy()
    next_date = (g["date"].iloc[-1] + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    m = is_test
    rmse = float(np.sqrt(np.mean((actual[m] - back_pred[m]) ** 2))) if m.any() else None
    mae = float(np.mean(np.abs(actual[m] - back_pred[m]))) if m.any() else None
    return {
        "label": label,
        "dates": dates, "precip": g["precip"].round(3).tolist(),
        "back_dates": back_dates, "back_actual": actual.round(3).tolist(),
        "back_pred": back_pred.round(3).tolist(), "is_test": is_test.tolist(),
        "next_date": next_date, "next_pred": round(next_pred, 3),
        "split_date": META["split_date"], "rmse_station": rmse, "mae_station": mae,
    }


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html", metrics=META["metrics"], timesteps=T,
                           n_stations=len(STATIONS), n_weeks=len(DATES),
                           date_min=str(pd.Timestamp(DATES[0]).date()),
                           date_max=str(pd.Timestamp(DATES[-1]).date()))


@app.route("/api/stations")
def stations():
    return jsonify([{"code": r.code, "label": r.label} for r in STATIONS.itertuples()])


@app.route("/api/forecast/<code>")
def forecast(code):
    code = code.upper()
    if code == "ALL":  # rata-rata semua stasiun (deret agregat)
        agg = DF.groupby("date")[FEATURES].mean().reset_index()
        # bobot rata-rata semua stasiun: prediksi tiap stasiun lalu dirata-ratakan
        preds = []
        for g in SERIES.values():
            preds.append(inv_target(predict_scaled(station_windows(g))))
        avg_p = np.mean(preds, axis=0)
        resp = build_response(agg, "Rata-rata seluruh stasiun")
        a = resp["back_actual"]; m = np.array(resp["is_test"])
        resp["back_pred"] = avg_p[:-1].round(3).tolist()
        resp["next_pred"] = round(float(avg_p[-1]), 3)
        bp = avg_p[:-1]
        resp["rmse_station"] = float(np.sqrt(np.mean((np.array(a)[m] - bp[m]) ** 2)))
        resp["mae_station"] = float(np.mean(np.abs(np.array(a)[m] - bp[m])))
        return jsonify(resp)
    if code not in SERIES:
        abort(404)
    row = STATIONS[STATIONS.code == code].iloc[0]
    return jsonify(build_response(SERIES[code], row["label"]))


if __name__ == "__main__":
    app.run(debug=True)

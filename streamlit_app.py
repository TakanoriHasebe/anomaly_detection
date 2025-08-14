# streamlit_app.py
from __future__ import annotations
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ========================= UI SETUP =========================
st.set_page_config(page_title="Credit Card Anomaly (Streamlit-Only)", page_icon="💳", layout="centered")
st.title("💳 Credit Card Anomaly Detection — Streamlit Community Cloud POC (No FastAPI)")
st.caption("Upload CSV → Train AE + OneClassSVM → Infer one transaction. All preprocessing is done in-app.")

# ========================= MODELS =========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        hidden = max(64, min(256, input_dim * 2))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

# ========================= FEATURE ENGINEERING =========================
REQUIRED_COLS = [
    "transaction_id","card_id","timestamp","amount","currency",
    "merchant_id","mcc","country","lat","lon","entry_mode","channel"
]
CYCLICAL_TWO_PI = 2 * np.pi

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    R = 6371.0
    p1, p2 = np.radians([lat1, lon1]), np.radians([lat2, lon2])
    dlat = p2[0] - p1[0]
    dlon = p2[1] - p1[1]
    a = np.sin(dlat/2)**2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)

def ensure_txn_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    for c in ["transaction_id","card_id","currency","merchant_id","mcc","country","entry_mode","channel"]:
        df[c] = df[c].astype(str).fillna("")
    df = df.dropna(subset=["timestamp"])  # keep only parsable timestamps
    df = df.sort_values(["card_id","timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["transaction_id"]) 
    return df

def top_k(series: pd.Series, k: int) -> List[str]:
    return series.value_counts().head(k).index.tolist()

def build_feature_maps(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    log_amt = np.log1p(df["amount"].clip(lower=0))
    global_mean = float(log_amt.mean())
    global_std = float(log_amt.std() + 1e-6)
    g = df.groupby("card_id")
    card_stats = {}
    for card, sub in g:
        la = np.log1p(sub["amount"].clip(lower=0))
        med = float(np.median(la))
        mad = float(np.median(np.abs(la - med)) + 1e-6)
        home_country = sub["country"].mode().iloc[0] if not sub["country"].mode().empty else ""
        hlat = float(sub["lat"].median()) if not sub["lat"].isna().all() else np.nan
        hlon = float(sub["lon"].median()) if not sub["lon"].isna().all() else np.nan
        card_stats[card] = {
            "median_log_amount": med,
            "mad_log_amount": mad,
            "home_country": home_country,
            "home_lat": hlat,
            "home_lon": hlon,
        }
    tokens = {
        "mcc": top_k(df["mcc"], int(cfg["topk_mcc"])),
        "merchant_id": top_k(df["merchant_id"], int(cfg["topk_merchant"])),
        "country": top_k(df["country"], int(cfg["topk_country"])),
        "currency": top_k(df["currency"], int(cfg["topk_currency"])),
        "channel": top_k(df["channel"].fillna(""), int(cfg["topk_channel"])),
        "entry_mode": top_k(df["entry_mode"].fillna(""), int(cfg["topk_entry"])),
    }
    return {
        "global_mean_log_amount": global_mean,
        "global_std_log_amount": global_std,
        "card_stats": card_stats,
        "tokens": tokens,
    }

def ohe(value: str, vocab: List[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=float)
    try:
        idx = vocab.index(value)
        vec[idx] = 1.0
    except ValueError:
        pass
    return vec

def featurize_txn(row: pd.Series, meta: Dict[str, Any]) -> np.ndarray:
    log_amount = float(np.log1p(max(row.get("amount", 0.0), 0.0)))
    g_mean = meta["global_mean_log_amount"]
    g_std = meta["global_std_log_amount"]
    z_global = (log_amount - g_mean) / g_std
    cs = meta["card_stats"].get(row.get("card_id", ""))
    if cs:
        z_card = (log_amount - cs["median_log_amount"]) / cs["mad_log_amount"]
        home_country = cs["home_country"]
        dist_km = haversine_km(cs["home_lat"], cs["home_lon"], row.get("lat", np.nan), row.get("lon", np.nan))
    else:
        z_card = 0.0
        home_country = row.get("country", "")
        dist_km = 0.0
    international = 1.0 if (row.get("country", "") != home_country and row.get("country", "") != "") else 0.0
    ts = pd.to_datetime(row.get("timestamp"))
    hour = ts.hour
    dow = ts.dayofweek
    hour_sin = np.sin(CYCLICAL_TWO_PI * hour / 24)
    hour_cos = np.cos(CYCLICAL_TWO_PI * hour / 24)
    dow_sin = np.sin(CYCLICAL_TWO_PI * dow / 7)
    dow_cos = np.cos(CYCLICAL_TWO_PI * dow / 7)
    T = meta["tokens"]
    mcc_vec = ohe(str(row.get("mcc", "")), T["mcc"])
    merch_vec = ohe(str(row.get("merchant_id", "")), T["merchant_id"])
    country_vec = ohe(str(row.get("country", "")), T["country"])
    currency_vec = ohe(str(row.get("currency", "")), T["currency"])
    channel_vec = ohe(str(row.get("channel", "")), T["channel"])
    entry_vec = ohe(str(row.get("entry_mode", "")), T["entry_mode"])
    numeric = np.array([
        log_amount, z_global, z_card, hour_sin, hour_cos, dow_sin, dow_cos, dist_km, international
    ], dtype=float)
    return np.concatenate([numeric, mcc_vec, merch_vec, country_vec, currency_vec, channel_vec, entry_vec], axis=0)

# ========================= TRAIN / INFER =========================
def train_ae_svm(df: pd.DataFrame, cfg: Dict[str, Any]):
    meta = build_feature_maps(df, cfg)
    feats = [featurize_txn(row, meta) for _, row in df.iterrows()]
    X = np.vstack(feats)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    ae = Autoencoder(input_dim=Xn.shape[1], latent_dim=int(cfg["latent_dim"]))
    optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    Xt = torch.tensor(Xn, dtype=torch.float32)
    ae.train()
    for _ in range(int(cfg["epochs"])):
        optimizer.zero_grad()
        recon = ae(Xt)
        loss = criterion(recon, Xt)
        loss.backward()
        optimizer.step()
    ae.eval()
    with torch.no_grad():
        Z = ae.encoder(Xt).cpu().numpy()
    svm = OneClassSVM(kernel="rbf", nu=float(cfg["nu"]), gamma="scale")
    svm.fit(Z)
    return {
        "scaler": scaler,
        "encoder_state": ae.encoder.state_dict(),
        "latent_dim": int(cfg["latent_dim"]),
        "input_dim": int(X.shape[1]),
        "svm": svm,
        "meta": meta,
    }

def infer_one(txn: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    scaler: StandardScaler = artifacts["scaler"]
    latent_dim: int = artifacts["latent_dim"]
    input_dim: int = artifacts["input_dim"]
    svm: OneClassSVM = artifacts["svm"]
    meta: Dict[str, Any] = artifacts["meta"]
    enc = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).encoder
    enc.load_state_dict(artifacts["encoder_state"])  # CPU
    enc.eval()
    row = pd.Series(txn)
    row = ensure_txn_df(pd.DataFrame([row])).iloc[0]
    x = featurize_txn(row, meta).reshape(1, -1)
    xn = scaler.transform(x)
    with torch.no_grad():
        z = enc(torch.tensor(xn, dtype=torch.float32)).cpu().numpy()
    score = float(svm.decision_function(z)[0])
    return {"anomaly_score": score, "is_anomaly": bool(score < 0.0)}

# ========================= SESSION STATE =========================
if "df" not in st.session_state:
    st.session_state.df = None
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None

# ========================= UI: INGEST =========================
st.header("1) Upload Transactions CSV")
st.write("Required headers: `transaction_id, card_id, timestamp, amount, currency, merchant_id, mcc, country, lat, lon, entry_mode, channel`")
up = st.file_uploader("CSV file", type=["csv"]) 
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Load CSV"):
        if up is None:
            st.warning("Upload a CSV first")
        else:
            df = pd.read_csv(up)
            df = ensure_txn_df(df)
            st.session_state.df = df
            st.success(f"Loaded rows: {df.shape[0]}  |  features will be computed at training time")
with col_b:
    if st.button("Use Small Synthetic Sample"):
        # Tiny synthetic set for demo (fast)
        cards = ["card_001","card_002","card_003"]
        rows = []
        base_ts = pd.Timestamp("2025-08-01")
        for i in range(200):
            card = np.random.choice(cards)
            ts = base_ts + pd.Timedelta(minutes=int(np.random.randint(0, 60*24)))
            amt = float(np.round(np.random.lognormal(mean=3.0, sigma=0.5), 2))
            rows.append({
                "transaction_id": f"tx_{i:05d}",
                "card_id": card,
                "timestamp": ts.isoformat(),
                "amount": amt,
                "currency": "JPY",
                "merchant_id": f"m_{np.random.randint(1,20):03d}",
                "mcc": np.random.choice(["5411","5812","5999","5541"]),
                "country": "JP",
                "lat": 35.68 + np.random.normal(0,0.1),
                "lon": 139.75 + np.random.normal(0,0.1),
                "entry_mode": np.random.choice(["chip","swipe","contactless"]),
                "channel": np.random.choice(["card_present","card_not_present"]),
            })
        df = pd.DataFrame(rows)
        st.session_state.df = ensure_txn_df(df)
        st.success("Synthetic sample loaded (200 rows)")

if st.session_state.df is not None:
    st.dataframe(st.session_state.df.head(20))

# ========================= UI: TRAIN =========================
st.header("2) Train Model")
with st.form("train_form"):
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("epochs", 5, 50, 15)
        latent = st.slider("latent_dim", 2, 64, 16)
        nu = st.slider("nu (OneClassSVM)", 1, 49, 10) / 100.0
    with col2:
        topk_mcc = st.slider("topk_mcc", 10, 200, 50)
        topk_merch = st.slider("topk_merchant", 10, 300, 100)
        topk_country = st.slider("topk_country", 5, 50, 20)
        topk_curr = st.slider("topk_currency", 2, 20, 10)
        topk_channel = st.slider("topk_channel", 2, 5, 3)
        topk_entry = st.slider("topk_entry", 2, 5, 3)
    submitted = st.form_submit_button("Train")

if st.session_state.df is not None and 'submitted' in locals() and submitted:
    cfg = {
        "epochs": epochs,
        "latent_dim": latent,
        "nu": nu,
        "topk_mcc": topk_mcc,
        "topk_merchant": topk_merch,
        "topk_country": topk_country,
        "topk_currency": topk_curr,
        "topk_channel": topk_channel,
        "topk_entry": topk_entry,
    }
    with st.spinner("Training AE + OneClassSVM..."):
        artifacts = train_ae_svm(st.session_state.df, cfg)
        st.session_state.artifacts = artifacts
    st.success(f"Trained! input_dim={artifacts['input_dim']}, latent_dim={artifacts['latent_dim']}")

# ========================= UI: INFER =========================
st.header("3) Inference (single transaction)")
def example_txn_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a JSON-serializable example transaction from the loaded DF.
    Converts pandas Timestamps -> ISO8601 strings and NaN -> None.
    """
    if df is None or df.empty:
        return {
            "transaction_id": "tx_9999",
            "card_id": "card_001",
            "timestamp": "2025-08-11T10:23:00",
            "amount": 120.5,
            "currency": "JPY",
            "merchant_id": "m_010",
            "mcc": "5411",
            "country": "JP",
            "lat": 35.6581,
            "lon": 139.7414,
            "entry_mode": "chip",
            "channel": "card_present",
        }
    row = df.iloc[0].to_dict()
    # timestamp -> ISO string
    ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
    row["timestamp"] = ts.isoformat() if pd.notna(ts) else "2025-08-11T00:00:00"
    # numeric casts
    row["amount"] = float(row.get("amount", 0.0) or 0.0)
    for f in ("lat", "lon"):
        v = row.get(f, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row[f] = None
        else:
            try:
                row[f] = float(v)
            except Exception:
                row[f] = None
    # force strings for categoricals
    for c in ("transaction_id","card_id","currency","merchant_id","mcc","country","entry_mode","channel"):
        if c in row and row[c] is not None:
            row[c] = str(row[c])
        else:
            row[c] = ""
    return row

example_json = json.dumps(example_txn_from_df(st.session_state.df), indent=2)
user_json = st.text_area("Transaction JSON", value=example_json, height=240)
if st.button("Infer"):
    if st.session_state.artifacts is None:
        st.warning("Train the model first.")
    else:
        try:
            obj = json.loads(user_json)
            out = infer_one(obj, st.session_state.artifacts)
            st.json(out)
            if 'anomaly_score' in out:
                st.metric("Anomaly score", f"{out['anomaly_score']:.4f}")
        except Exception as e:
            st.error(str(e))

st.markdown("---")
st.caption("Tip: Keep topK values moderate on Community Cloud (memory ~1GB). This app uses in-memory artifacts via session_state.")

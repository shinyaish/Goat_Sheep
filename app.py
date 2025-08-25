
import io, os, json, time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import plotly.express as px
import portalocker

st.set_page_config(page_title="Goat/Sheep PCA Plotter", layout="wide")

# -----------------------------
# Config & persistence helpers
# -----------------------------
DATA_DIR = os.environ.get("APP_DATA_DIR", ".")
SHARED_CSV = os.path.join(DATA_DIR, "shared_data.csv")
STATE_JSON = os.path.join(DATA_DIR, "app_state.json")
LOCKFILE = os.path.join(DATA_DIR, ".fslock")

os.makedirs(DATA_DIR, exist_ok=True)

def lock_path(path):
    return f"{path}.lock"

class FileLock:
    def __init__(self, path, timeout=10):
        self.path = lock_path(path)
        self.timeout = timeout
        self._fh = None
    def __enter__(self):
        self._fh = open(self.path, "a+")
        portalocker.lock(self._fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
        return self._fh
    def __exit__(self, exc_type, exc, tb):
        try:
            self._fh.flush()
        finally:
            portalocker.unlock(self._fh)
            self._fh.close()

# Admin password from st.secrets or env
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.environ.get("ADMIN_PASSWORD", ""))

# Default template data
template_cols = ["ID","種","体長(cm)","体高(cm)","胴回り(cm)","尾長(cm)","耳長(cm)",
                 "角","あごひげ","肉ぜん","顔の毛色","体の毛色"]
template_data = [
    ["ex1","Goat", 75, 60, 80, 10, 15, "有","有","無","黒っぽい","白っぽい"],
    ["ex2","Sheep", 85, 65, 95, 12, 14, "無","無","無","白っぽい","白っぽい"],
    ["ex3","Goat", 70, 58, 78, 11, 16, "有","有","有","その他","黒っぽい"],
]
template_df = pd.DataFrame(template_data, columns=template_cols)

# App state (maintenance mode, last init time, etc.)
def read_state():
    if not os.path.exists(STATE_JSON):
        return {"maintenance": False, "last_init": None}
    with FileLock(STATE_JSON):
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {"maintenance": False, "last_init": None}

def write_state(state: dict):
    with FileLock(STATE_JSON):
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

# Shared dataset load/save (for multi-user shared view)
def load_shared_df():
    if not os.path.exists(SHARED_CSV):
        return template_df.copy()
    with FileLock(SHARED_CSV):
        return pd.read_csv(SHARED_CSV)

def save_shared_df(df: pd.DataFrame):
    with FileLock(SHARED_CSV):
        df.to_csv(SHARED_CSV, index=False, encoding="utf-8")

# -----------------------------
# UI Header
# -----------------------------
st.title("🐐🐑 Morphometrics PCA — Goat vs Sheep")
st.caption("アンケート集計 → PCA二次元化 → 種で色分け。管理者の初期化・ロック機能つき。")

# -----------------------------
# Admin panel
# -----------------------------
with st.sidebar.expander("🔐 Settings (Admin)"):
    pwd = st.text_input("Admin Password", type="password", value="")
    auth = (pwd and ADMIN_PASSWORD and pwd == ADMIN_PASSWORD)
    if not ADMIN_PASSWORD:
        st.info("※ 管理者パスワードは `st.secrets['ADMIN_PASSWORD']` または環境変数 `ADMIN_PASSWORD` で設定してください。")
    if auth:
        st.success("認証済み")
        state = read_state()
        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("🔄 共有データをテンプレートで初期化"):
                save_shared_df(template_df.copy())
                state["last_init"] = time.strftime("%Y-%m-%d %H:%M:%S")
                write_state(state)
                st.toast("共有データを初期化しました", icon="✅")
        with colB:
            new_flag = st.toggle("メンテナンスロック（閲覧のみ）", value=state.get("maintenance", False), help="ON中、非管理者は編集不可")
            if new_flag != state.get("maintenance", False):
                state["maintenance"] = new_flag
                write_state(state)
                st.toast(f"メンテナンスロック: {'ON' if new_flag else 'OFF'}", icon="🔒" if new_flag else "🔓")
        with colC:
            if st.button("🧹 共有データを空にする"):
                empty = template_df.head(0).copy()
                save_shared_df(empty)
                state["last_init"] = time.strftime("%Y-%m-%d %H:%M:%S")
                write_state(state)
                st.toast("共有データを空にしました", icon="🗑️")
    else:
        st.caption("※ 認証されていないと管理操作はできません。")

state = read_state()
maintenance = state.get("maintenance", False)

# -----------------------------
# Data acquisition
# -----------------------------
st.subheader("📥 データ入力")
st.write("CSVを読み込むか、下の表で編集して解析します。**共有データ**をベースにすることで多人数で同時編集ができます。")
# Show and download template
with st.expander("📎 CSVテンプレートを表示/ダウンロード"):
    st.dataframe(template_df, use_container_width=True)
    buf = io.StringIO()
    template_df.to_csv(buf, index=False)
    st.download_button("テンプレートCSVをダウンロード", data=buf.getvalue(), file_name="template_goat_sheep.csv", mime="text/csv")

# Load shared dataset
shared_df = load_shared_df()
st.info(f"共有データ行数: {len(shared_df)}")

uploaded_file = st.file_uploader("CSVをアップロード（UTF-8, ヘッダー必須）", type=["csv"])

st.markdown("**または** 共有データを編集して保存：")
if maintenance and not (st.session_state.get("admin_auth", False) or (pwd and ADMIN_PASSWORD and pwd == ADMIN_PASSWORD)):
    st.warning("現在メンテナンスロック中のため、編集は無効化されています（管理者を除く）。")
    editable = False
else:
    editable = True

# Editable data editor (if editable, else read-only)
edited_df = st.data_editor(shared_df, num_rows="dynamic", use_container_width=True, disabled=not editable)

# Save button (guarded by lock)
save_cols = st.columns([1,1,4])
with save_cols[0]:
    if st.button("💾 共有データを保存", disabled=not editable):
        try:
            save_shared_df(edited_df)
            st.success("共有データを保存しました。")
        except portalocker.exceptions.LockException:
            st.error("ファイルロック取得に失敗しました。少し待って再実行してください。")

with save_cols[1]:
    if st.button("↩️ 共有データを再読込み"):
        st.rerun()

# Determine working df: uploaded > edited shared
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("アップロードデータを使用します。")
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
else:
    df = edited_df

# -----------------------------
# Validation & preprocessing
# -----------------------------
template_cols = ["ID","種","体長(cm)","体高(cm)","胴回り(cm)","尾長(cm)","耳長(cm)",
                 "角","あごひげ","肉ぜん","顔の毛色","体の毛色"]
required_cols = set(template_cols)
if df is None or len(df) == 0:
    st.warning("データ行がありません")
    st.stop()

missing = required_cols - set(df.columns)
if missing:
    st.error(f"必要な列が不足: {sorted(missing)}")
    st.stop()

df = df.dropna(how="all")
if len(df) == 0:
    st.warning("データ行がありません")
    st.stop()

# Normalize categorical fields
def norm_yesno(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in ["有", "あり", "Yes", "yes", "y", "Y", "1", "True", "true"]: return "有"
    if s in ["無","なし","No","no","n","N","0","False","false"]: return "無"
    return s

for col in ["角","あごひげ","肉ぜん"]:
    df[col] = df[col].map(norm_yesno)

def norm_color(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in ["黒","黒色","黒っぽい","dark","black"]: return "黒っぽい"
    if s in ["白","白色","白っぽい","white","light"]: return "白っぽい"
    return "その他"

for col in ["顔の毛色","体の毛色"]:
    df[col] = df[col].map(norm_color)

numeric_features = ["体長(cm)","体高(cm)","胴回り(cm)","尾長(cm)","耳長(cm)"]
binary_cats = ["角","あごひげ","肉ぜん"]
ordinal_map = {"有":1, "無":0}
for c in binary_cats:
    df[c] = df[c].map(ordinal_map)

categorical_features = ["顔の毛色","体の毛色"]
species_col = "種"
df[species_col] = df[species_col].astype(str).str.strip()
keep = df[species_col].isin(["Goat","Sheep"])
if not keep.all():
    st.warning("種は 'Goat' または 'Sheep' のみプロットします。その他は除外します。")
    df = df[keep]
if len(df) < 2:
    st.error("PCAには2個体以上が必要です。")
    st.stop()

X_num = df[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(method="pad")
# Impute remaining NaNs with median
X_num = X_num.fillna(X_num.median())

X_bin = df[binary_cats].fillna(0)
X_cat = df[categorical_features].astype("category")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("bin", "passthrough", binary_cats),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
])

pipeline = Pipeline(steps=[("pre", preprocessor), ("pca", PCA(n_components=2, random_state=42))])

# ColumnTransformer will select by names from the incoming DataFrame
Z = pipeline.fit_transform(df)

pc_df = pd.DataFrame(Z, columns=["PC1","PC2"])
out = pd.concat([df[["ID", species_col]].reset_index(drop=True), pc_df], axis=1)

# Sidebar metrics
pca_step = pipeline.named_steps["pca"]
evr = pca_step.explained_variance_ratio_
st.sidebar.metric("PC1 寄与率", f"{evr[0]*100:.1f}%")
st.sidebar.metric("PC2 寄与率", f"{evr[1]*100:.1f}%")

# Plot
fig = px.scatter(
    out, x="PC1", y="PC2",
    color=species_col, symbol=species_col, hover_data=["ID"],
    title="PCA 2D plot (colored by 種)",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("PCA座標 表"):
    st.dataframe(out, use_container_width=True)

out_csv = out.to_csv(index=False).encode("utf-8")
st.download_button("PCA座標をCSVでダウンロード", data=out_csv, file_name="pca_coords.csv", mime="text/csv")

with st.expander("前処理の詳細"):
    st.markdown("- 数値列: 標準化（平均0, 分散1）")
    st.markdown("- 2値列（角/ひげ/肉ぜん）: 有=1, 無=0 で数値化")
    st.markdown("- 色カテゴリ: One-Hotエンコーディング（未知カテゴリは無視）")
    st.markdown("- 欠損: 数値は列中央値で補完、2値は0（無）で補完")

st.success("プロット完了。共有データは左のボタンで保存できます。")

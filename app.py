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

st.set_page_config(page_title="Goat/Sheep PCA — Survey-first", layout="wide")

# =========================================
# Paths & configuration
# =========================================
DATA_DIR   = os.environ.get("APP_DATA_DIR", ".")
SHARED_CSV = os.path.join(DATA_DIR, "shared_data.csv")
STATE_JSON = os.path.join(DATA_DIR, "app_state.json")

os.makedirs(DATA_DIR, exist_ok=True)

TEMPLATE_COLS = [
    "ID","種","体長(cm)","体高(cm)","胴回り(cm)","尾長(cm)","耳長(cm)",
    "角","あごひげ","肉ぜん","顔の毛色","体の毛色"
]

TEMPLATE_DF = pd.DataFrame([
    ["ex1","Goat", 75, 60, 80, 10, 15, "有","有","無","黒っぽい","白っぽい"],
    ["ex2","Sheep", 85, 65, 95, 12, 14, "無","無","無","白っぽい","白っぽい"],
], columns=TEMPLATE_COLS)

# =========================================
# Lock helpers for concurrency
# =========================================
def _lock_path(path: str) -> str:
    return f"{path}.lock"

class FileLock:
    """Simple file lock using portalocker (exclusive lock)."""
    def __init__(self, path, timeout=10):
        self.path = _lock_path(path)
        self.timeout = timeout
        self._fh = None
    def __enter__(self):
        self._fh = open(self.path, "a+")
        # Non-blocking lock; if fail, raise LockException
        portalocker.lock(self._fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
        return self._fh
    def __exit__(self, exc_type, exc, tb):
        try:
            self._fh.flush()
        finally:
            portalocker.unlock(self._fh)
            self._fh.close()

# =========================================
# Admin password (secrets/env)
# =========================================
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.environ.get("ADMIN_PASSWORD", ""))

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

def load_shared_df() -> pd.DataFrame:
    if not os.path.exists(SHARED_CSV):
        return TEMPLATE_DF.copy()
    with FileLock(SHARED_CSV):
        return pd.read_csv(SHARED_CSV)

def save_shared_df(df: pd.DataFrame):
    with FileLock(SHARED_CSV):
        df.to_csv(SHARED_CSV, index=False, encoding="utf-8")

# Initialize state files if needed
if not os.path.exists(STATE_JSON):
    write_state({"maintenance": False, "last_init": None})
if not os.path.exists(SHARED_CSV):
    save_shared_df(TEMPLATE_DF.copy())

# =========================================
# Header
# =========================================
st.title("🐐🐑 Morphometrics PCA — Survey-first (Goat vs Sheep)")
st.caption("アンケート入力をメインに、補助としてCSVアップロードも利用できます。管理者の初期化・メンテナンスロック・排他ロックを実装。")

# =========================================
# Admin panel
# =========================================
with st.sidebar.expander("🔐 Settings (Admin)"):
    pwd = st.text_input("Admin Password", type="password", value="")
    is_admin = bool(pwd and ADMIN_PASSWORD and pwd == ADMIN_PASSWORD)
    if not ADMIN_PASSWORD:
        st.info("※ 管理者パスワードは `st.secrets['ADMIN_PASSWORD']` か環境変数 `ADMIN_PASSWORD` で設定。")
    if is_admin:
        st.success("認証済み")
        state = read_state()
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 共有データをテンプレートで初期化"):
                save_shared_df(TEMPLATE_DF.copy())
                state["last_init"] = time.strftime("%Y-%m-%d %H:%M:%S")
                write_state(state)
                st.toast("共有データをテンプレートで初期化しました。", icon="✅")
        with c2:
            new_maint = st.toggle("メンテナンスロック（閲覧のみ）", value=state.get("maintenance", False))
            if new_maint != state.get("maintenance", False):
                state["maintenance"] = new_maint
                write_state(state)
                st.toast(f"メンテナンスロック: {'ON' if new_maint else 'OFF'}", icon="🔒" if new_maint else "🔓")
        with c3:
            if st.button("🧹 共有データを空にする"):
                save_shared_df(TEMPLATE_DF.head(0).copy())
                state["last_init"] = time.strftime("%Y-%m-%d %H:%M:%S")
                write_state(state)
                st.toast("共有データを空にしました。", icon="🗑️")
    else:
        st.caption("※ 認証されていないと管理操作はできません。")

state = read_state()
maintenance = state.get("maintenance", False)

# =========================================
# Survey-first form
# =========================================
st.subheader("📝 個体データ入力フォーム（アンケート方式・共有データに追記）")

disabled_form = maintenance and not is_admin
with st.form("input_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        id_val = st.text_input("ID", help="例: G001, S012 など")
        species = st.selectbox("種", ["Goat", "Sheep"])
        body_len = st.number_input("体長(cm)", min_value=0, max_value=200, step=1, value=0)
        body_height = st.number_input("体高(cm)", min_value=0, max_value=200, step=1, value=0)
        girth = st.number_input("胴回り(cm)", min_value=0, max_value=300, step=1, value=0)
    with col2:
        tail_len = st.number_input("尾長(cm)", min_value=0, max_value=100, step=1, value=0)
        ear_len = st.number_input("耳長(cm)", min_value=0, max_value=100, step=1, value=0)
        horn = st.radio("角", ["有", "無"], horizontal=True)
        beard = st.radio("あごひげ", ["有", "無"], horizontal=True)
        dewlap = st.radio("肉ぜん", ["有", "無"], horizontal=True)
        face_color = st.selectbox("顔の毛色", ["黒っぽい", "白っぽい", "その他"])
        body_color = st.selectbox("体の毛色", ["黒っぽい", "白っぽい", "その他"])
    submitted = st.form_submit_button("➕ 共有データに追加", disabled=disabled_form)

if submitted:
    if not id_val.strip():
        st.error("ID は必須です。")
    else:
        try:
            cur = load_shared_df()
            if (cur["ID"].astype(str).str.strip() == id_val.strip()).any():
                st.error(f"ID '{id_val}' は既に存在します。別のIDを指定してください。")
            else:
                new_row = pd.DataFrame([[
                    id_val.strip(), species, body_len, body_height, girth,
                    tail_len, ear_len, horn, beard, dewlap, face_color, body_color
                ]], columns=TEMPLATE_COLS)
                updated = pd.concat([cur, new_row], ignore_index=True)
                save_shared_df(updated)
                st.success(f"{id_val} を追加しました。")
        except portalocker.exceptions.LockException:
            st.error("ファイルロック取得に失敗しました。少し待って再実行してください。")

# =========================================
# CSV upload (supplemental)
# =========================================
st.subheader("📥 CSVアップロード（補助的機能）")
uploaded_file = st.file_uploader("CSVをアップロード（UTF-8, ヘッダー必須）", type=["csv"])
uploaded_df = None
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.success("アップロードデータを読み込みました。")
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")

# =========================================
# View & manage shared data
# =========================================
st.subheader("📊 共有データのプレビュー")
shared_df = load_shared_df()
st.dataframe(shared_df, use_container_width=True)

# Download template and shared data
c_dl1, c_dl2 = st.columns(2)
with c_dl1:
    buf_tmp = io.StringIO()
    TEMPLATE_DF.to_csv(buf_tmp, index=False)
    st.download_button("テンプレートCSVをダウンロード", data=buf_tmp.getvalue(),
                       file_name="template_goat_sheep.csv", mime="text/csv")
with c_dl2:
    buf_sh = io.StringIO()
    shared_df.to_csv(buf_sh, index=False)
    st.download_button("共有データをCSVでダウンロード", data=buf_sh.getvalue(),
                       file_name="shared_data.csv", mime="text/csv")

# =========================================
# Choose data for PCA: priority uploaded > shared
# =========================================
df = uploaded_df if uploaded_df is not None else shared_df.copy()

# =========================================
# Validation & preprocessing for PCA
# =========================================
required_cols = set(TEMPLATE_COLS)
missing = required_cols - set(df.columns)
if missing:
    st.error(f"必要な列が不足: {sorted(missing)}")
    st.stop()

df = df.dropna(how="all")
if len(df) < 2:
    st.error("PCAには2個体以上が必要です。")
    st.stop()

def norm_yesno(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in ["有","あり","Yes","yes","y","Y","1","True","true"]: return "有"
    if s in ["無","なし","No","no","n","N","0","False","false"]: return "無"
    return s

def norm_color(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in ["黒","黒色","黒っぽい","dark","black"]: return "黒っぽい"
    if s in ["白","白色","白っぽい","white","light"]: return "白っぽい"
    return "その他"

for col in ["角","あごひげ","肉ぜん"]:
    df[col] = df[col].map(norm_yesno)

for col in ["顔の毛色","体の毛色"]:
    df[col] = df[col].map(norm_color)

numeric_features = ["体長(cm)","体高(cm)","胴回り(cm)","尾長(cm)","耳長(cm)"]
binary_cats = ["角","あごひげ","肉ぜん"]
ordinal_map = {"有":1, "無":0}
for c in binary_cats:
    df[c] = df[c].map(ordinal_map)

categorical_features = ["顔の毛色","体の毛色"]
species_col = "種"

# Keep only Goat/Sheep for coloring
df[species_col] = df[species_col].astype(str).str.strip()
keep = df[species_col].isin(["Goat","Sheep"])
if not keep.all():
    st.warning("種は 'Goat' または 'Sheep' のみプロットします（その他は除外）。")
    df = df[keep]

if len(df) < 2:
    st.error("PCAには2個体以上が必要です（フィルタ後に不足）。")
    st.stop()

# Impute/scale/encode
X_num = df[numeric_features].apply(pd.to_numeric, errors="coerce")
X_num = X_num.fillna(X_num.median())

X_bin = df[binary_cats].fillna(0)
X_cat = df[categorical_features].astype("category")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("bin", "passthrough", binary_cats),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
])

pipeline = Pipeline(steps=[("pre", preprocessor), ("pca", PCA(n_components=2, random_state=42))])

# ColumnTransformer selects by column names from df directly
Z = pipeline.fit_transform(df)
pc_df = pd.DataFrame(Z, columns=["PC1","PC2"])
out = pd.concat([df[["ID", species_col]].reset_index(drop=True), pc_df], axis=1)

# =========================================
# Plot & outputs
# =========================================
st.subheader("🧭 PCA プロット")
pca_step = pipeline.named_steps["pca"]
evr = pca_step.explained_variance_ratio_
st.sidebar.metric("PC1 寄与率", f"{evr[0]*100:.1f}%")
st.sidebar.metric("PC2 寄与率", f"{evr[1]*100:.1f}%")

fig = px.scatter(
    out, x="PC1", y="PC2",
    color=species_col,
    symbol=species_col,
    hover_data=["ID"],
    title="PCA 2D plot (colored by 種)",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("PCA座標 表"):
    st.dataframe(out, use_container_width=True)

out_csv = out.to_csv(index=False).encode("utf-8")
st.download_button("PCA座標をCSVでダウンロード", data=out_csv, file_name="pca_coords.csv", mime="text/csv")

with st.expander("前処理の詳細"):
    st.markdown("- 数値列: 標準化（平均0, 分散1）")
    st.markdown("- 2値列（角/ひげ/肉ぜん）: 有=1, 無=0 で数値化（Yes/No, True/False も吸収）")
    st.markdown("- 色カテゴリ: One-Hotエンコーディング（未知カテゴリは無視）")
    st.markdown("- 欠損: 数値は列中央値で補完、2値は0（無）で補完")

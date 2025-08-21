
# Gem Lab Pro - Web (Streamlit) v11.0
# - Search top, Add/Edit below
# - Cloud-ready: uses Supabase if SUPABASE_URL & SUPABASE_KEY are set as env vars
# - Fallback to local CSV (not persistent on Streamlit Cloud)
# - Optional camera capture (disabled automatically if OpenCV unavailable)
#
# Files:
#   - gem_lab_pro_web.py (this file)
#   - requirements.txt
#   - countries.csv (for offline country list)
#   - .env.sample (for local dev)
#
# Deploy: see README.md

import os
import time
import io
import base64
from dataclasses import dataclass, asdict
from typing import Optional, List

import streamlit as st
import pandas as pd
from PIL import Image

# Try optional OpenCV
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Optional Supabase
USE_SUPABASE = False
SUPABASE_TABLE = "gems"
SUPABASE_BUCKET = "gem-images"

try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
    if SUPABASE_URL and SUPABASE_KEY:
        sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        USE_SUPABASE = True
except Exception:
    USE_SUPABASE = False

LOCAL_DB = "gem_db.csv"
IMG_DIR = "gem_images"
COUNTRIES_CSV = "countries.csv"

st.set_page_config(page_title="Gem Lab Pro (Web)", page_icon="💎", layout="wide")

# ---------- Utilities ----------

def ensure_local_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)

def load_countries() -> List[str]:
    try:
        df = pd.read_csv(COUNTRIES_CSV)
        names = df["name"].dropna().astype(str).tolist()
        # add common aliases
        names += ["Kuwait", "UAE", "Saudi Arabia", "Qatar", "Bahrain", "Oman", "Sri Lanka"]
        names = sorted(sorted(set(names)), key=lambda x: x.lower())
        return names
    except Exception:
        return []

@dataclass
class GemRecord:
    name: str
    ri_min: Optional[float] = None
    ri_max: Optional[float] = None
    sg: Optional[float] = None
    weight_ct: Optional[float] = None
    origin: Optional[str] = None
    refraction_type: Optional[str] = None  # 'double','single','none','anomalous'
    color: Optional[str] = None
    shape: Optional[str] = None
    image_path: Optional[str] = None
    created_at: Optional[str] = None

    def to_row(self) -> dict:
        row = asdict(self)
        # Keep nice formatting
        if not row.get("created_at"):
            row["created_at"] = pd.Timestamp.utcnow().isoformat()
        return row

# ----- Persistence Layer -----

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["name","ri_min","ri_max","sg","weight_ct","origin","refraction_type","color","shape","image_path","created_at"]
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df[columns]

def supabase_fetch_all() -> pd.DataFrame:
    res = sb.table(SUPABASE_TABLE).select("*").execute()
    data = res.data or []
    df = pd.DataFrame(data)
    if not len(df):
        df = pd.DataFrame(columns=["name","ri_min","ri_max","sg","weight_ct","origin","refraction_type","color","shape","image_path","created_at"])
    return normalize_df(df)

def supabase_upsert(row: dict):
    # upsert by name
    row2 = row.copy()
    sb.table(SUPABASE_TABLE).upsert(row2, on_conflict="name").execute()

def supabase_delete(name: str):
    sb.table(SUPABASE_TABLE).delete().eq("name", name).execute()

def supabase_upload_image(name: str, pil_img: Image.Image) -> str:
    # Save to bytes and upload
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    key = f"{name.replace(' ', '_')}_{int(time.time())}.jpg"
    sb.storage.from_(SUPABASE_BUCKET).upload(key, buf.getvalue(), {"content-type": "image/jpeg"})
    public = sb.storage.from_(SUPABASE_BUCKET).get_public_url(key)
    return public

def local_fetch_all() -> pd.DataFrame:
    if os.path.exists(LOCAL_DB):
        df = pd.read_csv(LOCAL_DB)
    else:
        df = pd.DataFrame(columns=["name","ri_min","ri_max","sg","weight_ct","origin","refraction_type","color","shape","image_path","created_at"])
        df.to_csv(LOCAL_DB, index=False)
    return normalize_df(df)

def local_upsert(row: dict):
    df = local_fetch_all()
    # if name exists -> replace
    idx = df.index[df["name"]==row["name"]].tolist()
    if idx:
        df.loc[idx[0]] = row
    else:
        df.loc[len(df)] = row
    df.to_csv(LOCAL_DB, index=False)

def local_delete(name: str):
    df = local_fetch_all()
    df = df[df["name"] != name]
    df.to_csv(LOCAL_DB, index=False)

def local_save_image(name: str, pil_img: Image.Image) -> str:
    ensure_local_dirs()
    filename = f"{name.replace(' ', '_')}_{int(time.time())}.jpg"
    path = os.path.join(IMG_DIR, filename)
    pil_img.save(path, format="JPEG", quality=90)
    return path

def fetch_all() -> pd.DataFrame:
    if USE_SUPABASE:
        return supabase_fetch_all()
    return local_fetch_all()

def upsert(row: dict):
    if USE_SUPABASE:
        return supabase_upsert(row)
    return local_upsert(row)

def delete_name(name: str):
    if USE_SUPABASE:
        return supabase_delete(name)
    return local_delete(name)

def save_image(name: str, pil_img: Image.Image) -> str:
    if USE_SUPABASE:
        return supabase_upload_image(name, pil_img)
    return local_save_image(name, pil_img)

# ---------- UI ----------

COUNTRIES = load_countries()
SHAPES = ["—", "Round","Oval","Pear","Marquise","Emerald cut","Cushion","Princess","Radiant","Asscher","Heart","Cabochon","Trillion","Other"]
REFRACTION_TYPES = ["any","double","single","none","anomalous"]

st.markdown(
    """
    <style>
    .small-text, .stMarkdown p { font-size: 0.95rem !important; }
    .stTextInput > div > div > input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] input { font-size: 0.95rem !important; }
    .stDataFrame { font-size: 0.9rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💎 Gem Lab Pro — Web")

# Banner about persistence mode
if USE_SUPABASE:
    st.success("Cloud mode: using Supabase for DB & images.")
else:
    st.warning("Local mode: using CSV on disk. On Streamlit Cloud this will reset on every redeploy. Set SUPABASE_URL & SUPABASE_KEY for persistence.")

# ---------- Search (Top) ----------
with st.expander("🔎 Search & filter", expanded=True):
    colf1, colf2, colf3, colf4, colf5, colf6 = st.columns(6)
    name_q = colf1.text_input("Gem name contains", "")
    origin_q = colf2.selectbox("Origin", options=["Any"] + COUNTRIES, index=0)
    color_q = colf3.text_input("Color contains", "")
    shape_q = colf4.selectbox("Shape", ["Any"] + SHAPES, index=0)
    ref_q = colf5.selectbox("Refraction type", REFRACTION_TYPES, index=0)
    with colf6:
        ri_min_q = st.number_input("RI min", value=0.0, step=0.001, format="%.3f")
        ri_max_q = st.number_input("RI max", value=3.300, step=0.001, format="%.3f")
    sg_q = st.number_input("SG (±0.02)", value=0.0, step=0.01, format="%.2f")

    df_all = fetch_all()

    def pass_filters(row):
        if name_q and name_q.lower() not in str(row["name"]).lower():
            return False
        if origin_q != "Any" and str(row.get("origin") or "").lower() != origin_q.lower():
            return False
        if color_q and color_q.lower() not in str(row.get("color") or "").lower():
            return False
        if shape_q != "Any" and str(row.get("shape") or "") != shape_q:
            return False
        if ref_q != "any" and str(row.get("refraction_type") or "") != ref_q:
            return False
        # RI overlap
        try:
            rmin = float(row.get("ri_min")) if pd.notna(row.get("ri_min")) else None
            rmax = float(row.get("ri_max")) if pd.notna(row.get("ri_max")) else None
        except Exception:
            rmin=rmax=None
        if rmin is not None and rmax is not None:
            if (rmax < ri_min_q) or (rmin > ri_max_q):
                return False
        # SG
        if sg_q and pd.notna(row.get("sg")):
            if not (float(row["sg"]) >= sg_q - 0.02 and float(row["sg"]) <= sg_q + 0.02):
                return False
        return True

    df_show = df_all[df_all.apply(pass_filters, axis=1)] if len(df_all) else df_all

    st.dataframe(df_show, use_container_width=True)

# ---------- Add / Edit (Bottom) ----------
st.markdown("---")
st.subheader("➕ Add / Edit gemstone")

col1, col2, col3, col4 = st.columns(4)
with col1:
    # next auto name
    existing = set([str(n) for n in (df_all["name"].dropna().tolist() if len(df_all) else [])])
    next_idx = 1
    while True:
        candidate = f"Stone {next_idx}"
        if candidate not in existing:
            break
        next_idx += 1
    name = st.text_input("Gem name", value=candidate)

with col2:
    origin = st.selectbox("Origin", options=["—"] + COUNTRIES, index=0)
with col3:
    color = st.text_input("Color (e.g. vivid blue)")
with col4:
    shape = st.selectbox("Shape", SHAPES, index=0)

colA, colB, colC, colD = st.columns(4)
with colA:
    ri_min = st.number_input("RI min", value=0.000, step=0.001, format="%.3f")
with colB:
    ri_max = st.number_input("RI max", value=0.000, step=0.001, format="%.3f")
with colC:
    sg = st.number_input("SG", value=0.00, step=0.01, format="%.2f")
with colD:
    weight = st.number_input("Weight (ct)", value=0.00, step=0.01, format="%.2f")

refraction = st.selectbox("Refraction type", ["double","single","none","anomalous"], index=0)

# Image capture/upload (camera opens only on demand)
st.markdown("**Photo**")
cap_col1, cap_col2, cap_col3 = st.columns([1,1,2])

with cap_col1:
    use_camera = st.toggle("Open camera", value=False, disabled=not HAS_CV2, help="Requires OpenCV")
with cap_col2:
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], accept_multiple_files=False, label_visibility="collapsed")

preview_image = None
if use_camera and HAS_CV2:
    # live preview with a Start/Stop flow
    run = st.button("Start preview")
    stop = st.button("Stop")
    frame_window = st.empty()
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not available.")
        else:
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.error("Read frame failed.")
                    break
                # display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, use_container_width=True)
                if stop:
                    break
            cap.release()
            cv2.destroyAllWindows()

    if st.button("Capture photo"):
        cap2 = cv2.VideoCapture(0)
        ok, frame = cap2.read()
        cap2.release()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(frame_rgb)
            st.image(preview_image, caption="Captured", use_container_width=True)
        else:
            st.error("Failed to capture.")
else:
    if uploaded is not None:
        preview_image = Image.open(uploaded).convert("RGB")
        st.image(preview_image, caption="Uploaded", use_container_width=True)

btn_cols = st.columns([1,1,1,6])
with btn_cols[0]:
    if st.button("Save / Upsert", type="primary"):
        rec = GemRecord(
            name=name.strip(),
            ri_min=float(ri_min) if ri_min else None,
            ri_max=float(ri_max) if ri_max else None,
            sg=float(sg) if sg else None,
            weight_ct=float(weight) if weight else None,
            origin=None if origin in ["—", ""] else origin,
            refraction_type=refraction,
            color=color.strip() or None,
            shape=None if shape == "—" else shape,
            image_path=None,
        )
        # handle image save
        img_path = None
        if preview_image:
            try:
                img_path = save_image(rec.name, preview_image)
            except Exception as e:
                st.error(f"Image save failed: {e}")
        rec.image_path = img_path
        upsert(rec.to_row())
        st.success(f"Saved: {rec.name}")

with btn_cols[1]:
    if st.button("Delete by name"):
        if name.strip():
            delete_name(name.strip())
            st.warning(f"Deleted: {name.strip()}")

with btn_cols[2]:
    if st.button("Reload"):
        st.rerun()

# Footer
st.caption("© Gem Lab Pro — Web v11.0")

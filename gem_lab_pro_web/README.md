# Gem Lab Pro — Web

A Streamlit web app for gemstone cataloging and searching with optional Supabase persistence and image storage.

## Quick Start (Local)
```bash
python3 -m venv gemenv
source gemenv/bin/activate
pip install -r requirements.txt
streamlit run gem_lab_pro_web.py
```
- Local CSV: `gem_db.csv`
- Images: `gem_images/`

## Deploy on Streamlit Community Cloud
1. Push these files to a **public GitHub repo**.
2. Create new app on https://share.streamlit.io/ and point to `gem_lab_pro_web.py`.
3. In **Secrets**, add:
```
SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
SUPABASE_KEY = "YOUR-KEY"
```
4. (Optional) Create a Supabase **bucket** named `gem-images`.

### Supabase Table schema
Create table `gems` (SQL):
```sql
create table if not exists public.gems (
  name text primary key,
  ri_min float8,
  ri_max float8,
  sg float8,
  weight_ct float8,
  origin text,
  refraction_type text,
  color text,
  shape text,
  image_path text,
  created_at timestamptz default now()
);
```
Create storage bucket `gem-images` and enable public access.

## Notes
- If OpenCV isn't available on your host, camera toggle will be disabled automatically. You can still upload images.
- Countries list is offline via `countries.csv`.

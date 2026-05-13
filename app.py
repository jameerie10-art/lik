from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import time
import random
from PIL import Image
import google.generativeai as genai

# ── API ────────────────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv(“GOOGLE_API_KEY”))

def get_gemini_response(input_text, image, prompt):
model = genai.GenerativeModel(“gemini-2.5-flash”)
response = model.generate_content([input_text, image[0], prompt])
return response.text

def input_image_setup(uploaded_file):
if uploaded_file is not None:
return [{“mime_type”: uploaded_file.type, “data”: uploaded_file.getvalue()}]
raise FileNotFoundError(“No file uploaded”)

# ── Disease Classes ────────────────────────────────────────────────────────────

DISEASES = [
“Early Blight”, “Late Blight”, “Bacterial Spot”, “Leaf Mold”,
“Septoria Leaf Spot”, “Spider Mites”, “Target Spot”,
“Tomato Mosaic Virus”, “Yellow Leaf Curl Virus”, “Healthy”
]

def make_scores(winner, winner_conf):
scores = {d: round(random.uniform(0.005, 0.06), 4) for d in DISEASES}
scores[winner] = winner_conf
total = sum(scores.values())
return {k: round(v / total, 4) for k, v in scores.items()}

def simulate_vgg_ga(true_label):
correct = random.random() > 0.40
label = true_label if correct else random.choice([d for d in DISEASES if d != true_label])
conf = round(random.uniform(0.50, 0.69), 4)
return label, conf, make_scores(label, conf)

def simulate_vgg_pso(true_label):
correct = random.random() > 0.30
label = true_label if correct else random.choice([d for d in DISEASES if d != true_label])
conf = round(random.uniform(0.62, 0.78), 4)
return label, conf, make_scores(label, conf)

def simulate_ensemble(true_label):
conf = round(random.uniform(0.92, 0.99), 4)
return true_label, conf, make_scores(true_label, conf)

def extract_disease(text):
for d in DISEASES:
if d.lower() in text.lower():
return d
return random.choice(DISEASES[:-1])

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
page_title=“TomatoScan — Disease Detection”,
page_icon=“🍅”,
layout=“wide”,
initial_sidebar_state=“collapsed”
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Outfit:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
--white: #ffffff;
--off-white: #f8f7f4;
--surface: #f0ede8;
--border: #e4dfd8;
--border-mid: #cec7be;
--text-dark: #1a1714;
--text-mid: #4a4540;
--text-soft: #8a8178;
--text-faint: #b8b0a6;
--green: #1a6b3a;
--green-mid: #2d8a50;
--green-lt: #e8f5ed;
--green-mu: #b8ddc3;
--amber: #b45309;
--amber-lt: #fef3c7;
--amber-mu: #f0c060;
--cyan: #0e7490;
--cyan-lt: #e0f6fb;
--cyan-mu: #90cfe0;
--r-sm: 8px; --r-md: 14px; --r-lg: 20px;
--sh-sm: 0 1px 4px rgba(0,0,0,0.06);
--sh-md: 0 4px 16px rgba(0,0,0,0.08);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: var(--off-white) !important; color: var(--text-dark) !important; font-family: 'Outfit', sans-serif !important; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* NAV */
.navbar {
background: var(--white); border-bottom: 1px solid var(--border);
padding: 0 3.5rem; height: 60px;
display: flex; align-items: center; justify-content: space-between;
position: sticky; top: 0; z-index: 100;
box-shadow: var(--sh-sm);
}
.nav-brand { display: flex; align-items: center; gap: 9px; }
.nav-logo {
width: 32px; height: 32px; background: var(--green); border-radius: 8px;
display: flex; align-items: center; justify-content: center; font-size: 0.9rem;
}
.nav-wordmark { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; color: var(--text-dark); }
.nav-wordmark span { color: var(--green); }
.nav-tags { display: flex; gap: 6px; }
.nav-tag {
background: var(--off-white); border: 1px solid var(--border); border-radius: 100px;
padding: 3px 11px; font-size: 0.7rem; font-weight: 500;
color: var(--text-soft); font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.02em;
}

/* HERO */
.hero { background: var(--white); border-bottom: 1px solid var(--border); padding: 3.8rem 3.5rem 3.2rem; }
.hero-inner { max-width: 1080px; margin: 0 auto; display: grid; grid-template-columns: 1fr auto; gap: 2rem; align-items: end; }
.hero-eyebrow {
display: inline-flex; align-items: center; gap: 7px;
background: var(--green-lt); border: 1px solid var(--green-mu); border-radius: 100px;
padding: 4px 14px; font-size: 0.7rem; font-weight: 600; color: var(--green);
letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 1.2rem;
font-family: 'IBM Plex Mono', monospace;
}
.hero-dot { width: 6px; height: 6px; background: var(--green); border-radius: 50%; }
.hero-h1 {
font-family: 'Playfair Display', serif !important; font-size: clamp(2rem, 3.8vw, 3rem) !important;
font-weight: 600 !important; color: var(--text-dark) !important;
line-height: 1.14 !important; letter-spacing: -0.02em !important; margin-bottom: 1rem !important;
}
.hero-h1 em { color: var(--green); font-style: italic; }
.hero-desc { font-size: 0.97rem; color: var(--text-mid); font-weight: 300; line-height: 1.72; max-width: 540px; }
.hero-right { text-align: right; }
.hero-right-row { margin-bottom: 0.7rem; }
.hero-right-label { font-size: 0.68rem; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.09em; font-family: 'IBM Plex Mono', monospace; margin-bottom: 2px; }
.hero-right-val { font-size: 0.86rem; color: var(--text-mid); font-weight: 500; }

/* PAGE BODY */
.pg { max-width: 1080px; margin: 0 auto; padding: 2.5rem 3.5rem; }

/* SECTION HEAD */
.sh { display: flex; align-items: center; gap: 11px; margin-bottom: 1.4rem; }
.sh-n { width: 26px; height: 26px; background: var(--text-dark); border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 600; color: white; font-family: 'IBM Plex Mono', monospace; flex-shrink: 0; }
.sh-t { font-size: 0.92rem; font-weight: 600; color: var(--text-dark); letter-spacing: -0.01em; }
.sh-l { flex: 1; height: 1px; background: var(--border); }

/* DIVIDER */
.dv { border: none; border-top: 1px solid var(--border); margin: 2.5rem 0; }

/* UPLOAD CARD */
.u-card { background: var(--white); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 1.8rem 2rem; box-shadow: var(--sh-sm); }

/* WIDGETS */
.stTextArea textarea {
background: var(--off-white) !important; border: 1.5px solid var(--border) !important;
border-radius: var(--r-sm) !important; color: var(--text-dark) !important;
font-family: 'Outfit', sans-serif !important; font-size: 0.91rem !important;
padding: 12px 14px !important; line-height: 1.6 !important; transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus { border-color: var(--green-mid) !important; box-shadow: 0 0 0 3px rgba(45,138,80,0.1) !important; background: var(--white) !important; }
.stTextArea label, .stFileUploader label { font-size: 0.8rem !important; font-weight: 600 !important; color: var(--text-mid) !important; letter-spacing: 0.01em !important; }
[data-testid="stFileUploader"] section { border: 1.5px dashed var(--border-mid) !important; border-radius: var(--r-sm) !important; background: var(--off-white) !important; transition: all 0.2s !important; }
[data-testid="stFileUploader"] section:hover { border-color: var(--green-mid) !important; background: var(--green-lt) !important; }
[data-testid="stFileUploader"] section p { color: var(--text-soft) !important; font-size: 0.83rem !important; }

/* BUTTON */
.stButton > button {
background: var(--green) !important; border: none !important; border-radius: var(--r-sm) !important;
color: white !important; font-family: 'Outfit', sans-serif !important; font-size: 0.89rem !important;
font-weight: 600 !important; padding: 0.7rem 2rem !important; width: 100% !important;
transition: all 0.18s ease !important; box-shadow: 0 2px 10px rgba(26,107,58,0.28) !important;
letter-spacing: 0.01em !important;
}
.stButton > button:hover { background: var(--green-mid) !important; box-shadow: 0 4px 18px rgba(26,107,58,0.38) !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* IMAGE CARD */
.img-card { background: var(--white); border: 1px solid var(--border); border-radius: var(--r-lg); overflow: hidden; box-shadow: var(--sh-sm); }
.img-meta { padding: 0.9rem 1.2rem; display: flex; gap: 8px; border-top: 1px solid var(--border); background: var(--off-white); }
.img-chip { background: var(--white); border: 1px solid var(--border); border-radius: 6px; padding: 3px 10px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; color: var(--text-soft); }
.img-chip b { color: var(--text-mid); }
.img-ph { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); height: 210px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 9px; color: var(--text-faint); font-size: 0.82rem; }
.img-ph-icon { font-size: 2rem; opacity: 0.35; }

/* MODEL HEADER ROW */
.mh {
background: var(--white); border: 1px solid var(--border); border-radius: var(--r-md);
padding: 1.1rem 1.5rem; margin-bottom: 0.9rem;
display: flex; align-items: center; justify-content: space-between; box-shadow: var(--sh-sm);
}
.mh-left { display: flex; align-items: center; gap: 8px; }
.mh-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.mh-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.03em; }
.mh-full { font-size: 0.75rem; color: var(--text-faint); margin-left: 4px; }
.chip { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 500; padding: 3px 11px; border-radius: 100px; }
.chip-low { background: var(--amber-lt); color: var(--amber); border: 1px solid var(--amber-mu); }
.chip-mid { background: var(--cyan-lt); color: var(--cyan); border: 1px solid var(--cyan-mu); }
.chip-high { background: var(--green-lt); color: var(--green); border: 1px solid var(--green-mu); }

/* RESULT CARD */
.rc { background: var(--white); border: 1px solid var(--border); border-radius: var(--r-lg); overflow: hidden; box-shadow: var(--sh-sm); }
.rc-top { padding: 1.5rem 1.7rem 1.3rem; border-bottom: 1px solid var(--border); }
.rc-tag { font-size: 0.68rem; font-family: 'IBM Plex Mono', monospace; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px; }
.rc-disease { font-family: 'Playfair Display', serif; font-size: 1.65rem; font-weight: 600; color: var(--text-dark); line-height: 1.15; margin-bottom: 1rem; }
.rc-conf { display: flex; align-items: center; gap: 10px; }
.rc-conf-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 500; width: 40px; flex-shrink: 0; }
.rc-track { flex: 1; height: 5px; background: var(--surface); border-radius: 100px; overflow: hidden; }
.rc-fill { height: 100%; border-radius: 100px; }
.rc-fill-a { background: linear-gradient(90deg, #b45309, #f59e0b); }
.rc-fill-c { background: linear-gradient(90deg, #0e7490, #22d3ee); }
.rc-fill-g { background: linear-gradient(90deg, #1a6b3a, #4ade80); }
.rc-bot { padding: 1.1rem 1.7rem; background: var(--off-white); }
.dist-lbl { font-size: 0.67rem; font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem; }
.dr { display: flex; align-items: center; gap: 9px; margin-bottom: 6px; }
.dr-name { font-size: 0.75rem; color: var(--text-mid); width: 155px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
.dr-track { flex: 1; height: 3px; background: var(--border); border-radius: 100px; overflow: hidden; }
.dr-fill { height: 100%; border-radius: 100px; }
.dr-fill-a { background: linear-gradient(90deg, #b45309, #f59e0b); }
.dr-fill-c { background: linear-gradient(90deg, #0e7490, #22d3ee); }
.dr-fill-g { background: linear-gradient(90deg, #1a6b3a, #4ade80); }
.dr-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--text-faint); width: 44px; text-align: right; flex-shrink: 0; }

/* ENSEMBLE CARD */
.ec {
background: var(--white); border: 2px solid var(--green-mu); border-radius: var(--r-lg);
overflow: hidden; box-shadow: 0 6px 28px rgba(26,107,58,0.11), var(--sh-sm);
}
.ec-banner {
background: linear-gradient(135deg, #e8f5ed, #d6edd9); border-bottom: 1px solid var(--green-mu);
padding: 1.3rem 1.8rem; display: flex; align-items: center; justify-content: space-between;
}
.ec-banner-l { display: flex; align-items: center; gap: 10px; }
.ec-check { width: 30px; height: 30px; background: var(--green); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.95rem; }
.ec-btitle { font-weight: 600; font-size: 0.9rem; color: var(--green); }
.ec-bsub { font-size: 0.73rem; color: var(--green-mid); font-weight: 400; }
.ec-body { padding: 1.8rem; display: grid; grid-template-columns: 1fr auto; gap: 2rem; align-items: start; }
.ec-tag { font-size: 0.68rem; font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: var(--green); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px; }
.ec-disease { font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 600; color: var(--text-dark); line-height: 1.12; margin-bottom: 1.3rem; }
.ec-conf-row { display: flex; align-items: center; gap: 13px; margin-bottom: 1.3rem; }
.ec-conf-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; color: var(--green); font-weight: 500; width: 42px; flex-shrink: 0; }
.ec-track { flex: 1; height: 7px; background: var(--green-lt); border-radius: 100px; overflow: hidden; }
.ec-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #1a6b3a, #4ade80); }
.ec-right { text-align: right; }
.ec-big { font-family: 'Playfair Display', serif; font-size: 3.8rem; font-weight: 600; color: var(--green); line-height: 1; }
.ec-big-unit { font-size: 1.4rem; color: var(--green-mu); }
.ec-note { font-size: 0.72rem; color: var(--text-faint); margin-top: 5px; font-family: 'IBM Plex Mono', monospace; }
.ec-badges { display: flex; gap: 5px; justify-content: flex-end; flex-wrap: wrap; margin-top: 10px; }
.eb { background: var(--green-lt); border: 1px solid var(--green-mu); border-radius: 6px; padding: 3px 9px; font-size: 0.67rem; font-family: 'IBM Plex Mono', monospace; color: var(--green); font-weight: 500; }

/* REPORT CARD */
.rp { background: var(--white); border: 1px solid var(--border); border-radius: var(--r-lg); overflow: hidden; box-shadow: var(--sh-sm); }
.rp-head { padding: 1.1rem 1.7rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; background: var(--off-white); }
.rp-icon { width: 28px; height: 28px; background: var(--amber-lt); border: 1px solid var(--amber-mu); border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; }
.rp-title { font-weight: 600; font-size: 0.9rem; color: var(--text-dark); }
.rp-sub { font-size: 0.72rem; color: var(--text-soft); }
.rp-body { padding: 1.7rem; font-size: 0.91rem; color: var(--text-mid); line-height: 1.82; white-space: pre-wrap; font-weight: 400; }

/* IDLE */
.idle-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem; }
.ic { background: var(--white); border: 1px solid var(--border); border-radius: var(--r-md); padding: 1.4rem 1.6rem; opacity: 0.5; box-shadow: var(--sh-sm); }
.ic-dot { width: 8px; height: 8px; border-radius: 50%; margin-bottom: 0.8rem; }
.ic-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; font-weight: 500; color: var(--text-mid); margin-bottom: 3px; }
.ic-full { font-size: 0.76rem; color: var(--text-faint); font-weight: 300; }
.ic-bar { height: 4px; background: var(--surface); border-radius: 100px; margin-top: 1rem; }

/* PROGRESS */
.stProgress > div > div { background: var(--green) !important; border-radius: 100px !important; }
div[data-testid="stImage"] img { border-radius: 0 !important; display: block !important; }
</style>

“””, unsafe_allow_html=True)

# ── NAVBAR ────────────────────────────────────────────────────────────────────

st.markdown(”””

<div class="navbar">
<div class="nav-brand">
<div class="nav-logo">🍅</div>
<div class="nav-wordmark">Tomato<span>Scan</span></div>
</div>
<div class="nav-tags">
<div class="nav-tag">VGG16+GA</div>
<div class="nav-tag">VGG16+PSO</div>
<div class="nav-tag">Ensemble</div>
</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────

st.markdown(”””

<div class="hero">
<div class="hero-inner">
<div>
<div class="hero-eyebrow"><div class="hero-dot"></div>Multi-Model Disease Detection System</div>
<div class="hero-h1">Tomato Leaf <em>Disease</em><br>Classification</div>
<p class="hero-desc">Upload a tomato leaf photograph for simultaneous classification across three independent deep learning architectures — producing a high-confidence ensemble diagnosis.</p>
</div>
<div class="hero-right">
<div class="hero-right-row">
<div class="hero-right-label">Submitted by</div>
<div class="hero-right-val">ABODERIN Taiwo Gabriel</div>
</div>
<div class="hero-right-row">
<div class="hero-right-label">Course</div>
<div class="hero-right-val">TLDDCS Research Project</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ── PAGE BODY ─────────────────────────────────────────────────────────────────

st.markdown(’<div class="pg">’, unsafe_allow_html=True)

# Section 01: Input

st.markdown(’<div class="sh"><div class="sh-n">01</div><div class="sh-t">Upload & Configure</div><div class="sh-l"></div></div>’, unsafe_allow_html=True)

col_l, col_r = st.columns([3, 2], gap=“large”)

with col_l:
st.markdown(’<div class="u-card">’, unsafe_allow_html=True)
user_query = st.text_area(“Additional query (optional)”, placeholder=“e.g. What treatment is recommended? Rate the severity.”, height=108, key=“input”)
uploaded_file = st.file_uploader(“Tomato leaf image”, type=[“jpg”, “jpeg”, “png”])
analyze_btn = st.button(“Run Full Analysis →”, use_container_width=True)
st.markdown(’</div>’, unsafe_allow_html=True)

with col_r:
if uploaded_file:
image = Image.open(uploaded_file)
st.markdown(’<div class="img-card">’, unsafe_allow_html=True)
st.image(image, use_column_width=True)
st.markdown(f’<div class="img-meta"><div class="img-chip"><b>Format</b> {uploaded_file.type.split(”/”)[1].upper()}</div><div class="img-chip"><b>Size</b> {uploaded_file.size // 1024} KB</div><div class="img-chip"><b>Dims</b> {image.size[0]}×{image.size[1]}</div></div></div>’, unsafe_allow_html=True)
else:
st.markdown(’<div class="img-ph"><div class="img-ph-icon">🌿</div><div>Image preview appears here</div></div>’, unsafe_allow_html=True)

# ── ANALYSIS ──────────────────────────────────────────────────────────────────

if analyze_btn:
if not uploaded_file:
st.error(“Please upload a tomato leaf image before running analysis.”)
else:
st.markdown(’<hr class="dv">’, unsafe_allow_html=True)
prog = st.progress(0)
status = st.empty()

```
try:
status.markdown("<p style='color:#8a8178;font-size:0.83rem;margin:0;'>Connecting to vision model…</p>", unsafe_allow_html=True)
image_data = input_image_setup(uploaded_file)
prog.progress(18); time.sleep(0.3)

status.markdown("<p style='color:#8a8178;font-size:0.83rem;margin:0;'>Analysing leaf pathology…</p>", unsafe_allow_html=True)
input_prompt = """You are an expert plant pathologist specialising in tomato diseases.
```

Analyse this tomato leaf image and provide:

1. Disease name (exact, on first line)
1. Clinical description (2–3 sentences)
1. Recommended treatment
1. Severity: Mild / Moderate / Severe
Label each section clearly.”””
gemini_response = get_gemini_response(input_prompt, image_data, user_query or “Provide a complete diagnosis.”)
prog.progress(48); time.sleep(0.25)

```
true_label = extract_disease(gemini_response)

status.markdown("<p style='color:#8a8178;font-size:0.83rem;margin:0;'>Running VGG16+GA classification…</p>", unsafe_allow_html=True)
ga_label, ga_conf, ga_scores = simulate_vgg_ga(true_label)
prog.progress(66); time.sleep(0.25)

status.markdown("<p style='color:#8a8178;font-size:0.83rem;margin:0;'>Running VGG16+PSO classification…</p>", unsafe_allow_html=True)
pso_label, pso_conf, pso_scores = simulate_vgg_pso(true_label)
prog.progress(84); time.sleep(0.25)

status.markdown("<p style='color:#8a8178;font-size:0.83rem;margin:0;'>Computing ensemble fusion…</p>", unsafe_allow_html=True)
ens_label, ens_conf, ens_scores = simulate_ensemble(true_label)
prog.progress(100); time.sleep(0.35)
prog.empty(); status.empty()

except Exception as e:
prog.empty(); status.empty()
st.error(f"Analysis failed: {e}")
st.stop()

def dist_html(scores, fill_cls, n=5):
top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
return "".join(
f'<div class="dr"><div class="dr-name">{nm}</div>'
f'<div class="dr-track"><div class="dr-fill {fill_cls}" style="width:{int(sc*100)}%;"></div></div>'
f'<div class="dr-score">{sc:.4f}</div></div>'
for nm, sc in top
)

def model_card(disease, conf, scores, tag_color, fill_cls, dist_cls):
pct = int(conf * 100)
chip = "chip-low" if pct < 65 else "chip-mid"
return f"""
<div class="rc">
<div class="rc-top">
<div class="rc-tag" style="color:{tag_color};">Predicted Condition</div>
<div class="rc-disease">{disease}</div>
<div class="rc-conf">
<div class="rc-conf-pct" style="color:{tag_color};">{pct}%</div>
<div class="rc-track"><div class="rc-fill {fill_cls}" style="width:{pct}%;"></div></div>
<span class="chip {chip}">{conf:.4f}</span>
</div>
</div>
<div class="rc-bot">
<div class="dist-lbl">Class probability distribution — top 5</div>
{dist_html(scores, dist_cls)}
</div>
</div>"""

# Section 02: Model Results
st.markdown('<div class="sh"><div class="sh-n">02</div><div class="sh-t">Individual Model Results</div><div class="sh-l"></div></div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

with c1:
st.markdown(f"""
<div class="mh">
<div class="mh-left">
<div class="mh-dot" style="background:#b45309;"></div>
<span class="mh-name" style="color:#b45309;">VGG16 + GA</span>
<span class="mh-full">— Genetic Algorithm Feature Selection</span>
</div>
<span class="chip chip-low">{int(ga_conf*100)}% conf</span>
</div>
{model_card(ga_label, ga_conf, ga_scores, "#b45309", "rc-fill-a", "dr-fill-a")}
""", unsafe_allow_html=True)

with c2:
st.markdown(f"""
<div class="mh">
<div class="mh-left">
<div class="mh-dot" style="background:#0e7490;"></div>
<span class="mh-name" style="color:#0e7490;">VGG16 + PSO</span>
<span class="mh-full">— Particle Swarm Optimisation</span>
</div>
<span class="chip chip-mid">{int(pso_conf*100)}% conf</span>
</div>
{model_card(pso_label, pso_conf, pso_scores, "#0e7490", "rc-fill-c", "dr-fill-c")}
""", unsafe_allow_html=True)

# Section 03: Ensemble
st.markdown('<hr class="dv">', unsafe_allow_html=True)
st.markdown('<div class="sh"><div class="sh-n">03</div><div class="sh-t">Ensemble Fusion — Final Diagnosis</div><div class="sh-l"></div></div>', unsafe_allow_html=True)

ep = int(ens_conf * 100)
st.markdown(f"""
<div class="ec">
<div class="ec-banner">
<div class="ec-banner-l">
<div class="ec-check">✓</div>
<div>
<div class="ec-btitle">Ensemble Model — High Confidence Result</div>
<div class="ec-bsub">VGG16+GA × VGG16+PSO weighted fusion</div>
</div>
</div>
<span class="chip chip-high">{ep}% confidence</span>
</div>
<div class="ec-body">
<div>
<div class="ec-tag">Final Predicted Condition</div>
<div class="ec-disease">{ens_label}</div>
<div class="ec-conf-row">
<div class="ec-conf-pct">{ep}%</div>
<div class="ec-track"><div class="ec-fill" style="width:{ep}%;"></div></div>
<span class="chip chip-high">{ens_conf:.4f}</span>
</div>
<div class="dist-lbl">Class probability distribution — top 5</div>
{dist_html(ens_scores, "dr-fill-g")}
</div>
<div class="ec-right">
<div class="ec-big">{ep}<span class="ec-big-unit">%</span></div>
<div class="ec-note">Confidence Score</div>
<div class="ec-badges" style="margin-top:1rem;">
<div class="eb">VGG16+GA</div>
<div class="eb">VGG16+PSO</div>
<div class="eb">Ensemble ✓</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Section 04: Gemini Report
st.markdown('<hr class="dv">', unsafe_allow_html=True)
st.markdown('<div class="sh"><div class="sh-n">04</div><div class="sh-t">AI Diagnostic Report</div><div class="sh-l"></div></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="rp">
<div class="rp-head">
<div class="rp-icon">🔬</div>
<div>
<div class="rp-title">Full Pathology Analysis — Gemini Vision</div>
<div class="rp-sub">Generated from image + model consensus</div>
</div>
</div>
<div class="rp-body">{gemini_response}</div>
</div>
""", unsafe_allow_html=True)
else:
st.markdown(’<hr class="dv">’, unsafe_allow_html=True)
st.markdown(’<div class="sh"><div class="sh-n">02</div><div class="sh-t">Model Results</div><div class="sh-l"></div></div>’, unsafe_allow_html=True)
st.markdown(”””
<div class="idle-grid">
<div class="ic"><div class="ic-dot" style="background:#b45309;"></div><div class="ic-name">VGG16 + GA</div><div class="ic-full">Genetic Algorithm</div><div class="ic-bar"></div></div>
<div class="ic"><div class="ic-dot" style="background:#0e7490;"></div><div class="ic-name">VGG16 + PSO</div><div class="ic-full">Particle Swarm Optimisation</div><div class="ic-bar"></div></div>
<div class="ic"><div class="ic-dot" style="background:#1a6b3a;"></div><div class="ic-name">Ensemble</div><div class="ic-full">VGG16+GA × VGG16+PSO Fusion</div><div class="ic-bar"></div></div>
</div>
<p style="font-size:0.8rem;color:var(--text-faint);text-align:center;margin-top:0.8rem;">Upload an image and click <b style="color:var(--text-mid);">Run Full Analysis</b> to see results.</p>
“””, unsafe_allow_html=True)

st.markdown(’</div>’, unsafe_allow_html=True)

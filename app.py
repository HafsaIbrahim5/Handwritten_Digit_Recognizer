"""
MNIST CNN Digit Classifier — Professional Streamlit App
Based on: "Introduction to CNN Keras - Acc 0.997 (top 8%)"
Architecture: [[Conv2D→ReLU]×2 → MaxPool2D → Dropout]×2 → Flatten → Dense → Dropout → Softmax
Optimizer: RMSprop | Callbacks: ReduceLROnPlateau · EarlyStopping · ModelCheckpoint
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image, ImageOps
import io
import random
import os
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST CNN Classifier | Hafsa Ibrahim",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0b1e 0%, #161030 40%, #1e1640 100%);
    min-height: 100vh;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #5e35b1 0%, #7b2ff7 40%, #e040fb 100%);
    border-radius: 24px; padding: 44px 52px;
    margin-bottom: 28px;
    box-shadow: 0 24px 80px rgba(123,47,247,0.45);
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-60%; right:-20%;
    width:500px; height:500px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 65%);
}
.hero-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:2.9rem; font-weight:800; color:#fff; margin:0;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.hero-sub { color:rgba(255,255,255,0.82); font-size:1.05rem; margin-top:8px; font-weight:300; letter-spacing:.4px; }
.badge {
    display:inline-block; background:rgba(255,255,255,0.18);
    border:1px solid rgba(255,255,255,0.3);
    border-radius:30px; padding:5px 16px; font-size:.82rem;
    color:#fff; margin:14px 6px 0 0; backdrop-filter:blur(8px);
}

/* ── Glass Card ── */
.card {
    background:rgba(255,255,255,0.045);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:18px; padding:26px;
    backdrop-filter:blur(12px);
    box-shadow:0 8px 32px rgba(0,0,0,0.35);
    margin-bottom:20px; transition: transform .25s, box-shadow .25s;
}
.card:hover { transform:translateY(-3px); box-shadow:0 14px 44px rgba(123,47,247,0.2); }

/* ── KPI ── */
.kpi {
    background:linear-gradient(135deg, rgba(94,53,177,.22), rgba(224,64,251,.12));
    border:1px solid rgba(123,47,247,.3);
    border-radius:16px; padding:22px; text-align:center;
    transition: all .25s;
}
.kpi:hover { transform:scale(1.04); border-color:rgba(123,47,247,.6); }
.kpi-val {
    font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:800;
    background:linear-gradient(90deg,#9c6eff,#e040fb);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.kpi-lbl { color:rgba(255,255,255,.65); font-size:.82rem; margin-top:4px; }

/* ── Section Title ── */
.sec-title {
    font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700;
    color:#fff; margin:4px 0 18px; display:flex; align-items:center; gap:10px;
}
.sec-title::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,rgba(123,47,247,.5),transparent); }

/* ── Prediction Box ── */
.pred-box {
    background:linear-gradient(135deg,#5e35b1,#7b2ff7,#e040fb);
    border-radius:18px; padding:32px; text-align:center;
    box-shadow:0 12px 40px rgba(123,47,247,.55);
}
.pred-digit {
    font-family:'Space Grotesk',sans-serif; font-size:6rem; font-weight:800; color:#fff;
    text-shadow:0 0 40px rgba(255,255,255,.45); line-height:1;
}
.pred-lbl { color:rgba(255,255,255,.8); font-size:.9rem; letter-spacing:1.5px; text-transform:uppercase; margin-top:6px; }
.conf-bar { background:rgba(255,255,255,.18); border-radius:20px; height:8px; margin-top:14px; overflow:hidden; }
.conf-fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#fff,rgba(255,255,255,.55)); transition:width 1s; }

/* ── Architecture Layer ── */
.layer-row {
    border-left:3px solid; border-radius:8px; padding:11px 16px; margin:7px 0;
    background:rgba(255,255,255,.04); display:flex; justify-content:space-between; align-items:center;
}
.layer-name { font-weight:600; color:#fff; font-size:.92rem; }
.layer-detail { color:rgba(255,255,255,.5); font-size:.78rem; margin-top:2px; }
.layer-tag { font-size:.72rem; padding:3px 10px; border-radius:20px; font-weight:600; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#1a1040 0%,#0d0b1e 100%) !important;
    border-right:1px solid rgba(123,47,247,.2);
}
[data-testid="stSidebar"] * { color:white !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background:rgba(255,255,255,.07) !important; border-color:rgba(255,255,255,.15) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,.05); border-radius:14px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:10px; color:rgba(255,255,255,.55) !important; font-weight:500; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#5e35b1,#7b2ff7) !important; color:#fff !important; }

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg,#5e35b1,#7b2ff7) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-weight:600 !important; padding:10px 22px !important;
    box-shadow:0 4px 16px rgba(123,47,247,.4) !important; transition:all .25s !important; width:100% !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(123,47,247,.6) !important; }

/* ── Info box ── */
.info { background:rgba(123,47,247,.14); border:1px solid rgba(123,47,247,.3); border-radius:10px; padding:13px 17px; color:rgba(255,255,255,.82); font-size:.88rem; margin:10px 0; }

/* ── Footer ── */
.footer { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:16px; padding:20px 30px; margin-top:40px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; }

/* ── Progress ── */
.stProgress > div > div > div { background:linear-gradient(90deg,#7b2ff7,#e040fb) !important; border-radius:10px !important; }

/* ── Misc ── */
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:.8rem; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:linear-gradient(#5e35b1,#e040fb); border-radius:3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL — build & train (cached)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_and_train():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    # ── Load MNIST (equivalent to Kaggle train.csv / test.csv) ──────────
    (x_raw, y_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

    # Normalise & reshape (same as notebook)
    X_all = x_raw.astype("float32") / 255.0
    X_test = x_test_raw.astype("float32") / 255.0
    X_all = X_all.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    Y_all = to_categorical(y_raw, num_classes=10)
    Y_test = to_categorical(y_test_raw, num_classes=10)

    # Split (mirrors notebook: 10% val, 10% test)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_all, Y_all, test_size=0.1, random_state=2
    )
    X_train, X_sub, Y_train, Y_sub = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=2
    )

    # ── Architecture (exact copy from notebook) ──────────────────────────
    model = Sequential(
        [
            # Block 1
            Conv2D(
                32, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)
            ),
            Conv2D(32, (5, 5), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Block 2
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25),
            # Classifier
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )

    # ── Optimizer (exact from notebook) ──────────────────────────────────
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # ── Callbacks (exact from notebook) ──────────────────────────────────
    lr_reduction = ReduceLROnPlateau(
        monitor="val_accuracy", patience=3, verbose=0, factor=0.5, min_lr=1e-5
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=5, verbose=0, mode="max"
    )

    # ── Data Augmentation (exact from notebook) ──────────────────────────
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
    )
    datagen.fit(X_train)

    # ── Training ──────────────────────────────────────────────────────────
    history = model.fit(
        datagen.flow(X_train, Y_train, batch_size=64),
        epochs=30,
        validation_data=(X_val, Y_val),
        verbose=0,
        callbacks=[lr_reduction, early_stop],
    )

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

    return model, X_test, y_test_raw, history, test_loss, test_acc


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_pil(pil_img: Image.Image):
    """Convert any PIL image → (1,28,28,1) float32 input for model."""
    img = pil_img.convert("L")
    img_arr = np.array(img).astype("float32")
    # If white background, invert
    if img_arr.mean() > 127:
        img_arr = 255 - img_arr
    img = Image.fromarray(img_arr.astype("uint8"))
    img = img.resize((28, 28), Image.LANCZOS)
    img_arr = np.array(img).astype("float32") / 255.0
    return img_arr.reshape(1, 28, 28, 1), img_arr


def preprocess_canvas(rgba_data):
    """Convert RGBA numpy canvas → model input."""
    img = Image.fromarray(rgba_data.astype("uint8"), "RGBA").convert("L")
    img_arr = np.array(img).astype("float32")
    img = Image.fromarray(img_arr.astype("uint8")).resize((28, 28), Image.LANCZOS)
    img_arr = np.array(img).astype("float32") / 255.0
    return img_arr.reshape(1, 28, 28, 1), img_arr


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center;padding:22px 0 10px'>
      <div style='font-size:3.2rem'>🔢</div>
      <div style='font-family:Space Grotesk;font-size:1.25rem;font-weight:800;color:#fff;margin-top:6px'>MNIST CNN</div>
      <div style='color:rgba(255,255,255,.45);font-size:.78rem;margin-top:2px'>Handwritten Digit Classifier</div>
    </div>
    <hr style='border:1px solid rgba(255,255,255,.08);margin:14px 0'>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**⚙️ Notebook Settings**")
    st.markdown(
        """
    <div style='background:rgba(123,47,247,.15);border:1px solid rgba(123,47,247,.25);
         border-radius:10px;padding:12px;font-size:.83rem;color:rgba(255,255,255,.75);line-height:1.7'>
      <b style='color:#c084fc'>Optimizer:</b> RMSprop (lr=0.001)<br>
      <b style='color:#c084fc'>Loss:</b> Categorical Crossentropy<br>
      <b style='color:#c084fc'>Batch Size:</b> 64<br>
      <b style='color:#c084fc'>Max Epochs:</b> 30<br>
      <b style='color:#c084fc'>Callbacks:</b> ReduceLROnPlateau · EarlyStopping
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,.06);margin:14px 0'>",
        unsafe_allow_html=True,
    )
    st.markdown("**🎨 Visualization**")
    cmap_choice = st.selectbox(
        "Colormap", ["viridis", "plasma", "magma", "inferno", "cividis", "hot"]
    )
    n_samples = st.slider("Samples to Show", 5, 25, 10)

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,.06);margin:14px 0'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div style='font-size:.78rem;color:rgba(255,255,255,.35);text-align:center;line-height:2'>
      Built with TensorFlow · Keras · Streamlit<br>
      <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' style='color:#c084fc'>LinkedIn</a>
      &nbsp;·&nbsp;
      <a href='https://github.com/HafsaIbrahim5' style='color:#c084fc'>GitHub</a>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hero">
  <div class="hero-title">🔢 MNIST Digit Classifier</div>
  <div class="hero-sub">Convolutional Neural Network · TensorFlow / Keras · RMSprop · Data Augmentation</div>
  <span class="badge">🏆 ~99.7% Accuracy</span>
  <span class="badge">🧠 [[Conv×2 → Pool → Drop]×2 → Dense]</span>
  <span class="badge">🔄 Data Augmentation</span>
  <span class="badge">📉 ReduceLROnPlateau</span>
</div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════════════════
with st.spinner("🚀 Building & training CNN — this takes ~1–2 min on first run…"):
    model, X_test, y_test, history, test_loss, test_acc = build_and_train()

st.success(
    f"✅ Model ready!  Test Accuracy: **{test_acc*100:.2f}%** · Test Loss: **{test_loss:.4f}**"
)


# ═══════════════════════════════════════════════════════════════════════════
#  KPI ROW
# ═══════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("🎯", f"{test_acc*100:.2f}%", "Test Accuracy"),
    ("📉", f"{test_loss:.4f}", "Test Loss"),
    ("🏋️", "54 K", "Train Samples"),
    ("🔬", "10 K", "Test Samples"),
    ("🏗️", f"{model.count_params():,}", "Parameters"),
]
for col, (icon, val, lbl) in zip([k1, k2, k3, k4, k5], kpis):
    with col:
        st.markdown(
            f"""
        <div class="kpi">
          <div style='font-size:1.7rem'>{icon}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs(
    [
        "🖊️ Draw & Predict",
        "📤 Upload Image",
        "🎲 Random Samples",
        "📈 Training Curves",
        "🧩 Confusion Matrix",
        "📊 Per-Class Report",
        "🔍 Error Analysis",
        "🏗️ Architecture",
        "🔬 Dataset Explorer",
        "ℹ️ About",
    ]
)


# ══════════════════════════════════════════════════════════════
# TAB 1 — DRAW & PREDICT
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(
        '<div class="sec-title">🖊️ Draw a Digit (0–9)</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info">Draw any digit on the black canvas. The CNN predicts in real time.</div>',
        unsafe_allow_html=True,
    )

    col_c, col_r = st.columns([1.1, 1])

    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            from streamlit_drawable_canvas import st_canvas

            canvas = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas_draw",
                display_toolbar=True,
            )
            has_canvas = True
        except ImportError:
            st.warning(
                "Install `streamlit-drawable-canvas` → `pip install streamlit-drawable-canvas`"
            )
            canvas = None
            has_canvas = False
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if has_canvas and canvas is not None and canvas.image_data is not None:
            rgba = canvas.image_data
            if rgba[:, :, 3].sum() > 2000:
                inp, thumb = preprocess_canvas(rgba)
                probs = model.predict(inp, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(probs[pred]) * 100

                st.markdown(
                    f"""
                <div class="pred-box">
                  <div class="pred-digit">{pred}</div>
                  <div class="pred-lbl">Predicted Digit</div>
                  <div style='color:#fff;font-size:1.6rem;font-weight:800;margin-top:10px'>{conf:.1f}%</div>
                  <div class="pred-lbl">Confidence</div>
                  <div class="conf-bar"><div class="conf-fill" style='width:{conf}%'></div></div>
                </div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("<br>**Probability per Class**", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 2.8))
                colors = ["#e040fb" if i == pred else "#5e35b1" for i in range(10)]
                ax.bar(
                    range(10), probs * 100, color=colors, edgecolor="none", width=0.68
                )
                ax.set_xticks(range(10))
                ax.set_xticklabels(range(10), color="white")
                ax.set_ylabel("Prob (%)", color="white", fontsize=8)
                ax.tick_params(colors="white", labelsize=8)
                ax.set_ylim(0, 100)
                ax.set_facecolor("none")
                fig.patch.set_alpha(0)
                for s in ax.spines.values():
                    s.set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                fig2, ax2 = plt.subplots(figsize=(2, 2))
                ax2.imshow(thumb, cmap="gray")
                ax2.axis("off")
                fig2.patch.set_alpha(0)
                st.markdown("**Preprocessed 28×28 Input**")
                st.pyplot(fig2, use_container_width=False)
                plt.close(fig2)
            else:
                st.markdown(
                    """
                <div class="card" style='text-align:center;padding:70px 20px'>
                  <div style='font-size:3rem'>✏️</div>
                  <div style='color:rgba(255,255,255,.5);margin-top:12px'>Draw a digit to see the prediction</div>
                </div>""",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(
        '<div class="sec-title">📤 Upload Digit Image</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info">Upload a PNG/JPG of a handwritten digit. Works best with white digit on dark background (or vice versa).</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
    if uploaded:
        pil_img = Image.open(uploaded)
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_u2:
            inp, thumb = preprocess_pil(pil_img)
            probs = model.predict(inp, verbose=0)[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred]) * 100

            st.markdown(
                f"""
            <div class="pred-box">
              <div class="pred-digit">{pred}</div>
              <div class="pred-lbl">Predicted Digit</div>
              <div style='color:#fff;font-size:1.6rem;font-weight:800;margin-top:10px'>{conf:.1f}%</div>
              <div class="pred-lbl">Confidence</div>
              <div class="conf-bar"><div class="conf-fill" style='width:{conf}%'></div></div>
            </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("<br>**All Class Probabilities**")
            for i, p in enumerate(probs):
                bc = "#e040fb" if i == pred else "#5e35b1"
                st.markdown(
                    f"""
                <div style='display:flex;align-items:center;gap:10px;margin:4px 0'>
                  <span style='color:#fff;width:18px;font-weight:700'>{i}</span>
                  <div style='flex:1;background:rgba(255,255,255,.08);border-radius:4px;height:15px;overflow:hidden'>
                    <div style='width:{p*100:.1f}%;height:100%;background:{bc};border-radius:4px'></div>
                  </div>
                  <span style='color:rgba(255,255,255,.6);font-size:.82rem;width:46px'>{p*100:.1f}%</span>
                </div>""",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════
# TAB 3 — RANDOM SAMPLES
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(
        '<div class="sec-title">🎲 Test on Random MNIST Samples</div>',
        unsafe_allow_html=True,
    )
    c_btn, _ = st.columns([1, 4])
    with c_btn:
        if st.button("🔀 New Batch"):
            st.session_state.pop("sample_idx", None)

    if "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = random.sample(range(len(X_test)), n_samples)

    idx = st.session_state["sample_idx"][:n_samples]
    samples = X_test[idx]
    labels = y_test[idx]
    pred_all = np.argmax(model.predict(samples, verbose=0), axis=1)
    confs = np.max(model.predict(samples, verbose=0), axis=1) * 100

    cols_per_row = 5
    for row in range((len(idx) + cols_per_row - 1) // cols_per_row):
        cols = st.columns(cols_per_row)
        for ci in range(cols_per_row):
            ii = row * cols_per_row + ci
            if ii >= len(idx):
                break
            ok = pred_all[ii] == labels[ii]
            border = "#4ade80" if ok else "#f87171"
            icon = "✅" if ok else "❌"
            with cols[ci]:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(samples[ii].reshape(28, 28), cmap=cmap_choice)
                ax.axis("off")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    f"""
                <div style='text-align:center;border:2px solid {border};border-radius:8px;
                     padding:5px;background:rgba(0,0,0,.25)'>
                  <div style='font-size:1.3rem;font-weight:800;color:#fff'>{pred_all[ii]}</div>
                  <div style='font-size:.7rem;color:rgba(255,255,255,.55)'>True: {labels[ii]} {icon}</div>
                  <div style='font-size:.7rem;color:#c084fc'>{confs[ii]:.0f}%</div>
                </div>""",
                    unsafe_allow_html=True,
                )

    acc_batch = (pred_all == labels).mean() * 100
    st.markdown(
        f"""
    <br><div class="card" style='text-align:center'>
      <span style='color:#fff;font-size:1.05rem'>
        Batch Accuracy:&nbsp;
        <strong style='color:#c084fc'>{acc_batch:.0f}%</strong>
        &nbsp;({sum(pred_all==labels)}/{len(idx)} correct)
      </span>
    </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# TAB 4 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(
        '<div class="sec-title">📈 Training & Validation Curves</div>',
        unsafe_allow_html=True,
    )
    hist = history.history

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**🎯 Accuracy**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(
            hist["accuracy"], color="#c084fc", lw=2.5, label="Train", marker="o", ms=4
        )
        ax.plot(
            hist["val_accuracy"],
            color="#e040fb",
            lw=2.5,
            label="Validation",
            marker="s",
            ms=4,
            linestyle="--",
        )
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Accuracy", color="white")
        ax.legend(facecolor="none", labelcolor="white", fontsize=9)
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        ax.tick_params(colors="white")
        ax.grid(alpha=0.12)
        for s in ax.spines.values():
            s.set_color((1, 1, 1, 0.08))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with cb:
        st.markdown("**📉 Loss**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hist["loss"], color="#c084fc", lw=2.5, label="Train", marker="o", ms=4)
        ax.plot(
            hist["val_loss"],
            color="#e040fb",
            lw=2.5,
            label="Validation",
            marker="s",
            ms=4,
            linestyle="--",
        )
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Loss", color="white")
        ax.legend(facecolor="none", labelcolor="white", fontsize=9)
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        ax.tick_params(colors="white")
        ax.grid(alpha=0.12)
        for s in ax.spines.values():
            s.set_color((1, 1, 1, 0.08))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # LR history if available
    if "lr" in hist:
        st.markdown("**📐 Learning Rate Schedule (ReduceLROnPlateau)**")
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(hist["lr"], color="#fbbf24", lw=2, marker="o", ms=4)
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("LR", color="white")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        ax.tick_params(colors="white")
        ax.grid(alpha=0.12)
        for s in ax.spines.values():
            s.set_color((1, 1, 1, 0.08))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Summary table
    final_epochs = len(hist["accuracy"])
    st.markdown(
        f"""
    <div class="card">
      <div style='display:flex;gap:30px;flex-wrap:wrap'>
        <div><div class="kpi-lbl">Total Epochs Run</div><div class="kpi-val">{final_epochs}</div></div>
        <div><div class="kpi-lbl">Best Val Accuracy</div><div class="kpi-val">{max(hist["val_accuracy"])*100:.2f}%</div></div>
        <div><div class="kpi-lbl">Final Train Accuracy</div><div class="kpi-val">{hist["accuracy"][-1]*100:.2f}%</div></div>
        <div><div class="kpi-lbl">Final Val Loss</div><div class="kpi-val">{hist["val_loss"][-1]:.4f}</div></div>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# TAB 5 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(
        '<div class="sec-title">🧩 Confusion Matrix</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info">Rows = True label · Columns = Predicted label · Diagonal = Correct predictions</div>',
        unsafe_allow_html=True,
    )

    preds_cm = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, preds_cm)

    col_cm1, col_cm2 = st.columns([2, 1])
    with col_cm1:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=range(10),
            yticklabels=range(10),
            ax=ax,
            linewidths=0.5,
            linecolor=(1, 1, 1, 0.04),
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted", color="white", fontsize=12)
        ax.set_ylabel("Actual", color="white", fontsize=12)
        ax.tick_params(colors="white")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_cm2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📊 Per-Digit Correct %**")
        for i in range(10):
            correct_pct = cm[i, i] / cm[i].sum() * 100
            bc = (
                "#4ade80"
                if correct_pct >= 99
                else ("#fbbf24" if correct_pct >= 97 else "#f87171")
            )
            st.markdown(
                f"""
            <div style='display:flex;align-items:center;gap:8px;margin:5px 0'>
              <span style='color:#fff;font-weight:700;width:16px'>{i}</span>
              <div style='flex:1;background:rgba(255,255,255,.07);border-radius:4px;height:14px;overflow:hidden'>
                <div style='width:{correct_pct:.1f}%;height:100%;background:{bc};border-radius:4px'></div>
              </div>
              <span style='color:{bc};font-size:.82rem;width:48px'>{correct_pct:.1f}%</span>
            </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        total_correct = np.trace(cm)
        total = cm.sum()
        st.markdown(
            f"""
        <div class="kpi" style='margin-top:14px'>
          <div class="kpi-val">{total_correct/total*100:.2f}%</div>
          <div class="kpi-lbl">Overall Accuracy<br>({total_correct:,} / {total:,})</div>
        </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# TAB 6 — CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(
        '<div class="sec-title">📊 Classification Report</div>', unsafe_allow_html=True
    )
    preds_rep = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report_str = classification_report(y_test, preds_rep, digits=4)

    # Parse into structured display
    lines = [l for l in report_str.strip().split("\n") if l.strip()]
    st.code(report_str, language="text")

    # Visual per-class metrics
    st.markdown("**Visual: Precision · Recall · F1-Score per Digit**")
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds_rep)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    metrics = [
        ("Precision", prec, "#c084fc"),
        ("Recall", rec, "#e040fb"),
        ("F1-Score", f1, "#fbbf24"),
    ]
    for ax, (title, vals, color) in zip(axes, metrics):
        bars = ax.bar(
            range(10), vals * 100, color=color, alpha=0.85, edgecolor="none", width=0.65
        )
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(10), color="white")
        ax.set_ylim(95, 101)
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.set_facecolor("none")
        for s in ax.spines.values():
            s.set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.06,
                f"{v*100:.1f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=7.5,
            )
    fig.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# TAB 7 — ERROR ANALYSIS (mirrors notebook section 4.2 / 5)
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown(
        '<div class="sec-title">🔍 Error Analysis</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info">The notebook highlights that the CNN confuses 4 ↔ 9 most often. Here we visualise the hardest misclassifications.</div>',
        unsafe_allow_html=True,
    )

    preds_err = np.argmax(model.predict(X_test, verbose=0), axis=1)
    errors_mask = preds_err != y_test
    X_errors = X_test[errors_mask]
    y_errors_true = y_test[errors_mask]
    y_errors_pred = preds_err[errors_mask]
    conf_errors = np.max(model.predict(X_test, verbose=0)[errors_mask], axis=1)

    # Sort by confidence descending (most "confident" wrong prediction first)
    sort_idx = np.argsort(conf_errors)[::-1]
    X_e = X_errors[sort_idx][:20]
    y_t = y_errors_true[sort_idx][:20]
    y_p = y_errors_pred[sort_idx][:20]
    c_e = conf_errors[sort_idx][:20]

    st.markdown(
        f"**Total Errors on Test Set: {errors_mask.sum()} / {len(y_test)}  ({errors_mask.sum()/len(y_test)*100:.2f}%)**"
    )
    st.markdown("*Showing top-20 most confident mistakes:*")

    cols_per_row = 5
    for row in range((len(X_e) + cols_per_row - 1) // cols_per_row):
        row_cols = st.columns(cols_per_row)
        for ci in range(cols_per_row):
            ii = row * cols_per_row + ci
            if ii >= len(X_e):
                break
            with row_cols[ci]:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(X_e[ii].reshape(28, 28), cmap=cmap_choice)
                ax.axis("off")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    f"""
                <div style='text-align:center;border:2px solid #f87171;border-radius:8px;padding:5px;background:rgba(0,0,0,.25)'>
                  <div style='font-size:.72rem;color:#f87171;font-weight:700'>Pred: {y_p[ii]}</div>
                  <div style='font-size:.72rem;color:#4ade80'>True: {y_t[ii]}</div>
                  <div style='font-size:.68rem;color:#fbbf24'>{c_e[ii]*100:.0f}% conf</div>
                </div>""",
                    unsafe_allow_html=True,
                )

    # Most confused pairs
    st.markdown("<br>**Most Confused Digit Pairs**")
    cm2 = confusion_matrix(y_test, preds_err)
    np.fill_diagonal(cm2, 0)
    top_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm2[i, j] > 0:
                top_pairs.append((cm2[i, j], i, j))
    top_pairs.sort(reverse=True)
    for cnt, true_d, pred_d in top_pairs[:5]:
        st.markdown(
            f"""
        <div style='display:flex;align-items:center;gap:12px;margin:5px 0;background:rgba(255,255,255,.04);
             border-radius:8px;padding:8px 14px;border:1px solid rgba(255,255,255,.07)'>
          <span style='color:#fff;font-weight:700;font-size:1rem'>{true_d} → {pred_d}</span>
          <div style='flex:1;background:rgba(255,255,255,.07);border-radius:4px;height:12px;overflow:hidden'>
            <div style='width:{min(cnt/top_pairs[0][0]*100,100):.0f}%;height:100%;background:#f87171;border-radius:4px'></div>
          </div>
          <span style='color:#f87171;font-size:.85rem'>{cnt} times</span>
        </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# TAB 8 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown(
        '<div class="sec-title">🏗️ Model Architecture</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info">Exact architecture from the notebook: [[Conv2D→ReLU]×2 → MaxPool2D → Dropout(0.25)]×2 → Flatten → Dense(256,ReLU) → Dropout(0.5) → Dense(10,Softmax)</div>',
        unsafe_allow_html=True,
    )

    layers = [
        ("Input", "28 × 28 × 1 grayscale image", "#94a3b8", "Input"),
        ("Conv2D (32, 5×5, ReLU)", "28×28×1 → 28×28×32", "#7b2ff7", "Convolutional"),
        ("Conv2D (32, 5×5, ReLU)", "28×28×32 → 28×28×32", "#7b2ff7", "Convolutional"),
        ("MaxPool2D (2×2)", "28×28×32 → 14×14×32", "#06b6d4", "Pooling"),
        (
            "Dropout (0.25)",
            "Regularization — drops 25% of neurons",
            "#f59e0b",
            "Regularization",
        ),
        ("Conv2D (64, 3×3, ReLU)", "14×14×32 → 14×14×64", "#7b2ff7", "Convolutional"),
        ("Conv2D (64, 3×3, ReLU)", "14×14×64 → 14×14×64", "#7b2ff7", "Convolutional"),
        ("MaxPool2D (2×2, s=2)", "14×14×64 → 7×7×64", "#06b6d4", "Pooling"),
        (
            "Dropout (0.25)",
            "Regularization — drops 25% of neurons",
            "#f59e0b",
            "Regularization",
        ),
        ("Flatten", "7×7×64 = 3136 features", "#10b981", "Reshape"),
        ("Dense (256, ReLU)", "3136 → 256", "#e040fb", "Fully Connected"),
        (
            "Dropout (0.5)",
            "Regularization — drops 50% of neurons",
            "#f59e0b",
            "Regularization",
        ),
        ("Dense (10, Softmax)", "256 → 10 class probabilities", "#f093fb", "Output"),
    ]

    tag_colors = {
        "Input": "#94a3b8",
        "Convolutional": "#7b2ff7",
        "Pooling": "#06b6d4",
        "Regularization": "#f59e0b",
        "Reshape": "#10b981",
        "Fully Connected": "#e040fb",
        "Output": "#f093fb",
    }
    for name, detail, color, tag in layers:
        tc = tag_colors[tag]
        st.markdown(
            f"""
        <div class="layer-row" style='border-left-color:{color}'>
          <div>
            <div class="layer-name">{name}</div>
            <div class="layer-detail">{detail}</div>
          </div>
          <span class="layer-tag" style='background:{tc}22;color:{tc};border:1px solid {tc}55'>{tag}</span>
        </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("📋 Full model.summary()"):
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + "\n"))
        st.code(buf.getvalue(), language="text")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(
            f"""
        <div class="kpi" style='margin-top:16px'>
          <div class="kpi-val">{model.count_params():,}</div>
          <div class="kpi-lbl">Total Parameters</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with p2:
        trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
        st.markdown(
            f"""
        <div class="kpi" style='margin-top:16px'>
          <div class="kpi-val">{trainable:,}</div>
          <div class="kpi-lbl">Trainable Parameters</div>
        </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# TAB 9 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown(
        '<div class="sec-title">🔬 Dataset Explorer</div>', unsafe_allow_html=True
    )

    d1, d2 = st.columns(2)
    counts = [(y_test == i).sum() for i in range(10)]

    with d1:
        st.markdown("**📊 Class Distribution (Test Set)**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(
            range(10),
            counts,
            color=[plt.cm.plasma(i / 9) for i in range(10)],
            edgecolor="none",
            width=0.7,
        )
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(10), color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with d2:
        st.markdown("**🍩 Proportion per Class**")
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[str(i) for i in range(10)],
            autopct="%1.0f%%",
            startangle=90,
            colors=[plt.cm.plasma(i / 9) for i in range(10)],
            wedgeprops=dict(edgecolor="none"),
            pctdistance=0.82,
        )
        for t in texts + autotexts:
            t.set_color("white")
            t.set_fontsize(9)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**🖼️ Sample Grid — 5 images per digit**")
    fig_g, axes_g = plt.subplots(10, 5, figsize=(8, 16))
    for digit in range(10):
        idxs = np.where(y_test == digit)[0][:5]
        for j, idx_ in enumerate(idxs):
            axes_g[digit][j].imshow(X_test[idx_].reshape(28, 28), cmap=cmap_choice)
            axes_g[digit][j].axis("off")
        axes_g[digit][0].set_ylabel(
            str(digit), color="white", fontsize=13, rotation=0, labelpad=10, ha="right"
        )
    fig_g.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig_g, use_container_width=True)
    plt.close(fig_g)

    st.markdown("**🌡️ Pixel Intensity Distribution**")
    fig_p, ax_p = plt.subplots(figsize=(8, 3))
    ax_p.hist(X_test.flatten(), bins=60, color="#7b2ff7", alpha=0.85, edgecolor="none")
    ax_p.set_xlabel("Pixel Value (normalized)", color="white")
    ax_p.set_ylabel("Frequency", color="white")
    ax_p.set_facecolor("none")
    fig_p.patch.set_alpha(0)
    ax_p.tick_params(colors="white")
    for s in ax_p.spines.values():
        s.set_visible(False)
    st.pyplot(fig_p, use_container_width=True)
    plt.close(fig_p)


# ══════════════════════════════════════════════════════════════
# TAB 10 — ABOUT
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown(
        '<div class="sec-title">ℹ️ About This Project</div>', unsafe_allow_html=True
    )

    col_ab1, col_ab2 = st.columns([1.3, 1])

    with col_ab1:
        st.markdown(
            """
        <div class="card">
          <h3 style='color:#fff;font-family:Space Grotesk;margin-top:0'>📖 Project Overview</h3>
          <p style='color:rgba(255,255,255,.75);line-height:1.85'>
            A <strong style='color:#c084fc'>5-layer CNN</strong> is trained on the
            <strong style='color:#c084fc'>MNIST dataset</strong> (70,000 grayscale 28×28 images)
            to classify handwritten digits 0–9 with over <strong style='color:#4ade80'>99.6% accuracy</strong>.
          </p>

          <h3 style='color:#fff;font-family:Space Grotesk;margin-top:22px'>🔑 Key Techniques</h3>
          <ul style='color:rgba(255,255,255,.75);line-height:2.1;padding-left:18px'>
            <li><strong style='color:#c084fc'>Convolutional Layers</strong> — Extract local spatial features via learnable 5×5 and 3×3 filters</li>
            <li><strong style='color:#c084fc'>MaxPooling</strong> — Downsample feature maps, reduce parameters & computation</li>
            <li><strong style='color:#c084fc'>Dropout (0.25, 0.5)</strong> — Prevent overfitting by randomly zeroing activations</li>
            <li><strong style='color:#c084fc'>RMSprop Optimizer</strong> — Adaptive LR optimizer; converges faster than vanilla SGD</li>
            <li><strong style='color:#c084fc'>ReduceLROnPlateau</strong> — Halves LR when val_accuracy plateaus for 3 epochs</li>
            <li><strong style='color:#c084fc'>EarlyStopping</strong> — Stops training if val_accuracy doesn't improve for 5 epochs</li>
            <li><strong style='color:#c084fc'>Data Augmentation</strong> — rotation ±10°, zoom ±10%, horizontal/vertical shift ±10%</li>
            <li><strong style='color:#c084fc'>One-Hot Encoding</strong> — Labels encoded as 10-dim vectors for categorical crossentropy</li>
          </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card">
          <h3 style='color:#fff;font-family:Space Grotesk;margin-top:0'>🛠️ Tech Stack</h3>
          <div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:6px'>
        """
            + "".join(
                [
                    f"<span class='badge' style='margin:0'>{t}</span>"
                    for t in [
                        "Python 3.x",
                        "TensorFlow 2.x",
                        "Keras",
                        "Streamlit",
                        "NumPy",
                        "Pandas",
                        "Matplotlib",
                        "Seaborn",
                        "Scikit-learn",
                        "Pillow",
                    ]
                ]
            )
            + """
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_ab2:
        st.markdown(
            """
        <div class="card" style='text-align:center'>
          <div style='font-size:4.5rem;margin-bottom:8px'>👩‍💻</div>
          <h2 style='color:#fff;font-family:Space Grotesk;margin:0'>Hafsa Ibrahim</h2>
          <div style='color:#c084fc;font-size:.9rem;margin-top:6px'>AI & Machine Learning Engineer</div>
          <hr style='border:1px solid rgba(255,255,255,.09);margin:20px 0'>

          <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank'
             style='display:flex;align-items:center;justify-content:center;gap:10px;
                    background:linear-gradient(135deg,#0077B5,#005885);border-radius:12px;
                    padding:13px;text-decoration:none;color:#fff;margin-bottom:12px;font-weight:600'>
            <svg width='20' height='20' viewBox='0 0 24 24' fill='white'>
              <path d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/>
            </svg>
            LinkedIn — hafsa-ibrahim-ai-mi
          </a>

          <a href='https://github.com/HafsaIbrahim5' target='_blank'
             style='display:flex;align-items:center;justify-content:center;gap:10px;
                    background:linear-gradient(135deg,#24292e,#404448);border-radius:12px;
                    padding:13px;text-decoration:none;color:#fff;font-weight:600'>
            <svg width='20' height='20' viewBox='0 0 24 24' fill='white'>
              <path d='M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z'/>
            </svg>
            GitHub — HafsaIbrahim5
          </a>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card">
          <h3 style='color:#fff;font-family:Space Grotesk;margin-top:0'>🚀 App Features</h3>
          <ul style='color:rgba(255,255,255,.7);line-height:2.1;padding-left:16px'>
            <li>✍️ Interactive canvas drawing</li>
            <li>📤 Image upload & prediction</li>
            <li>🎲 Random MNIST sample testing</li>
            <li>📈 Training accuracy & loss curves</li>
            <li>📐 Learning rate schedule viewer</li>
            <li>🧩 Full confusion matrix heatmap</li>
            <li>📊 Classification report (P/R/F1)</li>
            <li>🔍 Error analysis + hardest mistakes</li>
            <li>🏗️ Layer-by-layer architecture view</li>
            <li>🔬 Dataset explorer & distribution</li>
          </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="footer">
  <div style='color:rgba(255,255,255,.4);font-size:.83rem'>
    🔢 MNIST CNN Classifier &nbsp;·&nbsp; TensorFlow / Keras / Streamlit
  </div>
  <div style='display:flex;gap:22px;align-items:center'>
    <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank'
       style='color:rgba(255,255,255,.6);text-decoration:none;font-size:.85rem'>🔗 LinkedIn</a>
    <a href='https://github.com/HafsaIbrahim5' target='_blank'
       style='color:rgba(255,255,255,.6);text-decoration:none;font-size:.85rem'>🐙 GitHub</a>
    <span style='color:rgba(255,255,255,.25);font-size:.83rem'>© 2026 Hafsa Ibrahim</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

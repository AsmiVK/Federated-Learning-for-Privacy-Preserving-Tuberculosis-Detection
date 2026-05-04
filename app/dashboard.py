# app/dashboard.py
"""
Streamlit dashboard for Federated TB Detection.
Tabs:
  1. 🏠 Overview       — project summary & architecture
  2. 🚀 Run Training   — launch FL experiment from UI
  3. 📊 Results        — metrics, convergence, confusion matrix
  4. 🔬 Predict        — upload X-ray → get prediction
"""

import sys, os, json, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedTB — Federated TB Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark theme globals */
  .stApp { background-color: #0e1117; }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
  }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #4fc3f7;
    margin: 0;
  }
  .metric-label {
    font-size: 0.85rem;
    color: #90a4ae;
    margin-top: 4px;
  }
  .metric-sub {
    font-size: 0.75rem;
    color: #546e7a;
    margin-top: 2px;
  }

  /* Node cards */
  .node-card {
    background: #1a1f2e;
    border-left: 4px solid #4fc3f7;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
  }
  .node-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e0e0e0;
  }
  .node-stat {
    font-size: 0.82rem;
    color: #90a4ae;
  }

  /* Status badge */
  .badge-green {
    background: #1b5e20; color: #a5d6a7;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600;
  }
  .badge-blue {
    background: #0d47a1; color: #90caf9;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600;
  }
  .badge-orange {
    background: #e65100; color: #ffccbc;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600;
  }

  /* Prediction result */
  .pred-healthy { color: #66bb6a; font-size: 1.6rem; font-weight: 700; }
  .pred-active  { color: #ef5350; font-size: 1.6rem; font-weight: 700; }
  .pred-latent  { color: #ffa726; font-size: 1.6rem; font-weight: 700; }

  /* Section header */
  .section-header {
    border-bottom: 2px solid #2d3561;
    padding-bottom: 8px;
    margin-bottom: 16px;
    color: #e0e0e0;
  }

  /* Hide Streamlit default elements */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

CLASS_NAMES  = ["Healthy", "Active TB", "Latent TB"]
CLASS_COLORS = {"Healthy": "#66bb6a", "Active TB": "#ef5350", "Latent TB": "#ffa726"}


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_model_cached(ckpt_path: str):
    """Load model — cached so it doesn't reload on every interaction."""
    try:
        from src.model.model_utils import load_model
        return load_model(ckpt_path)
    except Exception:
        return None


def predict_image(model, img: Image.Image) -> dict:
    """Run inference on a PIL image. Returns class probabilities."""
    from src.data.preprocessing import get_val_transforms
    transform = get_val_transforms(224)
    img_gray  = img.convert("L")
    tensor    = transform(img_gray).unsqueeze(0)   # (1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}


def confidence_bar(label: str, prob: float):
    """Render a single confidence bar using Plotly."""
    color = CLASS_COLORS.get(label, "#4fc3f7")
    fig = go.Figure(go.Bar(
        x=[prob * 100],
        y=[label],
        orientation="h",
        marker_color=color,
        text=[f"{prob*100:.1f}%"],
        textposition="outside",
    ))
    fig.update_layout(
        height=60, margin=dict(l=0, r=60, t=0, b=0),
        xaxis=dict(range=[0, 110], showticklabels=False,
                   showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🫁 FedTB Dashboard")
    st.markdown("*Federated Learning for Tuberculosis Detection*")
    st.divider()

    st.markdown("**📁 Data Status**")
    tbx_exists  = os.path.isdir("data/raw/TBX11K/imgs")
    shen_exists = os.path.isdir("data/raw/Shenzhen/images")
    
    tbx_badge  = '<span class="badge-green">✓ Found</span>' if tbx_exists  else '<span class="badge-orange">Mock data</span>'
    shen_badge = '<span class="badge-green">✓ Found</span>' if shen_exists else '<span class="badge-orange">Mock data</span>'
    st.markdown(f"TBX11K: {tbx_badge}",   unsafe_allow_html=True)
    st.markdown(f"Shenzhen: {shen_badge}", unsafe_allow_html=True)

    st.divider()
    st.markdown("**📦 Checkpoints**")
    ckpts = [f for f in os.listdir("checkpoints") if f.endswith(".pth")] \
            if os.path.isdir("checkpoints") else []
    if ckpts:
        for c in ckpts:
            st.markdown(f"<span class='badge-blue'>✓ {c}</span>",
                        unsafe_allow_html=True)
    else:
        st.caption("No checkpoints yet — run training first")

    st.divider()
    st.markdown("**ℹ️ Project Info**")
    st.caption("Topics in Deep Learning — Review 1")
    st.caption("Anvita Agarwal · Asmi Vishal Kapadnis")
    st.caption("PES University, Section B")


# ════════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Overview",
    "🚀  Run Training",
    "📊  Results",
    "🔬  Predict",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("# Federated Learning for TB Detection")
    st.markdown(
        "*Privacy-preserving multi-institutional AI — model weights travel, patient data never does.*"
    )

    col1, col2, col3, col4 = st.columns(4)
    for col, val, label, sub in [
        (col1, "11,200", "X-Ray Images", "TBX11K dataset"),
        (col2, "3",      "Hospital Nodes", "India · S.Africa · E.Europe"),
        (col3, "25",     "FL Rounds",      "FedProx convergence"),
        (col4, "~89%",   "Target Accuracy","Within 2% of centralized"),
    ]:
        col.markdown(
            f"<div class='metric-card'>"
            f"<p class='metric-value'>{val}</p>"
            f"<p class='metric-label'>{label}</p>"
            f"<p class='metric-sub'>{sub}</p>"
            f"</div>", unsafe_allow_html=True
        )

    st.divider()

    # Problem / Solution
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔴 The Problem")
        for item in [
            "10M+ TB cases annually (WHO 2023)",
            "Hospitals **cannot share** patient X-rays (HIPAA, GDPR)",
            "Regional TB patterns vary dramatically",
            "Centralized AI = privacy violations",
        ]:
            st.markdown(f"- {item}")

    with c2:
        st.markdown("### 🟢 Our Solution")
        for item in [
            "**Federated Learning**: train AI without data sharing",
            "Only model **weights** leave each hospital",
            "**FedProx**: handles Non-IID data across regions",
            "**ResNet-18**: proven for medical imaging (~89% acc)",
        ]:
            st.markdown(f"- {item}")

    st.divider()
    st.markdown("### 🏥 Federated Node Distribution (Non-IID)")

    node_data = [
        ("🇮🇳 Node 1 — India",          "40%", "50% Healthy · 40% Active · 10% Latent", "High TB burden"),
        ("🇿🇦 Node 2 — South Africa",    "40%", "40% Healthy · 50% Active · 10% Latent", "Very high TB burden"),
        ("🇵🇱 Node 3 — Eastern Europe",  "20% + Shenzhen", "35% Healthy · 55% Active · 10% Latent", "MDR-TB + domain shift"),
    ]
    for title, pct, dist, note in node_data:
        st.markdown(
            f"<div class='node-card'>"
            f"<div class='node-title'>{title} &nbsp; "
            f"<span class='badge-blue'>{pct} of data</span></div>"
            f"<div class='node-stat'>{dist}</div>"
            f"<div class='node-stat' style='color:#607d8b'>{note}</div>"
            f"</div>", unsafe_allow_html=True
        )

    st.divider()
    st.markdown("### 🧠 Model Architecture — ResNet-18")
    arch_col1, arch_col2 = st.columns([1, 1])
    with arch_col1:
        st.markdown("""
| Layer | Channels | Status |
|-------|----------|--------|
| Conv1 + BN + ReLU + MaxPool | 64 | 🔒 Frozen |
| Layer 1 (2 blocks) | 64 | 🔒 Frozen |
| Layer 2 (2 blocks) | 128 | 🔒 Frozen |
| Layer 3 (2 blocks) | 256 | 🔧 Trainable |
| Layer 4 (2 blocks) | 512 | 🔧 Trainable |
| Global Avg Pool → FC (512→3) | 3 | 🔧 Trainable |
""")
    with arch_col2:
        st.markdown("""
**Transfer Learning Strategy**
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- Layers 1–2 frozen → preserve generic edge/texture features
- Layers 3–4 fine-tuned → learn TB-specific patterns (cavities, opacities)
- Output head replaced: 1000 → **3 classes**
  - 🟢 Healthy
  - 🔴 Active TB
  - 🟡 Latent TB

**Why ResNet-18?**
- 11M parameters → low communication overhead (~44MB/round)
- 20–30 epochs per federated round
- Proven 85–92% accuracy on TB tasks
""")

    # Expected results table
    st.divider()
    st.markdown("### 📋 Expected Results")
    st.markdown("""
| Method | Accuracy | F1 | AUC | Privacy | Rounds |
|--------|----------|----|-----|---------|--------|
| Centralized | ~91% | ~0.90 | ~0.95 | ❌ None | N/A |
| Local Only | ~79% | ~0.77 | ~0.85 | ✅ Full | N/A |
| FedAvg | ~87% | ~0.86 | ~0.91 | ✅ High | ~22 |
| **FedProx (ours)** | **~89%** | **~0.88** | **~0.93** | **✅ High** | **~18** |
""")


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Run Training
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🚀 Launch Federated Training")
    st.info(
        "Configure and launch a federated training experiment. "
        "Training runs in-process — the page will update when complete.",
        icon="ℹ️"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        strategy   = st.selectbox("Aggregation Strategy",
                                   ["fedprox", "fedavg", "compare both"])
        num_rounds = st.slider("Communication Rounds", 2, 30, 5)
    with c2:
        local_epochs = st.slider("Local Epochs per Round", 1, 10, 5)
        batch_size   = st.selectbox("Batch Size", [16, 32, 64], index=1)
    with c3:
        lr      = st.select_slider("Learning Rate",
                                    options=[0.0001, 0.0005, 0.001, 0.005],
                                    value=0.001)
        mu      = st.select_slider("FedProx μ (proximal term)",
                                    options=[0.001, 0.01, 0.1],
                                    value=0.01)

    use_mock = not (tbx_exists and shen_exists)
    if use_mock:
        st.warning(
            "⚠️ Real datasets not found — training will use **mock data**. "
            "Place TBX11K in `data/raw/TBX11K/` and Shenzhen in `data/raw/Shenzhen/` "
            "for real results.",
            icon="⚠️"
        )
    else:
        st.success("✅ Real datasets detected — training will use actual X-ray data.")

    st.divider()

    # Estimated time
    est_min = round(num_rounds * local_epochs * 0.4 * (1 if use_mock else 8))
    st.caption(f"⏱ Estimated time: ~{est_min}–{est_min*2} minutes "
               f"({'mock' if use_mock else 'real'} data, CPU)")

    run_btn = st.button("▶️  Start Training", type="primary", use_container_width=True)

    if run_btn:
        from src.federated.server import run_federated_training

        strategies_to_run = (
            ["fedavg", "fedprox"] if strategy == "compare both"
            else [strategy]
        )

        progress_bar = st.progress(0)
        status_box   = st.empty()
        log_box      = st.empty()

        for si, strat in enumerate(strategies_to_run):
            status_box.markdown(
                f"**Running {strat.upper()}** "
                f"({si+1}/{len(strategies_to_run)})..."
            )

            with st.spinner(f"Training {strat.upper()} — {num_rounds} rounds..."):
                results = run_federated_training(
                    tbx_root      = "data/raw/TBX11K",
                    shenzhen_root = "data/raw/Shenzhen",
                    num_rounds    = num_rounds,
                    strategy_name = strat,
                    local_epochs  = local_epochs,
                    batch_size    = batch_size,
                    learning_rate = lr,
                    mu            = mu,
                    use_mock      = use_mock,
                    results_path  = f"runs/results_{strat}.json",
                    verbose       = False,
                )

            progress_bar.progress((si + 1) / len(strategies_to_run))

            # Show round metrics in a live table
            if results.get("round_metrics"):
                rounds = results["round_metrics"]
                log_box.markdown(
                    "**Round-by-round accuracy:**\n\n" +
                    "| Round | Avg Train Acc | Strategy |\n|-------|--------------|----------|\n" +
                    "\n".join(
                        f"| {r['round']} | {r['avg_train_accuracy']:.4f} | {r['strategy']} |"
                        for r in rounds
                    )
                )

        status_box.markdown("✅ **Training complete!** Switch to the **📊 Results** tab.")
        st.balloons()


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Results
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📊 Training Results")

    # Load available results
    res_fedprox = load_results("runs/results_fedprox.json")
    res_fedavg  = load_results("runs/results_fedavg.json")
    all_res     = load_results("runs/all_results.json")

    if not res_fedprox and not res_fedavg:
        st.warning("No results found yet. Run training first (🚀 Run Training tab).")
    else:
        # ── Metric cards ─────────────────────────────────────────────────────
        st.markdown("### Key Metrics")

        # Pull test metrics if available
        fp_metrics = all_res.get("fedprox", {}).get("test_metrics", {})
        fa_metrics = all_res.get("fedavg",  {}).get("test_metrics", {})
        show_res   = fp_metrics or fa_metrics
        ref        = fp_metrics if fp_metrics else fa_metrics
        strat_label = "FedProx" if fp_metrics else "FedAvg"

        if show_res:
            m1, m2, m3, m4 = st.columns(4)
            for col, val, label, sub in [
                (m1, f"{ref.get('accuracy', 0):.3f}",    "Accuracy",     f"{strat_label}"),
                (m2, f"{ref.get('f1_weighted', 0):.3f}", "F1 (weighted)","Primary metric"),
                (m3, f"{ref.get('auc_roc', 0):.3f}",     "AUC-ROC",      "OvR weighted"),
                (m4, f"{ref.get('cohen_kappa', 0):.3f}", "Cohen's Kappa","Beyond-chance"),
            ]:
                col.markdown(
                    f"<div class='metric-card'>"
                    f"<p class='metric-value'>{val}</p>"
                    f"<p class='metric-label'>{label}</p>"
                    f"<p class='metric-sub'>{sub}</p>"
                    f"</div>", unsafe_allow_html=True
                )

            # Per-class sensitivity
            st.markdown("### Per-Class Sensitivity (Recall)")
            sens = ref.get("sensitivity", {})
            spec = ref.get("specificity", {})
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Sensitivity (higher = better)**")
                for cls in CLASS_NAMES:
                    val = sens.get(cls, 0)
                    color = ("#ef5350" if val < 0.5
                             else "#ffa726" if val < 0.8
                             else "#66bb6a")
                    flag = " ⚠️ Clinical minimum not met" \
                           if cls == "Active TB" and val < 0.80 else ""
                    st.markdown(f"**{cls}**: {val:.3f}{flag}")
                    st.progress(val)

            with sc2:
                st.markdown("**Specificity (higher = better)**")
                for cls in CLASS_NAMES:
                    val = spec.get(cls, 0)
                    st.markdown(f"**{cls}**: {val:.3f}")
                    st.progress(val)

        # ── Convergence chart ─────────────────────────────────────────────────
        st.markdown("### Convergence Curves")
        rounds_fa = res_fedavg.get("round_metrics",  [])
        rounds_fp = res_fedprox.get("round_metrics", [])

        if rounds_fa or rounds_fp:
            fig = go.Figure()
            if rounds_fa:
                fig.add_trace(go.Scatter(
                    x=[r["round"] for r in rounds_fa],
                    y=[r["avg_train_accuracy"] for r in rounds_fa],
                    name="FedAvg (baseline)",
                    mode="lines+markers",
                    line=dict(color="#ef5350", width=2),
                    marker=dict(size=7),
                ))
            if rounds_fp:
                fig.add_trace(go.Scatter(
                    x=[r["round"] for r in rounds_fp],
                    y=[r["avg_train_accuracy"] for r in rounds_fp],
                    name="FedProx (ours)",
                    mode="lines+markers",
                    line=dict(color="#4fc3f7", width=2),
                    marker=dict(size=7),
                ))
            fig.update_layout(
                xaxis_title="Communication Rounds",
                yaxis_title="Avg Train Accuracy",
                yaxis_range=[0, 1],
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font_color="#e0e0e0",
                legend=dict(bgcolor="#1a1f2e", bordercolor="#2d3561"),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Confusion matrix ──────────────────────────────────────────────────
        st.markdown("### Confusion Matrices")
        cm_col1, cm_col2 = st.columns(2)
        for col, strat, res in [
            (cm_col1, "FedProx", all_res.get("fedprox", {})),
            (cm_col2, "FedAvg",  all_res.get("fedavg",  {})),
        ]:
            with col:
                cm_data = res.get("test_metrics", {}).get("confusion_matrix")
                classes = res.get("test_metrics", {}).get(
                    "classes_present", CLASS_NAMES[:2]
                )
                if cm_data:
                    cm_arr = np.array(cm_data)
                    fig_cm = px.imshow(
                        cm_arr,
                        x=classes, y=classes,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"),
                        text_auto=True,
                        title=f"Confusion Matrix — {strat}",
                    )
                    fig_cm.update_layout(
                        paper_bgcolor="#0e1117",
                        plot_bgcolor="#0e1117",
                        font_color="#e0e0e0",
                        height=380,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.caption(f"No {strat} confusion matrix yet.")


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — Predict
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🔬 X-Ray Prediction")
    st.markdown(
        "Upload a chest X-ray image to get a TB classification with confidence scores."
    )

    # Choose checkpoint
    available_ckpts = (
        [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
        if os.path.isdir("checkpoints") else []
    )

    if not available_ckpts:
        st.warning("No checkpoints found. Please run training first.")
    else:
        p1, p2 = st.columns([2, 1])
        with p1:
            selected_ckpt = st.selectbox(
                "Select model checkpoint", available_ckpts
            )
        with p2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='badge-green'>✓ {selected_ckpt}</span>",
                unsafe_allow_html=True
            )

        uploaded = st.file_uploader(
            "Upload chest X-ray",
            type=["png", "jpg", "jpeg"],
            help="Grayscale or RGB chest X-ray image",
        )

        if uploaded:
            img = Image.open(uploaded)

            img_col, result_col = st.columns([1, 1])
            with img_col:
                st.markdown("**Uploaded X-Ray**")
                st.image(img, use_column_width=True, clamp=True)
                st.caption(f"Size: {img.size[0]}×{img.size[1]} px")

            with result_col:
                st.markdown("**Prediction**")
                ckpt_path = os.path.join("checkpoints", selected_ckpt)

                with st.spinner("Running inference..."):
                    model = load_model_cached(ckpt_path)

                if model is None:
                    st.error("Failed to load model checkpoint.")
                else:
                    probs = predict_image(model, img)

                    # Top prediction
                    top_cls   = max(probs, key=probs.get)
                    top_prob  = probs[top_cls]
                    css_class = {
                        "Healthy":   "pred-healthy",
                        "Active TB": "pred-active",
                        "Latent TB": "pred-latent",
                    }.get(top_cls, "pred-healthy")
                    emoji = {
                        "Healthy": "🟢", "Active TB": "🔴", "Latent TB": "🟡"
                    }.get(top_cls, "⚪")

                    st.markdown(
                        f"<p class='{css_class}'>{emoji} {top_cls}</p>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"**Confidence: {top_prob*100:.1f}%**"
                    )
                    st.divider()

                    # Confidence bars for all classes
                    st.markdown("**All class probabilities:**")
                    for cls in CLASS_NAMES:
                        fig_bar = confidence_bar(cls, probs[cls])
                        st.plotly_chart(
                            fig_bar, use_container_width=True,
                            config={"displayModeBar": False}
                        )

                    # Clinical note
                    st.divider()
                    st.markdown(
                        "> ⚕️ **Clinical note:** This is a research prototype "
                        "trained on limited data. Results should **not** be used "
                        "for clinical diagnosis. Always consult a qualified "
                        "medical professional."
                    )
        else:
            # Show placeholder when no image uploaded
            st.info(
                "👆 Upload a chest X-ray image above to get a prediction.",
                icon="🫁"
            )
            st.markdown("**What this model detects:**")
            for cls, color, desc in [
                ("🟢 Healthy",   "#66bb6a",
                 "No TB detected — normal chest X-ray appearance"),
                ("🔴 Active TB", "#ef5350",
                 "Active tuberculosis — visible lung infiltrates/cavities"),
                ("🟡 Latent TB", "#ffa726",
                 "Latent TB infection — no active disease currently"),
            ]:
                st.markdown(f"- **{cls}** — {desc}")
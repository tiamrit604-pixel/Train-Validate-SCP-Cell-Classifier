"""
app.py  —  SCP Cell Audio Classifier
--------------------------------------
Classifies 27 cells of a Sandwich Composite Plate (SCP) as
  Good (bonded / healthy)  or  Bad (debonded / unhealthy)
using acoustic percussion signals.

Modes:
  1. Train & Validate  — upload train data + optional unseen robustness set
  2. Test on Model     — select any saved model, upload unlabelled files, classify
  3. Model Comparison  — charts across all trained classifiers
  4. About Pipeline    — methodology documentation

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os

# must be defined before any page code uses it
def plt_close(fig):
    plt.close(fig)

from audio_utils import process_uploaded_files
from ml_utils import (
    CLASSIFIER_CONFIGS, train_all, save_model, load_results,
    load_model_by_name, list_saved_models, model_exists, predict_files
)
from plot_utils import (
    plot_waveform, plot_psd, plot_mfcc,
    plot_confusion_matrix, plot_model_comparison,
    plot_metrics_comparison, plot_feature_distribution,
    plot_prediction_confidence
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SCP Cell Classifier",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0e0f14; color: #e8eaf6; }
[data-testid="stSidebar"] { background-color: #161820; border-right: 1px solid rgba(255,255,255,0.07); }
hr { border-color: rgba(255,255,255,0.07); }

.stButton > button {
    background-color: #5865f8; color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 8px 20px; transition: all 0.15s;
}
.stButton > button:hover { background-color: #7c87ff; transform: translateY(-1px); }
.stButton > button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

[data-testid="stFileUploader"] {
    background: #1e2030; border: 1.5px dashed rgba(255,255,255,0.15);
    border-radius: 10px; padding: 10px;
}
[data-testid="stDataFrame"] { background: #1e2030; }
.stTabs [data-baseweb="tab-list"] { background: #161820; border-radius: 8px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b90a8; border-radius: 6px; font-size: 13px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #252840 !important; color: #e8eaf6 !important; }
[data-testid="stExpander"] { background: #1e2030; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; }
.stProgress > div > div { background: linear-gradient(90deg, #5865f8, #7c87ff); }

/* Custom components */
.section-header {
    font-size: 11px; font-family: 'Space Mono', monospace; color: #5865f8;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 24px 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(88,101,248,0.25);
}
.info-box {
    background: rgba(88,101,248,0.08); border: 1px solid rgba(88,101,248,0.25);
    border-radius: 8px; padding: 14px 16px; font-size: 13px;
    color: #a5adff; margin-bottom: 16px; line-height: 1.6;
}
.warn-box {
    background: rgba(240,168,74,0.08); border: 1px solid rgba(240,168,74,0.25);
    border-radius: 8px; padding: 14px 16px; font-size: 13px;
    color: #f0a84a; margin-bottom: 16px; line-height: 1.6;
}
.result-good {
    background: rgba(34,211,160,0.08); border: 2px solid rgba(34,211,160,0.3);
    border-radius: 14px; padding: 28px 24px; text-align: center; margin: 12px 0;
}
.result-bad {
    background: rgba(240,92,92,0.08); border: 2px solid rgba(240,92,92,0.3);
    border-radius: 14px; padding: 28px 24px; text-align: center; margin: 12px 0;
}
.result-title { font-size: 26px; font-weight: 700; font-family: 'Space Mono', monospace; margin: 8px 0 4px; }
.stat-card {
    background: #1e2030; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 16px 18px; text-align: center;
}
.stat-label { font-size: 10px; font-family: 'Space Mono', monospace; color: #8b90a8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.stat-value { font-size: 26px; font-weight: 700; font-family: 'Space Mono', monospace; }
.model-select-card {
    background: #1e2030; border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 18px 20px; margin-bottom: 10px; cursor: pointer;
    transition: border 0.15s;
}
.model-select-card:hover { border-color: #5865f8; }
.model-select-card.selected { border-color: #5865f8; background: rgba(88,101,248,0.08); }
.unseen-badge {
    display: inline-block; background: rgba(240,168,74,0.15); color: #f0a84a;
    border-radius: 6px; padding: 2px 8px; font-size: 11px;
    font-family: 'Space Mono', monospace; font-weight: 700; margin-left: 8px;
}
.best-badge {
    display: inline-block; background: rgba(88,101,248,0.15); color: #7c87ff;
    border-radius: 6px; padding: 2px 8px; font-size: 11px;
    font-family: 'Space Mono', monospace; font-weight: 700; margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("training_results", None),
    ("trained_X", None),
    ("trained_y", None),
    ("selected_model_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔊 SCP Classifier")
    st.markdown(
        "<div style='font-size:11px; color:#8b90a8; margin-bottom:6px;'>"
        "Sandwich Composite Plate<br>Acoustic Cell QC — 27 cells</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🧪 Train & Validate", "🔬 Test on Model", "📊 Model Comparison", "ℹ️ About Pipeline"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Saved models status
    saved = list_saved_models()
    if saved:
        st.markdown(
            f"<div style='font-size:11px; color:#22d3a0; font-weight:700; margin-bottom:4px;'>✓ Saved models ({len(saved)})</div>",
            unsafe_allow_html=True
        )
        for n in saved:
            st.markdown(
                f"<div style='font-size:11px; color:#8b90a8; padding: 1px 0;'>· {n}</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div style='background:rgba(240,168,74,0.08); border:1px solid rgba(240,168,74,0.2);"
            "border-radius:8px; padding:10px; font-size:11px;'>"
            "<span style='color:#f0a84a; font-weight:700;'>⚠ No models saved</span><br>"
            "<span style='color:#8b90a8;'>Train models first</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#8b90a8; line-height:1.7;'>"
        "<b style='color:#e8eaf6;'>Labeling convention</b><br>"
        "<code style='font-size:10px; color:#22d3a0;'>*g.wav / *g.m4a</code> = Good (Bonded)<br>"
        "<code style='font-size:10px; color:#f05c5c;'>anything else</code> = Bad (Debonded)"
        "</div>",
        unsafe_allow_html=True
    )

    # ---- Reset section ----
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#8b90a8; font-weight:700; margin-bottom:8px;'>"
        "⚙ Reset</div>",
        unsafe_allow_html=True
    )

    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if not st.session_state.confirm_reset:
        if st.button("🗑 Reset Everything", use_container_width=True, key="reset_trigger"):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.markdown(
            "<div style='background:rgba(240,92,92,0.1); border:1px solid rgba(240,92,92,0.3);"
            "border-radius:8px; padding:10px; font-size:11px; color:#f05c5c; margin-bottom:8px;'>"
            "<b>Deletes all saved models and clears session data.</b><br>Cannot be undone."
            "</div>",
            unsafe_allow_html=True
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✓ Confirm", use_container_width=True, key="reset_confirm"):
                import glob as _glob
                _model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                for _pkl in _glob.glob(os.path.join(_model_dir, "*.pkl")):
                    try:
                        os.remove(_pkl)
                    except Exception:
                        pass
                _keys = [
                    "training_results", "trained_X", "trained_y",
                    "selected_model_name", "confirm_reset",
                    "last_pred_df", "last_pred_meta"
                ]
                for _k in _keys:
                    if _k in st.session_state:
                        del st.session_state[_k]
                st.rerun()
        with col_no:
            if st.button("✗ Cancel", use_container_width=True, key="reset_cancel"):
                st.session_state.confirm_reset = False
                st.rerun()


# ===========================================================================
# PAGE 1 — TRAIN & VALIDATE
# ===========================================================================
if page == "🧪 Train & Validate":

    st.title("Train & Validate — SCP Cell Classifier")
    st.markdown(
        "<div class='info-box'>"
        "Upload <b>labelled training audio</b> and (optionally) a separate <b>unseen robustness set</b>. "
        "The robustness set is never used for training or cross-validation — it is held out to test "
        "how well each model generalises to completely new data. All four classifiers are trained and "
        "saved individually so you can choose any of them in Mode 2."
        "</div>",
        unsafe_allow_html=True
    )

    # ---- Step 1: Training data ----
    st.markdown("<div class='section-header'>Step 1 — Upload Training / Validation Files</div>", unsafe_allow_html=True)
    st.markdown(
        "These files are split into **training** and **validation** sets and used for Grid Search CV. "
        "Label via filename: files ending `g.wav` / `g.m4a` = **Good**, everything else = **Bad**.",
        unsafe_allow_html=True
    )

    train_files = st.file_uploader(
        "Training audio files (WAV, M4A, MP3, OGG, FLAC)",
        type=["wav", "m4a", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="train_uploader",
        help="Files ending with 'g' before the extension = Good cell"
    )

    if train_files:
        good_tr = [f for f in train_files if f.name.lower().rstrip().endswith(
            ("g.wav","g.m4a","g.mp3","g.ogg","g.flac"))]
        bad_tr  = [f for f in train_files if f not in good_tr]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(train_files))
        c2.metric("Good (Bonded)", len(good_tr))
        c3.metric("Bad (Debonded)", len(bad_tr))
        if not good_tr or not bad_tr:
            st.warning("⚠️ Need at least one Good and one Bad file. Check naming convention.")
        with st.expander("View training file list"):
            st.dataframe(pd.DataFrame([{
                "Filename": f.name,
                "Label": "✅ Good" if f in good_tr else "❌ Bad",
                "Size": f"{f.size/1024:.1f} KB"
            } for f in train_files]), use_container_width=True, hide_index=True)

    # ---- Step 2: Unseen robustness set ----
    st.markdown("<div class='section-header'>Step 2 — Upload Unseen Robustness Test Files (Optional)</div>", unsafe_allow_html=True)
    st.markdown(
        "These files are **completely withheld** from training and CV. After training, each model "
        "is evaluated on this set to measure robustness / generalisation. Use the same labeling "
        "convention (`*g.wav` = Good). This is equivalent to the Set6 / unseen_test_set in the notebook.",
        unsafe_allow_html=True
    )

    unseen_files = st.file_uploader(
        "Unseen robustness test files (optional — same naming convention)",
        type=["wav", "m4a", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="unseen_uploader",
        help="Optional. Not used for training. Used only to measure generalisation."
    )

    if unseen_files:
        good_un = [f for f in unseen_files if f.name.lower().rstrip().endswith(
            ("g.wav","g.m4a","g.mp3","g.ogg","g.flac"))]
        bad_un  = [f for f in unseen_files if f not in good_un]
        u1, u2, u3 = st.columns(3)
        u1.metric("Unseen Total", len(unseen_files))
        u2.metric("Good", len(good_un))
        u3.metric("Bad", len(bad_un))
        with st.expander("View unseen file list"):
            st.dataframe(pd.DataFrame([{
                "Filename": f.name,
                "Label": "✅ Good" if f in good_un else "❌ Bad",
                "Size": f"{f.size/1024:.1f} KB"
            } for f in unseen_files]), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            "<div class='warn-box'>No unseen set uploaded — robustness testing will be skipped. "
            "Unseen Acc column will show N/A.</div>",
            unsafe_allow_html=True
        )

    # ---- Step 3: Configuration ----
    st.markdown("<div class='section-header'>Step 3 — Configure Grid Search</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_clfs = st.multiselect(
            "Classifiers to train",
            options=list(CLASSIFIER_CONFIGS.keys()),
            default=list(CLASSIFIER_CONFIGS.keys()),
        )
    with col2:
        cv_folds = st.select_slider("CV Folds", options=[3, 5, 10], value=5)
    with col3:
        test_size = st.select_slider(
            "Validation split",
            options=[0.2, 0.3, 0.4], value=0.3,
            format_func=lambda x: f"{int((1-x)*100)}/{int(x*100)} train/val"
        )

    # ---- Step 4: Run ----
    st.markdown("<div class='section-header'>Step 4 — Run Training</div>", unsafe_allow_html=True)

    can_train = bool(train_files) and len(selected_clfs) > 0
    train_btn = st.button("🚀 Start Training", disabled=not can_train)

    if train_btn and can_train:
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def progress_cb(frac, msg):
            progress_bar.progress(min(float(frac), 1.0))
            status_text.markdown(
                f"<div style='font-size:13px; color:#8b90a8;'>{msg}</div>",
                unsafe_allow_html=True
            )

        # --- Extract training features ---
        status_text.markdown(
            "<div style='font-size:13px; color:#8b90a8;'>Extracting features from training files...</div>",
            unsafe_allow_html=True
        )
        for f in train_files:
            f.seek(0)
        with st.spinner("Processing training audio..."):
            X, y, meta = process_uploaded_files(train_files, has_labels=True)

        if X.shape[0] < 4:
            st.error("❌ Not enough valid segments from training files (need ≥ 4). Check file formats.")
            st.stop()

        errs = [m for m in meta if m.get("error")]
        if errs:
            st.warning(f"⚠️ {len(errs)} training file(s) could not be decoded and were skipped.")

        st.success(f"✅ Training data: **{X.shape[0]} segments** from {len(train_files)} files · {X.shape[1]} features")

        # --- Extract unseen features ---
        X_unseen, y_unseen = np.array([]), np.array([])
        if unseen_files:
            status_text.markdown(
                "<div style='font-size:13px; color:#8b90a8;'>Extracting features from unseen robustness files...</div>",
                unsafe_allow_html=True
            )
            for f in unseen_files:
                f.seek(0)
            with st.spinner("Processing unseen audio..."):
                X_unseen, y_unseen, meta_un = process_uploaded_files(unseen_files, has_labels=True)
            errs_un = [m for m in meta_un if m.get("error")]
            if errs_un:
                st.warning(f"⚠️ {len(errs_un)} unseen file(s) skipped.")
            st.success(f"✅ Unseen robustness data: **{X_unseen.shape[0]} segments** from {len(unseen_files)} files")

        # --- Train ---
        with st.spinner("Running Grid Search CV across all classifiers..."):
            results = train_all(
                X, y, X_unseen, y_unseen,
                selected_classifiers=selected_clfs,
                cv_folds=cv_folds,
                test_size=test_size,
                progress_callback=progress_cb
            )

        progress_bar.progress(1.0)
        status_text.markdown(
            "<div style='font-size:13px; color:#22d3a0;'>✓ All models trained and saved!</div>",
            unsafe_allow_html=True
        )

        st.session_state.training_results = results
        st.session_state.trained_X = X
        st.session_state.trained_y = y

        best = results["best_model_name"]
        bacc = results["best_val_acc"]
        st.success(
            f"✅ **{len(selected_clfs)} models saved** individually to `models/` folder. "
            f"Best by val accuracy: **{best}** ({bacc*100:.1f}%)"
        )

    # ---- Step 5: Results ----
    results = st.session_state.training_results
    if results:
        st.markdown("<div class='section-header'>Step 5 — Results</div>", unsafe_allow_html=True)

        best_name  = results["best_model_name"]
        best_clf   = results["classifiers"][best_name]
        vm         = best_clf["val_metrics"]
        um         = best_clf["unseen_metrics"]
        has_unseen = results.get("has_unseen", False)

        # Top metrics row
        cols = st.columns(6 if has_unseen else 4)
        cols[0].metric("Best Val Acc",  f"{vm['accuracy']*100:.1f}%")
        cols[1].metric("F1 (val)",      f"{vm['f1']:.3f}")
        cols[2].metric("Precision",     f"{vm['precision']:.3f}")
        cols[3].metric("Recall",        f"{vm['recall']:.3f}")
        if has_unseen and um.get("accuracy") is not None:
            cols[4].metric("Unseen Acc",    f"{um['accuracy']*100:.1f}%")
            cols[5].metric("Unseen F1",     f"{um['f1']:.3f}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Summary Table",
            "🔀 Confusion Matrices",
            "🧪 Robustness Testing",
            "📈 Feature Distribution",
            "🔍 CV Details"
        ])

        # --- Tab 1: Summary table ---
        with tab1:
            df = results["summary_df"].copy()
            numeric_cols = [c for c in df.columns if "(%" in c or c in ("Precision","Recall","F1","Unseen F1")]
            numeric_cols = [c for c in numeric_cols if df[c].dtype != object]

            st.markdown(
                f"<div style='font-size:12px; color:#8b90a8; margin-bottom:10px;'>"
                f"All {len(results['classifiers'])} models saved individually. "
                f"Best by validation accuracy: <b style='color:#7c87ff;'>{best_name}</b></div>",
                unsafe_allow_html=True
            )

            try:
                styled = df.style.highlight_max(
                    subset=[c for c in ["Val Acc (%)","F1","Unseen Acc (%)","Unseen F1"] if c in df.columns and df[c].dtype != object],
                    color="rgba(88,101,248,0.25)"
                ).highlight_min(
                    subset=[c for c in ["Acc Drop (%)"] if c in df.columns and df[c].dtype != object],
                    color="rgba(34,211,160,0.15)"
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(df, use_container_width=True, hide_index=True)

        # --- Tab 2: Confusion Matrices (train + val) ---
        with tab2:
            for clf_name, clf_res in results["classifiers"].items():
                is_best = clf_name == best_name
                label = f"**{clf_name}**" + (" ★ Best" if is_best else "")
                st.markdown(label)
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_confusion_matrix(clf_res["cm_train"], f"{clf_name} — Training", cmap="Oranges")
                    st.pyplot(fig, use_container_width=True)
                    plt_close(fig)
                with col2:
                    fig = plot_confusion_matrix(clf_res["cm_val"], f"{clf_name} — Validation", cmap="Blues")
                    st.pyplot(fig, use_container_width=True)
                    plt_close(fig)
                st.markdown("---")

        # --- Tab 3: Robustness Testing ---
        with tab3:
            if not has_unseen:
                st.markdown(
                    "<div class='warn-box'>No unseen robustness set was uploaded during training. "
                    "Re-train and upload an unseen set in Step 2 to see robustness results.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='info-box'>"
                    "The unseen robustness set was never seen during training or cross-validation. "
                    "These results show how well each model generalises to completely new acoustic data — "
                    "the key indicator of real-world performance on the SCP inspection task."
                    "</div>",
                    unsafe_allow_html=True
                )

                # Robustness summary table
                rob_rows = []
                for clf_name, clf_res in results["classifiers"].items():
                    um2 = clf_res["unseen_metrics"]
                    if um2.get("accuracy") is not None:
                        rob_rows.append({
                            "Model":       clf_name,
                            "Unseen Acc (%)":  round(um2["accuracy"]*100, 2),
                            "Unseen Precision": round(um2["precision"], 3),
                            "Unseen Recall":    round(um2["recall"],    3),
                            "Unseen F1":        round(um2["f1"],        3),
                            "Val→Unseen Drop (%)": round(
                                (clf_res["val_metrics"]["accuracy"] - um2["accuracy"]) * 100, 2)
                        })

                if rob_rows:
                    rob_df = pd.DataFrame(rob_rows)
                    try:
                        styled_rob = rob_df.style.highlight_max(
                            subset=["Unseen Acc (%)","Unseen F1"], color="rgba(240,168,74,0.25)"
                        ).highlight_min(
                            subset=["Val→Unseen Drop (%)"], color="rgba(34,211,160,0.15)"
                        )
                        st.dataframe(styled_rob, use_container_width=True, hide_index=True)
                    except Exception:
                        st.dataframe(rob_df, use_container_width=True, hide_index=True)

                    # Robustness confusion matrices
                    st.markdown("**Confusion Matrices — Unseen Robustness Set**")
                    clf_names_list = list(results["classifiers"].keys())
                    cols_per_row = 2
                    for i in range(0, len(clf_names_list), cols_per_row):
                        row_cols = st.columns(cols_per_row)
                        for j, clf_name in enumerate(clf_names_list[i:i+cols_per_row]):
                            clf_res = results["classifiers"][clf_name]
                            if clf_res["cm_unseen"] is not None:
                                with row_cols[j]:
                                    fig = plot_confusion_matrix(
                                        clf_res["cm_unseen"],
                                        f"{clf_name} — Unseen Set",
                                        cmap="Greens"
                                    )
                                    st.pyplot(fig, use_container_width=True)
                                    plt_close(fig)

        # --- Tab 4: Feature distribution ---
        with tab4:
            X_tr = st.session_state.trained_X
            y_tr = st.session_state.trained_y
            if X_tr is not None:
                st.plotly_chart(plot_feature_distribution(X_tr, y_tr), use_container_width=True)
                st.caption(
                    "Mean normalised feature values for Good vs Bad cells (first 64 of 264 features). "
                    "Larger separation between lines = more discriminative feature."
                )

        # --- Tab 5: CV details ---
        with tab5:
            clf_sel = st.selectbox("Select classifier for CV details", list(results["classifiers"].keys()))
            cv_df   = results["classifiers"][clf_sel]["cv_results_df"]
            show_cols = [c for c in cv_df.columns if c.startswith("mean_test_") or c == "params"]
            st.dataframe(
                cv_df[show_cols].sort_values("mean_test_accuracy", ascending=False).head(20),
                use_container_width=True, hide_index=True
            )


# ===========================================================================
# PAGE 2 — TEST ON MODEL
# ===========================================================================
elif page == "🔬 Test on Model":

    st.title("Test on Model — SCP Cell Classification")
    st.markdown(
        "<div class='info-box'>"
        "Select any of your saved models, upload unlabelled audio files from the SCP inspection, "
        "and classify each cell as <b style='color:#22d3a0;'>Good (Bonded)</b> or "
        "<b style='color:#f05c5c;'>Bad (Debonded)</b>."
        "</div>",
        unsafe_allow_html=True
    )

    saved_models = list_saved_models()

    if not saved_models:
        st.warning("⚠️ No trained models found. Go to **Train & Validate** to train and save models first.")
        st.stop()

    # ---- Model Selection ----
    st.markdown("<div class='section-header'>Step 1 — Select Model</div>", unsafe_allow_html=True)

    saved_results = load_results()
    best_by_val   = saved_results.get("best_model_name") if saved_results else None

    # Show model cards
    col_count = min(len(saved_models), 4)
    model_cols = st.columns(col_count)
    for i, m_name in enumerate(saved_models):
        with model_cols[i % col_count]:
            is_best = m_name == best_by_val
            val_acc = None
            unseen_acc = None
            if saved_results and "classifiers" in saved_results:
                clf_data = saved_results["classifiers"].get(m_name, {})
                val_m    = clf_data.get("val_metrics", {})
                uns_m    = clf_data.get("unseen_metrics", {})
                val_acc    = val_m.get("accuracy")
                unseen_acc = uns_m.get("accuracy")

            val_str    = f"{val_acc*100:.1f}%" if val_acc is not None else "N/A"
            unseen_str = f"{unseen_acc*100:.1f}%" if unseen_acc is not None else "N/A"
            best_tag   = "<span class='best-badge'>★ BEST VAL</span>" if is_best else ""
            unseen_tag = f"<span class='unseen-badge'>Unseen: {unseen_str}</span>" if unseen_acc is not None else ""

            st.markdown(
                f"<div class='model-select-card'>"
                f"<div style='font-weight:700; font-size:14px; margin-bottom:6px;'>{m_name}{best_tag}</div>"
                f"<div style='font-size:12px; color:#8b90a8; margin-bottom:4px;'>Val Acc: <b style='color:#e8eaf6;'>{val_str}</b></div>"
                f"{unseen_tag}"
                f"</div>",
                unsafe_allow_html=True
            )

    selected_model_name = st.selectbox(
        "Choose model to use for classification",
        options=saved_models,
        index=saved_models.index(best_by_val) if best_by_val in saved_models else 0,
    )
    st.session_state.selected_model_name = selected_model_name

    model = load_model_by_name(selected_model_name)
    if model is None:
        st.error(f"❌ Could not load model file for '{selected_model_name}'. Re-train to regenerate.")
        st.stop()

    # Show selected model info
    if saved_results and "classifiers" in saved_results:
        clf_data   = saved_results["classifiers"].get(selected_model_name, {})
        val_m2     = clf_data.get("val_metrics", {})
        uns_m2     = clf_data.get("unseen_metrics", {})
        best_p     = clf_data.get("best_params", {})
        info_parts = [
            f"Active model: <b style='color:#7c87ff;'>{selected_model_name}</b>",
            f"Val Acc: <b style='color:#22d3a0;'>{val_m2.get('accuracy',0)*100:.1f}%</b>",
            f"Val F1: <b>{val_m2.get('f1',0):.3f}</b>",
        ]
        if uns_m2.get("accuracy") is not None:
            info_parts.append(f"Unseen Acc: <b style='color:#f0a84a;'>{uns_m2['accuracy']*100:.1f}%</b>")
        info_parts.append(f"Params: <code style='font-size:11px;'>{best_p}</code>")
        st.markdown(
            f"<div class='info-box'>{' &nbsp;·&nbsp; '.join(info_parts)}</div>",
            unsafe_allow_html=True
        )

    # ---- Upload test files ----
    st.markdown("<div class='section-header'>Step 2 — Upload Test Audio Files</div>", unsafe_allow_html=True)
    st.markdown(
        "Upload the audio files you want to classify. **No labeling needed** — "
        "the model will predict each cell's condition.",
        unsafe_allow_html=True
    )

    test_files = st.file_uploader(
        "Test audio files",
        type=["wav", "m4a", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="test_uploader"
    )

    if not test_files:
        st.stop()

    st.markdown(
        f"<div style='font-size:13px; color:#8b90a8; margin-bottom:12px;'>"
        f"{len(test_files)} file(s) loaded and ready</div>",
        unsafe_allow_html=True
    )

    classify_btn = st.button("🔬 Classify Files", use_container_width=False)

    if classify_btn:
        for f in test_files:
            f.seek(0)
        with st.spinner("Extracting features and classifying..."):
            X_test, _, meta = process_uploaded_files(test_files, has_labels=False)

        if X_test.shape[0] == 0:
            st.error("❌ No valid audio segments extracted. Check file formats.")
            st.stop()

        pred_df = predict_files(model, X_test, meta)
        if pred_df.empty:
            st.error("❌ Prediction failed.")
            st.stop()

        # Store for display
        st.session_state["last_pred_df"]   = pred_df
        st.session_state["last_pred_meta"] = meta

    # --- Display results ---
    pred_df = st.session_state.get("last_pred_df")
    meta    = st.session_state.get("last_pred_meta")

    if pred_df is not None and not pred_df.empty:

        st.markdown("<div class='section-header'>Classification Results</div>", unsafe_allow_html=True)

        good_count = int((pred_df["Label"] == 1).sum())
        bad_count  = int((pred_df["Label"] == 0).sum())
        total      = len(pred_df)
        avg_conf   = float(pred_df["Confidence"].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cells Tested", total)
        c2.metric("✅ Good (Bonded)",    good_count)
        c3.metric("❌ Bad (Debonded)",   bad_count)
        c4.metric("Avg Confidence",      f"{avg_conf:.1f}%")

        # SCP layout hint
        st.markdown(
            f"<div style='font-size:12px; color:#8b90a8; margin-bottom:12px;'>"
            f"Model used: <b style='color:#7c87ff;'>{selected_model_name}</b> "
            f"· {good_count}/{total} cells classified as Good "
            f"({good_count/total*100:.0f}% bonded)</div>",
            unsafe_allow_html=True
        )

        # Single file big result banner
        if total == 1:
            row   = pred_df.iloc[0]
            css   = "result-good" if row["Label"] == 1 else "result-bad"
            icon  = "✅" if row["Label"] == 1 else "❌"
            color = "#22d3a0" if row["Label"] == 1 else "#f05c5c"
            desc  = "BONDED — GOOD CELL" if row["Label"] == 1 else "DEBONDED — BAD CELL"
            st.markdown(
                f"<div class='{css}'>"
                f"<div style='font-size:40px;'>{icon}</div>"
                f"<div class='result-title' style='color:{color};'>{desc}</div>"
                f"<div style='color:#8b90a8; font-size:13px; margin-top:6px;'>"
                f"Confidence: {row['Confidence']}% · {row['Segments']} segment(s) analysed</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Confidence chart
        st.plotly_chart(plot_prediction_confidence(pred_df), use_container_width=True)

        # Detailed table
        st.markdown("<div class='section-header'>Detailed Per-File Results</div>", unsafe_allow_html=True)
        disp = pred_df[["Filename","Prediction","Confidence","Segments","Good Votes","Bad Votes"]].copy()
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Waveform inspector
        st.markdown("<div class='section-header'>Signal Inspector</div>", unsafe_allow_html=True)
        file_names_with_signal = list(dict.fromkeys(
            m["filename"] for m in meta if m.get("signal") is not None
        ))
        if file_names_with_signal:
            sel_file = st.selectbox("Select file to inspect", file_names_with_signal, key="inspect_sel")
            sel_meta = next((m for m in meta if m["filename"] == sel_file and m.get("signal") is not None), None)
            if sel_meta:
                file_row = pred_df[pred_df["Filename"] == sel_file]
                lbl      = int(file_row["Label"].values[0]) if not file_row.empty else -1
                pred_str = "✅ Good (Bonded)" if lbl == 1 else "❌ Bad (Debonded)"
                st.markdown(
                    f"<div style='font-size:13px; margin-bottom:8px;'>"
                    f"Prediction: <b>{pred_str}</b> · "
                    f"Confidence: <b>{file_row['Confidence'].values[0] if not file_row.empty else 'N/A'}%</b></div>",
                    unsafe_allow_html=True
                )
                tw, tp, tm = st.tabs(["📉 Waveform", "📊 PSD", "🎨 MFCC"])
                with tw:
                    fig = plot_waveform(sel_meta["signal"], sel_meta["sr"], title=sel_file, label=lbl)
                    st.pyplot(fig, use_container_width=True)
                    plt_close(fig)
                with tp:
                    fig = plot_psd(sel_meta["signal"], sel_meta["sr"], title=f"PSD — {sel_file}")
                    st.pyplot(fig, use_container_width=True)
                    plt_close(fig)
                with tm:
                    fig = plot_mfcc(sel_meta["signal"], sel_meta["sr"], title=f"MFCC — {sel_file}")
                    st.pyplot(fig, use_container_width=True)
                    plt_close(fig)

        # Export
        st.markdown("---")
        csv_buf = io.StringIO()
        pred_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv_buf.getvalue(),
            file_name=f"scp_results_{selected_model_name.replace(' ','_')}.csv",
            mime="text/csv"
        )


# ===========================================================================
# PAGE 3 — MODEL COMPARISON
# ===========================================================================
elif page == "📊 Model Comparison":

    st.title("Model Comparison — All Classifiers")

    results = st.session_state.training_results
    if results is None:
        results = load_results()

    if results is None or "summary_df" not in results:
        st.warning("⚠️ No training results found. Train models first.")
        st.stop()

    df         = results["summary_df"]
    best_name  = results.get("best_model_name", "")
    has_unseen = results.get("has_unseen", False)

    st.markdown(
        f"<div class='info-box'>"
        f"Comparing {len(results['classifiers'])} classifiers. "
        f"Best by validation accuracy: <b style='color:#7c87ff;'>{best_name}</b>"
        f"{' · Unseen robustness data included' if has_unseen else ' · No unseen robustness data'}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("**Accuracy — CV vs Training vs Validation" + (" vs Unseen" if has_unseen else "") + "**")
    st.plotly_chart(plot_model_comparison(df), use_container_width=True)

    st.markdown("**Precision / Recall / F1 (on validation set)**")
    st.plotly_chart(plot_metrics_comparison(df), use_container_width=True)

    st.markdown("**Full Results Table**")
    try:
        num_cols = [c for c in df.columns if df[c].dtype != object and "(%" in c]
        styled = df.style.highlight_max(
            subset=[c for c in ["Val Acc (%)","F1","Unseen Acc (%)","Unseen F1"] if c in df.columns and df[c].dtype != object],
            color="rgba(88,101,248,0.25)"
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)

    if has_unseen:
        st.markdown("**Val Accuracy vs Unseen Accuracy — Robustness Gap**")
        rob_rows = []
        for clf_name, clf_res in results["classifiers"].items():
            um = clf_res["unseen_metrics"]
            vm = clf_res["val_metrics"]
            if um.get("accuracy") is not None:
                rob_rows.append({
                    "Model":      clf_name,
                    "Val (%)":    round(vm["accuracy"]*100, 2),
                    "Unseen (%)": round(um["accuracy"]*100, 2),
                    "Gap (%)":    round((vm["accuracy"] - um["accuracy"])*100, 2)
                })
        if rob_rows:
            rob_df = pd.DataFrame(rob_rows)
            try:
                st.dataframe(
                    rob_df.style.highlight_min(subset=["Gap (%)"], color="rgba(34,211,160,0.2)"),
                    use_container_width=True, hide_index=True
                )
            except Exception:
                st.dataframe(rob_df, use_container_width=True, hide_index=True)
            st.caption("Lower Gap = more robust generalisation. Best generalising model has smallest Val→Unseen drop.")


# ===========================================================================
# PAGE 4 — ABOUT PIPELINE
# ===========================================================================
elif page == "ℹ️ About Pipeline":

    st.title("About the Pipeline")
    st.markdown(
        "End-to-end acoustic percussion signal classifier for Sandwich Composite Plate (SCP) cell inspection. "
        "Classifies 27 cells as **Good (bonded/healthy)** or **Bad (debonded/unhealthy)**. "
        "All signal processing replicates the original notebook exactly."
    )

    st.markdown("### Experimental Context")
    st.markdown(
        """
        The test object is a **sandwich composite plate** with **27 cells**. Each cell is struck
        with a percussion tool, and the resulting acoustic signal is recorded. A bonded (healthy) cell
        produces a distinctly different acoustic response to a debonded (unhealthy) one — this
        difference in resonance, energy distribution, and decay is captured via PSD and MFCC features.
        """
    )

    st.markdown("### Signal Processing Pipeline")
    steps = [
        ("1. Audio Loading",       "Files decoded with `librosa.load()` (WAV, M4A, MP3, OGG, FLAC). Converted to float32 mono."),
        ("2. Peak Detection",      "`scipy.signal.find_peaks` — height ≥ 0.3 × max amplitude, min distance 0.5 s. Each peak = one percussion event."),
        ("3. Segmentation",        "20 ms pre-peak + 200 ms post-peak window per peak. Each segment normalised to [-1, +1]."),
        ("4. PSD (200 features)",  "`scipy.signal.periodogram(nfft=4096, window='boxcar')`. Log of first 200 bins."),
        ("5. MFCC (64 features)",  "`librosa.feature.mfcc(n_mfcc=32)` — 32 mean + 32 std features. Captures spectral envelope."),
        ("6. Feature vector",      "PSD (200) + MFCC mean (32) + MFCC std (32) = **264 features** per segment."),
        ("7. StandardScaler",      "Z-score normalisation. Fitted on training data only — no leakage to validation or unseen sets."),
        ("8. Grid Search CV",      "5-fold stratified CV across KNN, SVM, Decision Tree, Logistic Regression. Refit by accuracy."),
        ("9. Unseen Testing",      "A completely held-out set (never seen during training/CV) is used to evaluate robustness — equivalent to Set6 in the notebook."),
        ("10. Majority Vote",      "For files with multiple percussion peaks, the final prediction is the majority vote across all segments."),
    ]
    for title, desc in steps:
        with st.expander(title):
            st.markdown(desc)

    st.markdown("### Classifier Hyperparameter Grids")
    grid_data = {
        "Classifier":    ["KNN", "SVM", "Decision Tree", "Logistic Regression"],
        "Grid Size":     ["7×2×2 = 28", "5 + 4×4 = 21", "5×3×3×2 = 90", "5×2 = 10"],
        "Key Params":    [
            "k∈{3,5,7,9,11,13,15}, weights∈{uniform,distance}, metric∈{euclidean,manhattan}",
            "kernel∈{linear,rbf}, C∈{0.01–100}, gamma∈{scale,0.01,0.1,1}",
            "max_depth∈{None,3,5,8,12}, criterion∈{gini,entropy}, min_samples_split∈{2,5,10}",
            "C∈{0.01,0.1,1,10,100}, penalty=l2, solver∈{lbfgs,liblinear}"
        ],
        "Needs Scaling": ["✅ Yes", "✅ Yes", "❌ No", "✅ Yes"],
        "Save File":     ["knn_model.pkl", "svm_model.pkl", "decision_tree_model.pkl", "logistic_regression_model.pkl"]
    }
    st.dataframe(pd.DataFrame(grid_data), use_container_width=True, hide_index=True)

    st.markdown("### Labeling Convention")
    st.markdown("""
| Filename example     | Class | Label |
|---|---|---|
| `cell_01g.wav`       | Good  | 1     |
| `cell_01g.m4a`       | Good  | 1     |
| `cell_02.wav`        | Bad   | 0     |
| `debonded_03.m4a`    | Bad   | 0     |

Files whose name ends with **`g`** immediately before the extension are Good (bonded). All others are Bad (debonded).
    """)

    st.markdown("### Running Locally")
    st.code("""
pip install -r requirements.txt
streamlit run app.py
    """, language="bash")

    st.markdown("### Deploying to Streamlit Cloud")
    st.code("""
# 1. Push this folder to GitHub
git init && git add . && git commit -m "SCP classifier"
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main

# 2. Go to share.streamlit.io
# 3. New app → select repo → main file: app.py → Deploy
# packages.txt (already included) installs ffmpeg automatically
    """, language="bash")

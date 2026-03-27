"""
plot_utils.py
-------------
All matplotlib / plotly figures used in the Streamlit app.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


PALETTE = {
    "good":    "#22d3a0",
    "bad":     "#f05c5c",
    "accent":  "#5865f8",
    "warn":    "#f0a84a",
    "bg":      "#1e2030",
    "surface": "#252840",
    "text":    "#e8eaf6",
    "muted":   "#8b90a8",
}


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

def plot_waveform(signal: np.ndarray, sr: int, title: str = "Waveform",
                  label: int = -1) -> plt.Figure:
    color = PALETTE["good"] if label == 1 else (PALETTE["bad"] if label == 0 else PALETTE["accent"])
    fig, ax = plt.subplots(figsize=(8, 2.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    t = np.linspace(0, len(signal) / sr, len(signal))
    ax.plot(t, signal, color=color, linewidth=0.6, alpha=0.9)
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Time (s)", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("Amplitude", color=PALETTE["muted"], fontsize=9)
    ax.set_title(title, color=PALETTE["text"], fontsize=10, pad=8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["surface"])
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

def plot_psd(signal: np.ndarray, sr: int, title: str = "Power Spectral Density") -> plt.Figure:
    from scipy.signal import periodogram
    x = signal.astype(np.float64)
    f, pxx = periodogram(x, fs=sr, nfft=4096, window='boxcar',
                         detrend=False, scaling='density', return_onesided=True)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(f, np.log(pxx + 1e-10), color=PALETTE["accent"], linewidth=0.9)
    ax.set_xlabel("Frequency (Hz)", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("Log Power", color=PALETTE["muted"], fontsize=9)
    ax.set_title(title, color=PALETTE["text"], fontsize=10, pad=8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.set_xlim(0, sr / 2)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["surface"])
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MFCC heatmap
# ---------------------------------------------------------------------------

def plot_mfcc(signal: np.ndarray, sr: int, title: str = "MFCC") -> plt.Figure:
    import librosa
    import librosa.display
    x = signal.astype(np.float64)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=32)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax, format='%+2.0f')
    ax.set_title(title, color=PALETTE["text"], fontsize=10, pad=8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.set_xlabel("Time (s)", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("MFCC Coeff", color=PALETTE["muted"], fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, title: str, cmap: str = "Blues") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    sns.heatmap(
        cm,
        annot=True, fmt='d', cmap=cmap,
        xticklabels=["Bad", "Good"],
        yticklabels=["Bad", "Good"],
        ax=ax,
        cbar=False,
        linewidths=0.5,
        linecolor=PALETTE["bg"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(title, color=PALETTE["text"], fontsize=10, pad=8)
    ax.set_xlabel("Predicted", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("Actual", color=PALETTE["muted"], fontsize=9)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart (Plotly)
# ---------------------------------------------------------------------------

def plot_model_comparison(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    metrics = ["CV Acc (%)", "Train Acc (%)", "Val Acc (%)"]
    colors  = [PALETTE["accent"], PALETTE["warn"], PALETTE["good"]]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=summary_df["Model"],
            y=summary_df[metric],
            marker_color=color,
            opacity=0.85,
        ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], size=11),
        legend=dict(bgcolor=PALETTE["bg"], bordercolor=PALETTE["surface"]),
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis=dict(range=[0, 105], ticksuffix="%",
                   gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=340,
    )
    return fig


def plot_metrics_comparison(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    metrics = ["Precision", "Recall", "F1"]
    colors  = [PALETTE["accent"], PALETTE["good"], PALETTE["warn"]]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=summary_df["Model"],
            y=summary_df[metric],
            marker_color=color,
            opacity=0.85,
        ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], size=11),
        legend=dict(bgcolor=PALETTE["bg"], bordercolor=PALETTE["surface"]),
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis=dict(range=[0, 1.05],
                   gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# Feature distribution (Good vs Bad)
# ---------------------------------------------------------------------------

def plot_feature_distribution(X: np.ndarray, y: np.ndarray) -> go.Figure:
    good_mean = X[y == 1].mean(axis=0) if (y == 1).any() else np.zeros(X.shape[1])
    bad_mean  = X[y == 0].mean(axis=0) if (y == 0).any() else np.zeros(X.shape[1])

    # Downsample to first 64 features for readability
    n = min(64, X.shape[1])
    x_axis = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=good_mean[:n], mode='lines',
        name='Good cells', line=dict(color=PALETTE["good"], width=1.5),
        fill='tozeroy', fillcolor='rgba(34,211,160,0.08)'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=bad_mean[:n], mode='lines',
        name='Bad cells', line=dict(color=PALETTE["bad"], width=1.5),
        fill='tozeroy', fillcolor='rgba(240,92,92,0.08)'
    ))

    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], size=11),
        legend=dict(bgcolor=PALETTE["bg"]),
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(title="Feature index", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Mean (scaled)", gridcolor="rgba(255,255,255,0.05)"),
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Confidence chart for test predictions
# ---------------------------------------------------------------------------

def plot_prediction_confidence(pred_df: pd.DataFrame) -> go.Figure:
    colors = [PALETTE["good"] if l == 1 else PALETTE["bad"]
              for l in pred_df["Label"]]

    fig = go.Figure(go.Bar(
        x=pred_df["Filename"],
        y=pred_df["Confidence"],
        marker_color=colors,
        text=pred_df["Prediction"],
        textposition='outside',
        opacity=0.85,
    ))

    fig.add_hline(y=50, line_dash="dash",
                  line_color=PALETTE["muted"], opacity=0.5,
                  annotation_text="Decision threshold (50%)")

    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], size=11),
        margin=dict(l=40, r=20, t=20, b=60),
        xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(range=[0, 110], ticksuffix="%",
                   gridcolor="rgba(255,255,255,0.05)"),
        height=320,
    )
    return fig

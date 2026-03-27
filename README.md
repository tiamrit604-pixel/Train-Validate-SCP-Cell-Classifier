# 🔋 Composite Structure Cell Classifier — Streamlit App

Battery cell quality classification from audio recordings.
Replicates the exact pipeline from `HW3_Code_-_newdataset_-_Grid_CV_Final.ipynb`.

---

## Project Structure

```
cell_classifier_app/
│
├── app.py              ← Main Streamlit application (run this)
├── audio_utils.py      ← Audio loading, peak detection, feature extraction
├── ml_utils.py         ← GridSearchCV training, model save/load, prediction
├── plot_utils.py       ← All matplotlib + plotly charts
├── requirements.txt    ← Python dependencies
├── .streamlit/
│   └── config.toml     ← Dark theme + upload size config
└── models/             ← Auto-created when you save a model
    ├── best_model.pkl
    └── last_results.pkl
```

---

## Setup & Run Locally

### 1. Install dependencies

```bash
cd cell_classifier_app
pip install -r requirements.txt
```

> **Note for M4A files:** Install `ffmpeg` system-wide for M4A support.
> - Windows: Download from https://ffmpeg.org/download.html, add to PATH
> - Mac: `brew install ffmpeg`
> - Linux: `sudo apt install ffmpeg`

### 2. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## How to Use

### Mode 1 — Train & Validate

1. Upload audio files with correct naming:
   - Files ending with `g.wav` / `g.m4a` → **Good cell (1)**
   - All other files → **Bad cell (0)**
   - Example: `cell_01g.wav` = Good, `cell_02.wav` = Bad

2. Choose classifiers (KNN, SVM, Decision Tree, Logistic Regression)
3. Set CV folds and validation split
4. Click **Start Training**
5. View results: model table, confusion matrices, feature distribution
6. Model is auto-saved to `models/best_model.pkl`

### Mode 2 — Test on Model

1. Upload new audio files (no labeling needed)
2. Click **Classify Files**
3. View per-file Good/Bad predictions with confidence scores
4. Inspect waveform, PSD, and MFCC for any file
5. Download results as CSV

---

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app**
5. Select your repo, branch `main`, and main file `app.py`
6. Click **Deploy**

> **Important for Streamlit Cloud:** Add `packages.txt` to your repo root:
> ```
> ffmpeg
> ```
> This installs ffmpeg for M4A support in the cloud environment.

---

## Feature Extraction (exact notebook logic)

| Feature | Method | Dim |
|---|---|---|
| Log PSD | `scipy.periodogram(nfft=4096)`, first 200 bins | 200 |
| MFCC mean | `librosa.feature.mfcc(n_mfcc=32)`.mean(axis=1) | 32 |
| MFCC std | `librosa.feature.mfcc(n_mfcc=32)`.std(axis=1) | 32 |
| **Total** | | **264** |

---

## Classifier Grid Search

| Model | CV Grid Size |
|---|---|
| KNN | 7 × 2 × 2 = 28 combinations |
| SVM | 5 (linear) + 4×4 (rbf) = 21 combinations |
| Decision Tree | 5 × 3 × 3 × 2 = 90 combinations |
| Logistic Regression | 5 × 2 = 10 combinations |

All use `scoring = {accuracy, f1, precision, recall}` with `refit='accuracy'`.

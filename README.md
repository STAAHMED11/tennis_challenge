# 🎾 Ball Hit and Bounce Detection System

This repository provides a comprehensive solution for detecting ball events — **Hits** (racket contacts) and **Bounces** (ground contacts) — from 2D trajectory data. It features two complementary approaches: a physics-based unsupervised method and a high-performance supervised machine learning model.

---

## ⚙️ Methodology

### 1. UNSUPERVISED (Physics-Based)
This method detects hits and bounces without labels, relying purely on physical principles derived from ball motion. Hits and bounces are inferred by analyzing discontinuities, slope changes, and acceleration spikes in the x/y trajectories.

**Core signals analyzed:**
* **Vertical Acceleration Patterns:** Sudden upward forces.
* **Sudden Velocity Changes:** Momentum shifts in both x and y.
* **Ball Height Dynamics:** Normalized height to detect ground contacts.
* **Characteristic Signatures:** Discontinuities and slope changes in motion curves.



### 2. SUPERVISED (Machine Learning)
This method uses provided "action" labels to train a supervised model that captures complex temporal dynamics.

* **🔹 LSTM (Default - Recommended):** * *Architecture:* Bidirectional LSTM with Temporal Attention.
    * *Strengths:* Better F1-Score, captures long-range temporal dependencies, superior at handling sequential patterns.
    * *Best for:* Imbalanced datasets where precision-recall balance matters.
* **🔹 ML (XGBoost):**
    * *Architecture:* Gradient Boosting Classifier.
    * *Strengths:* Higher raw accuracy, faster inference, interpretable feature importance.

---

> **⚠️ Note on Metrics:** With highly imbalanced classes (90%+ frames are "air"), Accuracy can be misleading. The **F1-Score** better reflects the model's ability to correctly identify the rare but critical hit/bounce events.

---
## 🚀 Usage

The `main.py` script provides a command-line interface for both methods.

### 1️⃣ Supervised Detection

Requires the `models/` directory containing pre-trained assets.

#### **Using LSTM (Default - Better F1-Score)**
```bash
python main.py supervised ball_data_i.json
```

Or explicitly:
```bash
python main.py supervised ball_data_i.json lstm output.json
```

#### **Using XGBoost ML Model (Better Accuracy)**
```bash
python main.py supervised ball_data_i.json ml output.json
```

#### **With Custom Model Directory**
```bash
python main.py supervised ball_data_i.json my_models/ lstm output.json
```

### 2️⃣ Unsupervised Detection

Runs purely on physics heuristics (no models required).
```bash
python main.py unsupervised ball_data_i.json output.json
```

---

## 📌 Supervised Model Requirements

The `models/` directory must contain:

### For LSTM Model:
- `trained_lstm_model.pth`  
- `lstm_model_config.pkl`
- `lstm_scaler.pkl`  
- `lstm_feature_columns.pkl`  
- `lstm_label_encoder.pkl`

### For ML Model:
- `trained_xgboost_model.pkl`  
- `scaler.pkl`  
- `feature_columns.pkl`  
- `label_encoder.pkl`

---

---

## 📂 Repository Structure
```
.
├── main.py                    # Feature engineering and detection logic
├── models/                    # Pre-trained model and preprocessing assets
│   ├── trained_lstm_model.pth
│   ├── lstm_*.pkl
│   ├── trained_xgboost_model.pkl
│   └── *.pkl
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📊 Features Extracted

The project implements an optimized feature engineering pipeline with **180+ features**.

### 🔹 Physical Derivatives
- **1st Order**: Horizontal / Vertical Velocity, Speed  
- **2nd Order**: Acceleration (magnitude, horizontal, vertical)
- **3rd Order**: Jerk Magnitude (rate of change of acceleration)

### 🔹 Temporal Features
- Rolling statistics (mean, std, min, max) over multiple windows
- Frame lags (past / future context windows)
- Velocity reversal detection
- Sustained event patterns

### 🔹 Geometric Features
- Trajectory curvature  
- Motion phase and phase changes
- Angular velocity  
- Height normalization

### 🔹 Composite Features
- Hit score (jerk × acceleration × angle change)
- Bounce score (vertical velocity change × height)
- Energy and momentum tracking

---

## 📝 Deliverables

### ✅ Two Detection Approaches:
1. **Supervised**: `supervised_hit_bounce_detection(json_path, model_dir, model_type='lstm')`
2. **Unsupervised**: `unsupervised_hit_bounce_detection(json_path)`

### ✅ Feature Engineering Pipeline:
- Converts raw `(x, y)` coordinates into 180+ physical and temporal signals

### ✅ Enriched JSON Output:
- Original trajectory data with predicted `action` per frame
- Format: `{"frame_id": {"x": ..., "y": ..., "visible": ..., "action": "air|hit|bounce"}}`

---

## 📋 Requirements

To run this project, install the required Python packages:
```bash
pip install -r requirements.txt
```


---

## 🎯 Model Selection Guide

**Choose LSTM if:**
- ✅ You need the best F1-Score (critical for imbalanced data)
- ✅ You want better precision-recall balance
- ✅ You have sequential/temporal data (which you do)

**Choose XGBoost if:**
- ✅ You need faster inference
- ✅ Raw accuracy is your primary metric
- ✅ You want feature interpretability

**Choose Unsupervised if:**
- ✅ No labeled training data available
- ✅ Need interpretable physics-based approach
- ✅ Quick prototyping without model training

---

## 🔧 Troubleshooting

### Pickle Compatibility Issues
If you encounter numpy pickle errors, the code includes automatic fallbacks:
- Attempts multiple loading strategies
- Applies numpy compatibility patches
- Creates runtime-fitted scalers if needed

### Model Not Found
Ensure the `models/` directory contains all required files for your chosen model type (LSTM or ML).

---

## 📈 Future Enhancements

- [ ] Ensemble method combining LSTM + XGBoost
- [ ] Additional training data collection
- [ ] Real-time inference optimization
- [ ] Temporal smoothing post-processing
- [ ] Confidence score calibration
- [ ] Cross-validation with different court/camera angles


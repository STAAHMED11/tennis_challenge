# 🎾 Ball Hit and Bounce Detection System

This repository provides a comprehensive solution for detecting ball events — **Hits** (racket contacts) and **Bounces** (ground contacts) — from **2D trajectory data**.

It features **two complementary approaches**:
- A **physics-based unsupervised method**
- A **high-performance supervised machine learning model**

---

## ⚙️ Methodology

### Method 1 — **UNSUPERVISED (Physics-Based)**

This method detects hits and bounces **without labels**, relying purely on physical principles derived from ball motion.

**Core signals analyzed:**
- **Vertical Acceleration Patterns**: sudden upward forces
- **Sudden Velocity Changes**: momentum shifts in both `x` and `y`
- **Ball Height Dynamics**: normalized height to detect ground contacts
- **Characteristic Signatures**: discontinuities and slope changes in motion curves

**Key Idea:**  
Hits and bounces are inferred by analyzing **discontinuities, slope changes, and acceleration spikes** in the `x/y` trajectories.

---

### Method 2 — **SUPERVISED (Machine Learning)**

This method uses the provided `"action"` labels to train a supervised model that captures complex temporal dynamics.

- **Model**: XGBoost Classifier
- **Feature Engineering**: 50+ physics-based and temporal features
- **Temporal Dynamics**:
  - Rolling statistics
  - Context before and after each frame
  - Sequence-aware features

---

## 🚀 Usage

The `main.py` script provides a **command-line interface** for both methods.

### 1️⃣ Supervised Detection
Requires the `models/` directory containing pre-trained assets.

```bash
python main.py supervised ball_data_i.json models/ output.json
```
### 2️⃣ Unsupervised Detection

Runs purely on physics heuristics (no models required).

```bash
python main.py unsupervised ball_data_i.json output.json
```
## 📌 Supervised Model Requirements

The `models/` directory must contain:

- trained_xgboost_model.pkl  
- scaler.pkl  
- feature_columns.pkl  
- label_encoder.pkl  

## 📂 Repository Structure

- main.py: Feature engineering and detection logic  
- models/: Pre-trained model and preprocessing assets  
- requirements.txt: Python dependencies  
- README.md: Project documentation  

## 📊 Features Extracted

The project implements an optimized feature engineering pipeline.

🔹 Physical Derivatives

- 1st Order: Horizontal / Vertical Velocity, Speed  
- 2nd Order: Acceleration  
- 3rd Order: Jerk Magnitude (rate of change of acceleration)

🔹 Temporal Features

- Rolling means and standard deviations  
- Frame lags (past / future context)

🔹 Geometric Features

- Trajectory curvature  
- Motion phase  
- Angular velocity  

## 📝 Deliverables

Two Detection Functions:

- unsupervised_hit_bounce_detection(ball_data_i.json)  
- supervised_hit_bounce_detection(ball_data_i.json)  

Feature Engineering Pipeline:

- Converts raw (x, y) coordinates into physical signals  

Enriched JSON Output:

- Original trajectory data with predicted action per frame  

## 📋 Requirements

To run this project, install the required Python packages.
```bash
pip install -r requirements.txt
```


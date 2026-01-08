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

### 1. Supervised Detection
Requires the `models/` directory containing pre-trained assets.

**Using LSTM (Default):**
```bash
python main.py supervised ball_data_i.json

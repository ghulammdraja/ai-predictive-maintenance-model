# AI-Based Predictive Maintenance Model

A simple web application that predicts machine failures by monitoring vibration, temperature, and operating conditions. Instead of relying on physical sensor hardware, this project uses machine learning trained on industrial telemetry datasets to detect abnormal wear and estimate Remaining Useful Life (RUL).

---

## Overview

In industrial setups, machines like motors, pumps, and CNC spindles degrade over time. Unexpected machine breakdowns lead to expensive downtime. 

This project simulates a condition-monitoring system where an operator can enter sensor values (or upload telemetry logs) to get an immediate assessment of machine health, failure probability, and maintenance recommendations.

### Key Features
- **Health Score & Failure Risk**: Calculates an overall health percentage (0–100%) and failure probability based on live sensor inputs.
- **Vibration Analysis**: Compares vibration levels (in mm/s) against ISO 10816 safety limits (Good, Acceptable, Warning, Danger).
- **Fault Diagnosis**: Identifies potential root causes such as bearing defects, unbalance, misalignment, or overheating.
- **Remaining Useful Life (RUL)**: Estimates how many operating hours the machine has left before service is required.
- **Built-in Presets**: One-click buttons to quickly test normal, warning, and critical machine states.

---

## How It Works

1. **Input Data**: The system takes in common sensor readings:
   - Vibration Level (mm/s RMS)
   - Motor Temperature (°C)
   - Rotational Speed (RPM)
   - Operating Hours
   - Motor Load Torque (Nm)

2. **Prediction Logic**:
   - Compares the telemetry against degradation models and safety thresholds.
   - Outputs an instant status badge (Healthy, Warning, or Critical).

3. **Actionable Advice**:
   - Provides clear guidance on what the operator should do (e.g. routine check, lubrication, or urgent shutdown).

---

## Project Structure

```text
├── app.py                      # Main Streamlit web application
├── train.py                    # Model training script
├── test_pipeline.py            # Automated test suite
├── fastapi_app.py              # Optional REST API backend
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container deployment setup
├── data/                       # Industrial telemetry datasets
│   ├── industrial_vibration_telemetry.csv
│   └── ai4i2020_raw.csv
├── models/                     # Saved trained models
└── src/                        # Core logic and helper functions
    ├── data_loader.py          # Data generation and loading
    ├── feature_engineering.py  # ISO severity & vibration calculations
    ├── models.py               # ML training and prediction pipeline
    └── vibration_simulator.py  # Signal and FFT simulator
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-predictive-maintenance.git
cd ai-predictive-maintenance
```

### 2. Install dependencies
Make sure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 3. Run the web app
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Testing & Retraining

- **Run tests**:
  ```bash
  python test_pipeline.py
  ```
- **Retrain models**:
  ```bash
  python train.py
  ```

---

## Tech Stack
- **Language**: Python 3.10+
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data & Signal Processing**: Pandas, NumPy, SciPy

---

## License
MIT License. Feel free to use and modify for your own academic or personal projects.

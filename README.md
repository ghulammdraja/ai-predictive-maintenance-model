# PREDIXION | Industrial Machinery Condition Monitoring & AI Predictive Maintenance

A software-based condition monitoring and asset reliability platform engineered for industrial rotating machinery (turbines, centrifugal pumps, CNC spindles, compressors).

Integrates **vibration spectral analysis ($1\times, 2\times, BPFO$)**, **Hilbert envelope demodulation (PeakVue)**, **ISO 10816-3/20816 severity standards**, **Remaining Useful Life (RUL) estimation**, and **automated CMMS Work Order dispatching**.

---

## 🏗️ Technical Architecture

```
                                  [ MULTI-CHANNEL TELEMETRY INGESTION ]
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
    [ TIME-DOMAIN KINEMATICS ]                                            [ SPECTRAL DEMODULATION ]
   • Overall RMS Velocity (mm/s)                                         • Running Speed Orders (1X, 2X, 3X)
   • True Peak-to-Peak Amplitude                                         • Bearing Defect Orders (BPFO / BPFI)
   • Impact Kurtosis Moment                                              • Hilbert Envelope Spectrum (PeakVue)
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    ▼
                                    [ AI DIAGNOSTIC & PROGNOSTICS CORE ]
                                                    │
         ┌───────────────────────────┬──────────────┴──────────────┬───────────────────────────┐
         ▼                           ▼                             ▼                           ▼
 [ ISOLATION FOREST ]    [ FAILURE CLASSIFIER ]         [ ROOT CAUSE DIAGNOSER ]       [ RUL REGRESSOR ]
  Anomaly Detection        XGBoost / Random Forest       Multi-Class Fault Isolation    Gradient Boosting
         │                           │                             │                           │
         └───────────────────────────┼─────────────────────────────┼───────────────────────────┘
                                     ▼                             ▼
                            [ HEALTH INDEX % ]           [ CMMS WORK ORDER ]
                          Composite Reliability           SAP PM / Maximo Ticket
                                     │                             │
                                     └──────────────┬──────────────┘
                                                    ▼
                                 [ INDUSTRIAL SCADA CONSOLE (app.py) ]
                                  [ & FASTAPI REST BACKEND GATEWAY ]
```

---

## 📊 Vibration & Mechanical Diagnostics Reference

| Parameter | Baseline Range | Fault Threshold | Diagnostic Interpretation |
| :--- | :--- | :--- | :--- |
| **Overall RMS Velocity** | $< 1.12\text{ mm/s}$ (Zone A) | $> 4.5\text{ mm/s}$ (Zone C/D) | Overall destructive mechanical vibration energy (ISO 10816-3) |
| **Statistical Kurtosis** | $2.8 - 3.2$ (Gaussian) | $> 4.5$ (Up to $15+$) | High sensitivity to repetitive shock impacts from bearing raceway fluting |
| **Crest Factor ($CF$)** | $2.0 - 2.5$ | $> 4.0$ | Peak-to-RMS ratio; indicator of early-stage fatigue spalling |
| **$1\times$ Order Peak** | $< 0.4\text{ mm/s}$ | Elevated | Rotor dynamic mass unbalance / eccentricity |
| **$2\times$ Order Peak** | $< 0.2\text{ mm/s}$ | Elevated | Angular or parallel shaft coupling misalignment |
| **Envelope Resonance** | Baseline noise | Discrete Peak at $BPFO$ | Bearing Outer Race Fluting / Rolling Element damage |

---

## 🌟 Application Features & Workspaces

1. **Digital Oscilloscope & Spectral Demodulator (`[CH-01]`)**:
   - High-speed time-domain waveform trace ($x(t)$).
   - Fast Fourier Transform (FFT) power spectrum with labeled running orders ($1\times, 2\times, BPFO$).
   - Hilbert Transform High-Frequency Envelope Demodulation (PeakVue) for bearing fault isolation.
2. **Asset Diagnostic Workbench**:
   - Multi-variable parameter bench for real-time what-if simulations.
   - Root-cause probability distribution.
   - Continuous Prognostic Counter (Remaining Useful Life in hours).
   - **Automated CMMS Maintenance Work Order Generator**: Formatted with Work Order ID, Priority ($P1-P4$), Safety LOTO Protocol, Action Codes, and Bill of Materials (BOM).
3. **Plant Fleet SCADA Matrix**:
   - Multi-machine surveillance board across Plant Lines A & B.
   - Vibration RMS vs Health Score reliability matrix.
4. **ML Benchmarks & Validation**:
   - Dual Dataset Engine: **Official UCI AI4I 2020 Dataset (10,000 Records)** + **Industrial Vibration Telemetry (5,000 Records)**.
   - Confusion matrices, ROC-AUC curves, and Gini feature importances.
5. **Batch Telemetry Ingestion**:
   - Drag-and-drop CSV batch processor with downloadable CMMS audit logs.

---

## 🚀 Quickstart

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Launch SCADA Dashboard
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501`.

### 3. Run Automated Verification Test Suite
```bash
python test_pipeline.py
```

### 4. REST API Backend (Optional)
```bash
python -m uvicorn fastapi_app:app --port 8000
```
API Documentation at `http://localhost:8000/docs`.

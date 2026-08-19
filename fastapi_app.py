"""
FastAPI REST API Backend for AI-Based Predictive Maintenance System
Provides programmatic endpoints for model inference, vibration simulation, and system telemetry health.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import PredictiveMaintenancePipeline, FEATURE_COLUMNS
from src.vibration_simulator import generate_vibration_signal
from src.feature_engineering import get_iso_vibration_severity

app = FastAPI(
    title="AI Predictive Maintenance REST API",
    description="Production REST API for Machine Vibration Monitoring, Failure Prediction & RUL Estimation",
    version="1.0.0"
)

# Enable CORS for external frontend or web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline at startup
pipeline: Optional[PredictiveMaintenancePipeline] = None

try:
    pipeline = PredictiveMaintenancePipeline.load()
except Exception:
    pipeline = None


class TelemetryInput(BaseModel):
    operating_hours: float = Field(120.0, description="Cumulative machine runtime hours")
    rotational_speed_rpm: float = Field(1800.0, description="Shaft rotational speed in RPM")
    load_torque_nm: float = Field(35.0, description="Operating torque in Nm")
    vibration_rms_mm_s: float = Field(0.95, description="ISO 10816-3 RMS vibration velocity in mm/s")
    vibration_peak_mm_s: float = Field(2.20, description="Peak vibration amplitude in mm/s")
    vibration_kurtosis: float = Field(2.90, description="Statistical kurtosis (peakedness / shocks)")
    vibration_crest_factor: float = Field(2.30, description="Crest factor (Peak/RMS)")
    fft_1x_unbalance: float = Field(0.30, description="1X running speed harmonic peak amplitude")
    fft_2x_misalignment: float = Field(0.15, description="2X running speed harmonic peak amplitude")
    hf_bearing_energy: float = Field(0.08, description="High-frequency bearing defect energy band")
    motor_temperature_c: float = Field(45.0, description="Motor casing / bearing temperature in °C")
    ambient_temperature_c: float = Field(22.0, description="Ambient air temperature in °C")
    acoustic_noise_db: float = Field(64.0, description="Acoustic sound pressure level in dB")
    tool_wear_min: float = Field(40.0, description="Tool / component cumulative wear in minutes")


class SimulationRequest(BaseModel):
    condition: str = Field("Healthy", description="Machine condition: Healthy, Unbalance (1X Fault), Misalignment (2X Fault), Mechanical Looseness, Bearing Fault (BPFO / Impact Ringing), Severe Failure (Multiple Faults)")
    duration_seconds: float = Field(0.1, description="Signal duration in seconds")
    sampling_rate: int = Field(4000, description="Sampling rate in Hz")
    rpm: float = Field(1800.0, description="Rotational speed in RPM")
    load_pct: float = Field(80.0, description="Operating load percentage")


@app.get("/")
def root():
    return {
        "system": "AI Predictive Maintenance Engine",
        "status": "Online",
        "endpoints": ["/health", "/docs", "/predict", "/simulate", "/metrics"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": bool(pipeline is not None),
        "feature_count": len(FEATURE_COLUMNS)
    }


@app.post("/predict")
def predict_machine_health(telemetry: TelemetryInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="ML Pipeline not initialized. Run train.py first.")
    
    input_dict = telemetry.model_dump()
    result = pipeline.predict_single(input_dict)
    return result


@app.post("/simulate")
def simulate_vibration(req: SimulationRequest):
    t, sig, freqs, amps, env_f, env_a, meta = generate_vibration_signal(
        condition=req.condition,
        duration_seconds=req.duration_seconds,
        sampling_rate=req.sampling_rate,
        rpm=req.rpm,
        load_pct=req.load_pct
    )
    
    return {
        "metadata": meta,
        "time_samples_ms": (t * 1000).tolist()[:500],
        "vibration_waveform": sig.tolist()[:500],
        "frequencies_hz": freqs[freqs <= 1000].tolist()[:300],
        "amplitudes": amps[freqs <= 1000].tolist()[:300],
        "envelope_frequencies_hz": env_f.tolist()[:200],
        "envelope_amplitudes": env_a.tolist()[:200]
    }


@app.get("/metrics")
def get_model_metrics():
    if pipeline is None:
        raise HTTPException(status_code=500, detail="ML Pipeline not initialized.")
    return pipeline.metrics


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

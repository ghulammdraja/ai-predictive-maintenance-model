"""
Simple AI-Based Predictive Maintenance Model
A clean, easy-to-use web application to monitor machine vibration and predict failures.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page setup
st.set_page_config(
    page_title="AI Machine Health Monitor",
    page_icon="⚙️",
    layout="wide"
)

# Custom minimal styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .healthy-box {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
    }
    .warning-box {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        color: #92400e;
    }
    .danger-box {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. SIMPLE MACHINE LEARNING MODEL & LOGIC
# -------------------------------------------------------------
def predict_machine_health(vibration, temperature, rpm, hours, torque):
    """
    Simple, clear AI prediction function based on vibration and sensor thresholds.
    """
    # Calculate health score (0 to 100%)
    # Vibration is the most important factor: normal is < 2.0 mm/s, critical is > 4.5 mm/s
    vibe_penalty = max(0, (vibration - 1.0) * 18.0)
    temp_penalty = max(0, (temperature - 50.0) * 1.2)
    wear_penalty = min(25, hours * 0.03)
    
    health_score = max(5, int(100 - (vibe_penalty + temp_penalty + wear_penalty)))
    failure_risk = max(0.0, min(100.0, 100.0 - health_score))
    
    # Remaining Useful Life estimation (in hours)
    if health_score > 75:
        rul_hours = int(max(400, 1000 - hours))
        status = "Healthy"
        issue = "None (Machine operating smoothly)"
        recommendation = "Continue normal operation. No maintenance required."
        box_class = "healthy-box"
        badge = "🟢 MACHINE STATUS: HEALTHY"
    elif health_score > 40:
        rul_hours = int(max(100, 350 - (hours * 0.3)))
        status = "Warning"
        if vibration > 3.0:
            issue = "Elevated Vibration / Early Bearing Wear"
        elif temperature > 65:
            issue = "Motor Running Warm / Cooling Issue"
        else:
            issue = "Normal Aging & Wear"
        recommendation = "Schedule routine inspection during next maintenance window."
        box_class = "warning-box"
        badge = "🟡 MACHINE STATUS: WARNING (DEGRADATION DETECTED)"
    else:
        rul_hours = int(max(10, 80 - (hours * 0.08)))
        status = "Critical"
        if vibration > 4.5 and temperature > 70:
            issue = "Severe Bearing Failure & Overheating"
        elif vibration > 4.5:
            issue = "Critical Vibration (Bearing / Unbalance Fault)"
        else:
            issue = "Severe Thermal Breakdown"
        recommendation = "URGENT: Stop machine and replace damaged parts immediately."
        box_class = "danger-box"
        badge = "🔴 MACHINE STATUS: CRITICAL (IMMINENT FAILURE)"
        
    return {
        "status": status,
        "badge": badge,
        "box_class": box_class,
        "health_score": health_score,
        "failure_risk": round(failure_risk, 1),
        "issue": issue,
        "rul_hours": rul_hours,
        "recommendation": recommendation
    }


# -------------------------------------------------------------
# 2. APP HEADER
# -------------------------------------------------------------
st.markdown('<div class="main-title">AI Predictive Maintenance Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter sensor readings (vibration, temperature, speed) to check machine health and predict potential failures.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. MAIN INTERFACE (2 COLUMNS)
# -------------------------------------------------------------
col_input, col_result = st.columns([1, 1], gap="large")

# State for input values
if "inp_vibe" not in st.session_state:
    st.session_state.inp_vibe = 1.1
    st.session_state.inp_temp = 45.0
    st.session_state.inp_rpm = 1800
    st.session_state.inp_hours = 120
    st.session_state.inp_torque = 35.0

with col_input:
    st.subheader("1. Enter Machine Sensor Data")
    
    # Preset sample buttons
    st.caption("Quick Test Samples:")
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 Normal Machine"):
        st.session_state.inp_vibe = 1.0
        st.session_state.inp_temp = 42.0
        st.session_state.inp_rpm = 1800
        st.session_state.inp_hours = 80
        st.session_state.inp_torque = 32.0
        st.rerun()
        
    if b2.button("🟡 Warning Vibration"):
        st.session_state.inp_vibe = 3.5
        st.session_state.inp_temp = 58.0
        st.session_state.inp_rpm = 1750
        st.session_state.inp_hours = 380
        st.session_state.inp_torque = 38.0
        st.rerun()
        
    if b3.button("🔴 Critical Failure"):
        st.session_state.inp_vibe = 6.2
        st.session_state.inp_temp = 78.0
        st.session_state.inp_rpm = 1720
        st.session_state.inp_hours = 620
        st.session_state.inp_torque = 45.0
        st.rerun()

    st.markdown("---")

    # Easy Sliders
    vibration_val = st.slider(
        "Vibration Level (mm/s)",
        min_value=0.1, max_value=10.0,
        value=float(st.session_state.inp_vibe),
        step=0.1,
        help="Normal is below 2.0 mm/s. Above 4.5 mm/s is dangerous."
    )
    
    temp_val = st.slider(
        "Motor Temperature (°C)",
        min_value=20.0, max_value=100.0,
        value=float(st.session_state.inp_temp),
        step=1.0,
        help="Normal operating temperature is between 35°C and 55°C."
    )
    
    rpm_val = st.slider(
        "Rotational Speed (RPM)",
        min_value=500, max_value=3600,
        value=int(st.session_state.inp_rpm),
        step=50
    )
    
    hours_val = st.slider(
        "Operating Hours (Total runtime)",
        min_value=0, max_value=1000,
        value=int(st.session_state.inp_hours),
        step=10
    )
    
    torque_val = st.slider(
        "Motor Load Torque (Nm)",
        min_value=10.0, max_value=80.0,
        value=float(st.session_state.inp_torque),
        step=1.0
    )

# Run Prediction
result = predict_machine_health(vibration_val, temp_val, rpm_val, hours_val, torque_val)

with col_result:
    st.subheader("2. AI Prediction Result")
    
    # Status Banner
    st.markdown(f"""
    <div class="result-box {result['box_class']}">
        <h2 style="margin:0; font-size:1.4rem;">{result['badge']}</h2>
        <p style="margin:0.5rem 0 0 0; font-size:1rem;"><strong>Diagnosed Issue:</strong> {result['issue']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 3 Summary Cards
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:0.85rem; color:#64748b;">HEALTH SCORE</div>
            <div style="font-size:1.8rem; font-weight:700; color:{'#10b981' if result['health_score']>70 else ('#f59e0b' if result['health_score']>40 else '#ef4444')};">
                {result['health_score']}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:0.85rem; color:#64748b;">FAILURE RISK</div>
            <div style="font-size:1.8rem; font-weight:700; color:{'#ef4444' if result['failure_risk']>50 else ('#f59e0b' if result['failure_risk']>25 else '#10b981')};">
                {result['failure_risk']}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:0.85rem; color:#64748b;">REMAINING LIFE</div>
            <div style="font-size:1.8rem; font-weight:700; color:#1e293b;">
                {result['rul_hours']} <span style="font-size:0.9rem; font-weight:400;">hrs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### 💡 Recommended Action:")
    st.info(result['recommendation'])

# -------------------------------------------------------------
# 4. SIMPLE VIBRATION TREND GRAPH
# -------------------------------------------------------------
st.markdown("---")
st.subheader("3. Vibration Level vs Safety Limits")

# Create a simple visual threshold chart
fig = go.Figure()

# Current machine reading
fig.add_trace(go.Bar(
    x=["Current Machine Reading"],
    y=[vibration_val],
    name="Your Vibration Level",
    marker_color="#ef4444" if vibration_val > 4.5 else ("#f59e0b" if vibration_val > 2.8 else "#10b981"),
    width=0.35,
    text=[f"{vibration_val} mm/s"],
    textposition="outside"
))

# Threshold reference lines
fig.add_hline(y=1.12, line_dash="dash", line_color="#10b981", annotation_text="Good (Zone A: < 1.12 mm/s)")
fig.add_hline(y=2.80, line_dash="dash", line_color="#3b82f6", annotation_text="Acceptable (Zone B: < 2.80 mm/s)")
fig.add_hline(y=4.50, line_dash="dash", line_color="#f59e0b", annotation_text="Warning Threshold (4.50 mm/s)")
fig.add_hline(y=7.10, line_dash="dash", line_color="#ef4444", annotation_text="Danger / Critical (Zone D: > 7.10 mm/s)")

fig.update_layout(
    yaxis_title="Vibration Velocity (mm/s)",
    yaxis=dict(range=[0, max(8.5, vibration_val + 2)]),
    height=320,
    margin=dict(l=40, r=40, t=30, b=30)
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 5. DATASET TABLE PREVIEW (SIMPLE)
# -------------------------------------------------------------
with st.expander("📄 Click to view sample training dataset table (CSV)"):
    csv_path = os.path.join("data", "industrial_vibration_telemetry.csv")
    if os.path.exists(csv_path):
        sample_df = pd.read_csv(csv_path)
        st.dataframe(sample_df[[
            "machine_id", "operating_hours", "rotational_speed_rpm",
            "vibration_rms_mm_s", "motor_temperature_c",
            "failure_risk_target", "failure_mode", "remaining_useful_life_hours"
        ]].head(15), use_container_width=True)
    else:
        st.write("Dataset file not found.")

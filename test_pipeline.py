"""
Test and Verification Suite for Industrial Predictive Maintenance System
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import PredictiveMaintenancePipeline, FEATURE_COLUMNS
from src.vibration_simulator import generate_vibration_signal
from src.feature_engineering import get_iso_vibration_severity, calculate_time_domain_features


class TestPredictiveMaintenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PredictiveMaintenancePipeline.load()
        self_df = pd.read_csv(os.path.join("data", "industrial_vibration_telemetry.csv"))
        cls.sample_df = self_df.head(20)

    def test_pipeline_loaded(self):
        self.assertIsNotNone(self.pipeline.failure_classifier)
        self.assertIsNotNone(self.pipeline.failure_mode_classifier)
        self.assertIsNotNone(self.pipeline.rul_regressor)
        self.assertIsNotNone(self.pipeline.anomaly_detector)

    def test_healthy_prediction(self):
        healthy_sample = {
            "operating_hours": 80,
            "rotational_speed_rpm": 1780.0,
            "load_torque_nm": 32.0,
            "vibration_rms_mm_s": 0.85,
            "vibration_peak_mm_s": 1.9,
            "vibration_kurtosis": 2.9,
            "vibration_crest_factor": 2.2,
            "fft_1x_unbalance": 0.25,
            "fft_2x_misalignment": 0.12,
            "hf_bearing_energy": 0.06,
            "motor_temperature_c": 42.0,
            "ambient_temperature_c": 22.0,
            "acoustic_noise_db": 62.0,
            "tool_wear_min": 25
        }
        res = self.pipeline.predict_single(healthy_sample)
        self.assertFalse(res["is_failure_predicted"])
        self.assertLess(res["failure_risk_probability"], 0.40)
        self.assertGreater(res["health_index"], 65.0)
        self.assertIn("Zone A", res["iso_severity"]["zone"])
        self.assertIn("work_order", res)

    def test_bearing_fault_prediction(self):
        fault_sample = {
            "operating_hours": 580,
            "rotational_speed_rpm": 1740.0,
            "load_torque_nm": 40.0,
            "vibration_rms_mm_s": 4.8,
            "vibration_peak_mm_s": 32.0,
            "vibration_kurtosis": 11.2,
            "vibration_crest_factor": 6.8,
            "fft_1x_unbalance": 0.55,
            "fft_2x_misalignment": 0.25,
            "hf_bearing_energy": 4.8,
            "motor_temperature_c": 68.0,
            "ambient_temperature_c": 24.0,
            "acoustic_noise_db": 82.0,
            "tool_wear_min": 280
        }
        res = self.pipeline.predict_single(fault_sample)
        self.assertTrue(res["is_failure_predicted"])
        self.assertGreater(res["failure_risk_probability"], 0.70)
        self.assertEqual(res["predicted_failure_mode"], "Bearing Raceway Fluting")
        self.assertLess(res["estimated_rul_hours"], 120.0)

    def test_batch_prediction(self):
        batch_res = self.pipeline.predict_batch(self.sample_df)
        self.assertIn("ai_failure_probability", batch_res.columns)
        self.assertIn("ai_predicted_failure", batch_res.columns)
        self.assertIn("ai_predicted_mode", batch_res.columns)
        self.assertIn("ai_predicted_rul_hours", batch_res.columns)
        self.assertIn("ai_health_score", batch_res.columns)
        self.assertEqual(len(batch_res), 20)

    def test_vibration_simulator(self):
        t, sig, freqs, amps, env_f, env_a, meta = generate_vibration_signal(
            condition="Bearing Fault (BPFO / Impact Ringing)",
            duration_seconds=0.2,
            sampling_rate=4000,
            rpm=1800.0
        )
        self.assertEqual(len(t), 800)
        self.assertEqual(len(sig), 800)
        self.assertGreater(meta["kurtosis"], 3.2)
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(env_f), 0)

    def test_iso_severity(self):
        iso_good = get_iso_vibration_severity(0.8)
        self.assertEqual(iso_good["severity_score"], 1)
        iso_crit = get_iso_vibration_severity(8.5)
        self.assertEqual(iso_crit["severity_score"], 4)


if __name__ == "__main__":
    unittest.main()

"""
Model Training Script for AI-Based Predictive Maintenance
Loads dataset, trains models, evaluates metrics, and serializes the pipeline.
"""

import sys
import os
import json
import time

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_loader import get_dataset
from src.models import PredictiveMaintenancePipeline


def main():
    print("=" * 70)
    print("🚀 AI-BASED PREDICTIVE MAINTENANCE MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Load / Generate dataset
    print("\n[1/4] Loading & Generating Industrial Vibration Telemetry Dataset...")
    df = get_dataset(force_recreate=True)
    print(f"  ✓ Loaded {len(df):,} records with {len(df.columns)} columns across {df['machine_id'].nunique()} machines.")
    print(f"  ✓ Failure risk positive class ratio: {df['failure_risk_target'].mean()*100:.1f}%")
    print(f"  ✓ Failure mode distribution:\n{df['failure_mode'].value_counts().to_string(header=False)}")
    
    # 2. Initialize and Train Pipeline
    print("\n[2/4] Initializing and Training ML Pipelines...")
    pipeline = PredictiveMaintenancePipeline()
    metrics = pipeline.train(df, test_size=0.2, random_state=42)
    
    # 3. Print Evaluation Metrics
    print("\n[3/4] Model Evaluation Metrics:")
    print("-" * 50)
    risk_m = metrics["binary_risk"]
    print("📊 [Binary Failure Classifier]")
    print(f"   - Accuracy  : {risk_m['accuracy']*100:.2f}%")
    print(f"   - Precision : {risk_m['precision']*100:.2f}%")
    print(f"   - Recall    : {risk_m['recall']*100:.2f}%")
    print(f"   - F1-Score  : {risk_m['f1_score']:.4f}")
    print(f"   - ROC-AUC   : {risk_m['roc_auc']:.4f}")
    print(f"   - Confusion Matrix: {risk_m['confusion_matrix']}")
    
    mode_m = metrics["failure_mode"]
    print("\n🔍 [Multi-Class Failure Mode Classifier]")
    print(f"   - Accuracy  : {mode_m['accuracy']*100:.2f}%")
    print(f"   - Classes   : {mode_m['classes']}")
    
    rul_m = metrics["rul_regression"]
    print("\n⏳ [Remaining Useful Life (RUL) Regressor]")
    print(f"   - R² Score  : {rul_m['r2_score']:.4f}")
    print(f"   - MAE       : {rul_m['mae_hours']:.2f} hours")
    print(f"   - RMSE      : {rul_m['rmse_hours']:.2f} hours")
    
    print("\n⭐ [Top 5 Influential Vibration & Sensor Features]")
    sorted_feat = sorted(metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_feat[:5]:
        print(f"   - {feat:<24}: {imp:.4f}")
    
    # 4. Save Model Artifacts
    print("\n[4/4] Serializing Trained Pipeline to Disk...")
    pipeline.save()
    elapsed = time.time() - start_time
    print(f"  ✓ Saved to 'models/predictive_maintenance_pipeline.joblib'")
    print(f"  ✓ Training completed in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()

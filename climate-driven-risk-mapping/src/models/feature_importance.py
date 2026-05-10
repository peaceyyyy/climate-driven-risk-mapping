import argparse
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/xgboost_tick_density.pkl')
    parser.add_argument('--data', type=str, default='data/engineered_features.csv')
    parser.add_argument('--output', type=str, default='models/feature_importance_shap.png')
    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.data):
        print("Model or data file missing.")
        return

    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    
    # Check what features the model expects
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        missing_cols = [c for c in expected_features if c not in df.columns]
        if missing_cols:
            print(f"Missing columns required by model: {missing_cols}")
            for c in missing_cols:
                df[c] = 0.0
        X = df[expected_features]
    else:
        cols_to_drop = ['log_density', 'density', 'estimatedTotalCount', 'siteID', 'year_month', 'taxonID', 'collectDate', 'domainID']
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        X = df.drop(columns=cols_to_drop)

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance: XGBoost Tick Density Model", fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"SHAP feature importance plot saved to {args.output}")
    
    # Calculate mean absolute SHAP values for ranking output
    if len(X) > 0:
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame(list(zip(X.columns, mean_shap)), columns=['Feature', 'SHAP_Importance'])
        feature_importance = feature_importance.sort_values('SHAP_Importance', ascending=False)
        
        print("\nFeature Importance Ranking:")
        print(feature_importance.head(10))
        
        # Append to report
        report_path = 'reports/model_evaluation.md'
        if os.path.exists(report_path):
            with open(report_path, 'a') as f:
                f.write("\n## Feature Importance (Top 10)\n")
                f.write(feature_importance.head(10).to_markdown(index=False))

if __name__ == "__main__":
    main()

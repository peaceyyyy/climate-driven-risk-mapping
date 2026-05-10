import argparse
import pandas as pd
import numpy as np
import os

def engineer_features(input_file, output_file):
    print(f"Loading data from {input_file} for feature engineering...")
    df = pd.read_csv(input_file)
    if df.empty:
        print("Data is empty.")
        return

    # Extract month
    df['year_month_dt'] = pd.to_datetime(df['year_month'], format='%Y-%m', errors='coerce')
    df['month'] = df['year_month_dt'].dt.month
    
    # Calculate Vapor Pressure Deficit (VPD)
    # Saturation Vapor Pressure (es) = 0.6108 * exp(17.27 * T / (T + 237.3))
    # Actual Vapor Pressure (ea) = RH / 100 * es
    # VPD = es - ea
    if 'T2M' in df.columns and 'RH2M' in df.columns:
        T = df['T2M']
        RH = df['RH2M']
        es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
        ea = (RH / 100.0) * es
        df['VPD'] = es - ea
    else:
        df['VPD'] = 0.0

    # Sort by site and date for rolling calculations
    df = df.sort_values(by=['siteID', 'year_month_dt'])

    # Lagged variables (simulated by rolling means due to monthly aggregation)
    # Using window=1, 2, 3 for approx 30, 60, 90 days.
    cols_to_lag = ['T2M', 'soil_moisture', 'NDVI']
    for col in cols_to_lag:
        if col in df.columns:
            df[f'{col}_lag30'] = df.groupby('siteID')[col].shift(1).fillna(df[col])
            df[f'{col}_lag60'] = df.groupby('siteID')[col].shift(2).fillna(df[col])
            df[f'{col}_lag90'] = df.groupby('siteID')[col].shift(3).fillna(df[col])
            
            # Rolling means
            df[f'{col}_roll30'] = df.groupby('siteID')[col].transform(lambda x: x.rolling(1, min_periods=1).mean())
            df[f'{col}_roll60'] = df.groupby('siteID')[col].transform(lambda x: x.rolling(2, min_periods=1).mean())
            df[f'{col}_roll90'] = df.groupby('siteID')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Winter minimum LST (prior Dec-Feb) as a survival proxy.
    # We will simulate this by keeping a rolling min of T2M over the past year.
    if 'T2M' in df.columns:
        df['winter_min_LST_proxy'] = df.groupby('siteID')['T2M'].transform(lambda x: x.rolling(12, min_periods=1).min())
    else:
        df['winter_min_LST_proxy'] = 0.0

    # Seasonal Indicators
    # 1: Winter (DJF), 2: Spring (MAM), 3: Summer (JJA), 4: Fall (SON)
    df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4)))
    df = pd.get_dummies(df, columns=['season'], prefix='season')
    
    # Target transformation: log1p(estimatedTotalCount / plot_area_km2)
    # Assuming plot_area_km2 is constant if not provided. We'll use 1.0 for simplicity.
    plot_area_km2 = 1.0
    df['density'] = df['estimatedTotalCount'] / plot_area_km2
    df['log_density'] = np.log1p(df['density'])

    # Clean up temporary column
    df = df.drop(columns=['year_month_dt'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Engineered features saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/joined_data.csv')
    parser.add_argument('--output', type=str, default='data/engineered_features.csv')
    args = parser.parse_args()
    engineer_features(args.input, args.output)

if __name__ == "__main__":
    main()

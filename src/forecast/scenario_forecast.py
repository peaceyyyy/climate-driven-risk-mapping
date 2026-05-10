import argparse
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
import joblib
import os

def create_mock_grid_features(model_features, num_cells=10000):
    df = pd.DataFrame(index=range(num_cells))
    for col in model_features:
        if 'Latitude' in col:
            df[col] = np.random.uniform(25.0, 49.0, num_cells)
        elif 'Longitude' in col:
            df[col] = np.random.uniform(-125.0, -67.0, num_cells)
        else:
            df[col] = np.random.uniform(0, 1, num_cells)
    return df

def apply_climate_shift(df, scenario='SSP2-4.5', year=2030):
    """
    Apply mocked delta shifts to temp and precip.
    SSP2-4.5 2030: +1.5C, +5% precip
    SSP5-8.5 2050: +3.0C, +10% precip
    """
    shifted = df.copy()
    if scenario == 'SSP2-4.5' and year == 2030:
        if 'T2M' in shifted.columns: shifted['T2M'] += 1.5
        if 'PRECTOTCORR' in shifted.columns: shifted['PRECTOTCORR'] *= 1.05
    elif scenario == 'SSP5-8.5' and year == 2050:
        if 'T2M' in shifted.columns: shifted['T2M'] += 3.0
        if 'PRECTOTCORR' in shifted.columns: shifted['PRECTOTCORR'] *= 1.10
    return shifted

def predict_and_export(model, grid_df, output_path, grid_size=100):
    preds_log = model.predict(grid_df)
    preds_density = np.expm1(preds_log)
    preds_density = np.maximum(preds_density, 0)
    
    density_grid = preds_density.reshape((grid_size, grid_size))
    transform = from_origin(-125.0, 49.0, (125-67)/grid_size, (49-25)/grid_size)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=grid_size,
        width=grid_size,
        count=1,
        dtype=density_grid.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(density_grid, 1)
        
    print(f"Exported prediction raster to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/xgboost_tick_density.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.model):
        print(f"Model {args.model} not found.")
        return
        
    model = joblib.load(args.model)
    expected_features = getattr(model, "feature_names_in_", None)
    
    grid_size = 100
    base_grid = create_mock_grid_features(expected_features, grid_size*grid_size)
    
    # 2030 Scenario (SSP2-4.5)
    print("Forecasting 2030 (SSP2-4.5)...")
    grid_2030 = apply_climate_shift(base_grid, 'SSP2-4.5', 2030)
    predict_and_export(model, grid_2030, os.path.join(args.output_dir, 'expansion_2030.tif'), grid_size)
    
    # 2050 Scenario (SSP5-8.5)
    print("Forecasting 2050 (SSP5-8.5)...")
    grid_2050 = apply_climate_shift(base_grid, 'SSP5-8.5', 2050)
    predict_and_export(model, grid_2050, os.path.join(args.output_dir, 'expansion_2050.tif'), grid_size)

if __name__ == "__main__":
    main()

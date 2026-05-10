import argparse
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
import joblib
import os

def create_mock_grid_features(model_features, num_cells=10000):
    # Create a simple synthetic feature set matching expected columns to simulate a US grid
    df = pd.DataFrame(index=range(num_cells))
    for col in model_features:
        if 'Latitude' in col:
            df[col] = np.random.uniform(25.0, 49.0, num_cells)
        elif 'Longitude' in col:
            df[col] = np.random.uniform(-125.0, -67.0, num_cells)
        else:
            df[col] = np.random.uniform(0, 1, num_cells)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/xgboost_tick_density.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs/monthly_risk_maps')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.model):
        print(f"Model {args.model} not found.")
        return
        
    model = joblib.load(args.model)
    expected_features = getattr(model, "feature_names_in_", None)
    
    if expected_features is None:
        print("Model has no feature_names_in_ attribute.")
        return
        
    print("Generating prediction grid...")
    # Simulate a 100x100 grid
    grid_size = 100
    num_cells = grid_size * grid_size
    grid_df = create_mock_grid_features(expected_features, num_cells)
    
    # Let's say we do this for a specific summer month, e.g., July
    if 'month' in grid_df.columns:
        grid_df['month'] = 7
        
    preds_log = model.predict(grid_df)
    preds_density = np.expm1(preds_log) # reverse log1p
    preds_density = np.maximum(preds_density, 0)
    
    # Reshape to grid
    density_grid = preds_density.reshape((grid_size, grid_size))
    
    # Create a mock GeoTIFF over US extent
    # Longitude -125 to -67, Latitude 49 to 25
    transform = from_origin(-125.0, 49.0, (125-67)/grid_size, (49-25)/grid_size)
    
    tif_path = os.path.join(args.output_dir, 'risk_map_july.tif')
    with rasterio.open(
        tif_path,
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
        
    print(f"Exported prediction raster to {tif_path}")

if __name__ == "__main__":
    main()

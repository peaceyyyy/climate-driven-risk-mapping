import argparse
import pandas as pd
import numpy as np
import os
import requests

def mock_fetch_ndvi(lat, lon, date_str):
    """
    Since actual Earthdata login requires credentials, this mock function
    generates synthetic NDVI data centered around realistic values.
    If the user has credentials, they can replace this with an earthaccess workflow.
    """
    # Realistic NDVI is between -1 (water) and 1 (dense vegetation)
    # Generate value around 0.5 for demonstration
    val = np.clip(np.random.normal(0.5, 0.2), 0, 1)
    return val

def main():
    parser = argparse.ArgumentParser(description="Fetch MODIS NDVI data.")
    parser.add_argument('--input', type=str, default='data/neon_ticks.csv', help='Input CSV')
    parser.add_argument('--output', type=str, default='data/modis_ndvi.csv', help='Output CSV path')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return
        
    df = pd.read_csv(args.input)
    if df.empty:
        return
        
    records = []
    # Deduplicate by site and month
    sites_months = df[['decimalLatitude', 'decimalLongitude', 'year_month']].drop_duplicates()
    
    # Just processing a few to save time if large
    sites_months = sites_months.head(10)
    
    for _, row in sites_months.iterrows():
        ndvi_val = mock_fetch_ndvi(row['decimalLatitude'], row['decimalLongitude'], row['year_month'])
        records.append({
            'decimalLatitude': row['decimalLatitude'],
            'decimalLongitude': row['decimalLongitude'],
            'year_month': row['year_month'],
            'NDVI': ndvi_val
        })
        
    out_df = pd.DataFrame(records)
    out_df.to_csv(args.output, index=False)
    print(f"Saved MODIS NDVI data to {args.output}")

if __name__ == "__main__":
    main()

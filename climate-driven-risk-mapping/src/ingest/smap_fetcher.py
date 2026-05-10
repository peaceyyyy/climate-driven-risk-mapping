import argparse
import pandas as pd
import numpy as np
import os

def mock_fetch_smap(lat, lon, date_str):
    """
    Mock function to simulate downloading SMAP soil moisture data.
    """
    # Simulate soil moisture (cm3/cm3) roughly between 0.05 and 0.4
    val = np.clip(np.random.normal(0.2, 0.1), 0.01, 0.6)
    return val

def mock_fetch_gedi(lat, lon):
    """
    Mock function to simulate GEDI canopy cover extraction.
    """
    # Simulate canopy cover (fraction 0-1)
    val = np.clip(np.random.normal(0.4, 0.2), 0, 1)
    return val

def main():
    parser = argparse.ArgumentParser(description="Fetch SMAP Soil Moisture and GEDI Canopy Cover.")
    parser.add_argument('--input', type=str, default='data/neon_ticks.csv', help='Input CSV')
    parser.add_argument('--output', type=str, default='data/smap_gedi.csv', help='Output CSV path')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return
        
    df = pd.read_csv(args.input)
    if df.empty:
        return
        
    records = []
    sites_months = df[['decimalLatitude', 'decimalLongitude', 'year_month']].drop_duplicates()
    
    # process subset
    sites_months = sites_months.head(10)
    
    for _, row in sites_months.iterrows():
        sm_val = mock_fetch_smap(row['decimalLatitude'], row['decimalLongitude'], row['year_month'])
        gedi_val = mock_fetch_gedi(row['decimalLatitude'], row['decimalLongitude'])
        records.append({
            'decimalLatitude': row['decimalLatitude'],
            'decimalLongitude': row['decimalLongitude'],
            'year_month': row['year_month'],
            'soil_moisture': sm_val,
            'canopy_cover': gedi_val
        })
        
    out_df = pd.DataFrame(records)
    out_df.to_csv(args.output, index=False)
    print(f"Saved SMAP and GEDI data to {args.output}")

if __name__ == "__main__":
    main()

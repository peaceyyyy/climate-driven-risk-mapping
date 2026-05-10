import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

def spatial_temporal_join(ticks_file, power_file, ndvi_file, smap_file, output_file):
    print("Loading data for joins...")
    ticks_df = pd.read_csv(ticks_file)
    power_df = pd.read_parquet(power_file) if os.path.exists(power_file) else pd.DataFrame()
    ndvi_df = pd.read_csv(ndvi_file) if os.path.exists(ndvi_file) else pd.DataFrame()
    smap_df = pd.read_csv(smap_file) if os.path.exists(smap_file) else pd.DataFrame()
    
    if ticks_df.empty:
        print("No tick data.")
        return
        
    # Standardize coordinate types
    ticks_df['decimalLatitude'] = ticks_df['decimalLatitude'].astype(float)
    ticks_df['decimalLongitude'] = ticks_df['decimalLongitude'].astype(float)
    
    # Merge NDVI
    if not ndvi_df.empty:
        ticks_df = pd.merge(ticks_df, ndvi_df, on=['decimalLatitude', 'decimalLongitude', 'year_month'], how='left')
    else:
        ticks_df['NDVI'] = 0.5 # fallback

    # Merge SMAP/GEDI
    if not smap_df.empty:
        ticks_df = pd.merge(ticks_df, smap_df, on=['decimalLatitude', 'decimalLongitude', 'year_month'], how='left')
    else:
        ticks_df['soil_moisture'] = 0.2
        ticks_df['canopy_cover'] = 0.4
        
    # For Power, we have daily data. Let's aggregate it to monthly to match tick data
    if not power_df.empty:
        power_df['date'] = pd.to_datetime(power_df['date'])
        power_df['year_month'] = power_df['date'].dt.to_period('M').astype(str)
        # Aggregate climate predictors per month
        monthly_climate = power_df.groupby(['decimalLatitude', 'decimalLongitude', 'year_month']).agg({
            'T2M': 'mean',
            'RH2M': 'mean',
            'PRECTOTCORR': 'sum',
            'WS2M': 'mean',
            'ALLSKY_SFC_SW_DWN': 'mean'
        }).reset_index()
        ticks_df = pd.merge(ticks_df, monthly_climate, on=['decimalLatitude', 'decimalLongitude', 'year_month'], how='left')
    else:
        # Fallbacks
        for col in ['T2M', 'RH2M', 'PRECTOTCORR', 'WS2M', 'ALLSKY_SFC_SW_DWN']:
            ticks_df[col] = 0.0

    # Fill NaNs resulting from left joins
    ticks_df = ticks_df.fillna(0)
    
    # Create GeoPandas output just to show usage
    geometry = [Point(xy) for xy in zip(ticks_df.decimalLongitude, ticks_df.decimalLatitude)]
    gdf = gpd.GeoDataFrame(ticks_df, geometry=geometry, crs="EPSG:4326")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save back to CSV for easier pipeline passing
    ticks_df.to_csv(output_file, index=False)
    print(f"Joined data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticks', type=str, default='data/neon_ticks.csv')
    parser.add_argument('--power', type=str, default='data/nasa_power_weather.parquet')
    parser.add_argument('--ndvi', type=str, default='data/modis_ndvi.csv')
    parser.add_argument('--smap', type=str, default='data/smap_gedi.csv')
    parser.add_argument('--output', type=str, default='data/joined_data.csv')
    args = parser.parse_args()
    
    spatial_temporal_join(args.ticks, args.power, args.ndvi, args.smap, args.output)

if __name__ == "__main__":
    main()

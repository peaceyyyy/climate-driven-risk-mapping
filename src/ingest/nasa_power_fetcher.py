import argparse
import requests
import pandas as pd
import os
import time

def fetch_nasa_power_for_site(lat, lon, start_date, end_date):
    """
    Fetch daily weather data from NASA POWER API for a given coordinate and date range.
    Parameters: T2M, RH2M, PRECTOTCORR, WS2M, ALLSKY_SFC_SW_DWN
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    print(f"Fetching NASA POWER for lat: {lat}, lon: {lon}, dates: {start_date} to {end_date}")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch data for {lat}, {lon}")
        return pd.DataFrame()
        
    data = response.json()
    if 'properties' not in data or 'parameter' not in data['properties']:
        return pd.DataFrame()
        
    parameters = data['properties']['parameter']
    
    df = pd.DataFrame(parameters)
    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['decimalLatitude'] = lat
    df['decimalLongitude'] = lon
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Download NASA POWER data for site coordinates.")
    parser.add_argument('--input', type=str, default='data/neon_ticks.csv', help='Input CSV with coordinates')
    parser.add_argument('--output', type=str, default='data/nasa_power_weather.parquet', help='Output Parquet path')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return
        
    df_sites = pd.read_csv(args.input)
    if df_sites.empty:
        print("No sites to process.")
        return
        
    sites = df_sites[['decimalLatitude', 'decimalLongitude']].drop_duplicates()
    
    all_weather = []
    
    # Restricting to fewer sites for test speed
    sites = sites.head(5)
    start_date = "20230101"
    end_date = "20231231"
    
    for _, row in sites.iterrows():
        lat = row['decimalLatitude']
        lon = row['decimalLongitude']
        weather_df = fetch_nasa_power_for_site(lat, lon, start_date, end_date)
        if not weather_df.empty:
            all_weather.append(weather_df)
        time.sleep(1) # Rate limit
        
    if all_weather:
        final_df = pd.concat(all_weather, ignore_index=True)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        final_df.to_parquet(args.output, index=False)
        print(f"Saved NASA POWER weather data to {args.output}")
    else:
        print("No weather data retrieved.")

if __name__ == "__main__":
    main()

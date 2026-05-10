import argparse
import requests
import pandas as pd
import os

def resolve_taxon_key(scientific_name="Ixodidae"):
    url = "https://api.gbif.org/v1/species/match"
    params = {"name": scientific_name}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('usageKey', 2190124)
    return 2190124

def fetch_gbif_neon_ticks(limit=1000):
    taxon_key = resolve_taxon_key()
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "taxonKey": taxon_key,
        "country": "US",
        "hasCoordinate": "true",
        "limit": limit
    }
    print(f"Fetching tick data from GBIF: {url} with taxonKey {taxon_key}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    records = []
    for result in data.get('results', []):
        count = result.get('organismQuantity', 1)
        try:
            count = float(count)
        except:
            count = 1.0
            
        records.append({
            'siteID': result.get('datasetKey', 'unknown_site'),
            'collectDate': result.get('eventDate', result.get('lastInterpreted', None)),
            'decimalLatitude': result.get('decimalLatitude', None),
            'decimalLongitude': result.get('decimalLongitude', None),
            'estimatedTotalCount': count,
            'taxonID': result.get('taxonKey', taxon_key)
        })
        
    df = pd.DataFrame(records)
    if df.empty:
        print("No records found.")
        return df

    cols_to_check = ['collectDate', 'decimalLatitude', 'decimalLongitude']
    for col in cols_to_check:
        if col not in df.columns:
            df[col] = None
    
    df = df.dropna(subset=cols_to_check)
    df['collectDate'] = pd.to_datetime(df['collectDate'], utc=True, errors='coerce')
    df = df.dropna(subset=['collectDate'])
    df['year_month'] = df['collectDate'].dt.to_period('M')
    
    agg_df = df.groupby(['siteID', 'year_month', 'decimalLatitude', 'decimalLongitude', 'taxonID']).agg({
        'estimatedTotalCount': 'sum'
    }).reset_index()
    
    agg_df['year_month'] = agg_df['year_month'].astype(str)
    
    return agg_df

def main():
    parser = argparse.ArgumentParser(description="Download NEON Tick data (using GBIF fallback).")
    parser.add_argument('--output', type=str, default='data/neon_ticks.csv', help='Output CSV path')
    parser.add_argument('--limit', type=int, default=1000, help='Number of records to fetch')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = fetch_gbif_neon_ticks(limit=args.limit)
    if not df.empty:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} aggregated tick records to {args.output}")

if __name__ == "__main__":
    main()

import argparse
import folium
import rasterio
import numpy as np
import os
import branca.colormap as cm
from matplotlib.colors import Normalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', type=str, default='outputs/monthly_risk_maps/risk_map_july.tif')
    parser.add_argument('--output', type=str, default='outputs/dashboard.html')
    args = parser.parse_args()

    if not os.path.exists(args.raster):
        print(f"Raster {args.raster} not found.")
        return

    print("Building Folium Dashboard...")
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    with rasterio.open(args.raster) as src:
        bounds = src.bounds
        data = src.read(1)
        
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        
        if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
            vmin, vmax = 0, 1
            
        colormap = cm.LinearColormap(colors=['blue', 'yellow', 'red'], vmin=vmin, vmax=vmax)
        colormap.caption = 'Predicted Tick Density (ticks/km²)'
        m.add_child(colormap)
        
        p95 = np.nanpercentile(data, 95)
        
        # Color mapping function that returns an RGBA tuple
        # folium colormap() returns a hex string, so we need to convert it to rgba tuple
        # or we just let folium handle hex string correctly. 
        # But wait, ImageOverlay expects rgba values from colormap function.
        import matplotlib.colors as mcolors
        
        def color_map_fn(x):
            if np.isnan(x):
                return (0, 0, 0, 0)
            if x >= p95:
                return (1.0, 0.0, 0.0, 0.8)  # Highlight high risk in strong red
            
            # Use matplotlib to map value to (r,g,b,a)
            norm_val = (x - vmin) / (vmax - vmin + 1e-6)
            c = mcolors.to_rgba(mcolors.hsv_to_rgb(((1.0-norm_val)*2/3, 1.0, 1.0)))
            return c
            
        folium.raster_layers.ImageOverlay(
            image=data,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.6,
            colormap=color_map_fn,
            name="July Risk Map"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"Dashboard saved to {args.output}")

if __name__ == "__main__":
    main()

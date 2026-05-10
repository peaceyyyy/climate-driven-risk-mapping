import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

def generate_architecture():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=400)
    ax.axis('off')
    
    # Define node positions
    pos = {
        # Data Sources
        "NEON/GBIF API\n(Tick Counts)": (0, 4),
        "NASA POWER API\n(Climate)": (0, 3),
        "MODIS LP DAAC\n(NDVI)": (0, 2),
        "SMAP / GEDI\n(Soil & Canopy)": (0, 1),
        
        # Ingestion
        "Data Ingestion Layer\n(src/ingest/)": (1.5, 2.5),
        
        # Feature Eng
        "Spatial Join &\nFeature Engineering\n(src/features/)": (3.5, 2.5),
        
        # Modeling
        "XGBoost Regression Model\n(src/models/)": (5.5, 2.5),
        
        # Output Maps & Forecast
        "Spatial Mapping\n(GeoTIFFs)\n(src/mapping/)": (7.5, 3.5),
        "Folium Heatmap\nDashboard": (7.5, 2.5),
        "Climate Scenario\nForecasting (SSP2/5)\n(src/forecast/)": (7.5, 1.5),
        
        "Outputs\n(.tif, .html, .pkl)": (9.5, 2.5)
    }

    # Define explicit graph
    G = nx.DiGraph()
    edges = [
        ("NEON/GBIF API\n(Tick Counts)", "Data Ingestion Layer\n(src/ingest/)"),
        ("NASA POWER API\n(Climate)", "Data Ingestion Layer\n(src/ingest/)"),
        ("MODIS LP DAAC\n(NDVI)", "Data Ingestion Layer\n(src/ingest/)"),
        ("SMAP / GEDI\n(Soil & Canopy)", "Data Ingestion Layer\n(src/ingest/)"),
        
        ("Data Ingestion Layer\n(src/ingest/)", "Spatial Join &\nFeature Engineering\n(src/features/)"),
        ("Spatial Join &\nFeature Engineering\n(src/features/)", "XGBoost Regression Model\n(src/models/)"),
        
        ("XGBoost Regression Model\n(src/models/)", "Spatial Mapping\n(GeoTIFFs)\n(src/mapping/)"),
        ("XGBoost Regression Model\n(src/models/)", "Folium Heatmap\nDashboard"),
        ("XGBoost Regression Model\n(src/models/)", "Climate Scenario\nForecasting (SSP2/5)\n(src/forecast/)"),
        
        ("Spatial Mapping\n(GeoTIFFs)\n(src/mapping/)", "Outputs\n(.tif, .html, .pkl)"),
        ("Folium Heatmap\nDashboard", "Outputs\n(.tif, .html, .pkl)"),
        ("Climate Scenario\nForecasting (SSP2/5)\n(src/forecast/)", "Outputs\n(.tif, .html, .pkl)"),
    ]
    G.add_edges_from(edges)

    # Draw arrows manually for solid primary flow
    for edge in G.edges():
        start = pos[edge[0]]
        end = pos[edge[1]]
        ax.annotate("",
                    xy=end, xycoords='data',
                    xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", lw=2.5, shrinkA=35, shrinkB=35))
                    
    # Draw Nodes as rounded rectangles (fancybox)
    for node, (x, y) in pos.items():
        if "API" in node or "MODIS" in node or "SMAP" in node:
            color = "#D9EAD3"  # light green
        elif "Output" in node or "Dashboard" in node or "GeoTIFF" in node:
            color = "#FCE5CD"  # light orange
        else:
            color = "#CFE2F3"  # light blue
            
        box = mpatches.FancyBboxPatch((x - 0.7, y - 0.3), 1.4, 0.6,
                                      boxstyle="round,pad=0.1,rounding_size=0.1",
                                      ec="black", fc=color, lw=2.0)
        ax.add_patch(box)
        ax.text(x, y, node, ha="center", va="center", fontsize=11, fontweight='bold', zorder=10)

    plt.title("Climate-Driven Tick Density Prediction Pipeline Architecture", fontsize=18, fontweight='bold', pad=20)
    plt.xlim(-1.5, 10.5)
    plt.ylim(0, 5)
    
    plt.savefig("system_architecture.png", bbox_inches='tight')
    print("Generated system_architecture.png")

if __name__ == "__main__":
    generate_architecture()

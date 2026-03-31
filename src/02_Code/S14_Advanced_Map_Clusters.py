
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

# Set style for publication quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Paths
EXCEL_FILE = "/Users/ishtiaq/Desktop/Chater 1 papers/temp/Confirmed Patieints Tier I-II Districts 2013 - 2025.xlsx"
OUTPUT_DIR = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/04_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 11 Cities/Districts
STUDY_CITIES = [
    'Lahore', 'Rawalpindi', 'Faisalabad', 'Gujranwala', 
    'Multan', 'Islamabad', 'Sargodha', 'Sheikhupura', 
    'Dera Ghazi Khan', 'Gujrat', 'Hafizabad'
]

# Load data
print("Loading data...")
cols_to_use = ['District', 'Hospital District', 'Tehsil', 'Entry Date', 'Latitude', 'Longitude']
df = pd.read_excel(EXCEL_FILE, usecols=cols_to_use)

# Standardize names and filter
df['District_Std'] = df['District'].fillna(df['Hospital District']).str.title()
df.loc[df['Tehsil'].str.contains('Isb', case=False, na=False), 'District_Std'] = 'Islamabad'
df.loc[df['Hospital District'] == 'Islamabad', 'District_Std'] = 'Islamabad'

df_filtered = df[df['District_Std'].isin(STUDY_CITIES)].copy()

# Date processing
df_filtered['Date'] = pd.to_datetime(df_filtered['Entry Date'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['Date', 'Latitude', 'Longitude'])
df_filtered['Year'] = df_filtered['Date'].dt.year

# Filter for 2013-2024
df_filtered = df_filtered[(df_filtered['Year'] >= 2013) & (df_filtered['Year'] <= 2024)]

def plot_advanced_city_clusters(city_name, ax, cmap_name):
    print(f"Generating advanced cluster map for {city_name}...")
    city_data = df_filtered[df_filtered['District_Std'] == city_name].copy()
    
    # Tighter spatial filtering for better zoom
    if city_name == 'Lahore':
        city_data = city_data[(city_data['Latitude'] > 31.35) & (city_data['Latitude'] < 31.65) &
                             (city_data['Longitude'] > 74.15) & (city_data['Longitude'] < 74.55)]
    elif city_name == 'Rawalpindi':
        # Focus on Rawalpindi core city areas
        city_data = city_data[(city_data['Latitude'] > 33.50) & (city_data['Latitude'] < 33.70) &
                             (city_data['Longitude'] > 72.95) & (city_data['Longitude'] < 73.20)]

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(city_data['Longitude'], city_data['Latitude'])]
    gdf = gpd.GeoDataFrame(city_data, geometry=geometry, crs="EPSG:4326")
    
    # Reproject to Web Mercator for contextily
    gdf_web = gdf.to_crs(epsg=3857)
    
    # Plotting the density (KDE) on the reprojected coordinates
    sns.kdeplot(
        x=gdf_web.geometry.x, y=gdf_web.geometry.y,
        cmap=cmap_name, fill=True, alpha=0.6, ax=ax, thresh=0.01, levels=15, zorder=2
    )
    
    # Add individual points with very low alpha
    ax.scatter(gdf_web.geometry.x, gdf_web.geometry.y, s=0.5, color='black', alpha=0.05, zorder=3)
    
    # Add base map using contextily - Using CartoDB Positron for clean English labels
    try:
        # CartoDB.Positron or CartoDB.Voyager are high quality and English-first
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager, zorder=1)
    except Exception as e:
        print(f"Warning: Could not add basemap for {city_name}: {e}")
        pass

    ax.set_title(f'Spatiotemporal Transmission Hotspots: {city_name}', fontsize=16, fontweight='bold', pad=15)
    ax.set_axis_off()

# Create separate figures for each city as requested
# 1. Lahore
fig_lhr, ax_lhr = plt.subplots(figsize=(12, 12))
plot_advanced_city_clusters('Lahore', ax_lhr, "Reds")
plt.tight_layout()
fig_lhr_path = os.path.join(OUTPUT_DIR, 'Fig_Advanced_Clusters_Lahore.png')
fig_lhr.savefig(fig_lhr_path, dpi=300, bbox_inches='tight')
plt.close(fig_lhr)

# 2. Rawalpindi
fig_rwp, ax_rwp = plt.subplots(figsize=(12, 12))
plot_advanced_city_clusters('Rawalpindi', ax_rwp, "Oranges")
plt.tight_layout()
fig_rwp_path = os.path.join(OUTPUT_DIR, 'Fig_Advanced_Clusters_Rawalpindi.png')
fig_rwp.savefig(fig_rwp_path, dpi=300, bbox_inches='tight')
plt.close(fig_rwp)

print(f"\nAdvanced cluster maps saved to: {OUTPUT_DIR}")
print(f"1. Lahore: Fig_Advanced_Clusters_Lahore.png")
print(f"2. Rawalpindi: Fig_Advanced_Clusters_Rawalpindi.png")

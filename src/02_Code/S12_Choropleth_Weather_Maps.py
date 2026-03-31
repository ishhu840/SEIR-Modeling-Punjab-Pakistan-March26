
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import os

# Professional plotting style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Paths
GEOJSON_FILE = "/Users/ishtiaq/Desktop/Article Coding/Dengue_Mathematical_Modeling_Thesis/Data/pak_districts_detailed.geojson"
WEATHER_FILE = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/01_Data/D1_Weekly_Cases_Weather_AllCities.xlsx"
OUTPUT_DIR = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/04_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Study cities mapping for GeoJSON
STUDY_CITIES_GEO = {
    'Lahore': 'Lahore', 'Faisalabad': 'Faisalabad', 'Rawalpindi': 'Rawalpindi',
    'Gujranwala': 'Gujranwala', 'Multan': 'Multan', 'Islamabad': 'Islamabad',
    'Sargodha': 'Sargodha', 'Sheikhupura': 'Sheikhupura', 'Gujrat': 'Gujrat',
    'Hafizabad': 'Hafizabad', 'Dera Ghazi Khan': 'Dera Ghazi Khan'
}

print("Loading data...")
gdf = gpd.read_file(GEOJSON_FILE)
df_weather = pd.read_excel(WEATHER_FILE)

# Calculate long-term average weather per city
weather_summary = df_weather.groupby('City').agg({
    'Temperature': 'mean',
    'Rainfall': 'mean'
}).reset_index()

# Merge weather data into GeoJSON
gdf['temp'] = gdf['NAME_3'].map(weather_summary.set_index('City')['Temperature'])
gdf['rain'] = gdf['NAME_3'].map(weather_summary.set_index('City')['Rainfall'])
gdf['is_study'] = gdf['NAME_3'].isin(STUDY_CITIES_GEO.values())

# Define zones for Punjab focus
punjab_gdf = gdf[gdf['NAME_1'] == 'Punjab'].copy()
# Note: Islamabad is often a separate territory but usually grouped with Punjab in these analyses
islamabad_gdf = gdf[gdf['NAME_1'] == 'Islamabad'].copy()
focus_gdf = pd.concat([punjab_gdf, islamabad_gdf])

def plot_weather_choropleth(metric, cmap, title, filename, unit, vmin, vmax):
    fig, ax = plt.subplots(figsize=(12, 14), facecolor='white')
    
    # Plot base map (all districts in light gray)
    gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)
    
    # Plot Punjab focus with color scale
    # Districts not in study are light gray but with boundaries
    focus_gdf.plot(ax=ax, color='#e8e8e8', edgecolor='#bdbdbd', linewidth=0.8)
    
    # Plot study cities with weather data
    study_gdf = focus_gdf[focus_gdf['is_study']].copy()
    study_gdf.plot(ax=ax, column=metric, cmap=cmap, edgecolor='black', 
                  linewidth=2, legend=False, vmin=vmin, vmax=vmax)
    
    # Add a clean colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label(f'Average {metric} ({unit})', fontsize=14, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    # Add city labels - Semi-transparent white boxes with rounded corners for better readability
    for _, row in study_gdf.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(text=f"{row['NAME_3']}\n{row[metric]:.1f}{unit}", 
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.6))

    # Map styling - Focus on Punjab/Islamabad area
    ax.set_xlim(69, 76)
    ax.set_ylim(28, 35)
    
    ax.set_title(title, fontsize=20, fontweight='bold', pad=30)
    ax.set_xlabel('Longitude (°E)', fontsize=14, labelpad=10)
    ax.set_ylabel('Latitude (°N)', fontsize=14, labelpad=10)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

# Define reasonable ranges for the color scales
t_min, t_max = weather_summary['Temperature'].min() - 1, weather_summary['Temperature'].max() + 1
r_min, r_max = weather_summary['Rainfall'].min() - 5, weather_summary['Rainfall'].max() + 5

# Generate the two maps in "News Weather Forecast" style
plot_weather_choropleth('temp', 'YlOrRd', 
                        'Regional Temperature Gradient: Punjab Study Cities', 
                        'Fig_Weather_Choropleth_Temp.png', '°C', t_min, t_max)

plot_weather_choropleth('rain', 'Blues', 
                        'Regional Rainfall Distribution: Punjab Study Cities', 
                        'Fig_Weather_Choropleth_Rain.png', 'mm', r_min, r_max)

print("\nChoropleth weather maps generated successfully.")

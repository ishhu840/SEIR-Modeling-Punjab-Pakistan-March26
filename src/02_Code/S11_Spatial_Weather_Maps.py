
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Professional plotting style
sns.set_theme(style="white", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Paths
DATA_FILE = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/01_Data/D1_Weekly_Cases_Weather_AllCities.xlsx"
OUTPUT_DIR = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/04_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Study Cities and their approximate Zones (as described)
# Zone 1: North (Islamabad, Rawalpindi, Gujrat, Hafizabad)
# Zone 2: Central (Lahore, Faisalabad, Gujranwala, Sargodha, Sheikhupura)
# Zone 3: South (Multan, Dera Ghazi Khan)
CITY_ZONES = {
    'Islamabad': 'North Punjab', 'Rawalpindi': 'North Punjab', 
    'Gujrat': 'North Punjab', 'Hafizabad': 'North Punjab',
    'Lahore': 'Central Punjab', 'Faisalabad': 'Central Punjab', 
    'Gujranwala': 'Central Punjab', 'Sargodha': 'Central Punjab', 
    'Sheikhupura': 'Central Punjab',
    'Multan': 'South Punjab', 'Dera Ghazi Khan': 'South Punjab'
}

# Approximate coordinates for the 11 cities to plot on a map
CITY_COORDS = {
    'Islamabad': (33.6844, 73.0479), 'Rawalpindi': (33.5651, 73.0169),
    'Gujrat': (32.5742, 74.0754), 'Hafizabad': (32.0709, 73.6853),
    'Lahore': (31.5204, 74.3587), 'Faisalabad': (31.4504, 73.1350),
    'Gujranwala': (32.1877, 74.1945), 'Sargodha': (32.0740, 72.6861),
    'Sheikhupura': (31.7131, 73.9783),
    'Multan': (30.1575, 71.5249), 'Dera Ghazi Khan': (30.0489, 70.6403)
}

print("Loading and processing weather data...")
df = pd.read_excel(DATA_FILE)

# Calculate long-term average weather per city
weather_summary = df.groupby('City').agg({
    'Temperature': 'mean',
    'Rainfall': 'mean'
}).reset_index()

# Add coordinates and zones
weather_summary['Lat'] = weather_summary['City'].map(lambda x: CITY_COORDS[x][0])
weather_summary['Lon'] = weather_summary['City'].map(lambda x: CITY_COORDS[x][1])
weather_summary['Zone'] = weather_summary['City'].map(CITY_ZONES)

def plot_spatial_weather(metric, cmap, title, filename, unit):
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Sort by latitude for better zone coloring
    sorted_df = weather_summary.sort_values('Lat', ascending=False)
    
    # Plot points with size and color based on metric
    scatter = ax.scatter(sorted_df['Lon'], sorted_df['Lat'], 
                        c=sorted_df[metric], cmap=cmap, 
                        s=sorted_df[metric]*15 if metric=='Temperature' else sorted_df[metric]*50,
                        edgecolor='black', linewidth=1.5, alpha=0.85, zorder=3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f'Average {metric} ({unit})', fontsize=12, fontweight='bold')
    
    # Add city labels and zone indicators
    for _, row in sorted_df.iterrows():
        ax.text(row['Lon']+0.15, row['Lat'], f"{row['City']}\n({row[metric]:.1f}{unit})", 
                fontsize=10, fontweight='bold', va='center')
    
    # Highlight Zones with background shading or labels
    zones = sorted_df['Zone'].unique()
    colors = ['#D5F5E3', '#FCF3CF', '#FADBD8'] # Light green, yellow, red for N, C, S
    for i, zone in enumerate(zones):
        zone_data = sorted_df[sorted_df['Zone'] == zone]
        ax.text(sorted_df['Lon'].min()-1.5, zone_data['Lat'].mean(), zone, 
                fontsize=14, fontweight='bold', rotation=90, va='center', 
                color=sns.color_palette("dark")[i])

    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Longitude (°E)', fontsize=14)
    ax.set_ylabel('Latitude (°N)', fontsize=14)
    
    # Expand limits for labels
    ax.set_xlim(sorted_df['Lon'].min()-2, sorted_df['Lon'].max()+2)
    ax.set_ylim(sorted_df['Lat'].min()-1, sorted_df['Lat'].max()+1)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

# Generate the two maps
plot_spatial_weather('Temperature', 'YlOrRd', 
                     'Spatial Distribution of Mean Temperature across Study Cities', 
                     'Fig_Spatial_Temperature_Zones.png', '°C')

plot_spatial_weather('Rainfall', 'Blues', 
                     'Spatial Distribution of Mean Weekly Rainfall across Study Cities', 
                     'Fig_Spatial_Rainfall_Zones.png', 'mm')

print("\nSpatial weather analysis complete.")

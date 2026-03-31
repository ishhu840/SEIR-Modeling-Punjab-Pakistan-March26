
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for publication quality figures
sns.set_theme(style="whitegrid", palette="muted")
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
# Loading only necessary columns for efficiency
cols_to_use = [
    'District', 'Hospital District', 'Tehsil', 'Confirmation Date', 'Entry Date',
    'Age', 'Gender', 'Latitude', 'Longitude'
]
df = pd.read_excel(EXCEL_FILE, usecols=cols_to_use)

# Standardize names and filter
df['District_Std'] = df['District'].fillna(df['Hospital District']).str.title()
# Special case for Islamabad which might be in Tehsil or Hospital District
df.loc[df['Tehsil'].str.contains('Isb', case=False, na=False), 'District_Std'] = 'Islamabad'
df.loc[df['Hospital District'] == 'Islamabad', 'District_Std'] = 'Islamabad'

df_filtered = df[df['District_Std'].isin(STUDY_CITIES)].copy()

# Date processing
# Use Entry Date as it covers 2013-2024 (Confirmation Date is mostly 2021+)
df_filtered['Date'] = pd.to_datetime(df_filtered['Entry Date'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['Date'])
df_filtered['Year'] = df_filtered['Date'].dt.year
df_filtered['Month'] = df_filtered['Date'].dt.month

# Filter for 2013-2024
df_filtered = df_filtered[(df_filtered['Year'] >= 2013) & (df_filtered['Year'] <= 2024)]

# Define Monsoon months (July-October based on manuscript)
MONSOON_MONTHS = [7, 8, 9, 10]
df_monsoon = df_filtered[df_filtered['Month'].isin(MONSOON_MONTHS)].copy()

# 1. Yearly Monsoon Case Numbers (2013-2024)
monsoon_counts = df_monsoon.groupby(['Year', 'District_Std']).size().reset_index(name='Cases')

# Mathematical Fix: Fill missing zero-case years for continuity
all_years = range(2013, 2025)
full_index = pd.MultiIndex.from_product([all_years, STUDY_CITIES], names=['Year', 'District_Std'])
monsoon_counts = monsoon_counts.set_index(['Year', 'District_Std']).reindex(full_index, fill_value=0).reset_index()

# For log scale, absolute 0 breaks. We substitute 0 with 0.5 so lines plunge below the "1" mark correctly
monsoon_counts['Cases_Plot'] = monsoon_counts['Cases'].replace(0, 0.5)

plt.figure(figsize=(14, 8))

# Custom color palette defined by user for maximum clarity
CITY_COLORS = {
    'Lahore': '#d84315',          # Dark Orange
    'Faisalabad': '#ff9800',      # Lighter Orange
    'Rawalpindi': '#2e7d32',      # Dark Green
    'Gujranwala': '#66bb6a',      # Lighter Green
    'Multan': '#a5d6a7',          # Softer Green
    'Sargodha': '#1565c0',        # Dark Blue
    'Sheikhupura': '#42a5f5',     # Lighter Blue
    'Dera Ghazi Khan': '#90caf9', # Softer Blue
    'Islamabad': '#8e24aa',       # Purple
    'Hafizabad': '#9e9e9e',       # Gray
    'Gujrat': '#9e9e9e'           # Gray
}

# Plot all cities dynamically
for city in STUDY_CITIES:
    city_data = monsoon_counts[monsoon_counts['District_Std'] == city]
    
    # Uniform thickness and styling for all lines
    plt.plot(city_data['Year'], city_data['Cases_Plot'], 
             label=city, color=CITY_COLORS[city], 
             linewidth=2.5, alpha=0.9, zorder=3, 
             marker='o', markersize=7)

plt.title('Inter-annual Variability of Dengue Cases During Monsoon Season (2013–2024)', 
          fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Total Confirmed Patient Count (Log Scale)', fontsize=14, fontweight='semibold')
plt.xlabel('Epidemiological Year', fontsize=14, fontweight='semibold')
plt.xticks(range(2013, 2025), fontsize=12)

# Set log scale and formatting
plt.yscale('log')
import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

# Set custom Y-ticks including our 0.5 baseline formatted visually as 0
plt.yticks([0.5, 1, 10, 100, 1000, 10000, 100000], ['0', '1', '10', '100', '1000', '10000', '100000'], fontsize=12)
plt.ylim(0.4, 200000) # Give it some breathing room at bottom and top

plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Place legend outside
plt.legend(title='Study Cities', title_fontsize='13', fontsize='12', 
           bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
# Save Figure 1
fig1_path = os.path.join(OUTPUT_DIR, 'Fig_Monsoon_Cases_2013_2024.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()

# 2. Age and Sex Distribution
# Clean Gender
df_filtered['Gender'] = df_filtered['Gender'].replace({'F': 'Female', 'M': 'Male'})
df_filtered = df_filtered[df_filtered['Gender'].isin(['Male', 'Female'])]

# Clean Age
def clean_age(age_str):
    if pd.isna(age_str): return np.nan
    try:
        # Extract numeric part from "50 Years" or similar
        return int(str(age_str).split()[0])
    except:
        return np.nan

df_filtered['Age_Clean'] = df_filtered['Age'].apply(clean_age)
# Filter for Age between 0 and 96 as requested
df_filtered = df_filtered[(df_filtered['Age_Clean'] >= 0) & (df_filtered['Age_Clean'] <= 96)]
df_filtered = df_filtered.dropna(subset=['Age_Clean', 'Gender'])

# Figure 2: Age and Sex Distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Sex Distribution (Pie Chart) - Enhanced colors and style
gender_counts = df_filtered['Gender'].value_counts()
colors = ['#5DADE2', '#EC7063'] # Professional blue and coral
axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=colors, explode=(0.05, 0), shadow=True,
            textprops={'fontsize': 14, 'fontweight': 'bold'})
axes[0].set_title('Gender Distribution of Study Population', fontsize=16, fontweight='bold', pad=20)

# Age Distribution (Histogram - Dodged/Side-by-side bars)
sns.histplot(data=df_filtered, x='Age_Clean', hue='Gender', multiple='dodge', 
             bins=np.arange(0, 101, 10), ax=axes[1], palette=colors, shrink=0.8, alpha=0.8)

axes[1].set_title('Age Stratification of Dengue Patients by Gender', fontsize=16, fontweight='bold', pad=20)
axes[1].set_xlabel('Patient Age (Years)', fontsize=14, fontweight='semibold')
axes[1].set_ylabel('Total Patient Count', fontsize=14, fontweight='semibold')

# Set specific x-ticks as requested: 0, 10, 20, 30...
axes[1].set_xticks(np.arange(0, 101, 10))
axes[1].tick_params(labelsize=12)
axes[1].set_xlim(0, 100)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, 'Fig_Age_Sex_Distribution.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()

# 3. Geo-location Cluster Map for Lahore and Rawalpindi
# Using a higher resolution density visualization
cities_to_map = ['Lahore', 'Rawalpindi']
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

for i, city in enumerate(cities_to_map):
    city_data = df_filtered[(df_filtered['District_Std'] == city) & 
                           (df_filtered['Latitude'].notna()) & 
                           (df_filtered['Longitude'].notna())].copy()
    
    # Precise spatial filtering
    if city == 'Lahore':
        city_data = city_data[(city_data['Latitude'] > 31.3) & (city_data['Latitude'] < 31.7) &
                             (city_data['Longitude'] > 74.1) & (city_data['Longitude'] < 74.6)]
        cmap = "YlOrRd"
    elif city == 'Rawalpindi':
        city_data = city_data[(city_data['Latitude'] > 33.4) & (city_data['Latitude'] < 33.8) &
                             (city_data['Longitude'] > 72.8) & (city_data['Longitude'] < 73.3)]
        cmap = "YlOrBr"

    # Using higher levels for smoother clusters
    sns.kdeplot(data=city_data, x='Longitude', y='Latitude', cmap=cmap, 
                fill=True, alpha=0.7, ax=axes[i], thresh=0.01, levels=15)
    
    # Add individual points with very low alpha to show underlying distribution
    axes[i].scatter(city_data['Longitude'], city_data['Latitude'], s=0.5, 
                    color='black', alpha=0.05, label='Patient Location')
    
    axes[i].set_title(f'Spatiotemporal Transmission Hotspots: {city}', fontsize=18, fontweight='bold', pad=15)
    axes[i].set_xlabel('Longitude (°E)', fontsize=14, fontweight='semibold')
    axes[i].set_ylabel('Latitude (°N)', fontsize=14, fontweight='semibold')
    axes[i].tick_params(labelsize=12)
    axes[i].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'Fig_Geo_Clusters_LHR_RWP.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print("\nAnalysis complete. Figures saved in:", OUTPUT_DIR)
print("\n=== MONSOON CASE NUMBERS (2013-2024) ===")
print(monsoon_counts.pivot(index='Year', columns='District_Std', values='Cases').fillna(0).astype(int))

print("\n=== AGE AND SEX SUMMARY ===")
print(df_filtered.groupby('Gender')['Age_Clean'].agg(['mean', 'median', 'std', 'count']))

print(f"\nFigure Captions:")
print(f"Fig 1: Yearly monsoon season (July–October) dengue case counts for 11 study cities in Punjab, Pakistan from 2013 to 2024. The data illustrates the inter-annual variability and localized nature of epidemics.")
print(f"Fig 2: Demographics of dengue patients across 11 study cities (2013–2024). (Left) Gender distribution showing the proportion of male and female cases. (Right) Age distribution histogram stacked by gender.")
print(f"Fig 3: Spatial density and clustering of confirmed dengue cases in Lahore and Rawalpindi. The heatmaps represent areas of high transmission intensity (hotspots) based on point-location data.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set professional publication style
sns.set_theme(style="white", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Paths
POP_FILE = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/01_Data/D2_Population_2017_2023.xlsx"
CASES_FILE = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/01_Data/D1_Weekly_Cases_Weather_AllCities.xlsx"
OUTPUT_DIR = "/Users/ishtiaq/Desktop/Chater 1 papers/Chapter 3/Dengue_SEIR_V4_Final/04_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df_pop = pd.read_excel(POP_FILE)
df_cases = pd.read_excel(CASES_FILE)

# Process data
cases_total = df_cases.groupby('City')['Number of Dengue Cases'].sum().reset_index()
merged = pd.merge(df_pop, cases_total, on='City')

# Use 2023 Population (convert to millions for clearer axis)
merged['Pop_Million'] = merged['Population 2023'] / 1_000_000
merged['Cases'] = merged['Number of Dengue Cases']

# Create the Bubble Plot
fig, ax = plt.subplots(figsize=(14, 10))

# --- Add Population Zone Highlights (Vertical Bands) ---
# Zone 1: Small Cities (1-3M)
ax.axvspan(1, 3, color='gray', alpha=0.05, label='Tier 1: <3M')
# Zone 2: Mid-Size Cities (3-5M)
ax.axvspan(3, 5, color='blue', alpha=0.05, label='Tier 2: 3-5M')
# Zone 3: Major Cities (5-7M) - Rawalpindi, Multan, Gujranwala
ax.axvspan(5, 7, color='green', alpha=0.08, label='Tier 3: 5-7M')
# Zone 4: Mega Cities (8M+)
ax.axvspan(8, 14, color='orange', alpha=0.05, label='Tier 4: >8M')

# Add labels for these zones at the top
ax.text(2, 85000, "Tier 1\n(<3M)", ha='center', fontsize=10, fontweight='bold', color='gray')
ax.text(4, 85000, "Tier 2\n(3-5M)", ha='center', fontsize=10, fontweight='bold', color='blue', alpha=0.6)
ax.text(6, 85000, "Tier 3\n(5-7M)", ha='center', fontsize=10, fontweight='bold', color='green', alpha=0.6)
ax.text(11, 85000, "Tier 4\n(>8M)", ha='center', fontsize=10, fontweight='bold', color='orange', alpha=0.6)

# Colors and bubble sizes
# Scale bubble sizes based on cases for visual impact, but keep it proportional
sizes = merged['Cases'] * 0.15 + 200 
colors = sns.color_palette("viridis", n_colors=len(merged))

# Scatter plot
scatter = ax.scatter(merged['Pop_Million'], merged['Cases'], 
                    s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidth=1.5)

# Add city labels with smart positioning to avoid overlap
for i, txt in enumerate(merged['City']):
    # Adjust position for Lahore specifically since it's on the edge
    offset_x, offset_y = (10, 10)
    if txt == 'Lahore':
        offset_x, offset_y = (-60, 10)
    
    ax.annotate(txt, (merged['Pop_Million'][i], merged['Cases'][i]), 
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=12, fontweight='bold', alpha=0.9)

# Formatting axes
ax.set_title('Relationship between Urban Population and Cumulative Dengue Burden (2013–2024)', 
             fontsize=18, fontweight='bold', pad=25)
ax.set_xlabel('City Population (Millions)', fontsize=14, fontweight='semibold', labelpad=15)
ax.set_ylabel('Total Cumulative Dengue Cases', fontsize=14, fontweight='semibold', labelpad=15)

# Expand limits so bubbles (like Lahore) aren't cut off
ax.set_xlim(0, 15)
ax.set_ylim(8, 100000)

# Log scale for Y-axis because Lahore and Rawalpindi are outliers
# This makes smaller values (like DG Khan, Gujrat) visible while keeping Lahore/Rawalpindi in view
ax.set_yscale('log')
# Set y-ticks for log scale manually for clarity
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_yticks([10, 50, 100, 500, 1000, 5000, 10000, 50000])

ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.3)

# Move the note to the footer (bottom of the figure)
plt.figtext(0.5, 0.01, "Note: Faisalabad exhibits a high population but a lower cumulative dengue burden compared to Lahore and Rawalpindi,\nreflecting the influence of city-specific environmental factors rather than population scaling alone.", 
            ha="center", fontsize=11, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.3, ec='none'))

plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make room for footer

# Save the figure
save_path = os.path.join(OUTPUT_DIR, 'Fig_Population_vs_Dengue_Bubble.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Professional bubble chart saved to: {save_path}")
print("\n=== DATA SUMMARY FOR BUBBLE CHART ===")
print(merged[['City', 'Pop_Million', 'Cases']].sort_values('Cases', ascending=False).to_string(index=False))

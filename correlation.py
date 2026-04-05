# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your datasets
df_uppsala = pd.read_csv("Cst_to_U.csv", index_col=False, encoding="utf-8")
df_malmo = pd.read_csv("Cst_to_M.csv", index_col=False, encoding="utf-8")

# 2. Ensure date column is datetime
df_uppsala['Datum_PAU'] = pd.to_datetime(df_uppsala['Datum_PAU'])
df_malmo['Datum_PAU'] = pd.to_datetime(df_malmo['Datum_PAU'])

# 3. Data Preparation & Aggregation WITH CLIPPING
# Calculate the daily average delay, but clip negative values to 0 FIRST.
# This ensures a day where trains were early on average is treated as a 0-delay day.

uppsala_daily = (df_uppsala.groupby('Datum_PAU')['AnkFörsening']
                 .apply(lambda x: x.clip(lower=0).mean()) # Clip then mean
                 .reset_index()
                 )
uppsala_daily.rename(columns={'AnkFörsening': 'avg_delay_uppsala'}, inplace=True)

malmo_daily = (df_malmo.groupby('Datum_PAU')['AnkFörsening']
               .apply(lambda x: x.clip(lower=0).mean()) # Clip then mean
               .reset_index()
               )
malmo_daily.rename(columns={'AnkFörsening': 'avg_delay_malmo'}, inplace=True)

# 4. Merge the two datasets on the 'Datum_PAU' key
daily_delays_df = pd.merge(uppsala_daily, malmo_daily, on='Datum_PAU', how='inner')

# ADD THIS: Create day of week information for coloring
daily_delays_df['day_of_week'] = daily_delays_df['Datum_PAU'].dt.dayofweek
daily_delays_df['is_weekday'] = daily_delays_df['day_of_week'] < 5  # Monday-Friday
daily_delays_df['day_type'] = daily_delays_df['is_weekday'].map({True: 'Weekday', False: 'Weekend'})

print(f"Merged data contains {len(daily_delays_df)} days of common data.")
print(f"Weekdays: {len(daily_delays_df[daily_delays_df['is_weekday']])}")
print(f"Weekends: {len(daily_delays_df[~daily_delays_df['is_weekday']])}")

# 5. Run the Spearman Rank Correlation Test
correlation, p_value = spearmanr(daily_delays_df['avg_delay_uppsala'], daily_delays_df['avg_delay_malmo'])

print("\nSpearman Correlation Analysis between Daily Mean DELAYS (Clipped < 0)")
print("----------------------------------------------------------------------")
print(f"Correlation Coefficient (ρ): {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the p-value
if p_value < 0.05:
    print("-> Statistically significant correlation (p < 0.05). Reject the null hypothesis.")
else:
    print("-> No statistically significant correlation (p >= 0.05).")

# 6. Create a scatter plot colored by day type
plt.figure(figsize=(12, 6))

# Create scatter plot with different colors for weekdays vs weekends
scatter = plt.scatter(daily_delays_df['avg_delay_uppsala'], 
                     daily_delays_df['avg_delay_malmo'],
                     c=daily_delays_df['is_weekday'].map({True: 'blue', False: 'red'}),
                     alpha=0.7, s=80)

plt.title('Correlation of Daily Average DELAYS (Clipped < 0)\nStockholm-Uppsala vs. Stockholm-Malmö', 
          fontweight='bold', pad=20)
plt.xlabel('Average Daily Delay - Uppsala Corridor (min)')
plt.ylabel('Average Daily Delay - Malmo Corridor (min)')

# Add the stats to the plot
text = f'Spearman ρ: {correlation:.3f}\np-value: {p_value:.4f}'
plt.gca().text(0.65, 0.95, text, transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add legend for weekdays vs weekends
import matplotlib.patches as mpatches
weekday_patch = mpatches.Patch(color='blue', label='Weekdays')
weekend_patch = mpatches.Patch(color='red', label='Weekends')
plt.legend(handles=[weekday_patch, weekend_patch], loc='lower right')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_plot_CLIPPED.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Additional: Calculate correlations separately for weekdays and weekends
weekday_data = daily_delays_df[daily_delays_df['is_weekday']]
weekend_data = daily_delays_df[~daily_delays_df['is_weekday']]

if len(weekday_data) > 1:
    weekday_corr, weekday_p = spearmanr(weekday_data['avg_delay_uppsala'], weekday_data['avg_delay_malmo'])
    print(f"\nWeekday correlation: ρ = {weekday_corr:.3f}, p = {weekday_p:.3f}")

if len(weekend_data) > 1:
    weekend_corr, weekend_p = spearmanr(weekend_data['avg_delay_uppsala'], weekend_data['avg_delay_malmo'])
    print(f"Weekend correlation: ρ = {weekend_corr:.3f}, p = {weekend_p:.3f}")




# 1. Create Time of Day categories (same as before)
def get_time_of_day(hour):
    if 6 <= hour < 10:
        return 'Morning Rush (6-10)'
    elif 10 <= hour < 16:
        return 'Midday (10-16)'
    elif 16 <= hour < 20:
        return 'Evening Rush (16-20)'
    else:
        return 'Night (20-6)'

# 2. Extract hour from PlanAnkTid (same as before)
def safe_extract_hour(time_str):
    try:
        if pd.notna(time_str):
            if isinstance(time_str, str):
                time_str = time_str.strip()
                for sep in [':', '.', '-', '']:
                    try:
                        time_obj = pd.to_datetime(time_str, format=f'%H{sep}%M', errors='raise')
                        return time_obj.hour
                    except:
                        continue
            time_obj = pd.to_datetime(time_str)
            return time_obj.hour
    except:
        pass
    return np.nan

# Apply hour extraction
df_uppsala['Hour'] = df_uppsala['PlanAnkTid'].apply(safe_extract_hour)
df_malmo['Hour'] = df_malmo['PlanAnkTid'].apply(safe_extract_hour)

# Apply time of day categorization
df_uppsala['Time_of_Day'] = df_uppsala['Hour'].apply(get_time_of_day)
df_malmo['Time_of_Day'] = df_malmo['Hour'].apply(get_time_of_day)

# 3. Calculate average delay per time window PER DAY FOR EACH ROUTE SEPARATELY
# Uppsala route
uppsala_time_daily = (df_uppsala.groupby(['Datum_PAU', 'Time_of_Day'])['AnkFörsening']
                      .apply(lambda x: x.clip(lower=0).mean())
                      .reset_index())
uppsala_time_daily.rename(columns={'AnkFörsening': 'avg_delay'}, inplace=True)
uppsala_time_daily['route'] = 'Uppsala'

# Malmö route
malmo_time_daily = (df_malmo.groupby(['Datum_PAU', 'Time_of_Day'])['AnkFörsening']
                    .apply(lambda x: x.clip(lower=0).mean())
                    .reset_index())
malmo_time_daily.rename(columns={'AnkFörsening': 'avg_delay'}, inplace=True)
malmo_time_daily['route'] = 'Malmö'

# 4. Combine both routes into one DataFrame for analysis
combined_time_delays = pd.concat([uppsala_time_daily, malmo_time_daily], ignore_index=True)

# 5. Analyze each route separately by time of day
time_windows = combined_time_delays['Time_of_Day'].unique()
routes = ['Uppsala', 'Malmö']
results = []

for route in routes:
    route_data = combined_time_delays[combined_time_delays['route'] == route]
    
    for window in time_windows:
        subset = route_data[route_data['Time_of_Day'] == window]
        if len(subset) > 3:
            # Calculate summary statistics for this route and time window
            mean_delay = subset['avg_delay'].mean()
            median_delay = subset['avg_delay'].median()
            std_delay = subset['avg_delay'].std()
            count_days = len(subset)
            
            results.append({
                'Route': route,
                'Time of Day': window,
                'n_days': count_days,
                'mean_delay': mean_delay,
                'median_delay': median_delay,
                'std_delay': std_delay
            })

# Create a summary DataFrame
results_df = pd.DataFrame(results)
print("=== Delay Statistics by Route and Time of Day ===")
print(results_df.round(4))

# 6. Visualization: Compare delays by time of day for both routes
plt.figure(figsize=(12, 8))

# Create subplots for different visualizations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Mean delays by time of day for both routes
time_windows = results_df['Time of Day'].unique()
x_pos = np.arange(len(time_windows))
width = 0.35

for i, route in enumerate(routes):
    route_data = results_df[results_df['Route'] == route]
    means = [route_data[route_data['Time of Day'] == window]['mean_delay'].values[0] 
             for window in time_windows]
    
    ax1.bar(x_pos + (i * width), means, width, label=route, alpha=0.8)

ax1.set_xlabel('Time of Day')
ax1.set_ylabel('Mean Delay (minutes)')
ax1.set_title('Mean Delays by Time of Day for Each Route')
ax1.set_xticks(x_pos + width/2)
ax1.set_xticklabels(time_windows, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Box plot comparison
import seaborn as sns
sns.boxplot(data=combined_time_delays, x='Time_of_Day', y='avg_delay', hue='route', ax=ax2)
ax2.set_xlabel('Time of Day')
ax2.set_ylabel('Delay (minutes)')
ax2.set_title('Delay Distribution by Time of Day and Route')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Route')

plt.tight_layout()
plt.savefig('delay_comparison_by_time_and_route.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Statistical comparison between routes for each time window
print("\n=== Statistical Comparison Between Routes ===")

for window in time_windows:
    uppsala_data = combined_time_delays[(combined_time_delays['route'] == 'Uppsala') & 
                                       (combined_time_delays['Time_of_Day'] == window)]['avg_delay']
    malmo_data = combined_time_delays[(combined_time_delays['route'] == 'Malmö') & 
                                     (combined_time_delays['Time_of_Day'] == window)]['avg_delay']
    
    if len(uppsala_data) > 3 and len(malmo_data) > 3:
        from scipy.stats import mannwhitneyu
        
        # Mann-Whitney U test for comparing two independent samples
        stat, p_value = mannwhitneyu(uppsala_data, malmo_data)
        
        print(f"\n{window}:")
        print(f"  Uppsala: n={len(uppsala_data)}, mean={uppsala_data.mean():.2f}, median={uppsala_data.median():.2f}")
        print(f"  Malmö: n={len(malmo_data)}, mean={malmo_data.mean():.2f}, median={malmo_data.median():.2f}")
        print(f"  Mann-Whitney U test: p={p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  -> Significant difference between routes (p < 0.05)")
        else:
            print(f"  -> No significant difference between routes")

# 8. Additional analysis: Peak vs Off-peak comparison
print("\n=== Peak vs Off-Peak Analysis ===")

# Define peak hours
peak_windows = ['Morning Rush (6-10)', 'Evening Rush (16-20)']
off_peak_windows = ['Midday (10-16)', 'Night (20-6)']

for route in routes:
    route_data = combined_time_delays[combined_time_delays['route'] == route]
    
    peak_data = route_data[route_data['Time_of_Day'].isin(peak_windows)]['avg_delay']
    off_peak_data = route_data[route_data['Time_of_Day'].isin(off_peak_windows)]['avg_delay']
    
    if len(peak_data) > 3 and len(off_peak_data) > 3:
        stat, p_value = mannwhitneyu(peak_data, off_peak_data)
        
        print(f"\n{route}:")
        print(f"  Peak hours: n={len(peak_data)}, mean={peak_data.mean():.2f}")
        print(f"  Off-peak hours: n={len(off_peak_data)}, mean={off_peak_data.mean():.2f}")
        print(f"  Mann-Whitney U test: p={p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  -> Significant difference between peak and off-peak (p < 0.05)")
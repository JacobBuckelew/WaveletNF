import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Define your file paths and model names
files = {
    'WENFlow-1': '../figures/WENFLOW_memory_values_0.01.csv',
    'WENFlow-5': '../figures/WENFLOW_memory_values_0.05.csv',
    'WENFlow-10': '../figures/WENFLOW_memory_values_0.1.csv',
    'WENFlow-25': '../figures/WENFLOW_memory_values_0.25.csv'
}

# Create the plot
plt.figure(figsize=(10, 6))

# Read and plot each file
for model_name, filepath in files.items():
    df = pd.read_csv(filepath)
    print(df)
    
    # Replace 'OOM' with NaN
    df['Memory'] = pd.to_numeric(df['Memory'], errors='coerce')
    
    # Split into valid and OOM
    valid_mask = df['Memory'].notna()
    valid_data = df[valid_mask]
    oom_data = df[~valid_mask]
    
    # Plot valid data
    plt.plot(valid_data['D'], valid_data['Memory'], 
             marker='o', 
             label=model_name, 
             linewidth=2)
    
    # If there are OOM values, add dashed line to indicate continuation
    if len(oom_data) > 0:
        last_valid = valid_data.iloc[-1]
        first_oom_d = oom_data.iloc[0]['D']
        
        # Get the current line color (last plotted line)
        current_color = plt.gca().get_lines()[-1].get_color()
        
        # Draw dashed line from last valid point
        plt.plot([last_valid['D'], first_oom_d], 
                [last_valid['Memory'], last_valid['Memory']], 
                linestyle='--', color=current_color, alpha=0.5, linewidth=2)
        
        # Add OOM marker
        plt.text(first_oom_d, last_valid['Memory'], ' OOM→', 
                fontsize=9, color=current_color, 
                verticalalignment='center')

plt.xlabel('Input Dimension (D)', fontsize=12)
plt.ylabel('Memory (MB)', fontsize=12)
plt.title('Memory Scaling Across Models', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/memory_scaling.png', dpi=300, bbox_inches='tight')
plt.show()
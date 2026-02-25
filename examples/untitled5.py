import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the component classification
data = {
    'Basic Event': [f'BE{i}' for i in range(1, 19)],
    'Importance': [1, 1, 1, 1, 3, 2, 1, 1, 1, 3, 3, 2, 1, 2, 2, 2, 1, 1],
    'Cost':       [1, 1, 1, 1, 2, 2, 3, 2, 3, 2, 1, 2, 2, 2, 3, 2, 1, 1]
}

df = pd.DataFrame(data)

# Add more jitter to prevent label overlap
np.random.seed(0)
jitter_strength = 0.25 # increased from 0.1
df['Importance_jittered'] = df['Importance'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
df['Cost_jittered'] = df['Cost'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))

# Create the plot
plt.figure(figsize=(8, 8))
plt.scatter(df['Cost_jittered'], df['Importance_jittered'], s=100, c='skyblue', edgecolor='black')

# Add labels with slight offset
for i, row in df.iterrows():
    plt.text(row['Cost_jittered'] + 0.07, row['Importance_jittered'] + 0.03, row['Basic Event'], fontsize=9)

# Add quadrant lines
plt.axhline(2, color='gray', linestyle='--')
plt.axvline(2, color='gray', linestyle='--')

# Axes and labels
plt.xticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.xlabel('Maintenance Cost Level', fontsize=12)
plt.ylabel('Reliability Importance Level', fontsize=12)
plt.title('Component Classification: Importance vs Cost', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0.5, 3.5)
plt.ylim(0.5, 3.5)
plt.tight_layout()
plt.show()

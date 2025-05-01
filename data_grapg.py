import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the DataFrame with your new structure
df = pd.DataFrame({
    'Column': ['height', 'weight', 'iq', 'fb_friends'],
    '0.25-0.75': [335, 321, 344, 317],
    '0.2-0.8': [339, 325, 344, 324],
    '0.15-0.85': [339, 329, 344, 330],
    'Total': [349, 338, 344, 346]
})

# Set up the x positions for each column
x = np.arange(len(df['Column']))
bar_width = 0.2

plt.figure(figsize=(10, 6))

# Plot bars for each value range
plt.bar(x - 1.5 * bar_width, df['0.25-0.75'], width=bar_width, label='0.25-0.75')
plt.bar(x - 0.5 * bar_width, df['0.2-0.8'], width=bar_width, label='0.2-0.8')
plt.bar(x + 0.5 * bar_width, df['0.15-0.85'], width=bar_width, label='0.15-0.85')
# Plot the Total bar with a different color
plt.bar(x + 1.5 * bar_width, df['Total'], width=bar_width, color='orange', label='Total')

# Set the x-axis labels and other plot properties
plt.xticks(x, df['Column'])
plt.xlabel('Column')
plt.ylabel('Data Count')
plt.title('Data Count by Value Range and Total per Column')
plt.legend()
plt.tight_layout()
plt.show()

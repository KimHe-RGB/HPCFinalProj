import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('time.csv',header=None)

ht = df.iloc[0, 1:].to_numpy(dtype=float)

# Extract other solutions
times = df.iloc[1:, 1:].to_numpy(dtype=float)

# Plot absolute difference between real solution and other solutions
plt.figure(figsize=(6, 6))
for i in range(times.shape[0]):
    if df.iloc[i+1, 0] == 'Fine':
        plt.plot(ht, times[i], label=df.iloc[i+1, 0], marker='o', color='black', linestyle=':')
    elif df.iloc[i+1, 0] == 'Coarse':
        plt.plot(ht, times[i], label=df.iloc[i+1, 0], marker='o', color='blue', linestyle=':')
    else:
        plt.plot(ht, times[i], label=df.iloc[i+1, 0], marker='o')

plt.xlabel('ht')
plt.ylabel('seconds')

plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

plt.savefig('timing.eps', format='eps')

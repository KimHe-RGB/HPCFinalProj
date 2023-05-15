import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the csv file
data = pd.read_csv('solutions.csv',header=None)

# Extract xs and real solutions
xs = data.iloc[0, 1:].to_numpy(dtype=float)
real_solutions = data.iloc[1, 1:].to_numpy(dtype=float)

# Extract other solutions
other_solutions = data.iloc[2:, 1:].to_numpy(dtype=float)

# Plot absolute difference between real solution and other solutions
plt.figure(figsize=(6, 6))
for i in range(other_solutions.shape[0]):
    abs_difference = np.abs(other_solutions[i] - real_solutions)
    plt.plot(xs, abs_difference, label=data.iloc[i+2, 0])
    #plt.plot(xs, other_solutions[i], label=data.iloc[i+2, 0])

plt.xlabel('xs')
plt.ylabel('Absolute difference from real solution')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('error1.eps', format='eps')

CN_solutions = data.iloc[2, 1:].to_numpy(dtype=float)
other_solutions = data.iloc[3:, 1:].to_numpy(dtype=float)
# Plot absolute difference between real solution and other solutions
plt.figure(figsize=(6, 6))
for i in range(other_solutions.shape[0]):
    abs_difference = np.abs(other_solutions[i] - CN_solutions)
    plt.plot(xs, abs_difference, label=data.iloc[i+3, 0])
    #plt.plot(xs, other_solutions[i], label=data.iloc[i+2, 0])

plt.xlabel('xs')
plt.ylabel('Absolute difference from CN')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('error2.eps', format='eps')


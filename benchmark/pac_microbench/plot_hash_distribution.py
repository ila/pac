#!/usr/bin/env python3
"""Plot hash bit distribution vs expected binomial."""

import matplotlib.pyplot as plt
import numpy as np

# Data from binomial.out
num_ones = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
observed = [4692, 10057, 19764, 35913, 61227, 98015, 147038, 206522, 273576, 338837, 395711, 434562, 446854, 432363, 395100, 339178, 272061, 205717, 147131, 97415, 61223, 35483, 19665, 10050, 4879]
expected = [4786.1, 10028.1, 19600.4, 35792.0, 61144.7, 97831.5, 146747.3, 206533.3, 272919.0, 338796.0, 395262.0, 433513.1, 447060.4, 433513.1, 395262.0, 338796.0, 272919.0, 206533.3, 146747.3, 97831.5, 61144.7, 35792.0, 19600.4, 10028.1, 4786.1]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot observed as bars
bars = ax.bar(num_ones, observed, width=0.8, alpha=0.7, color='#3498db',
              edgecolor='black', linewidth=0.5, label='DuckDB hash(c_custkey)')

# Plot expected as line
ax.plot(num_ones, expected, 'o-', color='#e74c3c', linewidth=2, markersize=5,
        label='Binomial(n=64, p=0.5)')

ax.set_xlabel("Number of 1's in 64-bit hash", fontsize=12)
ax.set_ylabel('Observed frequency', fontsize=12)
ax.set_title('Hash Bit Distribution vs Expected Binomial (TPCH SF=30 c_custkey)', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/hash_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/hash_distribution.png")

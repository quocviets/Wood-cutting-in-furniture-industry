import matplotlib.pyplot as plt

# Data for First Fit and Best Fit
labels = ["Waste", "Reward"]
first_fit = [4400, -80]
best_fit = [2700, -140]

# Create a more visually appealing line chart
fig, ax = plt.subplots(figsize=(7, 5))

# Plot lines with enhanced styling
ax.plot(labels, first_fit, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8, label="First Fit")
ax.plot(labels, best_fit, marker='s', linestyle='--', color='green', linewidth=2, markersize=8, label="Best Fit")

# Labels and title with better styling
ax.set_ylabel("Values", fontsize=12)
ax.set_title("Comparison of First Fit vs Best Fit", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc="best")

# Annotate values on the graph for better readability
for i, label in enumerate(labels):
    ax.annotate(f'{first_fit[i]}', (label, first_fit[i]), textcoords="offset points", xytext=(-5,8), ha='center', color='blue', fontsize=11, fontweight='bold')
    ax.annotate(f'{best_fit[i]}', (label, best_fit[i]), textcoords="offset points", xytext=(-5,-15), ha='center', color='green', fontsize=11, fontweight='bold')

# Grid for better visualization
ax.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.show()

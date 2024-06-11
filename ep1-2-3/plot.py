import matplotlib.pyplot as plt

# Data points
points = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]

# Separate the points based on their classes
class_0 = [p for p in points if p[2] == 0]
class_1 = [p for p in points if p[2] == 1]

# Plot the points
plt.scatter([p[0] for p in class_0], [p[1] for p in class_0], color='blue', label='Class 0 (Blue Circle)')
plt.scatter([p[0] for p in class_1], [p[1] for p in class_1], color='red', label='Class 1 (Red Circle)')

# Plot the separating line (x + y - 1.5 = 0)
plt.plot([0, 1.5], [1.5, 0], color='black', linestyle='-', linewidth=1, label='Separating Line')

# Set labels and legend
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('AND Gate Visualization')
plt.legend()

# Show plot
plt.grid(True)
plt.axis('equal')
plt.show()

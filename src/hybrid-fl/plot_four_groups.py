import matplotlib.pyplot as plt
import random
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# Function to generate points around a center with some deviation, constrained to 0-1 range
def generate_points(center, deviation, num_points):
    return [min(max(random.uniform(center[0] - deviation, center[0] + deviation), 0), 1) for _ in range(num_points)], \
           [min(max(random.uniform(center[1] - deviation, center[1] + deviation), 0), 1) for _ in range(num_points)]

# Generate points in four groups
num_points_per_group = 5

# Group 1: Near 0,0
x_group1, y_group1 = generate_points((0, 0), 0.2, num_points_per_group)

# Group 2: Near 1,0
x_group2, y_group2 = generate_points((1, 0), 0.2, num_points_per_group)

# Group 3: Near 1,1 (painted black)
x_group3, y_group3 = generate_points((1, 1), 0.2, num_points_per_group)
x_group3_black, y_group3_black = generate_points((1, 1), 0.2, num_points_per_group)

# Group 4: Near 0,1
x_group4, y_group4 = generate_points((0, 1), 0.2, num_points_per_group)

# Plotting the points
plt.figure(figsize=(8, 6))

plt.scatter(x_group1, y_group1, s=100, color='blue', alpha=0.6)#, label='Group 1')
plt.scatter(x_group2, y_group2, s=100, color='blue', alpha=0.6)#, label='Group 2')
plt.scatter(x_group4, y_group4, s=100, color='blue', alpha=0.6)#, label='Group 4')
plt.scatter(x_group3, y_group3, s=200, color='black', label='Clientes de interesse', alpha=0.6)

# Overlay the black points for Group 3
#plt.scatter(x_group3_black, y_group3_black, color='black')

plt.legend (loc='upper center', 
            #bbox_to_anchor=(0,0,1,0.7),
            ncol=1, frameon=True, markerfirst=True, labelcolor='black')
#plt.title('Points Grouped Around Specific Coordinates')
plt.ylabel("Velocidade em uma rodada (1/$t_n$)", fontsize=20)
plt.xlabel("Número de amostras", fontsize=20)
#plt.xlim (0, 1)
#plt.ylim (0, 1)
plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
#plt.legend()
#plt.show()
plt.savefig ('four_groups.pdf')
print ('saved')

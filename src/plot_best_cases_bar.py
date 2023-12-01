import matplotlib.pyplot as plt
import numpy as np

# Three lists of values
list1 = [0.9801999926567078, 0.9767000079154968, 0.9624999761581421, 0.9668999910354614, 0.9010999798774719]
list2 = [0.9778000116348267, 0.9817000031471252, 0.9822999835014343, 0.9812999963760376, 0.9812999963760376]
list3 = [0.9560999870300293, 0.9699000120162964, 0.9476000070571899, 0.9661999940872192, 0.89410001039505]

# Calculate means and standard deviations
means = [np.mean(list1), np.mean(list2), np.mean(list3)]
std_devs = [np.std(list1), np.std(list2), np.std(list3)]

# Number of groups and the width for the bars
num_groups = len(means)
bar_width = 0.35

# The x-axis positions for the groups
index = np.arange(num_groups)

# Creating the bar chart
plt.bar(index, means, bar_width, yerr=std_devs, capsize=5, color='skyblue', label='Valor médio nas últimas 5 rodadas')

# Adding mean values on top of each bar
for i, v in enumerate(means):
    plt.text(i, v/2, str(round(v, 2)), ha='center', va='bottom')

# Customizing the chart
plt.xlabel('Cenário')
plt.ylabel('Acurácia')
#plt.title('XPTO')
plt.xticks(index, ['FedAVG', '3FL', 'DOFL'])
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

FedAVG = [37111]

THREE_FL_6  = [24637]
THREE_FL_12 = [24124]
THREE_FL_19 = [23831]

DOFL_6  = [32819]
DOFL_12 = [34061]
DOFL_19 = [37069]


hybrid_6  = [28325]
hybrid_12 = [34894]
hybrid_19 = [35887]

# All data lists grouped together
data = [FedAVG, THREE_FL_6, THREE_FL_12, THREE_FL_19, DOFL_6, DOFL_12, DOFL_19, hybrid_6, hybrid_12, hybrid_19]

# Calculate means and standard deviations for each list
means = [int(np.mean(lst)) for lst in data]

# Number of groups and the width for the bars
num_groups = len(means)
bar_width = 0.55

# The x-axis positions for the groups
index = np.arange(num_groups)

# Defining colors for each group of bars
colors = ['lightcoral', 'lightskyblue', 'lightskyblue', 'lightskyblue', 'lightgreen', 'lightgreen', 'lightgreen', 'lightpink', 'lightpink', 'lightpink']

# Creating the bar chart with specified colors for each group
plt.bar(index, means, bar_width, capsize=8, color=colors)#, label='Acurácia média nas útlimas 10 rodadas')

# Adding mean values on top of each bar
for i, v in enumerate(means):
    plt.text(int(i), v/2, str(round(v, 2)), ha='center', va='bottom', fontsize=16, rotation=90)

# Customizing the chart
plt.gcf().set_size_inches(12, 6)
plt.xlabel('Cenário', fontsize=20)
plt.ylabel('Latência em segundos', fontsize=20)
plt.xticks(index, ['FedAVG', '3FL 25%', '3FL 50%', '3FL 75%', 'DOFL 25%', 'DOFL 50%', 'DOFL 75%', 'Hybrid-FL 25%', 'Hybrid-FL 50%', 'Hybrid-FL 75%'], rotation=45)
#plt.legend()

# Show the plot
plt.tight_layout()
#plt.show()
filename='latencias_all.pdf'
plt.savefig (filename)
print ('saved')

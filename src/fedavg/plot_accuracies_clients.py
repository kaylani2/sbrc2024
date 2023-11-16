### Plot centralized accuracies (measured by the server)
import re
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


logfiles = [
  '5clients/client_main_1_5_clients.log',
  '5clients/client_main_2_5_clients.log',
  '5clients/client_main_3_5_clients.log',
  '5clients/client_main_4_5_clients.log',
  '5clients/client_main_5_5_clients.log',
]

accuracies=[]
for logfile in zip(logfiles):
  # Read the log file
  with open(logfile, 'r') as file:
      log_content = file.read()

  # Regular expression pattern to match accuracy values
  accuracy_pattern = r"accuracy=(\d+\.\d+)"

  # Find all accuracy values in the log content
  accuracy_values = re.findall(accuracy_pattern, log_content)

  # Convert the accuracy values to floats and store in a list
  accuracy_list = [float(value) for value in accuracy_values]

  accuracies.append(accuracy_list)


### K: TESTE
#import random
#accuracies[0] = [x + random.uniform(-0.5, 0.5) for x in accuracies[0]]
#accuracies[1] = [x + random.uniform(-0.5, 0.5) for x in accuracies[1]]


# Calculate mean and standard deviation across the lists
mean_values = np.mean(accuracies, axis=0)
std_dev = np.std(accuracies, axis=0)
# Plot
plt.plot(np.arange(len(mean_values)), mean_values, label='Média', color='blue')
plt.fill_between(np.arange(len(mean_values)), mean_values - std_dev, mean_values + std_dev, color='blue', alpha=0.2, label='Desvio padrão')


plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend (loc='lower right', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.xlim (0, 101)
plt.tight_layout()
plt.show()
#plt.savefig ('client_accuracy_fedavg_100rounds.pdf')
#print ('saved')

### Plot centralized accuracies (measured by the server)
### Just define all logs and all configs
import re
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


logfiles_5clients  = [f"5clients/client_main_{i}_5_clients.log" for i in range(1, 5+1)]
logfiles_10clients = [f"10clients/client_main_{i:02d}_10_clients.log" for i in range(1, 10+1)]
logfiles_15clients = [f"15clients/client_main_{i:02d}_15_clients.log" for i in range(1, 15+1)]
logfiles_25clients = [f"25clients/client_main_{i:02d}_25_clients.log" for i in range(1, 25+1)]
logfiles_50clients = [f"50clients/client_main_{i:02d}_50_clients.log" for i in range(1, 50+1)]

configs = [
  {'label1': 'Média 5 clientes', 'label2': 'Desvio padrão 5 clientes', 'color': 'blue',},
  {'label1': 'Média 10 clientes', 'label2': 'Desvio padrão 10 clientes', 'color': 'red',},
  {'label1': 'Média 15 clientes', 'label2': 'Desvio padrão 15 clientes', 'color': 'green',},
  {'label1': 'Média 25 clientes', 'label2': 'Desvio padrão 25 clientes', 'color': 'pink',},
  {'label1': 'Média 50 clientes', 'label2': 'Desvio padrão 50 clientes', 'color': 'purple',},
]

log_groups = [logfiles_5clients, logfiles_10clients, logfiles_15clients, logfiles_25clients, logfiles_50clients]

for log_group, config in zip (log_groups, configs):
  accuracies=[]
  for logfile in log_group:
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

  # Calculate mean and standard deviation across the lists
  mean_values = np.mean(accuracies, axis=0)
  std_dev = np.std(accuracies, axis=0)
  # Plot
  plt.plot(np.arange(len(mean_values)), mean_values, label=config['label1'], color=config['color'])
  plt.fill_between(np.arange(len(mean_values)), mean_values - std_dev, mean_values + std_dev, color=config['color'], alpha=0.2, label=config['label2'])


plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend(loc='center right', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.gcf().set_size_inches(10, 7)  # Adjust the figure size (width, height) to fit the legend
plt.xlim (0, 101)
plt.tight_layout()
#plt.show()
plt.savefig ('client_accuracy_fedavg_100rounds.pdf')
print ('saved')

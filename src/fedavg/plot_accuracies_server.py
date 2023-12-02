### Plot centralized accuracies (measured by the server)
import re
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


logfiles = [
  'server_main_100rounds_2clients_fedavg.log',
  'server_main_100rounds_5clients_fedavg.log',
  'server_main_100rounds_10clients_fedavg.log',
  'server_main_100rounds_15clients_fedavg.log',
  'server_main_100rounds_25clients_fedavg.log',
  'server_main_100rounds_50clients_fedavg.log',
]
labels = [
  '2 clientes',
  '5 clientes',
  '10 clientes',
  '15 clientes',
  '25 clientes',
  '50 clientes',
]

colors = [
  'cyan',
  'blue',
  'red',
  'green',
  'black',
  'purple',
]

for logfile, label, color in zip(logfiles, labels, colors):
  # Read the log file
  with open(logfile, 'r') as file:
    log_content = file.read()

  # Regular expression pattern to match accuracy values
  accuracy_pattern = r"'accuracy': (\d+\.\d+)"

  # Find all accuracy values in the log content
  accuracy_values = re.findall(accuracy_pattern, log_content)

  # Convert the accuracy values to floats and store in a list
  accuracy_list = [float(value) for value in accuracy_values]

  x = list(range(1, len(accuracy_list)+ 1))
  plt.plot(x, accuracy_list, label=label, #marker='.', 
           markersize=10.0, alpha=0.8, linewidth=3.0, color=color)

plt.xlabel("Rodada")
plt.ylabel("Acurácia")

plt.legend (loc='center right', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.gcf().set_size_inches(10, 7)  # Adjust the figure size (width, height) to fit the legend
plt.xlim (0, 101)
plt.tight_layout()
plt.show()
#plt.savefig ('server_accuracy_fedavg_100rounds.pdf')
#print ('saved')

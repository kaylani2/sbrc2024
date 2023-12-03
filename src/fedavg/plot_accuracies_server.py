### Plot centralized accuracies (measured by the server)
import re
import matplotlib
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

markers =[
  'o',
  'x',
  's',
  'P',
  '.',
  ',',
]

lss =[
  'solid',
  'dotted',
  'dashed',
  'dashdot',
  (5, (10, 3)),
  (0, (3, 8, 1, 8, 1, 8)),
]

for logfile, label, color, marker, ls in zip(logfiles, labels, colors, markers, lss):
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
  plt.plot(x, accuracy_list, label=label,
           #marker=marker, 
           #markersize=7.0, 
           linestyle='solid',#ls,
           alpha=0.8,
           linewidth=3.0,
           color=color)


plt.xlabel("Rodada", fontsize=20)
plt.ylabel("Acurácia", fontsize=20)

plt.legend (loc='center right', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.gcf().set_size_inches(12, 6)
plt.xlim (0, 101)
plt.tight_layout()

#plt.show()
plt.savefig ('server_accuracy_fedavg_100rounds.pdf')
print ('saved')

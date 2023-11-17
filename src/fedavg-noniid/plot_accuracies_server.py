### Plot centralized accuracies (measured by the server)
import re
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


logfiles = [
  'server_main_500rounds_5clients_fedavg.log',
]
labels = [
  '5 clientes',
]

for logfile, label in zip(logfiles, labels):
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
  plt.plot(x, accuracy_list, label=label, marker='.', alpha=0.5)

plt.xlabel("Rodada")
plt.ylabel("Acurácia")

plt.legend (loc='lower right', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.xlim (0, 501)
plt.gcf().set_size_inches(12, 6)  # Adjust the figure size (width, height) to fit the legend
plt.tight_layout()
plt.show()
#plt.savefig ('server_accuracy_fedavg_100rounds.pdf')
#print ('saved')

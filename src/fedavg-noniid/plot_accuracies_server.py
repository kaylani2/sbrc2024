### Plot centralized accuracies (measured by the server)
import re
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

filename = 'server_accuracy_fedavg-noniid_500rounds_custom.pdf'

logfiles = [
  'logs_plot_custom/server_main_500rounds_2clients_fedavg.log',
  'logs_plot_custom/server_main_500rounds_5clients_fedavg.log',
  'logs_plot_custom/server_main_500rounds_10clients_fedavg.log',
  'logs_plot_custom/server_main_500rounds_15clients_fedavg.log',
  'logs_plot_custom/server_main_500rounds_25clients_fedavg.log',
  'logs_plot_custom/server_main_500rounds_50clients_fedavg.log',
]



#### plotting mobilenet
#filename = 'server_accuracy_fedavg-noniid_500rounds_mobilenetv2.pdf'
#logfiles = [
#  'mobilenetv2_logs/server_main_500rounds_2clients_fedavg.log',
#  'mobilenetv2_logs/server_main_500rounds_5clients_fedavg.log',
#  'mobilenetv2_logs/server_main_500rounds_10clients_fedavg.log',
#  'mobilenetv2_logs/server_main_500rounds_15clients_fedavg.log',
#  'mobilenetv2_logs/server_main_500rounds_25clients_fedavg.log',
#  'mobilenetv2_logs/server_main_500rounds_50clients_fedavg.log',
#]

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
  plt.plot(x, accuracy_list, label=label,
           #marker=marker, 
           #markersize=7.0, 
           linestyle='solid',#ls,
           alpha=0.8,
           linewidth=1.5,
           color=color)

plt.xlabel("Rodada", fontsize=20)
plt.ylabel("Acurácia", fontsize=20)

#plt.legend (loc='upper left', bbox_to_anchor=(0,-0.09,1,1), ncol=1, frameon=False, markerfirst=True, labelcolor='black') ### mobilnetv2
plt.legend (loc='upper left', bbox_to_anchor=(0,-0.09,1,1), ncol=1, frameon=False, markerfirst=True, labelcolor='black')

plt.xlim (0, 501)
plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
#plt.show()
plt.savefig (filename)
print ('saved')

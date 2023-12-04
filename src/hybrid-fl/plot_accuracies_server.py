### Plot centralized accuracies (measured by the server)
import re
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
filename='server_accuracy_hybridfl_500rounds_custom_mnist.pdf'
first_logfiles = [
  'logs_plot/6-6-25.log',
  'logs_plot/12-12-25.log',
  'logs_plot/19-19-25.log',
]
second_logfiles = [
  'logs_plot/6-25-25.log',
  'logs_plot/12-25-25.log',
  'logs_plot/19-25-25.log',
]
labels = [
  'Clientes de interesse primeiro (25%)',
  'Clientes de interesse primeiro (50%)',
  'Clientes de interesse primeiro (75%)',
]

colors = [
  'blue',
  'red',
  'green',
]


all_clients = 'logs_plot/25-25-25.log'

for first_logfile, second_logfile, label, color in zip(first_logfiles, second_logfiles, labels, colors):
  # Read the log file
  with open(first_logfile, 'r') as file:
    log_content = file.read()

  # Regular expression pattern to match accuracy values
  accuracy_pattern = r"'accuracy': (\d+\.\d+)"

  # Find all accuracy values in the log content
  accuracy_values = re.findall(accuracy_pattern, log_content)

  # Convert the accuracy values to floats and store in a list
  accuracy_list1 = [float(value) for value in accuracy_values]

  with open(second_logfile, 'r') as file2:
    log_content = file2.read()

  # Regular expression pattern to match accuracy values
  accuracy_pattern = r"'accuracy': (\d+\.\d+)"

  # Find all accuracy values in the log content
  accuracy_values = re.findall(accuracy_pattern, log_content)

  # Convert the accuracy values to floats and store in a list
  accuracy_list2 = [float(value) for value in accuracy_values]


  accuracy_list = accuracy_list1[:250] + accuracy_list2[:250]

  ### plot only the fastests
  #accuracy_list = accuracy_list1
  #filename='server_accuracy_only_fastest_clients_500rounds_custom_mnist.pdf'

  x = list(range(1, len(accuracy_list)+ 1))
  plt.plot(x, accuracy_list, 
           label=label, marker='', alpha=0.8,
           color=color,
           linestyle='dashed',
           linewidth=1.5,
           )


### Plot all 25 clients (regular fedavg)
with open(all_clients, 'r') as file:
  log_content = file.read()

# Regular expression pattern to match accuracy values
accuracy_pattern = r"'accuracy': (\d+\.\d+)"

# Find all accuracy values in the log content
accuracy_values = re.findall(accuracy_pattern, log_content)

# Convert the accuracy values to floats and store in a list
accuracy_list = [float(value) for value in accuracy_values]

x = list(range(1, len(accuracy_list)+ 1))
plt.plot(x, accuracy_list, label='FedAVG (25 clientes)', linestyle='solid', color='black', alpha=0.8)

#plt.text(5.0, 0.07, 'Clientes mais rápidos apenas', fontsize=16, color='black')
#plt.text(400.0, 0.77, 'Todos os clientes', fontsize=16, color='black')
plt.axvline(x=250, color='black', linestyle='dotted', linewidth=3.0)#, label='Rodada 250')

plt.xlabel("Rodada", fontsize=20)
plt.ylabel("Acurácia", fontsize=20)


### K: Plotando o 250 pra ficar mais claro.
# Get the current x-axis ticks
existing_xticks = plt.gca().get_xticks().tolist()
# Add the specific x-axis value you want to display
desired_xtick = 250
all_xticks = existing_xticks + [desired_xtick]
# Set the x-axis ticks
plt.xticks(all_xticks)

plt.legend (loc='lower right', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.xlim (0, 501)
plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
#plt.show()
plt.savefig (filename)
print ('saved')

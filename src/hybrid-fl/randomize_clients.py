import random
from collections import Counter
import matplotlib.pyplot as plt
import math
CLIENTS_TO_INCLUDE=[6,12,19]

#samples = {
#  '0': 5923,
#  '1': 6742,
#  '2': 5958,
#  '3': 6131,
#  '4': 5842,
#  '5': 5421,
#  '6': 5918,
#  '7': 6265,
#  '8': 5851,
#  '9': 5949,
#}
#
#
#
## Create a list of 25 clients initialized with '0'
#clients = []
#for i in range(1,26):
#  fake_index = (i - 1) % 5 + 1 
#  if (fake_index == 1):
#    client = {'index':i, 'num_samples':(samples['0']+samples['1'])//5, 'speed':20, 'final_num_samples':(samples['0']+samples['1'])//5} ### K: 25 clients.
#  if (fake_index == 2):
#    client = {'index':i, 'num_samples':(samples['2']+samples['3'])//5, 'speed':20, 'final_num_samples':(samples['2']+samples['3'])//5} ### K: 25 clients.
#  if (fake_index == 3):
#    client = {'index':i, 'num_samples':(samples['4']+samples['5'])//5, 'speed':20, 'final_num_samples':(samples['4']+samples['5'])//5} ### K: 25 clients.
#  if (fake_index == 4):
#    client = {'index':i, 'num_samples':(samples['6']+samples['7'])//5, 'speed':20, 'final_num_samples':(samples['6']+samples['7'])//5} ### K: 25 clients.
#  if (fake_index == 5):
#    client = {'index':i, 'num_samples':(samples['8']+samples['9'])//5, 'speed':20, 'final_num_samples':(samples['8']+samples['9'])//5} ### K: 25 clients.
#
#  ### randomly augment the dataset
#  multiplier = random.gauss(1.5, 0.25)  # Mean = 1.5, Standard deviation = 0.25
#  multiplier = max(1, min(2, multiplier))  # Ensure multiplier is between 1 and 2
#  client['num_samples'] = int(client['num_samples']*multiplier)
#  
#  clients.append(client)
#  
#for client in clients:
#  random_number = random.randint(0, 2)
#  if (random_number == 0):
#    client['speed'] = 60
#  if (random_number == 1):
#    client['speed'] = 40
#
#
## Extract num_samples values
#num_samples_values = [client['num_samples'] for client in clients]
## Find the minimum and maximum values of num_samples
#min_samples = min(num_samples_values)
#max_samples = max(num_samples_values)
## Normalize the num_samples values between 0 and 1
#for client in clients:
#  client['num_samples'] = (client['num_samples'] - min_samples) / (max_samples - min_samples)
#
## Extract speed values
#speed = [client['speed'] for client in clients]
## Find the minimum and maximum values of speed
#min_samples = min(speed)
#max_samples = max(speed)
## Normalize the speed values between 0 and 1
#for client in clients:
#  client['speed'] = (client['speed'] - min_samples) / (max_samples - min_samples)


clients =[
  {'index': 9, 'num_samples': 0.8132474701011959, 'speed': 1.0, 'final_num_samples': 2436},
  {'index': 7, 'num_samples': 0.6651333946642134, 'speed': 1.0, 'final_num_samples': 2417},
  {'index': 8, 'num_samples': 0.5873965041398344, 'speed': 1.0, 'final_num_samples': 2252},
  {'index': 11, 'num_samples': 1.0, 'speed': 0.5, 'final_num_samples': 2533},
  {'index': 16, 'num_samples': 0.46550137994480223, 'speed': 1.0, 'final_num_samples': 2533},
  {'index': 21, 'num_samples': 0.4153633854645814, 'speed': 1.0, 'final_num_samples': 2533},
  {'index': 18, 'num_samples': 0.35510579576816925, 'speed': 1.0, 'final_num_samples': 2252},
  {'index': 3, 'num_samples': 0.08003679852805888, 'speed': 1.0, 'final_num_samples': 2252},
  {'index': 4, 'num_samples': 0.9806807727690893, 'speed': 0.0, 'final_num_samples': 2436},
  {'index': 1, 'num_samples': 0.8955841766329347, 'speed': 0.0, 'final_num_samples': 2533},
  {'index': 23, 'num_samples': 0.6522539098436062, 'speed': 0.5, 'final_num_samples': 2252},
  {'index': 14, 'num_samples': 0.6453541858325667, 'speed': 0.5, 'final_num_samples': 2436},
  {'index': 6, 'num_samples': 0.8026678932842686, 'speed': 0.0, 'final_num_samples': 2533},
  {'index': 17, 'num_samples': 0.7543698252069917, 'speed': 0.0, 'final_num_samples': 2417},
  {'index': 2, 'num_samples': 0.5358785648574057, 'speed': 0.5, 'final_num_samples': 2417},
  {'index': 25, 'num_samples': 0.5349586016559338, 'speed': 0.5, 'final_num_samples': 2360},
  {'index': 5, 'num_samples': 0.5087396504139834, 'speed': 0.5, 'final_num_samples': 2360},
  {'index': 12, 'num_samples': 0.4806807727690892, 'speed': 0.5, 'final_num_samples': 2417},
  {'index': 22, 'num_samples': 0.4659613615455382, 'speed': 0.5, 'final_num_samples': 2417},
  {'index': 10, 'num_samples': 0.3813247470101196, 'speed': 0.5, 'final_num_samples': 2360},
  {'index': 19, 'num_samples': 0.5910763569457221, 'speed': 0.0, 'final_num_samples': 2436},
  {'index': 20, 'num_samples': 0.26954921803127874, 'speed': 0.5, 'final_num_samples': 2360},
  {'index': 15, 'num_samples': 0.5133394664213431, 'speed': 0.0, 'final_num_samples': 2360},
  {'index': 24, 'num_samples': 0.4816007359705612, 'speed': 0.0, 'final_num_samples': 2436},
  {'index': 13, 'num_samples': 0.0, 'speed': 0.0, 'final_num_samples': 2252}
]



# Extracting num_samples and speed values
num_samples = [client['num_samples'] for client in clients]
speed = [client['speed'] for client in clients]


# Extracting num_samples and speed values
num_samples = [client['num_samples'] for client in clients]
speed = [client['speed'] for client in clients]

# Calculate Euclidean distance from each point to the origin (0, 0)
distances = [math.sqrt(x ** 2 + y ** 2) for x, y in zip(num_samples, speed)]



for oregon in CLIENTS_TO_INCLUDE:
  # Get the indices of the 5 furthest points from the origin
  furthest_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[:oregon]

  # Create the scatter plot
  plt.figure(figsize=(8, 6))
  for i, (x, y) in enumerate(zip(num_samples, speed)):
      if i in furthest_indices:
          plt.scatter(x, y, color='red', s=100, edgecolors='black')  # Highlight furthest points
      else:
          plt.scatter(x, y, color='blue', s=50, alpha=0.7, edgecolors='black')  # Other points


  # Adding labels and title
  plt.xlabel('Velocidade de processamento normalizado')
  plt.ylabel('Número de amostras normalizado')
  #plt.title('Scatter Plot of num_samples vs speed')

  #plt.grid(True)
  #plt.show()
  filename = 'random_clients_with_'+str(oregon)+'_furthests.pdf'
  plt.savefig(filename)
  print('saved')


# Sort the clients based on the distances
sorted_clients = [client for _, client in sorted(zip(distances, clients), key=lambda x: x[0], reverse=True)]

# Print the sorted list of clients
for client in sorted_clients:
  print(client)

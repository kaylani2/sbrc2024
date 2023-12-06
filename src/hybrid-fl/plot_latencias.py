### K: Tempos em s
import random
from random import randint
from random import seed
from matplotlib import pyplot as plt
import numpy as np
import sys
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

RODADAS = 500
TOTAL_CLIENTES = 25
TEMPOS_CLIENTES_LENTOS = [55, 68, 72]
TEMPOS_CLIENTES_RAPIDOS = [27, 23, 19]
TIMEOUT = 100


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




fast_timeout = 32
slow_timeout = 128
fast_mean = 21
fast_dev = 3.6
slow_mean = 59
slow_dev = 6.3

middle_mean = 41
middle_dev = 4.2
middle_timeout = 64

##################################################################################################################################
y1 = []; latencia_total = 0; rodada = 1
while rodada < RODADAS + 1:
  if (rodada < 250):
    TIMEOUT_PROB = 0.01
    latencia_rodada = np.random.choice([middle_timeout, np.random.normal(middle_mean, middle_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  else:
    TIMEOUT_PROB = 0.10
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  latencia_total += latencia_rodada
  if (latencia_rodada == fast_timeout or latencia_rodada == slow_timeout or latencia_rodada == middle_timeout):
    rodada -=1
  rodada +=1
  y1.append (latencia_total)
y1[0] = 1+2+4+8+16+32
x = [*range (1, len(y1)+1)]
plt.plot(x, y1, 'b-', markersize=6, label='Clientes mais rápidos primeiro (25%)', linestyle='dashed')

##################################################################################################################################
y2 = []; latencia_total = 0; rodada = 1
while rodada < RODADAS + 1:
  if (rodada < 250):
    TIMEOUT_PROB = 0.04
    #latencia_rodada = np.random.choice([fast_timeout, np.random.normal(fast_mean, fast_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  else:
    TIMEOUT_PROB = 0.10
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  latencia_total += latencia_rodada
  if (latencia_rodada == fast_timeout or latencia_rodada == slow_timeout):
    rodada -=1
  rodada +=1
  y2.append (latencia_total)
y2[0] = 1+2+4+8+16+32
x = [*range (1, len(y2)+1)]
plt.plot(x, y2, 'r-', markersize=6, label='Clientes mais rápidos primeiro (50%)', linestyle='dashed')

##################################################################################################################################
y3 = []; latencia_total = 0; rodada = 1
while rodada < RODADAS + 1:

  if (rodada < 250):
    TIMEOUT_PROB = 0.08
    #latencia_rodada = np.random.choice([fast_timeout, np.random.normal(fast_mean, fast_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  else:
    TIMEOUT_PROB = 0.10
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  latencia_total += latencia_rodada
  if (latencia_rodada == fast_timeout or latencia_rodada == slow_timeout):
    rodada -=1
  rodada +=1
  y3.append (latencia_total)
y3[0] = 1+2+4+8+16+32
x = [*range (1, len(y3)+1)]
plt.plot(x, y3, 'g-', markersize=6, label='Clientes mais rápidos primeiro (75%)', linestyle='dashed')

##################################################################################################################################
y4 = []; latencia_total = 0; rodada = 1

while rodada < RODADAS + 1:
  TIMEOUT_PROB=0.10

  latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  if (latencia_rodada == fast_timeout or latencia_rodada == slow_timeout):
    rodada-=1
  rodada+=1
  latencia_total += latencia_rodada
  y4.append (latencia_total)
x = [*range (1, len(y4)+1)]
plt.plot(x, y4, 'k-', markersize=6, label='Latência convencional')


offsets = [-10, -5, 0, 10]  # Adjust these offsets for appropriate spacing
for i, var in enumerate((y1[-1], y2[-1], y3[-1], y4[-1])):
    plt.annotate('%0.0f' % int(var), xy=(1, int(var)), xytext=(8, offsets[i]),  xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.legend (loc='upper left', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.gcf().set_size_inches(12, 6)
plt.xlabel("Rodada", fontsize=20)
plt.ylabel("Latência total em segundos", fontsize=20)
plt.xlim (0, max(len(y1), len(y2), len(y3), len(y4)))
plt.tight_layout()
#plt.show()
plt.savefig ('latencia_hybrid.pdf')
print ('saved')

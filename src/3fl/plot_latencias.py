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

#TAMANHO_CNN_MNIST = 1.53 # <MB>
#L_ts = [53.2*(10**6), 51.8*(10**6), 94.0*(10**6)]  # <Mb> * [<b/s>] ### K: Velocidades medidas com o iPerf.
#L_ts = [TAMANHO_MODELO / x for x in L_ts] # [<b/s>] / b = <s>


fast_timeout = 32
slow_timeout = 128
fast_mean = 21
fast_dev = 3.6
slow_mean = 59
slow_dev = 6.3


##################################################################################################################################
y1 = []; latencia_total = 0; rodada = 1
while rodada < RODADAS + 1:
  if (rodada < 250):
    TIMEOUT_PROB = 0.01
    latencia_rodada = np.random.choice([fast_timeout, np.random.normal(fast_mean, fast_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  else:
    TIMEOUT_PROB = 0.10
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  latencia_total += latencia_rodada
  if (latencia_rodada == fast_timeout or latencia_rodada == slow_timeout):
    rodada -=1
  rodada +=1
  y1.append (latencia_total)
x = [*range (1, len(y1)+1)]
plt.plot(x, y1, 'b-', markersize=6, label='Clientes mais rápidos primeiro (25%)', linestyle='dashed')

##################################################################################################################################
y2 = []; latencia_total = 0; rodada = 1
while rodada < RODADAS + 1:
  if (rodada < 250):
    TIMEOUT_PROB = 0.04
    latencia_rodada = np.random.choice([fast_timeout, np.random.normal(fast_mean, fast_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  else:
    TIMEOUT_PROB = 0.10
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  latencia_total += latencia_rodada
  if (latencia_rodada == fast_timeout or latencia_rodada == slow_timeout):
    rodada -=1
  rodada +=1
  y2.append (latencia_total)
x = [*range (1, len(y2)+1)]
plt.plot(x, y2, 'r-', markersize=6, label='Clientes mais rápidos primeiro (50%)', linestyle='dashed')

##################################################################################################################################
y3 = []; latencia_total = 0; rodada = 1
while rodada < RODADAS + 1:

  if (rodada < 250):
    TIMEOUT_PROB = 0.08
    latencia_rodada = np.random.choice([fast_timeout, np.random.normal(fast_mean, fast_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  else:
    TIMEOUT_PROB = 0.10
    latencia_rodada = np.random.choice([slow_timeout, np.random.normal(slow_mean, slow_dev)], p=[TIMEOUT_PROB, 1 - TIMEOUT_PROB])
  latencia_total += latencia_rodada
  if (latencia_rodada == fast_timeout or latencia_rodada == slow_timeout):
    rodada -=1
  rodada +=1
  y3.append (latencia_total)
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


offsets = [20, 0, -20, 0]  # Adjust these offsets for appropriate spacing
for i, var in enumerate((y1[-1], y2[-1], y3[-1], y4[-1])):
    plt.annotate('%0.0f' % int(var), xy=(1, int(var)), xytext=(8, offsets[i]),  xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.legend (loc='upper left', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.gcf().set_size_inches(12, 6)
plt.xlabel("Rodada", fontsize=20)
plt.ylabel("Latência total em segundos", fontsize=20)
plt.xlim (0, max(len(y1), len(y2), len(y3), len(y4)))
plt.tight_layout()
#plt.show()
plt.savefig ('latencia_3fl.pdf')
print ('saved')
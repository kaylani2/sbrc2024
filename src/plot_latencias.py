### K: Tempos em s
import random
from random import randint
from random import seed
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

RODADAS = 500
TOTAL_CLIENTES = 50
CLIENTES_RAPIDOS = 10
CLIENTES_LENTOS = TOTAL_CLIENTES - CLIENTES_RAPIDOS
TEMPOS_CLIENTES_LENTOS = [10, 12, 13] ### TODO: refinar com tempos empiricos
TEMPOS_CLIENTES_RAPIDOS = [2, 4, 5] ### TODO: refinar com tempos empiricos

x = [*range (1, 501)]; y = []; latencia_total = 0
for rodada in range (RODADAS):
  if (rodada < 100):
    latencia_rodada = random.choice (TEMPOS_CLIENTES_RAPIDOS * CLIENTES_RAPIDOS)
  else:
    latencia_rodada = random.choice (TEMPOS_CLIENTES_LENTOS * CLIENTES_LENTOS)
  latencia_total += latencia_rodada
  y.append (latencia_total)
plt.plot(x, y, 'r-', markersize=6, label='Clientes mais rápidos primeiro')


x = [*range (1, 501)]; y = []; latencia_total = 0
for rodada in range (RODADAS):
  latencia_rodada = random.choice (TEMPOS_CLIENTES_LENTOS * CLIENTES_LENTOS + TEMPOS_CLIENTES_RAPIDOS * CLIENTES_RAPIDOS)
  latencia_total += latencia_rodada
  y.append (latencia_total)
plt.plot(x, y, 'b-', markersize=6, label='Latência convencional')


plt.legend (loc='upper center', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           #ncol=2, mode="expand", borderaxespad=0.)
#plt.legend(loc='upper center', bbox_to_anchor=(1, 1))
plt.xlabel ('Rodada')
plt.ylabel ('Latência total em segundos')
plt.xlim (0, 502)
#plt.ylim (0, 500)
plt.tight_layout()
plt.show()
#plt.savefig ('unified_monte_carlo.pdf')
#print ('saved')

import random
from random import randint
from random import seed
from matplotlib import pyplot as plt
import sys
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


import re
accuracies = []
with open('centralized.log', 'r') as file:
  for line in file:
    match = re.search(r'val_accuracy: ([0-9.]+)', line)
    if match:
      val_accuracy = float(match.group(1))
      accuracies.append(val_accuracy)
      print(f'Found val_accuracy: {val_accuracy}!!!')


acc1, acc2, acc3, acc4, acc5, acc6 = [accuracies[i:i + 10] for i in range(0, len(accuracies), 10)]


x = list(range(1, 11))
plt.plot(x, acc1, label=r'B_c=64, $\eta$=1e-2')
plt.plot(x, acc2, label=r'B_c=128, $\eta$=1e-4')
plt.plot(x, acc3, label=r'B_c=256, $\eta$=1e-2')
plt.plot(x, acc4, label=r'B_c=64, $\eta$=1e-4')
plt.plot(x, acc5, label=r'B_c=128, $\eta$=1e-2')
plt.plot(x, acc6, label=r'B_c=256, $\eta$=1e-4')


plt.xlabel("Época")
plt.ylabel("Acurácia")

plt.legend (loc='lower right', ncol=1, frameon=False, markerfirst=True, labelcolor='black')
plt.xlim (0, 10)
plt.tight_layout()
#plt.show()
plt.savefig ('centralized.pdf')
print ('saved')

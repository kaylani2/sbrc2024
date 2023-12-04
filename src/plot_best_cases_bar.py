import matplotlib.pyplot as plt
import numpy as np
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

FedAVG = [0.9544000029563904, 0.972000002861023, 0.9735999703407288, 0.9593999981880188, 0.9761999845504761, 0.9801999926567078, 0.9767000079154968, 0.9624999761581421, 0.9668999910354614, 0.9010999798774719]

THREE_FL_6  = [0.9592999815940857, 0.9814000129699707, 0.9760000109672546, 0.9613000154495239, 0.9724000096321106, 0.9728999733924866, 0.9790999889373779, 0.9537000060081482, 0.9702000021934509, 0.9807999730110168]
THREE_FL_12 = [0.9078999757766724, 0.9168999791145325, 0.9435999989509583, 0.9157000184059143, 0.9207000136375427, 0.9064000248908997, 0.9171000123023987, 0.9283000230789185, 0.9552000164985657, 0.9240000247955322]
THREE_FL_19 = [0.9641000032424927, 0.9509999752044678, 0.9562000036239624, 0.9782999753952026, 0.9814000129699707, 0.9778000116348267, 0.9817000031471252, 0.9822999835014343, 0.9812999963760376, 0.9812999963760376]

DOFL_6  = [0.9779000282287598, 0.9326000213623047, 0.9717000126838684, 0.9779999852180481, 0.9634000062942505, 0.9645000100135803, 0.9724000096321106, 0.9718000292778015, 0.9689000248908997, 0.9714999794960022]
DOFL_12 = [0.973800003528595, 0.9828000068664551, 0.9858999848365784, 0.98580002784729, 0.9510999917984009, 0.9706000089645386, 0.963699996471405, 0.9832000136375427, 0.9473999738693237, 0.984000027179718]
DOFL_19 = [0.9779000282287598, 0.8925999999046326, 0.9545999765396118, 0.9196000099182129, 0.9689000248908997, 0.9560999870300293, 0.9699000120162964, 0.9476000070571899, 0.9661999940872192, 0.89410001039505]


hybrid_6  = [0.9815000295639038, 0.9767000079154968, 0.9617999792098999, 0.9789000153541565, 0.9858999848365784, 0.9829999804496765, 0.9740999937057495, 0.9794999957084656, 0.9761000275611877, 0.9790999889373779]
hybrid_12 = [0.8982999920845032, 0.9627000093460083, 0.896399974822998, 0.9110000133514404, 0.9638000130653381, 0.927299976348877, 0.932699978351593, 0.9546999931335449, 0.8992000222206116, 0.9343000054359436]
hybrid_19 = [0.9793000221252441, 0.9778000116348267, 0.965399980545044, 0.9617999792098999, 0.9700000286102295, 0.9711999893188477, 0.9754999876022339, 0.9616000056266785, 0.9693999886512756, 0.9696999788284302]

# All data lists grouped together
data = [FedAVG, THREE_FL_6, THREE_FL_12, THREE_FL_19, DOFL_6, DOFL_12, DOFL_19, hybrid_6, hybrid_12, hybrid_19]

# Calculate means and standard deviations for each list
means = [np.mean(lst) for lst in data]
std_devs = [np.std(lst) for lst in data]

# Number of groups and the width for the bars
num_groups = len(means)
bar_width = 0.55

# The x-axis positions for the groups
index = np.arange(num_groups)

# Defining colors for each group of bars
colors = ['lightcoral', 'lightskyblue', 'lightskyblue', 'lightskyblue', 'lightgreen', 'lightgreen', 'lightgreen', 'lightpink', 'lightpink', 'lightpink']

# Creating the bar chart with specified colors for each group
plt.bar(index, means, bar_width, yerr=std_devs, capsize=8, color=colors)#, label='Acurácia média nas útlimas 10 rodadas')

# Adding mean values on top of each bar
for i, v in enumerate(means):
    plt.text(i, v/2, str(round(v, 2)), ha='center', va='bottom', fontsize=16)

# Customizing the chart
plt.gcf().set_size_inches(12, 6)
plt.xlabel('Cenário', fontsize=20)
plt.ylabel('Acurácia', fontsize=20)
plt.xticks(index, ['FedAVG', '3FL 25%', '3FL 50%', '3FL 75%', 'DOFL 25%', 'DOFL 50%', 'DOFL 75%', 'Hybrid-FL 25%', 'Hybrid-FL 50%', 'Hybrid-FL 75%'], rotation=45)
#plt.legend()

# Show the plot
plt.tight_layout()
#plt.show()
filename='all_results.pdf'
plt.savefig (filename)
print ('saved')

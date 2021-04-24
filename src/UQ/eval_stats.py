import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import norm

#pathway to the data 
path = '../../data/'

#Viremia data
dadosViremia = pd.read_csv(path+'dataset_viremia.csv',',')
virus_mean=np.log10(dadosViremia['Mean']+1)
virus_max=np.log10(dadosViremia['Max']+1)-np.log10(dadosViremia['Mean']+1)
virus_min=np.log10(dadosViremia['Mean']+1)-np.log10(dadosViremia['Min']+1)

#Antibodies Data
dadosAnticorposLog2_avg = pd.read_csv(path+'IgG_IgM_data.csv',',')
antibody_g_mean = np.log2(dadosAnticorposLog2_avg['IgG']+1)
antibody_m_mean = np.log2(dadosAnticorposLog2_avg['IgM']+1)

antibody_g_min=antibody_g_mean-np.log2(dadosAnticorposLog2_avg['IgG_25']+1)
antibody_g_max=np.log2(dadosAnticorposLog2_avg['IgG_975']+1)-antibody_g_mean
antibody_m_min=antibody_m_mean-np.log2(dadosAnticorposLog2_avg['IgM_25']+1)
antibody_m_max=np.log2(dadosAnticorposLog2_avg['IgM_975']+1)-antibody_m_mean

#cytokine data
dadosCitocina = pd.read_csv(path+'dataset_il6.csv',',')
cytokineSurvivor = dadosCitocina['Survivor']
cytokineSurvivor_min = cytokineSurvivor - dadosCitocina['MinSurvivor']
cytokineSurvivor_max = dadosCitocina['MaxSurvivor'] - cytokineSurvivor

cytokineNonSurvivor = dadosCitocina['NonSurvivor']
cytokineNonSurvivor_min = cytokineNonSurvivor- dadosCitocina['MinNonSurvivor']
cytokineNonSurvivor_max = dadosCitocina['MaxNonSurvivor']- cytokineNonSurvivor

#table 1 paper
n_survivor = 137
n_nonsurvivor = 54 



std_survivor = (dadosCitocina['MaxSurvivor'] - dadosCitocina['MinSurvivor'])/(2.0*norm.ppf((n_survivor-0.375)/(n_survivor+0.25)))
std_nonsurvivor = (dadosCitocina['MaxNonSurvivor'] - dadosCitocina['MinNonSurvivor'])/(2.0*norm.ppf((n_nonsurvivor-0.375)/(n_nonsurvivor+0.25)))



print(std_survivor)

print(dadosCitocina)

dadosCitocina=dadosCitocina.assign(STD_Survivor=std_survivor)
dadosCitocina=dadosCitocina.assign(STD_NonSurvivor=std_nonsurvivor)

print(dadosCitocina)

dadosCitocina.to_csv('teste.csv', index=False)
'''
dadosIgs = pd.read_csv(path+'IgG_IgM_21_1b.csv')



arr = []
arr.append(dadosIgs.loc[dadosIgs['Interval'] == '0-7'])
arr.append(dadosIgs.loc[dadosIgs['Interval'] == '8-14'])
arr.append(dadosIgs.loc[dadosIgs['Interval'] == '15-21'])
arr.append(dadosIgs.loc[dadosIgs['Interval'] == '22-27'])

arr_igg_std = []
arr_igm_std = []

for i in range(len(arr)):
	arr_igg_std.append(arr[i]['IgG'].std())
	arr_igm_std.append(arr[i]['IgM'].std())
	
print(arr_igg_std)

dadosAnticorposLog2_avg = dadosAnticorposLog2_avg.assign(IgG_STD=arr_igg_std)
dadosAnticorposLog2_avg = dadosAnticorposLog2_avg.assign(IgM_STD=arr_igm_std)
'''


'''
#x_from_max = (np.log10(dadosViremia['Max']+1)-np.log10(dadosViremia['Mean']+1))
x_from_max = (dadosViremia['Max']-dadosViremia['Mean'])
#print(x+dadosViremia['Mean'])

#x_from_min = (np.log10(dadosViremia['Mean']+1)-np.log10(dadosViremia['Min']+1))
x_from_min = (dadosViremia['Mean']-dadosViremia['Min'])
#x_from_min = 10**x_from_min-1
#print(x+dadosViremia['Mean'])

x = (virus_min+virus_max)/2.0
#x = 10.0**x-1

dadosViremia = dadosViremia.assign(Log10_STD = x)
print(dadosViremia)



plt.figure();
plt.title("Virus")
plt.errorbar(dadosViremia['Day'], virus_mean, yerr=[virus_min, virus_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)
plt.errorbar(dadosViremia['Day'], virus_mean, yerr=[x, x],linestyle='None', label='Data', fmt='o', color='blue', capsize=4, elinewidth=2)
#plt.yscale('log');
plt.legend(loc="best",prop={'size': 13})
plt.grid(True)
plt.tight_layout() 
plt.show()   
'''

#dadosViremia.to_csv('teste.csv', index=False);

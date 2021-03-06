#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:06:01 2019

@author: carlabonin
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from NovoModelo import *
from scipy.stats.mstats import gmean
import pylab
import math
sns.set_style("whitegrid")

output_path= './output/'


V0 = 4.59722506e+01
Ap0 = 1.0e6#0.6e6
Apm0 = 0.0
Ai0=0
C0=0.0
Thn0 = (1.0e6)#*0.5
The0 = 0.0  #### Convertendo de ul para ml
Tkn0 = (1.0e3)*500.0#(1.0e3)*500.0
Tke0 = 0.0
B0 =  (1.0e3)*250.0
Ps0 = 0.0
Pl0 = 0.0
Bm0 = 0.0
A0_M = 0.0  
A0_G = 0.0
P0 = [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G,Ai0,C0]


pi_v = 1.091710061112672880e-01
c_v1 = 2.63
c_v2 = 0.60
k_v1 = 5.600298025616778555e-09
k_v2 = 6.011588247777179580e-09
alpha_Ap = 1.87E-06*0.4
beta_Ap = 2.00E-05
k_ap1 = 0.8  
k_ap2 = 40.0

delta_Apm = 8.14910996e+00 
alpha_Tn =2.17E-04 
pi_T = 1.431849023090428446e-05
k_te1 = 1.0E-08 
delta_te = 0.0003
alpha_B = 3.578236584371140339e+02
pi_B1 = 8.979145365768647095e-05
pi_B2 = 1.27E-8

beta_S = 0.000672 
beta_L = 5.61E-06 
beta_Bm = 1.0E-06
delta_S = 1.5
delta_L = 0.8
gamma_M = (1.95E-06)*500.0
k_bm1 = 1.0e-5      
k_bm2 = 2500.0 
pi_AS = 2.850370072424884479e-02
pi_AL = 6.304459239904726120e-01
delta_A_G = 3.650482092015642221e-01
delta_A_M = 6.873347140815699419e+00
c11 = 2.17E-04
c12 = 1.0E-07
c13 = 1.0E-08  
c14 = 0.22
Ap0 = 1.0e6
Thn0 = 1.0e6
Tkn0 = 5.0e5
B0 = 2.5E5
pi_c_apm = 4.52515051e-01
pi_c_i = 1.96382616e-03
pi_c_tke = 0.04730172
delta_c = 9.94728039e+00
k_apm = 1.51576968e-1
k_v3 = 3.25225701e-09
k_tk = 1.80727212e-01

model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1, k_ap2,
    delta_Apm, alpha_Tn, pi_T, k_te1, delta_te, alpha_B, pi_B1, pi_B2, 
    beta_S, beta_L, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_A_G, delta_A_M, c11, c12, c13, c14, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, k_apm, k_v3, k_tk)
    



dias_de_simulação = 35

t=range(dias_de_simulação)    
y,d=integrate.odeint(immune_response_v3, P0, t, args=(model_args), full_output=1, printmessg=True)


#######   Viremia  log10 copias/ml ##########
dadosViremiaLog10 = pd.read_csv('../../data/Viral_load_paper.csv',',')
dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 
dadosCitocinaObitos = pd.read_csv('../../data/IL6_non-survivors_19.csv',',')
dadosCitocinaSobreviventes = pd.read_csv('../../data/IL6_survivors_19.csv',',')


dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM.csv',',')
antibody_g = dadosAnticorposLog2_avg['IgG']
antibody_m = dadosAnticorposLog2_avg['IgM']

media_igG =[]
media_igM = []

#correto calcular media geometrica de valores em log2?
for x,df in dadosAnticorposLog2.groupby(['Interval']):
    media_igG.append({'grupo':x,
                    'm_igg':gmean(df['IgG'])})
    media_igM.append({'grupo':x,
                    'm_igm':gmean(df['IgM'])})

media_igG = pd.DataFrame(media_igG)
media_igM = pd.DataFrame(media_igM)
means_igG= [media_igG.iloc[0,1],media_igG.iloc[1,1],media_igG.iloc[2,1],media_igG.iloc[3,1]]
means_igM= [media_igM.iloc[0,1],media_igM.iloc[1,1],media_igM.iloc[2,1],media_igM.iloc[3,1]]

#tirando dia 0 para a media geometrica dos 7 primeiros dias ser diferente de zero
n=4
means_numerica_igM =[np.log2(gmean(y[1*n:7*n,11])+1),np.log2(gmean(y[8*n:14*n,11])+1),np.log2(gmean(y[15*n:21*n,11])+1),np.log2(gmean(y[22*n:27*n,11])+1)]
means_numerica_igG =[np.log2(gmean(y[1*n:7*n,12])+1),np.log2(gmean(y[8*n:14*n,12])+1),np.log2(gmean(y[15*n:21*n,12])+1),np.log2(gmean(y[22*n:27*n,12])+1)]
dataset=pd.melt(dadosAnticorposLog2,id_vars=['Interval'], var_name='Antibody Type',value_name='Log 2 (Antibody Level)')
sns.boxplot(x='Interval', y='Log 2 (Antibody Level)',data=dataset,hue='Antibody Type', palette="Set3",showfliers=False, order=['0-7','8-14','15-21','22-27'])
pylab.scatter([-0.2,0.8,1.8,2.8], means_igG,marker='D',color='red',label='GMT Dados IgG')
pylab.scatter([0.2,1.2,2.2,3.2], means_igM,marker='D',color='blue',label='GMT Dados IgM')
pylab.scatter([-0.2,0.8,1.8,2.8], means_numerica_igG,marker='D',color='green',label='GMT Numerico IgG')
pylab.scatter([0.2,1.2,2.2,3.2], means_numerica_igM,marker='D',color='yellow',label='GMT Numerico IgM')
plt.legend()
plt.savefig(output_path+'AnticorposBoxPlot.pdf',bbox_inches='tight',dpi = 300)    


plt.figure('Anticorpos')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log2(y[:,11]+1),label='IgM Modelo',linewidth=1.5, linestyle="-")
plt.plot(t, np.log2(antibody_m+1), 'o', label='IgM data', linewidth=4)
plt.plot(t,np.log2(y[:,12]+1),label='IgG Modelo',linewidth=1.5, linestyle="-")
plt.plot(t, np.log2(antibody_g+1), 'o', label='IgG data', linewidth=4)
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('Anticorpos - log 2 S/CO')
plt.legend()
plt.savefig(output_path+'Anticorpos.pdf',bbox_inches='tight',dpi = 300)


plt.figure('Viremia')
#dadosViremiaLog10.plot.scatter(x='Day',y='Viral_load',color='m',label='Dados experimentais')
plt.plot(t, dadosViremiaLog10['Plot'], 'o', label='data', linewidth=4)
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log10(y[:,0]),label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Viremia')
plt.legend()
plt.savefig(output_path+'Viremia.pdf',bbox_inches='tight',dpi = 300)


plt.figure('Ap')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,1],label='Ap',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Apresentadores')
plt.legend()
plt.savefig(output_path+'Ap.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Apm')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,2],label='Apm',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Apresentadores Maduras')
plt.legend()
plt.savefig(output_path+'Apm.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Thn')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,3],label='Thn',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Th Naive')
plt.legend()
plt.savefig(output_path+'Thn.pdf',bbox_inches='tight',dpi = 300)

plt.figure('The')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,4],label='The',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Th Efetora')
plt.legend()
plt.savefig(output_path+'The.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Tkn')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,5],label='Tkn',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Tk Naive')
plt.legend()
plt.savefig(output_path+'Tkn.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Tke')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,6],label='Tke',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Tk Efetora')
plt.legend()
plt.savefig(output_path+'Tke.pdf',bbox_inches='tight',dpi = 300)

plt.figure('B')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,7],label='B',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('B')
plt.legend()
plt.savefig(output_path+'B.pdf',bbox_inches='tight',dpi = 300)

np.savetxt(output_path+'b.txt',y[:,7])

plt.figure('Ps')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,8],label='Ps',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Ps')
plt.legend()
plt.savefig(output_path+'Ps.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Pl')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,9],label='Pl',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Pl')
plt.legend()
plt.savefig(output_path+'Pl.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Bm')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,10],label='Bm',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Bm')
plt.legend()
plt.savefig(output_path+'Bm.pdf',bbox_inches='tight',dpi = 300)


plt.figure('A_M')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,11],label='Ai',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('A_M')
plt.legend()
plt.savefig(output_path+'A_M.pdf',bbox_inches='tight',dpi = 300)

plt.figure('A_M')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,12],label='Ai',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('A_G')
plt.legend()
plt.savefig(output_path+'A_G.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Ai')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,13],label='Ai',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('APC infectada')
plt.legend()
plt.savefig(output_path+'Ai.pdf',bbox_inches='tight',dpi = 300)

plt.figure('C')
#dadosCitocinaSobreviventes.plot.scatter(x='Day',y='IL6(pg/mL)',color='g',label='Dados experimentais(Sobreviventes)')
plt.plot(dadosCitocinaSobreviventes['Day']+5, dadosCitocinaSobreviventes['IL6(pg/mL)'], 'o', label='data', linewidth=4)
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,14],label='C',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('C')
plt.legend()
plt.savefig(output_path+'C.pdf',bbox_inches='tight',dpi = 300)

plt.figure('C')
dadosCitocinaObitos.plot.scatter(x='Day',y='IL6(pg/mL)',color='y',label='Dados experimentais(Obito)')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,14],label='C',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('C')
plt.legend()
plt.savefig(output_path+'C2.pdf',bbox_inches='tight',dpi = 300)

#plt.show() 

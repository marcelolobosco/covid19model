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

#######  CONDIÇÕES INICIAIS PARA AJUSTE DE UNIDADES
V0 = 1.0#  infectious dose # 27476.0 IU/0.5ml  ---->  5 IU/ml diluído em 5500ml de sangue do corpo  ----> 
###  x1.91 = 9.55 PFU/ml no corpo = 0.98log10 PFU/ml no corpo   ------> 3.89 log10 copias/ml no corpos (*) = 7728.0 copias/ml
###(*)  log10 PFU/ml = [0.974 x log10 copias/ml] - 2.807
Ap0 = 1.0e6#0.6e6
Apm0 = 0.0
Ai0=0
C0=0
Thn0 = (1.0e6)#*0.5
The0 = 0.0  #### Convertendo de ul para ml
Tkn0 = (1.0e3)*500.0#(1.0e3)*500.0
Tke0 = 0.0
B0 =  (1.0e3)*250.0#125000.0#
Ps0 = 0.0
Pl0 = 0.0
Bm0 = 0.0
A0_M = 0.0  
A0_G = 0.0
# pi_v, c_v1, c_v2, k_v1, k_v2 antes era 7.48e-7, alpha_Ap, beta_Ap,
# k_ap1, k_ap2, delta_Apm, alpha_Tn, pi_T, k_te1, delta_te,
# alpha_B, pi_B1, pi_B2, beta_S, beta_L, beta_Bm,
# delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
# pi_AL, delta_A_G, delta_A_M antes 0.1, c11, c12, c13, c14, Ap0, Thn0, Tkn0, B0, pi_c_apm
# pi_c_i,pi_c_tke,delta_c, k_apm, k_v3
model_args = (1.2, 2.63, 0.60, 4.5e-05, 4.5e-05, 1.87E-06*0.4, 2.00E-03, 0.8, 40.0, 8.14910996e+00, 2.17E-04, 1.9E-05, 1.0E-08, 0.1*0.003, 6.55248840e+01, 2.826E-06, 1.27E-08, 0.000672, 5.61E-07, 1.0E-08, 1.5, 0.8, (1.95E-06)*500.0, 1.0e-5, 2500.0, 0.004, 0.0005, 8.00694162e-02, 5.06889922e-00, 2.17E-04, 1.0E-04, 1.0E-08, 0.22,1.0e6, 1.0e6, 5.0e5, 2.5E5,0.000015,0.015,0.000015,0.1,0.000007,1.0e-6)

#1.3, 2.63, 0.60, 0.00004821287, 2.50E-05, 2.50E-03, 5.5e-01, 
#              0.8, 40.0, 5.38E-01, 2.17E-04, 1.0E-05, 1.0E-08,  0.0003,   
#              6.0E+00, 4.826E-06, 1.27E-8, 6.72E-05, 5.61E-06, 1.0E-06,   
#              0.2, 2.3976E-06, 9.75E-04, 1.0e-5, 2500.0, 0.015, 
#              0.0009, 0.095, 1.05, 2.17E-04, 1.0E-07, 1.0E-08, 0.22, 1.0e6, 1.0e6, 5.0e5, 2.5E5,0.000000,
#              0.0,0.0,0.0,0.0,0.0)

#1.25, 2.63, 0.60, 9.80167723e-02, 6.40994768e-05, 1.87E-06*0.4, 2.50E-03, 
#0.8, 40.0, 8.14910996e+00, 2.17E-04, 1.0E-05, 1.0E-08, 0.1*0.003, 
#6.55248840e+01, 4.826E-06, 1.27E-10*100.0, 0.000672, 5.61E-06, 1.0E-06, 
#2.0, (2.22E-04)*1.8*0.6, (1.95E-06)*500.0, 1.0e-5, 2500.0, 0.002, 
#0.00068, 8.00694162e-02, 3.06889922e-01, 2.17E-04, 1.0E-07, 1.0E-08, 0.22)

dias_de_simulação = 45

t=np.linspace(0,dias_de_simulação,180)    
y,d=integrate.odeint(immune_response, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G,Ai0,C0], t, args=(model_args), full_output=1, printmessg=True)


#######   Viremia  log10 copias/ml ##########
dadosViremiaLog10 = pd.read_csv('../../data/Viral_load_10_2.csv',',')
dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 
dadosCitocinaObitos = pd.read_csv('../../data/IL6_non-survivors_19.csv',',')
dadosCitocinaSobreviventes = pd.read_csv('../../data/IL6_survivors_19.csv',',')

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
plt.savefig('AnticorposBoxPlot.pdf',bbox_inches='tight',dpi = 300)    


plt.figure('Anticorpos')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log2(y[:,11]+1),label='IgM Modelo',linewidth=1.5, linestyle="-")
plt.plot(t,np.log2(y[:,12]+1),label='IgG Modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('Anticorpos - log 2 S/CO')
plt.legend()
plt.savefig('Anticorpos.pdf',bbox_inches='tight',dpi = 300)


plt.figure('Viremia')
dadosViremiaLog10.plot.scatter(x='Day',y='Viral_load',color='m',label='Dados experimentais')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log10(y[:,0]+1),label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Viremia')
plt.legend()
plt.savefig('Viremia.pdf',bbox_inches='tight',dpi = 300)


plt.figure('Ap')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,1],label='Ap',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Apresentadores')
plt.legend()
plt.savefig('Ap.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Apm')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,2],label='Apm',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Apresentadores Maduras')
plt.legend()
plt.savefig('Apm.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Thn')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,3],label='Thn',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Th Naive')
plt.legend()
plt.savefig('Thn.pdf',bbox_inches='tight',dpi = 300)

plt.figure('The')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,4],label='The',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Th Efetora')
plt.legend()
plt.savefig('The.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Tkn')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,5],label='Tkn',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Tk Naive')
plt.legend()
plt.savefig('Tkn.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Tke')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,6],label='Tke',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Tk Efetora')
plt.legend()
plt.savefig('Tke.pdf',bbox_inches='tight',dpi = 300)

plt.figure('B')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,7],label='B',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('B')
plt.legend()
plt.savefig('B.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Ps')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,8],label='Ps',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Ps')
plt.legend()
plt.savefig('Ps.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Pl')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,9],label='Pl',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Pl')
plt.legend()
plt.savefig('Pl.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Bm')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,10],label='Bm',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Bm')
plt.legend()
plt.savefig('Bm.pdf',bbox_inches='tight',dpi = 300)

plt.figure('Ai')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,13],label='Ai',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('APC infectada')
plt.legend()
plt.savefig('Ai.pdf',bbox_inches='tight',dpi = 300)

plt.figure('C')
dadosCitocinaSobreviventes.plot.scatter(x='Day',y='IL6(pg/mL)',color='g',label='Dados experimentais(Sobreviventes)')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,14],label='C',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('C')
plt.legend()
plt.savefig('C.pdf',bbox_inches='tight',dpi = 300)

plt.figure('C')
dadosCitocinaObitos.plot.scatter(x='Day',y='IL6(pg/mL)',color='y',label='Dados experimentais(Obito)')
plt.xlim(0.0,dias_de_simulação)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,14],label='C',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('C')
plt.legend()
plt.savefig('C2.pdf',bbox_inches='tight',dpi = 300)

#plt.show() 

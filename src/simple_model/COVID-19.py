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
from Modelo import *
from scipy.stats.mstats import gmean
import pylab
import math
sns.set_style("whitegrid")

#######  CONDIÇÕES INICIAIS PARA AJUSTE DE UNIDADES
V0 = 10 #  infectious dose # 27476.0 IU/0.5ml  ---->  5 IU/ml diluído em 5500ml de sangue do corpo  ----> 
###  x1.91 = 9.55 PFU/ml no corpo = 0.98log10 PFU/ml no corpo   ------> 3.89 log10 copias/ml no corpos (*) = 7728.0 copias/ml
###(*)  log10 PFU/ml = [0.974 x log10 copias/ml] - 2.807
Ap0 = 1.0e6
Apm0 = 0.0
Thn0 = (1.0e6)#*0.5
The0 = 0.0  #### Convertendo de ul para ml
Tkn0 = 5.0e5
Tkn0 = (1.0e3)*500.0#(1.0e3)*500.0
Tke0 = 0.0
B0 =  (1.0e3)*250.0#125000.0#
Ps0 = 0.0
Pl0 = 0.0
Bm0 = 0.0
A0_M = 0.0  
A0_G = 0.0

model_args = (1.25, 2.63, 0.60, 0.000120532191*0.4, 1.87E-06*0.4, 2.50E-03, 5.5e-01,
                     0.8, 40.0, 5.38E-01, 2.17E-04, 1.0E-05, 1.0E-08, 0.1*0.003,
                     6.0E+00, 4.826E-06, 1.27E-10*100.0, 0.000672, 5.61E-06, 1.0E-06,
                     2.0, (2.22E-04)*1.8*0.6, (1.95E-06)*500.0, 1.0e-5, 2500.0, 0.002,
                     0.00068, 0.04, 0.08, 2.17E-04, 1.0E-07, 1.0E-08, 0.22)

t=np.linspace(0,45,20000)    
y,d=integrate.odeint(immune_response, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G], t, args=(model_args), full_output=True, printmessg=True)


#######   Viremia  log10 copias/ml ##########
dadosViremiaLog10 = pd.read_csv('../../data/Viral_load_10_2.csv',';')
dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 

media_igG =[]
media_igM = []
for x,df in dadosAnticorposLog2.groupby(['Interval']):
    media_igG.append({'grupo':x,
                    'm_igg':gmean(df['IgG'])})
    media_igM.append({'grupo':x,
                    'm_igm':gmean(df['IgM'])})

media_igG = pd.DataFrame(media_igG)
media_igM = pd.DataFrame(media_igM)
means_igG= [media_igG.iloc[0,1],media_igG.iloc[1,1],media_igG.iloc[2,1],media_igG.iloc[3,1]]
means_igM= [media_igM.iloc[0,1],media_igM.iloc[1,1],media_igM.iloc[2,1],media_igM.iloc[3,1]]

dataset=pd.melt(dadosAnticorposLog2,id_vars=['Interval'], var_name='Antibody Type',value_name='Log 2 (Antibody Level)')

sns.boxplot(x='Interval', y='Log 2 (Antibody Level)',data=dataset,hue='Antibody Type', palette="Set3",showfliers=False, order=['0-7','8-14','15-21','22-27'])
pylab.scatter([-0.2,0.8,1.8,2.8], means_igG,marker='D',color='red',label='GMT Dados IgG')
pylab.scatter([0.2,1.2,2.2,3.2], means_igM,marker='D',color='blue',label='GMT Dados IgM')
    
plt.figure('CurvaAjuste1')
plt.xlim(0.0,45.0)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log2(y[:,11]),label='IgM Modelo',linewidth=1.5, linestyle="-")
plt.plot(t,np.log2(y[:,12]),label='IgG Modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('Anticorpos - log 2 S/CO')
plt.legend()
#plt.savefig('FigurasTese/Primovacinados/AnticorposPrimovacinadosCurvaAjuste1.pdf',bbox_inches='tight',dpi = 300)

#plt.figure('CurvaAjuste2')
dadosViremiaLog10.plot.scatter(x='Day',y='Viral_load',color='m',label='Dados experimentais')
plt.xlim(0.0,45.0)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log10(y[:,0]),label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Viremia')
plt.legend()
#plt.savefig('FigurasTese/Primovacinados/AnticorposPrimovacinadosCurvaAjuste2.pdf',bbox_inches='tight',dpi = 300)

plt.show() 

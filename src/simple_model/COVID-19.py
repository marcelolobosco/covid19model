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

t=np.linspace(0,45,20000)    
y,d=integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0], t, full_output=True, printmessg=True)


#######   Viremia  log10 copias/ml ##########
dadosViremiaLog10 = pd.read_csv('../../data/Viral_load_10_2.csv',';')
dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 

media_igG =[]
media_igM = []
for x,df in dadosAnticorposLog2.groupby(['Group']):
    media_igG.append({'grupo':x,
                    'm_igg':gmean(df['IgG'])})
    media_igM.append({'grupo':x,
                    'm_igm':gmean(df['IgM'])})

media_igG = pd.DataFrame(media_igG)
media_igM = pd.DataFrame(media_igM)
means_igG= [media_igG.iloc[0,1],media_igG.iloc[1,1],media_igG.iloc[2,1],media_igG.iloc[3,1]]
means_igM= [media_igM.iloc[0,1],media_igM.iloc[1,1],media_igM.iloc[2,1],media_igM.iloc[3,1]]  

sns.boxplot(x="Group", y="IgG",data=dadosAnticorposLog2, palette="BuGn_r",showfliers=False,order=['0-7','8-14','15-21','22-27'])
sns.boxplot(x="Group", y="IgM",data=dadosAnticorposLog2, palette="BuGn_r",showfliers=False,order=['0-7','8-14','15-21','22-27'])
pylab.scatter([0,1,2,3], means_igG,marker='D',color='red',label='GMT Dados IgG')
pylab.scatter([0,1,2,3], means_igM,marker='D',color='blue',label='GMT Dados IgM')
    
plt.figure('CurvaAjuste1')
plt.xlim(0.0,45.0)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log10(y[:,11]),label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('Anticorpos - log 10 mUI/ml')
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

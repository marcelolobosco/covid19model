#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:06:01 2019

@author: carlabonin
"""
#python3 COVID-19.py n => executa o modelo para os non-survivors
#python3 COVID-19.py => executa o modelo para os survivors

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ModeloCitocina import *
from scipy.stats.mstats import gmean
import pylab
import math
import sys
sns.set_style("whitegrid")

#######  CONDIÇÕES INICIAIS PARA AJUSTE DE UNIDADES
V0 = 100 #  infectious dose # 27476.0 IU/0.5ml  ---->  5 IU/ml diluído em 5500ml de sangue do corpo  ----> 
###  x1.91 = 9.55 PFU/ml no corpo = 0.98log10 PFU/ml no corpo   ------> 3.89 log10 copias/ml no corpos (*) = 7728.0 copias/ml
###(*)  log10 PFU/ml = [0.974 x log10 copias/ml] - 2.807
Ap0 = 1.0e6
Apm0 = 0.0
Thn0 = (1.0e6)#*0.5
The0 = 0.0  #### Convertendo de ul para ml
Tkn0 = (1.0e3)*500.0#(1.0e3)*500.0
Tke0 = 0.0
B0 =  0.0#(1.0e3)*250.0#125000.0#
Ps0 = 0.0
Pl0 = 0.0
Bm0 = 0.0
A0_M = 0.0  
A0_G = 0.0
C0 = 0.0

#            (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1, k_ap2, delta_Apm, alpha_Tn, pi_T, k_te1, delta_te, alpha_B, pi_B1, pi_B2, beta_S, beta_L, beta_Bm, delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS, pi_AL, delta_A_G, delta_A_M, c11, c12, c13, c14, pi_capm, pi_cthe, pi_ctke, pi_cb, delta_c, Ap0, Thn0, Tkn0, B0)

#Modelo sem citocina, argumentos:
#model_args = (1.32, 2.63, 0.60, 0.000120532191*0.4, 1.87E-06*0.4, 2.50E-03, 5.5e-01, 0.8, 40.0,      5.38E-01, 2.17E-04, 1.0E-05, 1.0E-08,  0.0003,   6.0E+00, 4.826E-06, 1.27E-8, 0.000672, 5.61E-06, 1.0E-06,   2.0, 2.3976E-04, 9.75E-04, 1.0e-5, 2500.0, 0.002,0.00068,     0.01,       4.826E-06, 2.17E-04, 1.0E-07, 1.0E-08, 0.22, 1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e6, 5.0e5, 2.5E5)#,2.63, 0.60, 9.80167723e-02, 6.40994768e-05, 1.87E-06*0.4, 2.50E-03, 5.5e-01, 0.8, 40.0, 8.14910996e+00, 2.17E-04, 1.0E-05, 1.0E-08, 0.1*0.003, 6.55248840e+01, 4.826E-06, 1.27E-10*100.0, 0.000672, 5.61E-06, 1.0E-06, 2.0, (2.22E-04)*1.8*0.6, (1.95E-06)*500.0, 1.0e-5, 2500.0, 0.002, 0.00068, 8.00694162e-02, 3.06889922e-01, 2.17E-04, 1.0E-07, 1.0E-08, 0.22)

#modelo com citocina-naoSobrevivente, argumentos:
if sys.argv[-1] == 'n':
    #            (pi_v, c_v1, c_v2, k_v1              , k_v2        , alpha_Ap, beta_Ap, k_ap1, k_ap2, delta_Apm, alpha_Tn,  pi_T  , k_te1  , delta_te, alpha_B,   pi_B1  , pi_B2  , beta_S  , beta_L  , beta_Bm, delta_S,  delta_L  , gamma_M , k_bm1 , k_bm2 , pi_AS, pi_AL , delta_A_G, delta_A_M,   c11   ,  c12   ,   c13  , c14 , pi_capm, pi_cthe, pi_ctke, pi_cb, delta_c, Ap0  , Thn0 , Tkn0 , B0)
    model_args = (1.32, 2.63, 0.60, 0.000120532191*0.4, 1.87E-06*0.4, 2.50E-03, 5.5e-01, 0.8  , 40.0 , 5.38E-01 , 2.17E-04, 1.0E-05, 1.0E-08, 0.0003  , 6.0E+00, 4.826E-06, 1.27E-8, 0.000672, 5.61E-06, 1.0E-06,   2.0  , 2.3976E-04, 9.75E-04, 1.0e-5, 2500.0, 0.002,0.00068,   0.01   , 4.826E-06, 2.17E-04, 1.0E-07, 1.0E-08, 0.22, 8.0e-6 , 1.0e-5 , 1.0e-6 ,  3.0e-6 , 0.2    , 1.0e6, 1.0e6, 5.0e5, 2.5E5)
#experimentando
#pi_camp: muito efeito, range:(1.0e-6 - 5.0e-6) 
#pi_cthe: range(8.0e-6, 1.0e-5) 
#pi_ctke: range(7.0e-7, 1.0e-6) 
#pi_cb: range(1.0e-6, 9.0e-6) 
#delta_c: range(0.1-1.0)

#modelo com citocina-sobrevivente, argumentos:
else:
    #            (pi_v, c_v1 , c_v2  , k_v1  , k_v2   , alpha_Ap, beta_Ap, k_ap1, k_ap2, delta_Apm, alpha_Tn,  pi_T  , k_te1  ,delta_te,alpha_B,   pi_B1  , pi_B2  , beta_S   , beta_L  , beta_Bm, delta_S,  delta_L  , gamma_M , k_bm1 , k_bm2 , pi_AS, pi_AL , delta_A_G, delta_A_M,   c11   ,  c12   ,   c13  , c14 , pi_capm, pi_cthe, pi_ctke, pi_cb ,delta_c, Ap0  , Thn0 , Tkn0 , B0)
    model_args = (1.02, 32.63, 0.099, 4.82e-6, 8.18e-8, 2.50E-03, 5.5e-01, 0.8  , 40.0 , 5.38E-01 , 2.17E-04, 1.0E-05, 1.0E-08, 0.003  ,  6.0  , 4.826E-06, 1.27E-8, 0.0000672, 5.61E-06, 1.0E-06,   2.0  , 2.3976E-04, 9.75E-04, 1.0e-5, 2500.0, 0.002,0.00068,   0.01   , 4.826E-06, 2.17E-04, 1.0E-07, 1.0E-08, 0.22, 4.0e-6 , 1.0e-5 , 1.5e-6 , 9.0e-8,  0.27  , 1.0e6, 1.0e6, 5.0e5, 2.5E5)
#experimentando
#pi_camp: range:(3.5e-6,5.5e-6)
#pi_cthe: range(7.0e-6,1.5e-5)
#pi_ctke: range(9.5e-7,1.3e-6) 
#pi_cb: range(1.0e-6,6.0e-6) 
#delta_c: range(0.1,0.35)

t=np.linspace(0,100,360000)    
y,d=integrate.odeint(immune_response, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G, C0], t, args=(model_args), full_output=1, printmessg=True)


#######   Viremia  log10 copias/ml ##########
dadosViremiaLog10 = pd.read_csv('../../data/Viral_load_10_2.csv',',')
dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 
if sys.argv[-1] == 'n':
    dadosCitocina = pd.read_csv('../../data/IL6_non-survivors_19.csv',',')
else:
    dadosCitocina = pd.read_csv('../../data/IL6_survivors_19.csv',',')

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
means_numerica_igG =[np.log2(gmean(y[1*n:7*n,11])+1),np.log2(gmean(y[8*n:14*n,11])+1),np.log2(gmean(y[15*n:21*n,11])+1),np.log2(gmean(y[22*n:27*n,11])+1)]
means_numerica_igM =[np.log2(gmean(y[1*n:7*n,12])+1),np.log2(gmean(y[8*n:14*n,12])+1),np.log2(gmean(y[15*n:21*n,12])+1),np.log2(gmean(y[22*n:27*n,12])+1)]

dataset=pd.melt(dadosAnticorposLog2,id_vars=['Interval'], var_name='Antibody Type',value_name='Log 2 (Antibody Level)')

##plotando boxplot anticorpo
#sns.boxplot(x='Interval', y='Log 2 (Antibody Level)',data=dataset,hue='Antibody Type', palette="Set3",showfliers=False, order=['0-7','8-14','15-21','22-27'])
#pylab.scatter([-0.2,0.8,1.8,2.8], means_igG,marker='D',color='red',label='GMT Dados IgG')
#pylab.scatter([0.2,1.2,2.2,3.2], means_igM,marker='D',color='blue',label='GMT Dados IgM')
#pylab.scatter([-0.2,0.8,1.8,2.8], means_numerica_igG,marker='D',color='green',label='GMT Numerico IgG')
#pylab.scatter([0.2,1.2,2.2,3.2], means_numerica_igM,marker='D',color='yellow',label='GMT Numerico IgM')
#plt.legend()

#plotando viremia
dadosViremiaLog10.plot.scatter(x='Day',y='Viral_load',color='m',label='Dados experimentais')
plt.xlim(0.0,50.0)
plt.plot(t,np.log10(y[:,0]),label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Viremia')
plt.legend()

ts = 50.0
ti = 0.0
#plotando Apresentadora
plt.figure('CurvaAjusteAp')
plt.xlim(ti,ts)
plt.plot(t,y[:,1],label='Ap',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('Apm e Ap')

#plotando Apresentadora Madura
plt.xlim(ti,ts)
plt.plot(t,y[:,2],label='Apm',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()

#plotando T helper naive
plt.figure('CurvaAjusteTh')
plt.xlim(ti,ts)
plt.plot(t,y[:,3],label='Thn',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('Thn e The')

#plotando T helper efetora
plt.xlim(ti,ts)
plt.plot(t,y[:,4],label='The',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('Thn e The')

#plotando T killer naive
plt.figure('CurvaAjusteTk')
plt.xlim(ti,ts)
plt.plot(t,y[:,5],label='Tkn',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('Tkn e Tke')

#plotando T killer efetora
plt.xlim(ti,ts)
plt.plot(t,y[:,6],label='Tke',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()

#plotando B
plt.figure('CurvaAjusteB')
plt.xlim(ti,ts)
plt.plot(t,y[:,7],label='B',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('célula B')

#plotando plasma-short
plt.figure('CurvaAjustePs')
plt.xlim(ti,ts)
plt.plot(t,y[:,8],label='Ps',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('célula plasma-short')

#plotando plasma-long
plt.figure('CurvaAjustePl')
plt.xlim(ti,ts)
plt.plot(t,y[:,9],label='Pl',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('célula plasma-long')


#plotando B de memoria
plt.figure('CurvaAjusteBm')
plt.xlim(ti,ts)
plt.plot(t,y[:,10],label='Bm',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('numero de celulas')
plt.legend()
plt.title('célula B memória')

#plotando IgM e IgG
plt.figure('CurvaAnticorpo')
plt.xlim(ti,ts)
plt.plot(t,np.log2(y[:,11]+1),label='IgM Modelo',linewidth=1.5, linestyle="-")
plt.plot(t,np.log2(y[:,12]+1),label='IgG Modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('Anticorpos - log 2 S/CO')
plt.legend()

#plotando Citocina
dadosCitocina.plot.scatter(x='Day',y='IL6(pg/mL)',color='m',label='Dados experimentais')
plt.xlim(ti,ts)
plt.plot(t,y[:,13],label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('Cytokine')
plt.legend()
plt.title('Citokine')

plt.show() 

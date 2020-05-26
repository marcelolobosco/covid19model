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

t=np.linspace(0,50,20000)    
y,d=integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0], t, full_output=True, printmessg=True)

   


#######   PrimoVacinados  log10 mUI/ml ##########
#n=4
dadosViremiaLog10 = pd.read_csv('../../data/ViralLoad.csv',';')
#dfUmaDuasDosesLog10 = pd.read_csv('Dados/UmaDuasDosesLog10teste.csv',';')

#media_gPrimovacinados =[]
#for x,df in dfUmaDuasDosesLog10_2.groupby(['Grupo']):
#    media_gPrimovacinados.append({'grupo':x,
#                    'm_g':gmean(df['PRNTLOG10mUI_mL'])})

#media_gPrimovacinados = pd.DataFrame(media_gPrimovacinados)
#meansPrimovacinados= [media_gPrimovacinados.iloc[0,1],media_gPrimovacinados.iloc[1,1],media_gPrimovacinados.iloc[2,1],media_gPrimovacinados.iloc[3,1],media_gPrimovacinados.iloc[4,1]]
  

#sns.violinplot(x="Grupo", y="PRNTLOG10mUI_mL",data=dfUmaDuasDosesLog10, palette="Set3",showfliers=False,split=True,scale="count", inner="quartile")
#pylab.scatter([0,1,2,3,4], meansPrimovacinados,marker='D',color='red',label='GMT Dados')
#plt.title("Anticorpos Primovacinados - PRNT (log10 mUI/ml)")
#plt.suptitle('')
#plt.scatter(0,np.log10(y[0*n,11]), marker='o',color='blue',label='GMT Modelo')#  NV 0
#plt.scatter(1,np.log10(y[44*n,11]), marker='o',color='blue')# PV 30-45
#plt.scatter(2,np.log10(y[1367*n,11]), marker='o',color='blue')# PV 1-5
#plt.scatter(3,np.log10(y[2609*n,11]), marker='o',color='blue')# PV 5-9
#plt.scatter(4,np.log10(y[4081*n,11]), marker='o',color='blue')# PV 10
#plt.xlabel('Tempo pós-vacinação')
#plt.ylabel('Anticorpos - log 10 mUI/ml')
#plt.legend()
#plt.savefig('FigurasTese/Primovacinados/AnticorposPrimovacinadosViolinPlot.pdf',bbox_inches='tight', dpi = 300)

#pv30_45log = np.log10((y[42*n,11]+y[45*n,11]+y[46*n,11]+y[49*n,11])/4)
#pv30_45log2 = gmean((np.log10(y[42*n,11]),np.log10(y[45*n,11]),np.log10(y[46*n,11]),np.log10(y[49*n,11])))
#pv1_5log = np.log10((y[537*n,11]+y[573*n,11]+y[599*n,11]+y[675*n,11]+y[767*n,11]+y[801*n,11]+y[1092*n,11]+y[1135*n,11]+y[1162*n,11]+y[1170*n,11]+y[1171*n,11]+y[1382*n,11]+y[1406*n,11]+y[1415*n,11]+y[1416*n,11]+y[1470*n,11]+y[1472*n,11]+y[1499*n,11]+y[1500*n,11]+y[1584*n,11]+y[1585*n,11]+y[1587*n,11]+y[1588*n,11]+y[1589*n,11]+y[1591*n,11]+y[1691*n,11]+y[1768*n,11]+y[1782*n,11]+y[1805*n,11]+y[1833*n,11])/30)          
#pv1_5log2 = gmean((np.log10(y[537*n,11]),np.log10(y[573*n,11]),np.log10(y[599*n,11]),np.log10(y[675*n,11]),np.log10(y[767*n,11]),np.log10(y[801*n,11]),np.log10(y[1092*n,11]),np.log10(y[1135*n,11]),np.log10(y[1162*n,11]),np.log10(y[1170*n,11]),np.log10(y[1171*n,11]),np.log10(y[1382*n,11]),np.log10(y[1406*n,11]),np.log10(y[1415*n,11]),np.log10(y[1416*n,11]),np.log10(y[1470*n,11]),np.log10(y[1472*n,11]),np.log10(y[1499*n,11]),np.log10(y[1500*n,11]),np.log10(y[1584*n,11]),np.log10(y[1585*n,11]),np.log10(y[1587*n,11]),np.log10(y[1588*n,11]),np.log10(y[1589*n,11]),np.log10(y[1591*n,11]),np.log10(y[1691*n,11]),np.log10(y[1768*n,11]),np.log10(y[1782*n,11]),np.log10(y[1805*n,11]),np.log10(y[1833*n,11])))
#pv5_9log = np.log10((y[1882*n,11]+y[2318*n,11]+y[2425*n,11]+y[2462*n,11]+y[2573*n,11]+y[2715*n,11]+y[2722*n,11]+y[2729*n,11]+y[2779*n,11]+y[3406*n,11])/10)
#pv5_9log2 = gmean((np.log10(y[1882*n,11]),np.log10(y[2318*n,11]),np.log10(y[2425*n,11]),np.log10(y[2462*n,11]),np.log10(y[2573*n,11]),np.log10(y[2715*n,11]),np.log10(y[2722*n,11]),np.log10(y[2729*n,11]),np.log10(y[2779*n,11]),np.log10(y[3406*n,11])))
#pv10log = np.log10((y[3721*n,11]+y[3723*n,11]+y[3839*n,11]+y[3937*n,11]+y[3951*n,11]+y[4074*n,11]+y[4080*n,11]+y[4082*n,11]+y[4083*n,11]+y[4087*n,11]+y[4088*n,11]+y[4093*n,11]+y[4095*n,11]+y[4096*n,11]+y[4097*n,11]+y[4103*n,11]+y[4104*n,11]+y[4110*n,11]+y[4111*n,11]+y[4114*n,11]+y[4115*n,11]+y[4129*n,11]+y[4140*n,11]+y[4142*n,11]+y[4143*n,11]+y[4167*n,11]+y[4326*n,11]+y[4414*n,11])/28)
#pv10log2 = gmean((np.log10(y[3721*n,11]),np.log10(y[3723*n,11]),np.log10(y[3839*n,11]),np.log10(y[3937*n,11]),np.log10(y[3951*n,11]),np.log10(y[4074*n,11]),np.log10(y[4080*n,11]),np.log10(y[4082*n,11]),np.log10(y[4083*n,11]),np.log10(y[4087*n,11]),np.log10(y[4088*n,11]),np.log10(y[4093*n,11]),np.log10(y[4095*n,11]),np.log10(y[4096*n,11]),np.log10(y[4097*n,11]),np.log10(y[4103*n,11]),np.log10(y[4104*n,11]),np.log10(y[4110*n,11]),np.log10(y[4111*n,11]),np.log10(y[4114*n,11]),np.log10(y[4115*n,11]),np.log10(y[4129*n,11]),np.log10(y[4140*n,11]),np.log10(y[4142*n,11]),np.log10(y[4143*n,11]),np.log10(y[4167*n,11]),np.log10(y[4326*n,11]),np.log10(y[4414*n,11])))


#plt.figure('AnticorposPrimovacinadosBoxPlot')
#sns.boxplot(x="Grupo", y="PRNTLOG10mUI_mL",data=dfUmaDuasDosesLog10, palette="BuGn_r",showfliers=False,order=['NV(dia 0)','PV(30-45 dias)','PV(1-5 anos)','PV(>5-9 anos)','PV(10 anos)'])
#pylab.scatter([0,1,2,3,4], meansPrimovacinados,marker='D',color='red',label='GMT Dados')
#plt.title("Anticorpos Primovacinados - PRNT (log10 mUI/ml)")
#plt.suptitle('')
#plt.scatter(0,np.log10(y[0*n,11]), marker='o',color='blue',label='GMT Modelo')#  NV 0
#plt.scatter(1,pv30_45log2, marker='o',color='blue')# PV 30-45
#plt.scatter(2,pv1_5log2, marker='o',color='blue')# PV 1-5
#plt.scatter(3,pv5_9log2, marker='o',color='blue')# PV 5-9
#plt.scatter(4,pv10log2, marker='o',color='blue')# PV 10
#plt.xlabel('Tempo pós-vacinação')
#plt.ylabel('Anticorpos - log 10 mUI/ml')
#plt.legend()
#plt.savefig('FigurasTese/Primovacinados/AnticorposPrimovacinadosBoxPlot.pdf',bbox_inches='tight', dpi = 300)

#mediaModelo = [np.log10(y[0*n,11]),pv30_45log2,pv1_5log2,pv5_9log2,pv10log2] 
#dif = []
#for i in range(len(mediaModelo)):
#    dif.append(meansPrimovacinados[i]/mediaModelo[i])

#dif2 = []
#for i in range(len(mediaModelo)):
#    dif2.append((meansPrimovacinados[i]-mediaModelo[i])/meansPrimovacinados[i])

    
plt.figure('CurvaAjuste1')
plt.xlim(0.0,50.0)
#plt.ylim(0.0,8.0)
plt.plot(t,np.log10(y[:,11]),label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-infecção (dias)')
plt.ylabel('Anticorpos - log 10 mUI/ml')
plt.legend()
#plt.savefig('FigurasTese/Primovacinados/AnticorposPrimovacinadosCurvaAjuste1.pdf',bbox_inches='tight',dpi = 300)

plt.figure('CurvaAjuste2')
dadosViremiaLog10.plot.scatter("TEMPO_DIAS",'LOG10COPIAS_mL',color='m',label='Dados experimentais')
plt.xlim(0.0,50.0)
#plt.ylim(0.0,8.0)
plt.plot(t,y[:,0],label='Curva gerada pelo modelo',linewidth=1.5, linestyle="-")
plt.xlabel('Tempo pós-vacinação (dias)')
plt.ylabel('Viremia')
plt.legend()
#plt.savefig('FigurasTese/Primovacinados/AnticorposPrimovacinadosCurvaAjuste2.pdf',bbox_inches='tight',dpi = 300)

plt.show() 

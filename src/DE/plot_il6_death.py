#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from math import factorial as fat
from scipy.optimize import fmin,differential_evolution
import math

# Bibliotecas proprias
from expmmq import *
from mix import *
from NovoModelo import *

#pathway to the data 
path = '../../data/'

# global data
# ajustar pela viremia e anticorpos
dadosViremiaLog10 = pd.read_csv(path+'Viral_load_paper.csv',',')
dadosAnticorposLog2 = pd.read_csv(path+'IgG_IgM.csv',',')
dadosIL6 = pd.read_csv(path+'IL6_storm_ajuste.csv',',')


#Viremia data
dadosViremia = pd.read_csv('../../data/dataset_viremia.csv',',')
virus_mean=np.log10(dadosViremia['Mean']+1)
virus_max=np.log10(dadosViremia['Max']+1)-np.log10(dadosViremia['Mean']+1)
virus_min=np.log10(dadosViremia['Mean']+1)-np.log10(dadosViremia['Min']+1)

#Antibodies Data
dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM_data.csv',',')
antibody_g_mean = np.log2(dadosAnticorposLog2_avg['IgG']+1)
antibody_m_mean = np.log2(dadosAnticorposLog2_avg['IgM']+1)

antibody_g_min=antibody_g_mean-np.log2(dadosAnticorposLog2_avg['IgG_25']+1)
antibody_g_max=np.log2(dadosAnticorposLog2_avg['IgG_975']+1)-antibody_g_mean
antibody_m_min=antibody_m_mean-np.log2(dadosAnticorposLog2_avg['IgM_25']+1)
antibody_m_max=np.log2(dadosAnticorposLog2_avg['IgM_975']+1)-antibody_m_mean

#cytokine data
dadosCitocina = pd.read_csv('../../data/dataset_il6.csv',',')
cytokineSurvivor = dadosCitocina['Survivor']
cytokineSurvivor_min = cytokineSurvivor - dadosCitocina['MinSurvivor']
cytokineSurvivor_max = dadosCitocina['MaxSurvivor'] - cytokineSurvivor

cytokineNonSurvivor = dadosCitocina['NonSurvivor']
cytokineNonSurvivor_min = cytokineNonSurvivor- dadosCitocina['MinNonSurvivor']
cytokineNonSurvivor_max = dadosCitocina['MaxNonSurvivor']- cytokineNonSurvivor


first_day = 0
'''
virus = np.power(10,dadosViremiaLog10['Viral_load'])
antibody_g = np.power(2,dadosAnticorposLog2['IgG'])-1
antibody_m = np.power(2,dadosAnticorposLog2['IgM'])-1

'''

virus = np.power(10, dadosViremiaLog10['Viral_load'])
virus_plot = np.power(10, dadosViremiaLog10['Plot'])
antibody_g = dadosAnticorposLog2['IgG']
antibody_m = dadosAnticorposLog2['IgM']
il6_data = dadosIL6['IL6(pg/mL)']


# ones in days to be considered and zero otherwise
mask_virus     =[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
mask_antibodies=[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]
mask_cytokine  =[0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0]

execution_de =  []

if __name__ == "__main__":    
        	
    
    path='./output/'
    case='all_';
    
    V = np.loadtxt(path+'output_nonsurvivor_virus.txt');
    A_g = np.loadtxt(path+'output_nonsurvivor_igg.txt');
    A_m = np.loadtxt(path+'output_nonsurvivor_igm.txt');
    il6 = np.loadtxt(path+'output_nonsurvivor_c.txt');
    
    V_tke = np.loadtxt(path+'tke_output_nonsurvivor_virus.txt');
    A_g_tke = np.loadtxt(path+'tke_output_nonsurvivor_igg.txt');
    A_m_tke = np.loadtxt(path+'tke_output_nonsurvivor_igm.txt');
    il6_tke = np.loadtxt(path+'tke_output_nonsurvivor_c.txt');
    
    V_amp = np.loadtxt(path+'amp_output_nonsurvivor_virus.txt');
    A_g_amp = np.loadtxt(path+'amp_output_nonsurvivor_igg.txt');
    A_m_amp = np.loadtxt(path+'amp_output_nonsurvivor_igm.txt');
    il6_amp = np.loadtxt(path+'amp_output_nonsurvivor_c.txt');
    ts=range(len(virus)) 
    
    
    plt.style.use('estilo/PlotStyle.mplstyle')
    plt.close('all')
    
        
    plt.figure();
    #plt.title("Virus")
    plt.plot(ts, np.log10(V+1), label='Complete', linewidth=1.5)
    plt.plot(ts, np.log10(V_amp+1), label='beta_apm = 0', linewidth=1.5)
    plt.plot(ts, np.log10(V_tke+1), label='beta_tke = 0', linewidth=1.5)
    plt.errorbar(dadosViremia['Day']-1, virus_mean, yerr=[virus_min, virus_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)
    plt.legend(loc="best",prop={'size': 13})
    plt.xlabel('Time (days)')
    plt.ylabel('Viremia - $log_{10}$(copies/ml + 1)')
    plt.grid(True)
    plt.tight_layout()    
    plt.savefig(path+case+'output_nonsurvivor_virus.pdf',bbox_inches='tight',dpi = 300)
    
    
    plt.figure();
    #plt.title("IgG")
    plt.plot(ts, np.log2(A_g+1), label='Complete', linewidth=1.5)
    plt.plot(ts, np.log2(A_g_amp+1), label='beta_apm = 0', linewidth=1.5)
    plt.plot(ts, np.log2(A_g_tke+1), label='beta_tke = 0', linewidth=1.5)
    plt.errorbar(dadosAnticorposLog2_avg['Day'], antibody_g_mean, yerr=[antibody_g_min, antibody_g_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)    
    plt.legend(loc="best",prop={'size': 13})
    plt.xlabel('Time (days)')
    plt.ylabel('IgG - log$_2$(S/CO+1)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+case+'output_nonsurvivor_igg.pdf',bbox_inches='tight',dpi = 300)
    
    plt.figure();
    #plt.title("IgM")
    plt.plot(ts, np.log2(A_m+1), label='Complete', linewidth=1.5)
    plt.plot(ts, np.log2(A_m_amp+1), label='beta_apm = 0', linewidth=1.5)
    plt.plot(ts, np.log2(A_m_tke+1), label='beta_tke = 0', linewidth=1.5)
    plt.errorbar(dadosAnticorposLog2_avg['Day'], antibody_m_mean, yerr=[antibody_m_min, antibody_m_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)
    plt.legend(loc="best",prop={'size': 13})
    plt.xlabel('Time (days)')
    plt.ylabel('IgM - log$_2$(S/CO+1)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+case+'output_nonsurvivor_igm.pdf',bbox_inches='tight',dpi = 300)
    
    plt.figure();
    #plt.title("Cytokines")
    plt.plot(ts, il6, label='Complete', linewidth=1.5)
    plt.plot(ts, il6_amp, label='beta_apm = 0', linewidth=1.5)
    plt.plot(ts, il6_tke, label='beta_tke = 0', linewidth=1.5)
    plt.errorbar(dadosCitocina['Day'], cytokineNonSurvivor, yerr=[cytokineNonSurvivor_min, cytokineNonSurvivor_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)    
    plt.legend(loc="best",prop={'size': 13})
    plt.xlabel('Time (days)')
    plt.ylabel('Cytokines - (pg/mL)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+case+'output_nonsurvivor_c.pdf',bbox_inches='tight',dpi = 300)

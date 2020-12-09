#!/usr/bin/python

import sys
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import odeint, solve_ivp
#from scipy.optimize import fmin,differential_evolution
import pandas as pd

# datas
#import matplotlib.dates as mdates
#import datetime as dt

# barra de progresso =)
# sudo pip install tqdm
from tqdm import tqdm
from tqdm import tqdm_gui

# Bibliotecas proprias

from uqtools import *
from NovoModelo import *

#sobol
#from SALib.sample import saltelli
#from SALib.analyze import sobol
#from SALib.test_functions import Ishigami



#################################LEITURA DE DADOS#############################

#from COVID-19_all
#Viremia data
dadosViremia = pd.read_csv('../../data/dataset_viremia.csv',',')
virus_mean=np.log10(dadosViremia['Mean']+1)
virus_max=np.log10(dadosViremia['Max']+1)-np.log10(dadosViremia['Mean']+1)
virus_min=np.log10(dadosViremia['Mean']+1)-np.log10(dadosViremia['Min']+1)

#Antibodies Data
dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM_data.csv',',')
antibody_g = np.log2(dadosAnticorposLog2_avg['IgG']+1)
antibody_m = np.log2(dadosAnticorposLog2_avg['IgM']+1)

antibody_g_min=antibody_g-np.log2(dadosAnticorposLog2_avg['IgG_25']+1)
antibody_g_max=np.log2(dadosAnticorposLog2_avg['IgG_975']+1)-antibody_g
antibody_m_min=antibody_m-np.log2(dadosAnticorposLog2_avg['IgM_25']+1)
antibody_m_max=np.log2(dadosAnticorposLog2_avg['IgM_975']+1)-antibody_m

#cytokine data
dadosCitocina = pd.read_csv('../../data/dataset_il6.csv',',')
cytokineSurvivor = dadosCitocina['Survivor']
cytokineSurvivor_min = cytokineSurvivor - dadosCitocina['MinSurvivor']
cytokineSurvivor_max = dadosCitocina['MaxSurvivor'] - cytokineSurvivor

cytokineNonSurvivor = dadosCitocina['NonSurvivor']
cytokineNonSurvivor_min = cytokineNonSurvivor- dadosCitocina['MinNonSurvivor']
cytokineNonSurvivor_max = dadosCitocina['MaxNonSurvivor']- cytokineNonSurvivor



dadosCitocinaObitos = pd.read_csv('../../data/IL6_non-survivors_19.csv',',')
dadosCitocinaSobreviventes = pd.read_csv('../../data/IL6_survivors_19.csv',',')
dadosIL6= pd.read_csv('../../data/IL6_storm_ajuste.csv',',')
#il6_data = dadosIL6['IL6(pg/mL)']



#TEMPO
dias_de_simulação = 35
t=range(dias_de_simulação)

##############################################################################

def plot_confidence_interval_log(ax, time, evals, linecolor, textlabel):
    """
    Compute and plot statistical moments (mean,std)
    """
    mean = np.mean(evals, axis=0)
    sdev = np.std(evals, axis=0)
    perc_min = 2.5
    perc_max = 97.5
    percentile = np.percentile(evals, [perc_min, perc_max], axis=0)
    ax.plot(time, np.log2(mean+1), lw=2, color=linecolor, label=textlabel)
    ax.fill_between(time, np.log2(percentile[0,:]+1) , np.log2(percentile[1,:]+1), alpha=0.5, color=linecolor)
    return mean



def eval_model(x):
 

	#CONDICOES INICIAIS FROM COVID-19_ALL
    V0 = x[0] 
    Ap0 = 1.0e6
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
    P0 = [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G,Ai0,C0]


    #Model Parameters    
    pi_v = x[8]
    c_v1 = 2.63
    c_v2 = 0.60
    k_v1 = x[9]
    k_v2 = x[10]
    alpha_Ap = 1.0
    beta_Ap = x[11]
    c_ap1 = 8.0
    c_ap2 = 8.08579916e+06

    delta_Apm = 4.0e-2
    alpha_Tn =1.0
    beta_tk = 1.431849023090428446e-05
    pi_tk = 1.0E-08
    delta_tk = 0.0003*100
    alpha_B = 3.578236584371140339e+02
    pi_B1 = 8.979145365768647095e-05
    pi_B2 = 1.27E-8

    beta_ps = 6.0e-6
    beta_pl = 5.0e-6
    beta_Bm = 1.0E-06
    delta_S = 2.5
    delta_L = 0.35
    gamma_M = (1.95E-06)*500.0
    k_bm1 = 1.0e-5      
    k_bm2 = 2500.0 
    pi_AS = 0.087
    pi_AL = 0.001
    delta_ag = 0.07
    delta_am = 0.07
    alpha_th = 2.17E-04
    beta_th = 1.8e-5
    pi_th = 1.0E-08
    delta_th = 0.3
    #Ap0 = 1.0e6
    Thn0 = 1.0e6
    Tkn0 = 5.0e5
    B0 = 2.5E5


    pi_c_apm = x[1]
    pi_c_i = x[2]
    pi_c_tke = x[3]
    delta_c = x[4]
    beta_apm = x[5]
    k_v3 = x[6]
    beta_tke = x[7]

       
    model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, c_ap1, c_ap2,
    delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk, alpha_B, pi_B1, pi_B2, 
    beta_ps, beta_pl, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke) 
    
    
    #Ps= odeint(immune_response_v3, P0, ts, args=(model_args))  
    #V=Ps[:,0] # virus
    #A_m=Ps[:,11] # antibody
    #A_g=Ps[:,12] # antibody
    #il6 = Ps[:,14] #citocine

    def teste(t, y):
        return immune_response_v3_2(t, y, *model_args)
    
    Ps= solve_ivp(teste,(0, dias_de_simulação), P0, t_eval=t, method='Radau') 
    y = Ps.y.transpose()

    #ordem de retorno 
    #V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, C = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]
    

    return y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]

if __name__ == "__main__":



    # uso
    '''if(len(sys.argv) != 6):
        print("Usage: sirmodel [arquivo_samples] [caso] [opt_forecast] [path_output] [output extension]")
        sys.exit(0)'''
        
    
    ext = '.pdf'
    itvl = 6
    caso = 'caso.'
    opt_storm = True
    ns = 10000

    #arqpar = "./cluster/execution_de_non_survivor.txt"
    #para executar os vivos
    if (opt_storm):
        arqpar = "./execution_de_non_survivor.txt"
        output_path = './output_monte_carlo/nonsurvivors/'
    else:
        arqpar = "./execution_de_survivor.txt"
        output_path = './output_monte_carlo/survivors/'
    
    print('Arquivo de Samples: ', arqpar)
    print('Nome do output: ', output_path)
    print('Extensao do output: ', ext)

    # ajeita os dados para o caso
    data = None

    
    label_param = ("pi_v", "c_v1", "c_v2", "k_v1", "k_v2", 
        "beta_Ap", "c_ap1", "c_ap2", "delta_Apm", "beta_tk",
        "pi_tk", "delta_tk", "alpha_B","pi_B1", "pi_B2", 
        "beta_ps", "beta_pl", "beta_Bm","delta_S", "delta_L", "gamma_M", "k_bm1", "k_bm2", "pi_AS",
        "pi_AL", "delta_ag", "delta_am", "alpha_th", "beta_th", "pi_th", "delta_th", "Ap0", "Thn0", "Tkn0", "B0",  
        "pi_c_apm", "pi_c_i","pi_c_tke","delta_c", "beta_apm", "k_v3", "beta_tke")
    
    
    #--------------------MATRIZ COVARIANCIA, DISTRIBUICAO---------------------#


    label_covid = ['v0', 'pi_c_apm', 'pi_c_i', 'pi_c_tke', 'delta_c', 'beta_apm',
        'k_v3','beta_tke','pi_v','k_v1','k_v2','beta_Ap']


     #--------------------Para vivos---------------------#
    m = np.loadtxt(arqpar, comments='#')
    num_rows, num_cols = m.shape
    print(num_rows, num_cols)
    
    m=m[m[:,0].argsort()]
    m=m[:25000]
    num_rows, num_cols = m.shape
    print(num_rows, num_cols)    
    #exit()
    M= m[:,1:]

    '''
    Cov_matrix = np.cov(M.T)
    mean_vector = np.mean(M, axis=0)
    std_vector = np.std(M, axis=0)
    print("mean = ", mean_vector)
    print("std = ", std_vector)

    dist = cp.MvNormal(mean_vector, Cov_matrix)
    
    '''
    '''
    #--------------------Para mortos---------------------#
    m2 = np.loadtxt(arqpar2, comments='#')
    M2= m2[:,:-1]

    
    Cov_matrix_2 = np.cov(M2.T)
    mean_vector_2 = np.mean(M2, axis=0)
    std_vector_2 = np.std(M2, axis=0)
 
    print("mean = ", mean_vector)
    print("std = ", std_vector)

    dist2 = cp.MvNormal(mean_vector_2, Cov_matrix_2)
    
    '''
    
    #-------------------APENAS1DIST----------------------#
    #C = np.concatenate ((M,M2))
    #print("Concatenacao: ")
    #print(C)
    
    Cov_matrix = np.cov(M.T)
    mean_vector = np.mean(M, axis=0)
    std_vector = np.std(M, axis=0)
    print("mean = ", mean_vector)
    print("std = ", std_vector)

    dist = cp.MvNormal(mean_vector, Cov_matrix)
    
    #-------------------APENAS1DIST----------------------#
    #-------------------***********----------------------#
   
    #numero de parametros
    npar = len(dist)
    #npar2 = len(dist2)

    #grau da aproximacao polinomial
    degpc = 2
    print("number of input parameters two txts  %d" % npar)
    #print("number of input parameters txt 2 %d" % npar2)
    print("polynomial degree %d" % degpc)
    print("min number of samples %d" % ns)

    # create samples and evaluate them
    #samples = distribution.sample(ns,"L")
    #-----------------------------------------------------------------------------------------------------------------------#

    #----------------------------------------------------EVALS--------------------------------------------------------------#
    evals_virus = []
    evals_Ap = []
    evals_Apm = []
    evals_Thn = []
    evals_The = []
    evals_Tkn = []
    evals_Tke = []
    evals_B = []
    evals_Ps = []
    evals_Pl = []
    evals_Bm = []
    evals_AM = []
    evals_AG= []
    evals_I = []
    evals_C = []
    
    samples = dist.sample(ns,"L")
    #samples2 = dist2.sample(ns,"L")
    k = 0
    lpts = range(ns)
    samp = samples.T
    #samp2 = samples2.T
    
    print("evaluating samples: ")
    for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    #for s in samples.T:
        s = samp[k]
        #s2 = samp2[k]
        k = k+1
        #s = np.concatenate ((s,s2))
        virus,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = eval_model(s)
        evals_virus.append(virus)
        evals_Ap.append(Ap) 
        evals_Apm.append(Apm) 
        evals_Thn.append(Thn)
        evals_The.append(The) 
        evals_Tkn.append(Tkn) 
        evals_Tke.append(Tke)
        evals_B.append(B)
        evals_Ps.append(Ps)
        evals_Pl.append(Pl) 
        evals_Bm.append(Bm) 
        evals_AM.append(A_M) 
        evals_AG.append(A_G)
        evals_I.append(I) 
        evals_C.append(C)
    #-----------------------------------------------------------------------------------------------------------------------#  
    
    evals_virus = np.array(evals_virus)
    evals_Ap =  np.array( evals_Ap)
    evals_Apm = np.array(evals_Apm)
    evals_Thn = np.array(evals_Thn)
    evals_The = np.array(evals_The)
    evals_Tkn = np.array(evals_Tkn)
    evals_Tke = np.array(evals_Tke)
    evals_B =  np.array(evals_B)
    evals_Ps =  np.array(evals_Ps)
    evals_Pl =  np.array(evals_Pl)
    evals_Bm =  np.array(evals_Bm)
    evals_AM =  np.array(evals_AM)
    evals_AG= np.array(evals_AG)
    evals_I =  np.array(evals_I)
    evals_C =  np.array(evals_C)
        
    #-----------------------------------------------------------------------------------------------------------------------# 
    #
    # Plot data
    #
    plt.style.use('estilo/PlotStyle.mplstyle')
    plt.close('all')


        
    #saving the evals
   # if opt_save_evals:
        #np.savetxt(out_path+caso+'_evals_virus.txt',evals_virus)
        #np.savetxt(out_path+caso+'_evals_Ap.txt',evals_Ap)
        #np.savetxt(out_path+caso+'_evals_Apm.txt',evals_Apm)
        #np.savetxt(out_path+caso+'_evals_Thn.txt',evals_Thn)
        #np.savetxt(out_path+caso+'_evals_The.txt',evals_The)
        #np.savetxt(out_path+caso+'_evals_Tkn.txt',evals_Tkn)
        #np.savetxt(out_path+caso+'_evals_Tke.txt',evals_Tke)
        #np.savetxt(out_path+caso+'_evals_B.txt',evals_B)
        #np.savetxt(out_path+caso+'_evals_Ps.txt',evals_Ps)
        #np.savetxt(out_path+caso+'_evals_Pl.txt',evals_Pl)
        #np.savetxt(out_path+caso+'_evals_Bm.txt',evals_Bm)
        #np.savetxt(out_path+caso+'_evals_AM.txt',evals_AM)
        #np.savetxt(out_path+caso+'_evals_AG.txt',evals_AG)
        #np.savetxt(out_path+caso+'_evals_I.txt',evals_I)
        #np.savetxt(out_path+caso+'_evals_C.txt',evals_C)


    #--------------------------------------MEANS-------------------------------------------------------------------#
    #para copiar e colar caso necessario
    '''
    mean_virus = plot_confidence_interval(ax, t, evals_virus, 'red', 'Modelo: Virus')
    mean_Ap = plot_confidence_interval(ax, t, evals_Ap, 'green', 'Modelo: Ap')
    mean_Apm = plot_confidence_interval(ax, t, evals_Apm, 'blue', 'Modelo: Apm')
    mean_Thn = plot_confidence_interval(ax, t, evals_Thn, 'black', 'Modelo: Thn')
    mean_The = plot_confidence_interval(ax, t, evals_The, 'purple', 'Modelo: The')
    mean_Tkn = plot_confidence_interval(ax, t, evals_Tkn, 'orange', 'Modelo: Tkn')
    mean_Tke = plot_confidence_interval(ax, t, evals_Tke, 'red', 'Modelo: Tke')
    mean_B = plot_confidence_interval(ax, t, evals_B, 'green', 'Modelo: B')
    mean_Ps = plot_confidence_interval(ax, t, evals_Ps, 'blue', 'Modelo: Ps')
    mean_Pl = plot_confidence_interval(ax, t, evals_Pl, 'black', 'Modelo: Pl')
    mean_Bm = plot_confidence_interval(ax, t, evals_Bm, 'purple', 'Modelo: Bm')
    mean_AM = plot_confidence_interval(ax, t, evals_AM, 'orange', 'Modelo: A_M')
    mean_AG = plot_confidence_interval(ax, t, evals_AG, 'red', 'Modelo: A_G')
    mean_I = plot_confidence_interval(ax, t, evals_I, 'green', 'Modelo: I')
    mean_C = plot_confidence_interval(ax, t, evals_C, 'blue', 'Modelo: C')
    '''
    #---------------------------------------------------------------------------------------------------------------#
    #usuais de plot
    '''
    ax.set_xlabel('dia')
    ax.set_ylabel('população')
    ax.legend(loc='upper left', prop={'size':13})
    plt.title("Solution -")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path+caso+'_output_R'+ext)
    plt.show()  
    '''  

    #---------------------------------------------------------------------------------------------------------------#
    #example
    '''
       mean_IgM = plot_confidence_interval(plt, t, evals_IgM, 'r', 'Modelo: testando IgM')
       plt.grid(True)
       plt.xticks(rotation=45)
       plt.title("Solution IgM")
       plt.legend(loc='best')
       plt.tight_layout()
       plt.savefig(out_path+'_output_IgM'+ext)
       plt.show()'''
    
    
    #-------------------------VIRUS--------------------------------------


  
    plt.figure('Viremia')
    #dadosViremiaLog10.plot.scatter(x='Day',y='Viral_load',color='m',label='Dados experimentais')
    #plt.plot(dadosViremiaLog10['day'], np.log10(dadosViremiaLog10['Virus']+1), 'o', label='Data', linewidth=4)
    plt.errorbar(dadosViremia['Day']-1, virus_mean, yerr=[virus_min, virus_max],linestyle='None', label='Data', fmt='o', color='brown', capsize=4, elinewidth=2)
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Viremia - $log_{10}$(copies/ml + 1)')
    plt.grid(True)
    #plt.title("Solution Virus")
    mean_virus= plot_confidence_interval(plt, t, np.log10(evals_virus+1), 'red', 'Model Virus')
    plt.legend(loc='best', prop={'size':13})
    #plot_mean_std(plt, t, evals_virus, 'blue', 'Model Virus std')
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path+'Virus'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------C SOBREVIVENTES---------------------------------------------
    
    plt.figure('C')
    #dadosCitocinaSobreviventes.plot.scatter(x='Day',y='IL6(pg/mL)',color='g',label='Dados experimentais(Sobreviventes)')
    #plt.plot(dadosCitocinaSobreviventes['Day']+5, dadosCitocinaSobreviventes['IL6(pg/mL)'], 'o', label='Data', linewidth=4)
    if (opt_storm):
        plt.errorbar(dadosCitocina['Day'], cytokineNonSurvivor, yerr=[cytokineNonSurvivor_min, cytokineNonSurvivor_max],linestyle='None', label='Data', fmt='o', color='darkblue', capsize=4, elinewidth=2)
    else:
        plt.errorbar(dadosCitocina['Day'], cytokineSurvivor, yerr=[cytokineSurvivor_min, cytokineSurvivor_max],linestyle='None', label='Data', fmt='o', color='darkblue', capsize=4, elinewidth=2)    
    plt.xlim(0.0,dias_de_simulação)
    plt.grid(True)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Cytokines - (pg/mL)')
    #plt.title("Solution Cytokine")
    #plt.xticks(rotation=45)
    plt.tight_layout()
    mean_C = plot_confidence_interval(plt, t, evals_C, 'blue', 'Model Cytokines')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'C'+ext,bbox_inches='tight',dpi = 300)
    
    #-------------------------AM---------------------------------------------
    
    plt.figure('A_M')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('IgM - log$_2$(S/CO+1)')
    plt.grid(True)
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.errorbar(dadosAnticorposLog2_avg['Day'], antibody_m, yerr=[antibody_m_min, antibody_m_max],linestyle='None', label='Data', fmt='o', color='darkgoldenrod', capsize=4, elinewidth=2)
    mean_AM = plot_confidence_interval(plt, t, np.log2(evals_AM+1), 'orange', 'Model IgM')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'A_M'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------AG---------------------------------------------
   
    plt.figure('A_G')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('IgG - log$_2$(S/CO+1)')
    plt.grid(True)
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.errorbar(dadosAnticorposLog2_avg['Day'], antibody_g, yerr=[antibody_g_min, antibody_g_max],linestyle='None', label='Data', fmt='o', color='saddlebrown', capsize=4, elinewidth=2)    
    mean_AG = plot_confidence_interval(plt, t, np.log2(evals_AG+1), 'chocolate', 'Model IgG')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'A_G'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()

    #-------------------------ANTICORPOS--------------------------------------
    
    plt.figure('Anticorpos')
    plt.xlim(0.0,dias_de_simulação)
    
    plt.grid(True)
    #plt.title("Solution Anticorpos")
    #plt.xticks(rotation=45)
    plt.plot(dadosAnticorposLog2_avg['Day'], antibody_m, 'o', color='goldenrod', label='Data IgM', linewidth=4)
    #plt.errorbar(dadosAnticorposLog2_avg['Day'], np.log2(antibody_m+1), yerr=[np.log2(antibody_m_min+1), np.log2(antibody_m_max+1)],linestyle='None', label='Data IgM', fmt='o', color='goldenrod', capsize=4)
    plt.plot(dadosAnticorposLog2_avg['Day'], antibody_g, 'o', color='brown',label='Data IgG', linewidth=4)
    #plt.errorbar(dadosAnticorposLog2_avg['Day'], np.log2(antibody_g+1), yerr=[np.log2(antibody_g_min+1), np.log2(antibody_g_max+1)],linestyle='None', label='Data IgG', fmt='o', color='brown', capsize=4)
    plt.xlabel('Time (days)')
    plt.ylabel('Antibody Level ($log_2$(S/CO+1)')
    mean_AM = plot_confidence_interval(plt, t, np.log2(evals_AM+1), 'orange', 'Model IgM')
    mean_AG = plot_confidence_interval(plt, t, np.log2(evals_AG+1), 'red', 'Model IgG')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Anticorpos'+ext,bbox_inches='tight',dpi = 300)
    
    
    
    #-------------------------AP--------------------------------------
    
    plt.figure('Ap')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('APCs')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Ap = plot_confidence_interval(plt, t, evals_Ap, 'green', 'Model APCs')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Ap'+ext,bbox_inches='tight',dpi = 300)
    
    #plt.show()
    
    #-------------------------APM---------------------------------------------
   
    plt.figure('Apm')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Mature APCs')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Apm = plot_confidence_interval(plt, t, evals_Apm, 'blue', 'Model Mature APCs')
    plt.legend(loc='lower right', prop={'size':13})
    plt.savefig(output_path+'Apm'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------Thn---------------------------------------------
    
    plt.figure('Thn')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Th Naive')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Thn = plot_confidence_interval(plt, t, evals_Thn, 'black', 'Model Thn')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Thn'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------The---------------------------------------------
    
    plt.figure('The')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Th effector')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_The = plot_confidence_interval(plt, t, evals_The, 'purple', 'Model The')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'The'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------Tkn---------------------------------------------    
    
    plt.figure('Tkn')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Tk Naive')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Tkn = plot_confidence_interval(plt, t, evals_Tkn, 'orange', 'Model Tkn')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Tkn'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------Tke---------------------------------------------
   
    plt.figure('Tke')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Tk effector')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Tke = plot_confidence_interval(plt, t, evals_Tke, 'red', 'Model Tke')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Tke'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------B---------------------------------------------
    
    plt.figure('B')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('B')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_B = plot_confidence_interval(plt, t, evals_B, 'green', 'Model B')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'B'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #np.savetxt(output_path+'b.txt',y[:,7])
    
    #-------------------------Ps---------------------------------------------
    
    plt.figure('Ps')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Plasma Short')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Ps = plot_confidence_interval(plt, t, evals_Ps, 'blue', 'Model Ps')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Ps'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------Pl---------------------------------------------
    
    plt.figure('Pl')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Plasma Long')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Pl = plot_confidence_interval(plt, t, evals_Pl, 'black', 'Model Pl')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Pl'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    
    #-------------------------Bm---------------------------------------------
    
    plt.figure('Bm')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Bm')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Bm = plot_confidence_interval(plt, t, evals_Bm, 'purple', 'Model Bm')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Bm'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    

    
    #-------------------------I---------------------------------------------
    
    plt.figure('Ai')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Immune Cells')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_I = plot_confidence_interval(plt, t, evals_I, 'green', 'Model I')
    plt.legend(loc='best', prop={'size':13})
    plt.savefig(output_path+'Ai'+ext,bbox_inches='tight',dpi = 300)
    #plt.show()
    

    #plt.show()
    
    print("done!")
    
    
    
    
    
    #-------------------------C OBITOS---------------------------------------------
    
    '''
    plt.figure('C2')
    #dadosCitocinaObitos.plot.scatter(x='Day',y='IL6(pg/mL)',color='y',label='Dados experimentais(Obito)')
    #plt.plot(dadosIL6['Day']+5, dadosIL6['IL6(pg/mL)'], 'o', label='data', linewidth=4)
    plt.errorbar(dadosCitocina['Day'], cytokineNonSurvivor, yerr=[cytokineNonSurvivor_min, cytokineNonSurvivor_max],linestyle='None', label='Data', fmt='o', color='darkblue', capsize=4, elinewidth=2)    
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Time (days)')
    plt.ylabel('Cytokines (pg/mL)')
    plt.legend()
    mean_C = plot_confidence_interval(plt, t, evals_C, 'blue', 'Model C')
    plt.savefig(output_path+'C2.png',bbox_inches='tight',dpi = 300)
    #plt.show()
    '''


    
#Fim

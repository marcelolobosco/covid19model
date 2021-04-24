#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import factorial as fat
from scipy.optimize import differential_evolution
import math

# Bibliotecas proprias
from NovoModelo import *

#pathway to the data 
path = '../../data/'

# global data
# ajustar pela viremia e anticorpos
dadosViremiaLog10 = pd.read_csv(path+'Viral_load_paper.csv',',')
dadosAnticorposLog2 = pd.read_csv(path+'IgG_IgM.csv',',')
dadosIL6 = pd.read_csv(path+'IL6_ajuste.csv',',')


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


erro_max = 0.45

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

execution_de = []


# ones in days to be considered and zero otherwise
mask_virus     =[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
mask_antibodies=[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]
mask_cytokine  =[0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0]


def model(x):
    ts=range(len(virus))  
    
    #Initial Condition
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
    

    def teste(t, y):
        return immune_response_v3_2(t, y, *model_args)
    
    Ps= solve_ivp(teste,(0, len(virus)), P0, t_eval=ts, method='Radau') 
    #print(Ps)
    V=Ps.y[0,:] # virus
    A_m=Ps.y[11,:] # antibody
    A_g=Ps.y[12,:] # antibody
    il6 = Ps.y[14,:] #citocine
    
    erro_V = 0
    erro_IgG = 0
    erro_IgM = 0
    erro_il6 = 0
    vnorm=2.0
    
    sum_v = np.sum(V)
    sum_am = np.sum(A_m)
    sum_ag = np.sum(A_g)
    sum_il6 = np.sum(il6)
    
    if (math.isnan(sum_v) or math.isinf(sum_v) or 
    math.isnan(sum_am) or math.isinf(sum_am) or 
    math.isnan(sum_ag) or math.isinf(sum_ag) or 
    math.isnan(sum_il6) or math.isinf(sum_il6) or 
    (math.isinf(np.sum(np.log10(V)))) or (min(np.log10(V+1))<-0.1)):
        erro = 1e12
    else:
        '''
        Ps= odeint(immune_response_v3, P0, ts, args=(model_args)) 

        V=Ps[:,0] # virus
        A_m=Ps[:,11] # antibody
        A_g=Ps[:,12] # antibody
        il6 = Ps[:,14] #citocine
        '''
        #Viremia error 
        #virus represents experimental data, V the numerical one 
        
        #virus_aux = np.multiply(virus, mask_virus)
        V_aux = np.multiply(np.log10(V[first_day:]), mask_virus[first_day:])
        erro_V = np.linalg.norm(np.log10(virus[first_day:])-V_aux, vnorm)/np.linalg.norm(np.log10(virus[first_day:]), vnorm)
        #print(np.linalg.norm(V_aux,vnorm))
        if (math.isnan(erro_V) or math.isinf(erro_V)):
            #print("Com NaN ou Inf", x)
            erro_V = 1e12
        
        #antibody G
        #print(len(antibody_g), len(mask_antibodies))
        #print(antibody_g)
        
        antibody_g_aux = np.multiply(antibody_g[first_day:], mask_antibodies[first_day:])
        IgG_aux = np.multiply(A_g[first_day:], mask_antibodies[first_day:])
        erro_IgG = np.linalg.norm(antibody_g_aux-IgG_aux, vnorm)/np.linalg.norm(antibody_g_aux, vnorm)
        #print(np.linalg.norm(antibody_g_aux[first_day:]-IgG_aux, vnorm))
        if (math.isnan(erro_IgG) or math.isinf(erro_IgG)):
            erro_IgG = 1e12
        
        #antibody M
        antibody_m_aux = np.multiply(antibody_m[first_day:], mask_antibodies[first_day:])
        IgM_aux = np.multiply(A_m[first_day:], mask_antibodies[first_day:])
        erro_IgM = np.linalg.norm(antibody_m_aux-IgM_aux, vnorm)/np.linalg.norm(antibody_m_aux, vnorm)
        #print(antibody_m_aux[first_day:]-IgM_aux)
        if (math.isnan(erro_IgM) or math.isinf(erro_IgM)):
            erro_IgM = 1e12

        #cytokine il-6
        il6data_aux = np.multiply(il6_data[first_day:], mask_cytokine[first_day:])
        il6_aux = np.multiply(il6[first_day:], mask_cytokine[first_day:])
        #print(il6_aux)
        erro_il6 = np.linalg.norm(il6data_aux-il6_aux, vnorm)/np.linalg.norm(il6data_aux, vnorm)
        #print(antibody_m_aux[first_day:]-IgM_aux)
        if (math.isnan(erro_il6) or math.isinf(erro_il6)):
            erro_il6 = 1e12    
        
        
        weight = 0.1
        erro = weight*erro_IgG + weight*erro_IgM + erro_V + erro_il6
        erro_inf = max(erro_V, erro_il6, erro_IgM, erro_IgG)

    if (erro_inf <= erro_max):        
        #print(x)
        ind = []
        ind.append(erro)
        for v in x:
            ind.append(v)
        execution_de.append(ind)


    return [erro, V, A_m, A_g,il6, ts, erro_V, erro_IgM, erro_IgG, erro_il6]

def model_adj(x):
    result = model(x)
    return result[0]

if __name__ == "__main__":    
        	
    #define os bounds para cada um dos parâmetros

    opt_de = False
    opt_storm = True
    opt_name = 'survivor'
    
    if opt_storm:
        dadosIL6 = pd.read_csv(path+'IL6_storm_ajuste.csv',',')
        il6_data = dadosIL6['IL6(pg/mL)']
        erro_max = 0.4
        opt_name = 'nonsurvivor'
        
    if opt_de:
        #Best fit for all parameters survivor
        
        v0=6.102197486860535491e+01
        pi_c_apm=3.280626433780918774e+02
        pi_c_tke=1.783480640636802769e-02
        delta_c=7.042258813317423574e+02
        k_v3=6.451858595261411589e-02
        pi_v =1.474205503564698594e+00
        k_v1= 9.823907861836385022e-03
        k_v2=6.100010515847981629e-05
        beta_Ap =1.790443713053306851e-01	
        if (opt_storm):
            beta_apm=1.509602565560082960e-02
            beta_tke=9.999999999999999547e-07
            pi_c_i=9.963909138628952722e-02
        else:
            beta_tke=3.499578657824721993e-06
            beta_apm=1.333640061614344702e-02
            pi_c_i=6.440689397431548051e-03 
        

        min_bound = 0.9
        max_bound = 1.1
    
        fit_args = (v0, pi_c_apm, pi_c_i, pi_c_tke, delta_c, beta_apm,
        k_v3,beta_tke,pi_v,k_v1,k_v2,beta_Ap)
        
        vbounds = []
        for i in range(len(fit_args)):
            vbounds.append([fit_args[i]*min_bound, fit_args[i]*max_bound])


        #chama a evolução diferencial que o result contém o melhor individuo
        result = differential_evolution(model_adj, vbounds, strategy='best1bin', maxiter=30, popsize=20, disp=True)
        print('Params order: ')
        print ('...')
        print(result.x)
        #saving the best offspring...
        np.savetxt('params_best_sample.txt',result.x)
        best=result.x
        
        #saving the samples for UQ
        if (opt_storm):
            np.savetxt('execution_de_non_survivor.txt',execution_de)
        else:    
            np.savetxt('execution_de_survivor.txt',execution_de)
    else:
        best = np.loadtxt('params_best_sample.txt')
    
    
   
    erro, V, A_m, A_g, il6, ts, erro_V, erro_IgM, erro_IgG, erro_il6 = model(best)
    print("RELATIVE ERROR")
    print("Erro viremia: ", erro_V)
    print("Erro IgM: ", erro_IgM)
    print("Erro IgG: ", erro_IgG)
    print("Erro il6: ", erro_il6)
    
    
    '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.set_size_inches(12, 25)
   
   
    ax1.set_title('Viremia, Antibodies and Cytokines')
    
    #Plot active infected cases
    ax1.plot(ts, np.log10(V), label='Viremia model', linewidth=4)
    ax1.plot(ts, np.log10(virus_plot), 'o', label='data', linewidth=4)
    ax1.set_xlabel('day')
    ax1.set_ylabel('log10 (copies/ml)')    
    ax1.legend()
    ax1.grid()
        
    #Plot IgG
    ax2.plot(ts, np.log2(A_g+1), label='IgG model', linewidth=4)
    ax2.plot(ts, np.log2(antibody_g+1), 'o', label='data', linewidth=4)
    ax2.set_xlabel('day')
    ax2.set_ylabel('log2(S/CO+1)')
    ax2.legend()
    ax2.grid()

    #Plot igM 
    ax3.plot(ts, np.log2(A_m+1), label='igM model', linewidth=4)
    ax3.plot(ts, np.log2(antibody_m+1), 'o', label='data', linewidth=4)
    ax3.set_xlabel('day')
    ax3.set_ylabel('log2(S/CO+1)')
    ax3.legend()
    ax3.grid()
    
    #Plot IL-6
    ax4.plot(ts, il6, label='IL-6 model', linewidth=4)
    ax4.plot(ts,il6_data, 'o', label='data', linewidth=4)
    ax4.set_xlabel('day')
    ax4.set_ylabel('(pg/mL)')
    ax4.legend()
    ax4.grid()

    plt.savefig('output_fit.pdf')
    '''
    #plt.style.use('../estilo/PlotStyle.mplstyle')
    #plt.close('all')
        
    plt.figure();
    plt.title("Virus")
    plt.plot(ts, np.log10(V), label='Viremia model', linewidth=1.5)
    plt.errorbar(dadosViremia['Day']-1, virus_mean, yerr=[virus_min, virus_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)
    plt.legend(loc="best",prop={'size': 13})
    plt.grid(True)
    plt.tight_layout()    
    plt.savefig('output_'+opt_name+'_virus.pdf',bbox_inches='tight',dpi = 300)
    
    plt.figure();
    plt.title("IgG")
    plt.plot(ts, np.log2(A_g+1), label='IgG model', linewidth=1.5)
    plt.errorbar(dadosAnticorposLog2_avg['Day'], antibody_g_mean, yerr=[antibody_g_min, antibody_g_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)    
    plt.legend(loc="best",prop={'size': 13})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_'+opt_name+'_igg.pdf',bbox_inches='tight',dpi = 300)
    
    plt.figure();
    plt.title("IgM")
    plt.plot(ts, np.log2(A_m+1), label='igM model', linewidth=1.5)
    plt.errorbar(dadosAnticorposLog2_avg['Day'], antibody_m_mean, yerr=[antibody_m_min, antibody_m_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)
    plt.legend(loc="best",prop={'size': 13})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_'+opt_name+'_igm.pdf',bbox_inches='tight',dpi = 300)
    
    if opt_storm:
        plt.figure();
        plt.title("Cytokines")
        plt.plot(ts, il6, label='IL-6 model', linewidth=1.5)
        plt.errorbar(dadosCitocina['Day'], cytokineNonSurvivor, yerr=[cytokineNonSurvivor_min, cytokineNonSurvivor_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)    
        plt.legend(loc="best",prop={'size': 13})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output_'+opt_name+'_c.pdf',bbox_inches='tight',dpi = 300)
    else:
        plt.figure();
        plt.title("Cytokines")
        plt.plot(ts, il6, label='IL-6 model', linewidth=1.5)
        plt.errorbar(dadosCitocina['Day'], cytokineSurvivor, yerr=[cytokineSurvivor_min, cytokineSurvivor_max],linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=2)    
        plt.legend(loc="best",prop={'size': 13})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output_'+opt_name+'_c.pdf',bbox_inches='tight',dpi = 300)
    #plt.show()


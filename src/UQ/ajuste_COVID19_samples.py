#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import factorial as fat
from scipy.optimize import fmin,differential_evolution
import math

# Bibliotecas proprias
from NovoModelo import *

#pathway to the data 
path = '../../data/'

# global data
# ajustar pela viremia e anticorpos
dadosViremiaLog10 = pd.read_csv(path+'Viral_load.csv',',')
dadosAnticorposLog2 = pd.read_csv(path+'IgG_IgM.csv',',')
dadosIL6 = pd.read_csv(path+'IL6_ajuste.csv',',')

erro_max = 0.28

#dadosIL6 = pd.read_csv(path+'IL6_storm_ajuste.csv',',')

first_day = 0
'''
virus = np.power(10,dadosViremiaLog10['Viral_load'])
antibody_g = np.power(2,dadosAnticorposLog2['IgG'])-1
antibody_m = np.power(2,dadosAnticorposLog2['IgM'])-1

'''

virus = dadosViremiaLog10['Viral_load']
antibody_g = dadosAnticorposLog2['IgG']
antibody_m = dadosAnticorposLog2['IgM']
il6_data = dadosIL6['IL6(pg/mL)']

execution_de = []


# ones in days to be considered and zero otherwise
mask_virus     =[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
mask_antibodies=[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]
mask_cytokine  =[0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0]

execution_de =  []

def model(x):
    '''
    Params description:   

    '''
    
    
    ts=range(len(virus))  
    
    #Initial Condition
    V0 = 2.18808824e+00
    Ap0 = 1.0e6
    Apm0 = 0.0
    Ai0=0
    C0=0.0
    Thn0 = (1.0e6)
    The0 = 0.0  
    Tkn0 = (1.0e3)*500.0
    Tke0 = 0.0
    B0 =  (1.0e3)*250.0
    Ps0 = 0.0
    Pl0 = 0.0
    Bm0 = 0.0
    A0_M = 0.0  
    A0_G = 0.0
    
    P0 = [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G,Ai0,C0]
  
    
    #Model Parameters    
    pi_v = x[0]#0.1955 #fit
    c_v1 = 2.63
    c_v2 = 0.60
    k_v1 = 3.5e-3
    k_v2 = 9.5e-5
    alpha_Ap = 1.87E-06*0.4
    beta_Ap = x[1]#2.00E-03 #fit
    c_ap1 = x[2]#0.8  #fit
    c_ap2 = x[3]#40.0 #fit

    delta_Apm = x[4]#8.14910996e+00  #fit
    alpha_Tn =2.17E-04 
    beta_tk = x[5]#1.431849023090428446e-05 #fit
    pi_tk = 1.0E-08 
    delta_tk = 0.0003
    alpha_B = 3.578236584371140339e+02
    pi_B1 = 8.979145365768647095e-05
    pi_B2 = 1.27E-8

    beta_ps = x[6]#6.0e-6 #fit
    beta_pl = x[7]#5.0e-6 #fit
    beta_Bm = 1.0E-06
    delta_S = x[8]#2.5 #fit
    delta_L = x[9]#0.35 #fit
    gamma_M = (1.95E-06)*500.0
    k_bm1 = 1.0e-5      
    k_bm2 = 2500.0 
    pi_AS = x[10]#0.087 #fit
    pi_AL = x[11]#0.001 #fit
    delta_ag = 0.07
    delta_am = 0.07
    alpha_th = x[12]#2.17E-04 #fit
    beta_th = x[13]#1.8e-5 #fit
    pi_th = 1.0E-08  
    delta_th = 0.3
    Ap0 = 1.0e6
    Thn0 = 1.0e6
    Tkn0 = 5.0e5
    B0 = 2.5E5


    pi_c_apm = 7.43773673e-01
    pi_c_i = 1.97895565e-02
    pi_c_tke = x[14]#0.04730172 #fit
    delta_c = x[15]#8.26307952e+00 #fit
    beta_apm = x[16]#5.36139617e-01 #fit
    k_v3 = 3.08059068e-03
    beta_tke = x[17]#2.10152618e-01 #fit   

       
    model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, c_ap1, c_ap2,
    delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk, alpha_B, pi_B1, pi_B2, 
    beta_ps, beta_pl, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke)     
    
    
    Ps= odeint(immune_response_v3, P0, ts, args=(model_args)) 

    V=Ps[:,0] # virus
    A_m=Ps[:,11] # antibody
    A_g=Ps[:,12] # antibody
    il6 = Ps[:,14] #Citocine
    
    erro_V = 0;
    erro_IgG = 0;
    erro_IgM = 0;
    vnorm=2
    
	
    #Viremia error 
    #virus represents experimental data, V the numerical one 
    
    #virus_aux = np.multiply(virus, mask_virus)
    
    V_aux = np.multiply(V, mask_virus[first_day:])
    erro_V = np.linalg.norm(virus[first_day:]-V_aux, vnorm)/np.linalg.norm(virus[first_day:], vnorm)
    #print(np.linalg.norm(V_aux,vnorm))
    if (math.isnan(erro_V) or math.isinf(erro_V)):
        erro_V = 1e12
    
    #antibody G
    #print(len(antibody_g), len(mask_antibodies))
    #print(antibody_g)
    
    antibody_g_aux = np.multiply(antibody_g, mask_antibodies)
    IgG_aux = np.multiply(A_g, mask_antibodies[first_day:])
    erro_IgG = np.linalg.norm(antibody_g_aux[first_day:]-IgG_aux, vnorm)/np.linalg.norm(antibody_g_aux[first_day:], vnorm)
    #print(np.linalg.norm(antibody_g_aux[first_day:]-IgG_aux, vnorm))
    if (math.isnan(erro_IgG) or math.isinf(erro_IgG)):
        erro_IgG = 1e12
    
    #antibody M
    antibody_m_aux = np.multiply(antibody_m, mask_antibodies)
    IgM_aux = np.multiply(A_m, mask_antibodies[first_day:])
    erro_IgM = np.linalg.norm(antibody_m_aux[first_day:]-IgM_aux, vnorm)/np.linalg.norm(antibody_m_aux[first_day:], vnorm)
    #print(antibody_m_aux[first_day:]-IgM_aux)
    if (math.isnan(erro_IgM) or math.isinf(erro_IgM)):
        erro_IgM = 1e12

    #cytokine il-6
    il6data_aux = np.multiply(il6_data, mask_cytokine)
    il6_aux = np.multiply(il6, mask_cytokine)
    #print(il6_aux)
    erro_il6 = np.linalg.norm(il6data_aux[first_day:]-il6_aux, vnorm)/np.linalg.norm(il6data_aux[first_day:], vnorm)
    #print(antibody_m_aux[first_day:]-IgM_aux)
    if (math.isnan(erro_il6) or math.isinf(erro_il6)):
        erro_il6 = 1e12    
    
    weight = 0.5
    erro = weight*erro_IgG + weight*erro_IgM + erro_V + erro_il6
    
    if (max(erro_IgG, erro_IgM, erro_V, erro_il6) <= erro_max):
        
        ind = []
        for v in x:
            ind.append(v)
        ind.append(erro)
        
        execution_de.append(ind)
        
    
    '''
    print("RELATIVE ERROR")
    print("Erro viremia: ", erro_V)
    print("Erro IgM: ", erro_IgM)
    print("Erro IgG: ", erro_IgG)
    print("Erro il6: ", erro_il6)
    '''
    #erro = erro_V


    return [erro, V, A_m, A_g,il6, ts, erro_V, erro_IgM, erro_IgG, erro_il6]

def model_adj(x):
    result = model(x)
    return result[0]

if __name__ == "__main__":    
        	
    #define os bounds para cada um dos parâmetros

    opt_de = True
    opt_storm = True
    
    if opt_storm:
        dadosIL6 = pd.read_csv(path+'IL6_storm_ajuste.csv',',')
        il6_data = dadosIL6['IL6(pg/mL)']
        erro_max = 0.3
        
    if opt_de:
        #Best fit for all parameters survivor
        
        pi_v = 0.1955 #fit
        c_v1 = 2.63
        c_v2 = 0.60
        k_v1 = 3.5e-3
        k_v2 = 9.5e-5
        alpha_Ap = 1.87E-06*0.4
        beta_Ap = 2.00E-03 #fit
        c_ap1 = 0.8  #fit
        c_ap2 = 40.0 #fit

        delta_Apm = 8.14910996e+00  #fit
        alpha_Tn =2.17E-04 
        beta_tk = 1.431849023090428446e-05 #fit
        pi_tk = 1.0E-08 
        delta_tk = 0.0003
        alpha_B = 3.578236584371140339e+02
        pi_B1 = 8.979145365768647095e-05
        pi_B2 = 1.27E-8

        beta_ps = 6.0e-6 #fit
        beta_pl = 5.0e-6 #fit
        beta_Bm = 1.0E-06
        delta_S = 2.5 #fit
        delta_L = 0.35 #fit
        gamma_M = (1.95E-06)*500.0
        k_bm1 = 1.0e-5      
        k_bm2 = 2500.0 
        pi_AS = 0.087 #fit
        pi_AL = 0.001 #fit
        delta_ag = 0.07
        delta_am = 0.07
        alpha_th = 2.17E-04 #fit
        beta_th = 1.8e-5 #fit
        pi_th = 1.0E-08  
        delta_th = 0.3
        Ap0 = 1.0e6
        Thn0 = 1.0e6
        Tkn0 = 5.0e5
        B0 = 2.5E5


        pi_c_apm = 7.43773673e-01
        pi_c_i = 1.97895565e-02
        pi_c_tke = 0.04730172 #fit
        delta_c = 8.26307952e+00 #fit
        k_v3 = 3.08059068e-03
		
        if (opt_storm):
            beta_apm = 1.48569967 #fit
            beta_tke = 0.10171796#fit
        else:
            beta_apm = 5.36139617e-01 #fit
            beta_tke = 2.10152618e-01 #fit
        

        min_bound = 0.9
        max_bound = 1.1
    
        fit_args = (pi_v, beta_Ap, c_ap1, c_ap2, delta_Apm, beta_tk, 
            beta_ps, beta_pl, delta_S, delta_L, pi_AS, pi_AL, alpha_th, 
            beta_th, pi_c_tke, delta_c, beta_apm, beta_tke)
        
        vbounds = []
        for i in range(len(fit_args)):
            vbounds.append([fit_args[i]*min_bound, fit_args[i]*max_bound])


        #chama a evolução diferencial que o result contém o melhor individuo
        result = differential_evolution(model_adj, vbounds, strategy='best1bin', popsize=100, disp=True)
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
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.set_size_inches(12, 25)
   
   
    ax1.set_title('Viremia, Antibodies and Cytokines')
    
    #Plot active infected cases
    ax1.plot(ts, V, label='Viremia model', linewidth=4)
    ax1.plot(ts, virus[first_day:], 'o', label='data', linewidth=4)
    ax1.set_xlabel('day')
    ax1.set_ylabel('(copies/ml)')    
    ax1.legend()
    ax1.grid()
        
    #Plot IgG
    ax2.plot(ts, np.log2(A_g+1), label='IgG model', linewidth=4)
    ax2.plot(ts, np.log2(antibody_g[first_day:]+1), 'o', label='data', linewidth=4)
    ax2.set_xlabel('day')
    ax2.set_ylabel('log2(S/CO+1)')
    ax2.legend()
    ax2.grid()

    #Plot igM 
    ax3.plot(ts, np.log2(A_m+1), label='igM model', linewidth=4)
    ax3.plot(ts, np.log2(antibody_m[first_day:]+1), 'o', label='data', linewidth=4)
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


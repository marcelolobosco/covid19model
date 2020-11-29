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

def model(x):
    '''
    Params description:   
    x[0]=>pi_v 
    x[1]=>k_v3
    x[2]=>delta_Apm
    x[3]=>alpha_B
    x[4]=>delta_A_G
    x[5]=>delta_A_M
    x[6]=>pi_c_apm, 
    x[7]=>pi_c_i,
    x[8]=>pi_c_tke,
    x[9]=>delta_c, 
    x[10]=>k_apm
    x[11]=>V0
    x[12]=>Am0
    x[13]=>Ag0
    '''

    ts=range(len(virus))  
    #       V0,   Ap0,Apm0,  Thn0,The0,  Tkn0,,Tke0,     B0, Ps0, Pl0, Bm0, A0_M, A0_G Ai C
    #P0 = [9.971841136161140184e+02, 1.0e6, 0.0, 1.0e6, 0.0, 5.0e5, 0.0, 1.25E5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    V0 =  3.02514449e+01 #x[0]#2.136323495622534097e+00 
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
    P0 = [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G,Ai0,C0]
  
    
    '''
    pi_v = x[8]#0.1955#1.091710061112672880e-01
    c_v1 = 2.63
    c_v2 = 0.60
    k_v1 = x[9]#3.5e-3#5.600298025616778555e-05
    k_v2 = x[10]#9.5e-5#6.011588247777179580e-05
    alpha_Ap = 1.87E-06*0.4
    beta_Ap = x[11]#2.00E-03
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

    beta_S = 6.0e-6#0.000672 
    beta_L = 5.0e-6#5.61E-06 
    beta_Bm = 1.0E-06
    delta_S = 2.5
    delta_L = 0.35
    gamma_M = (1.95E-06)*500.0
    k_bm1 = 1.0e-5      
    k_bm2 = 2500.0 
    pi_AS = 0.087#2.850370072424884479e-02
    pi_AL = 0.001#6.304459239904726120e-01
    delta_A_G = 0.07#3.650482092015642221e-01
    delta_A_M = 0.07#6.873347140815699419e+00
    c11 = 2.17E-04
    c12 = 1.8e-5#1.0E-07
    c13 = 1.0E-08  
    c14 = 0.3#0.22
    Ap0 = 1.0e6
    Thn0 = 1.0e6
    Tkn0 = 5.0e5
    B0 = 2.5E5

    
    pi_c_apm = x[1]#0.000015#  #ajuste
    pi_c_i = x[2]#0.015#ajuste
    pi_c_tke = 0.04730172#x[3]#0.000015 #ajuste
    delta_c = x[4]#0.1#ajuste
    k_apm = x[5]#0.000007 #ajuste
    k_v3 = x[6]#1.0e-6#ajuste
    k_tk = x[7]
    
    
    model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1, k_ap2,
    delta_Apm, alpha_Tn, pi_T, k_te1, delta_te, alpha_B, pi_B1, pi_B2, 
    beta_S, beta_L, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_A_G, delta_A_M, c11, c12, c13, c14, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, k_apm, k_v3, k_tk)  
    '''
    
    
    
      #Model Parameters    
    pi_v = 1.239122007490762956e+00#x[8]#0.1955#1.091710061112672880e-01
    c_v1 = 2.63
    c_v2 = 0.60
    k_v1 = 4.074914859019632130e-02
    k_v2 = 3.664348294667734058e-05#x[10]#9.5e-5#6.011588247777179580e-05
    alpha_Ap = 1.87E-06*0.4
    beta_Ap = 5.748390263404423717e-02#x[11]#2.00E-03
    c_ap1 = 8.0#x[10]#0.8
    c_ap2 = 8.08579916e+04#x[11]#40.0

    delta_Apm = 8.14910996e+00
    alpha_Tn =2.17E-04 
    beta_tk = 1.431849023090428446e-05
    pi_tk = 1.0E-08 
    delta_tk = 0.0003
    alpha_B = 3.578236584371140339e+02#incluir
    pi_B1 = 8.979145365768647095e-05
    pi_B2 = 1.27E-8

    beta_ps = 2.006520080920685608e-06
    beta_pl = 1.189263128657398219e-06
    beta_Bm = 1.0E-06
    delta_S = 2.5
    delta_L = 0.35
    gamma_M = (1.95E-06)*500.0
    k_bm1 = 1.0e-5      
    k_bm2 = 2500.0 
    pi_AS = 4.422081842730109930e-02
    pi_AL = 7.021486442581976670e-04
    delta_ag = 0.07
    delta_am = 0.07
    alpha_th = 2.17E-04
    beta_th = 1.8e-5
    pi_th = 1.0E-08  
    delta_th = 0.3
    Ap0 = 1.0e6
    Thn0 = 1.0e6
    Tkn0 = 5.0e5
    B0 = 2.5E5


    pi_c_apm = 3.30598859e+01#x[1]
    pi_c_i = 7.27465600e-02#x[2]
    pi_c_tke = 9.47083313e-02#x[3]#0.04730172
    delta_c = 6.076707938151348571e+02#x[4]
    beta_apm = x[0]#4.492678675586297965e-02#x[5]
    k_v3 = 3.117300692366209436e-03#x[6]
    beta_tke = x[1]#2.703419803803602406e+01#x[7] 

       
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
    #print(np.log10(V))
    #print(sum_am)
    #print(sum_ag)
    #print(sum_il6)
    
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
        
        
        #penalizacoes
        #if (erro_V>0.3):
        #    erro_V=erro_V*5
        #tot=np.sum(mask_antibodies)+np.sum(mask_cytokine)+np.sum(mask_virus)
        #wigg =  wigm= np.sum(mask_antibodies)/tot
        #wil6 = np.sum(mask_cytokine)/tot
        #wv = np.sum(mask_virus)/tot
        weight = 0.5
        #erro = weight*erro_IgG + weight*erro_IgM + erro_V + erro_il6
        #erro = max(wv*erro_V, wil6*erro_il6, wigg*erro_IgM, wigm*erro_IgG)
        erro = max(erro_V, erro_il6, erro_IgM, erro_IgG)
        #print(x)
    
    if (erro <= 0.2):
        
        ind = []
        ind.append(erro)
        for v in x:
            ind.append(v)
        
        print("Entrei",ind);
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
    if opt_de:

        bounds = [
        (1e-4,1e1),
        (1e-4,1e1)
        ]

        #chama a evolução diferencial que o result contém o melhor individuo
        result = differential_evolution(model_adj, bounds, strategy='best1bin', maxiter=50,popsize=40, disp=True, workers=3)
        print('Params order: ')
        print ('...')
        print(result.x)
        print(result.success)
        #saving the best offspring...
        np.savetxt('params_ajuste_covid19_de_il6_death.txt',result.x)
        
        best=result.x
    else:
        best = np.loadtxt('params_ajuste_covid19_de_il6_death.txt')
    
    #saving the samples for UQ
    #np.savetxt('execution_de.txt',execution_de)
   
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

    plt.show()

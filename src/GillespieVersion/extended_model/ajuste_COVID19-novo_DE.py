#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import factorial as fat
from scipy.optimize import fmin,differential_evolution

# Bibliotecas proprias
from expmmq import *
from mix import *
from NovoModelo import *

#pathway to the data 
path = '../../data/'

# global data
# ajustar pela viremia e anticorpos
dadosViremiaLog10 = pd.read_csv(path+'Viral_load.csv',',')
dadosAnticorposLog2 = pd.read_csv(path+'IgG_IgM.csv',',')

dia = dadosViremiaLog10['Day']
first_day = 0
virus = np.power(10,(dadosViremiaLog10['Viral_load']))
antibody_g = np.power(2,dadosAnticorposLog2['IgG'])
antibody_m = np.power(2,dadosAnticorposLog2['IgM'])


# ones in days to be considered and zero otherwise
mask_virus     =[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
mask_antibodies=[0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]

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

    ts=range(len(mask_virus))  
    #       V0,   Ap0,Apm0,  Thn0,The0,  Tkn0,,Tke0,     B0, Ps0, Pl0, Bm0, A0_M, A0_G Ai C
    #P0 = [9.971841136161140184e+02, 1.0e6, 0.0, 1.0e6, 0.0, 5.0e5, 0.0, 1.25E5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    P0 = [x[11], 1.0e6, 0.0, 1.0e6, 0.0, 5.0e5, 0.0, 1.25E5, 0.0, 0.0, 0.0, x[12], x[13], 0.0, 0.0]
    
    #            pi_v,  c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap,k_ap1,k_ap2,delta_Apm, alpha_Tn,    pi_T,   k_te1,delta_te,alpha_B,     pi_B1,   pi_B2,
    #beta_S,   beta_L, beta_Bm,delta_S,    delta_L, gamma_M, k_bm1,    k_bm2, pi_AS,  pi_AL,delta_A_G, delta_A_M,       c11,     c12,  c13,  c14, Ap0, Thn0,
    #Tkn0
    model_args = (x[0], 2.63, 0.60, 9.783944251204167164e-02, 4.164999863476288964e-05, 2.50E-03, 5.5e-01, 0.8, 40.0,
    x[2], 2.17E-04, 1.0E-05, 1.0E-08,  0.0003,   x[3], 4.826E-06, 1.27E-8, 
    0.000672, 5.61E-06, 1.0E-06,   2.0, 2.3976E-04, 9.75E-04, 1.0e-5, 2500.0, 0.002,0.00068,     x[4],       
    x[5], 2.17E-04, 1.0E-07, 1.0E-08, 0.22, 1.0e6, 1.0e6, 5.0e5, 2.5E5, x[6], x[7], x[8], x[9], x[10], x[1])

    Ps,d = odeint(immune_response, P0, ts, args=(model_args), full_output=True, printmessg=True) 

    V=Ps[:,0] # virus
    A_m=Ps[:,11] # antibody
    A_g=Ps[:,12] # antibody
    erro_V = 0;
    erro_IgG = 0;
    erro_IgM = 0;
    
    
	
    #Viremia error 
    #virus represents experimental data, V the numerical one 
    
    #virus_aux = np.multiply(virus, mask_virus)
    vnorm=2
    V_aux = np.multiply(V, mask_virus[first_day:])
    erro_V = np.linalg.norm(virus[first_day:]-V_aux, vnorm)/np.linalg.norm(virus[first_day:], vnorm)
    #print(np.linalg.norm(V_aux,vnorm))
    '''
    #antibody G
    #print(len(antibody_g), len(mask_antibodies))
    #print(antibody_g)
    antibody_g_aux = np.multiply(antibody_g, mask_antibodies)
    IgG_aux = np.multiply(np.log2(A_g), mask_antibodies[first_day:])
    erro_IgG = np.linalg.norm(antibody_g_aux[first_day:]-IgG_aux, np.inf)/np.linalg.norm(antibody_g_aux[first_day:], np.inf)
    
    #antibody M
    antibody_m_aux = np.multiply(antibody_m, mask_antibodies)
    IgM_aux = np.multiply(np.log2(A_m), mask_antibodies[first_day:])
    erro_IgM = np.linalg.norm(antibody_m_aux[first_day:]-IgM_aux, np.inf)/np.linalg.norm(antibody_m_aux[first_day:], np.inf)
    '''
    
    weight = 0.5
    #erro = weight*erro_IgG + weight*erro_IgM + erro_V
    erro = erro_V


    return [erro, V, A_m, A_g, ts, erro_V, erro_IgM, erro_IgG]

def model_adj(x):
    result = model(x)
    return result[0]

if __name__ == "__main__":    
        	
    opt_last =  42 # os ultimos opt_last dias não serão contados para o ajuste se o valor for negativo, coloque o tamanho do vetor (len(infected)) caso queira considerar todos
    
    #define os bounds para cada um dos parâmetros
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
    x[6]=>V0
    x[7]=>Am0
    x[8]=>Ag0
    '''
    bounds = [
    (1e-7,2), 
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1e-7,1e3),
    (1,1e4),
    (1,1e4),
    (1,1e4)
    ]

    #chama a evolução diferencial que o result contém o melhor individuo
    result = differential_evolution(model_adj, bounds, strategy='best1bin', popsize=20, disp=True,workers=3)
    print('Params order: ')
    print ('V0, pi_v,k_v1, k_v2, delta_Apm, alpha_B, delta_A_G,delta_A_M')
    print(result.x)
    #saving the best offspring...
    np.savetxt('params_simple.txt',result.x)
    
    #saving the samples for UQ
    #np.savetxt('execution_de_100_ge.txt',execution_de)
   
    erro, V, A_m, A_g, ts, erro_V, erro_IgM, erro_IgG = model(result.x)
    print("RELATIVE ERROR")
    print("Erro viremia: ", erro_V)
    print("Erro IgM: ", erro_IgM)
    print("Erro IgG: ", erro_IgG)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.set_size_inches(12, 25)
   
   
    ax1.set_title('Viremia and Antibodies')
    
    #Plot active infected cases
    ax1.plot(ts, V, label='Viremia model', linewidth=4)
    ax1.plot(ts, virus[first_day:], label='data', linewidth=4)
    ax1.set_xlabel('day')
    ax1.set_ylabel('log 10(copies/ml)')    
    ax1.legend()
    ax1.grid()
        
    #Plot death cases 
    ax2.plot(ts, np.log2(A_g), label='IgG model', linewidth=4)
    ax2.plot(ts, antibody_g[first_day:], label='data', linewidth=4)
    ax2.set_xlabel('day')
    ax2.set_ylabel('log 2(S/CO)')
    ax2.legend()
    ax2.grid()

    #Plot recovery cases 
    ax3.plot(ts, log2(A_m), label='model', linewidth=4)
    ax3.plot(ts, antibody_m[first_day:], label='data', linewidth=4)
    ax3.set_xlabel('day')
    ax3.set_ylabel('log 2(S/CO)')
    ax3.legend()
    ax3.grid()
    
    plt.show()

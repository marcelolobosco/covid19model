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
import sys

# Bibliotecas proprias
from expmmq import * #ajuste da curva pelo MMQ(return np.exp(x[0]), x[1])
from mix import * #find nearest(return idx)
#from Modelo import *
from ModeloCitocina import *

#pathway to the data 
path = '../../data/'

# global data
# ajustar pela viremia, anticorpos e citocinas
dadosViremiaLog10 = pd.read_csv(path+'Viral_load_10_2.csv',',')
dadosAnticorposLog2 = pd.read_csv(path+'IgG_IgM_21_1b_average.csv',',')
if sys.argv[-1] == 'n':
    dadosCitocina = pd.read_csv(path+'IL6_non-survivors_19.csv',',')
else:
    dadosCitocina = pd.read_csv(path+'IL6_survivors_19.csv',',')

#casos = np.loadtxt(path+'active_world.txt');
dia = dadosViremiaLog10['Day']
#total_population = 83.02e6
first_day = 0
#confirmed = np.loadtxt(path+'confirmed_Germany.txt');
virus = dadosViremiaLog10['Viral_load']
antibody_g = dadosAnticorposLog2['IgG']
antibody_m = dadosAnticorposLog2['IgM']
cytokine = dadosCitocina['IL6(pg/mL)']
#deaths = np.loadtxt(path+'death_Germany.txt');
#recovery = np.loadtxt(path+'recovered_Germany.txt');

# ones in days to be considered and zero otherwise
mask_virus     =[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
mask_antibodies=[0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
mask_cytokine  =[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]#ateh o 20o dia

execution_de =  []

def model(x):
    '''
    Params description:
    x[0]=> V0    
    x[1]=> pi_v 
    x[2]=> k_v1
    x[3]=> k_v2
    x[4]=>delta_Apm
    x[5]=>alpha_B
    x[6]=>delta_A_G
    x[7]=>delta_A_M
    x[8]=>pi_capm 5.0e-6
    x[9]=>pi_cthe 1.0e-5
    x[10]=>pi_ctke 1.0e-6
    x[11]=>pi_cb 3.5e-6
    x[12]=>delta_c 0.3
    delta_ps, delta_th, beta_pl, pi_pl, delta_apm
    
    '''

    ts=np.linspace(0,42,20000)#range(45)
    #       V0,   Ap0,Apm0, Thn0 ,The0, Tkn0 ,Tke0,   B0  , Ps0, Pl0, Bm0,A0_M,A0_G, C0
    P0 = [x[0], 1.0e6, 0.0, 1.0e6, 0.0, 5.0e5, 0.0, 1.25E5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if sys.argv[-1] == 'n':
        #             pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap,k_ap1,k_ap2,delta_Apm, alpha_Tn,    pi_T,   k_te1,delta_te,alpha_B,     pi_B1,   pi_B2,   beta_S,   beta_L, beta_Bm,delta_S,    delta_L, gamma_M , k_bm1 , k_bm2 , pi_AS,  pi_AL,delta_A_G, delta_A_M,      c11,     c12,  c13   ,  c14, pi_capm, pi_cthe, pi_ctke, pi_cb, delta_c, Ap0  , Thn0 , Tkn0, B0
        model_args = (x[1], 2.63, 0.60, x[2], x[3], 2.50E-03, 5.5e-01, 0.8 , 40.0,   x[4]  , 2.17E-04, 1.0E-05, 1.0E-08,  0.0003,   x[5], 4.826E-06, 1.27E-8, 0.000672, 5.61E-06, 1.0E-06,   2.0 , 2.3976E-04, 9.75E-04, 1.0e-5, 2500.0, 0.002,0.00068,     x[6],   x[7]   , 2.17E-04, 1.0E-07, 1.0E-08, 0.22, x[8]   ,  x[9]   ,  x[10]  , x[11] ,  x[12]  , 1.0e6, 1.0e6, 5.0e5, 2.5E5)
    else:
        #            (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1, k_ap2, delta_Apm, alpha_Tn,  pi_T  , k_te1 , delta_te, alpha_B,   pi_B1  , pi_B2  , beta_S  , beta_L  , beta_Bm, delta_S,  delta_L  , gamma_M , k_bm1 , k_bm2 , pi_AS, pi_AL , delta_A_G, delta_A_M,   c11   ,  c12   ,   c13  , c14 , pi_capm, pi_cthe, pi_ctke, pi_cb, delta_c, Ap0  , Thn0 , Tkn0 , B0)
#modelo com citocina-sobrevivente, argumentos:
        model_args = (x[1], 2.63, 0.60, x[2], x[3], 2.50E-03, 5.5e-01, 0.8  , 40.0 ,  x[4]   , 2.17E-04, 1.0E-05, 1.0E-08, 0.0003  ,  x[5]  , 4.826E-06, 1.27E-8, 0.000672, 5.61E-06, 1.0E-06,   2.0  , 2.3976E-04, 9.75E-04, 1.0e-5, 2500.0, 0.002,0.00068,   x[6]   ,   x[7]   , 2.17E-04, 1.0E-07, 1.0E-08, 0.22,  x[8]  ,  x[9]  ,  x[10] , x[11],  x[12] , 1.0e6, 1.0e6, 5.0e5, 2.5E5)
    Ps,d = odeint(immune_response, P0, ts, args=(model_args), full_output=True, printmessg=True) 

    V  =Ps[:,0] # virus
    A_m=Ps[:,11] # antibody
    A_g=Ps[:,12] # antibody
    C  =Ps[:,13] # cytokine 

    #Viremia error 
    #virus represents experimental data, V the numerical one 
    virus_aux = np.multiply(virus, mask_virus)
    V_aux = np.multiply(np.log10(V[:opt_last]), mask_virus[first_day:opt_last])
    erro_V = np.linalg.norm(virus_aux[first_day:opt_last]-V_aux, np.inf)/np.linalg.norm(virus_aux[first_day:opt_last], np.inf)
    
    #antibody G
    antibody_g_aux = np.multiply(antibody_g, mask_antibodies)
    IgG_aux = np.multiply(np.log2(A_g[:opt_last]+1), mask_antibodies[first_day:opt_last])
    erro_IgG = np.linalg.norm(antibody_g_aux[first_day:opt_last]-IgG_aux, np.inf)/np.linalg.norm(antibody_g_aux[first_day:opt_last], np.inf)

    #antibody M
    antibody_m_aux = np.multiply(antibody_m, mask_antibodies)
    IgM_aux = np.multiply(np.log2(A_m[:opt_last]+1), mask_antibodies[first_day:opt_last])
    erro_IgM = np.linalg.norm(antibody_m_aux[first_day:opt_last]-IgM_aux, np.inf)/np.linalg.norm(antibody_m_aux[first_day:opt_last], np.inf)

    #cytokine
    cytokine_aux = np.multiply(cytokine, mask_cytokine)  
    C_aux = np.multiply(C[:opt_last_c], mask_cytokine[first_day:opt_last_c])
    erro_C = np.linalg.norm(C_aux[first_day:opt_last_c]-C_aux, np.inf)/np.linalg.norm(C_aux[first_day:opt_last_c], np.inf)
    
    weight = 0.5
    erro = weight*erro_IgG + weight*erro_IgM + erro_V + erro_C


    return [erro, V, A_m, A_g, ts, erro_V, erro_IgM, erro_IgG, erro_C]

def model_adj(x):
    result = model(x)
    return result[0]

if __name__ == "__main__":    
        	
    opt_last =  42 #os ultimos opt_last dias não serão contados para o ajuste se o valor for negativo, coloque o tamanho do vetor (len(infected)) caso queira considerar todos
    opt_last_c = 20 #opt para a citocina

    #for use in P(t)32 = c0*exp(c1*t)
    c0,c1 = exp_mmq(dia[:opt_last], virus[:opt_last]) #expqm.py
    
    #define os bounds para cada um dos parâmetros
    #           V0    ,   PI_V   ,    k_v1        , k_v2           , delta_Apm        ,  alpha_B         , delta_A_G , delta_A_M ,     pi_camp    ,    pi_cthe    ,    pi_ctke    ,      pi_cb    , delta_c  
    bounds = [(1,1000), (1.1,1.4), (1.0E-4,1.0E-1), (1.0E-6,1.0E-4), (1.0E-01, 1.08E1), (1.0E+00,1.0E+02), (0.01,1.0), (0.01,1.0),(3.5e-6,5.5e-6),(7.0e-6,1.5e-5),(9.5e-7,1.3e-6),(1.0e-6,6.0e-6),(0.1,0.35)]

    #chama a evolução diferencial que o result contém o melhor individuo
    result = differential_evolution(model_adj, bounds, strategy='best1bin', popsize=20, workers=2)
    print('Params order: ')
    print ('V0, pi_v,k_v1, k_v2, delta_Apm, alpha_B, delta_A_G, delta_A_M, pi_camp, pi_cthe, pi_ctke, pi_cb,delta_c')
    print(result.x)
    #saving the best offspring...
    np.savetxt('params_simple.txt',result.x)
    
    #saving the samples for UQ
    np.savetxt('execution_de_100_ge.txt',execution_de)
   
    erro, V, A_m, A_g, ts, erro_V, erro_IgM, erro_IgG, erro_C = model(result.x)
    print("RELATIVE ERROR")
    print("Erro viremia: ", erro_V)
    print("Erro IgM: ", erro_IgM)
    print("Erro IgG: ", erro_IgG)
    print("Erro cytokine: ", erro_C)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.set_size_inches(12, 25)
   
   
    ax1.set_title('Viremia and Antibodies')
    
    #Plot active infected cases
    ax1.plot(ts, np.log10(V), label='Viremia model', linewidth=4)
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

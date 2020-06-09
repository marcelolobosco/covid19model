#!/usr/bin/python
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
from Modelo import *

#pathway to the data 
path = '../../data/'

# global data
# ajustar pela viremia e anticorpos
dadosViremiaLog10 = pd.read_csv(path+'Viral_load_10_2.csv',';')
dadosAnticorposLog2 = pd.read_csv(path+'IgG_IgM_21_1b_average.csv',',')

#casos = np.loadtxt(path+'active_world.txt');
dia = dadosViremiaLog10['Day']
#total_population = 83.02e6
first_day = 0
#confirmed = np.loadtxt(path+'confirmed_Germany.txt');
virus = dadosViremiaLog10['Viral_load']
antibody_g = dadosAnticorposLog2['IgG']
antibody_m = dadosAnticorposLog2['IgM']

#deaths = np.loadtxt(path+'death_Germany.txt');
#recovery = np.loadtxt(path+'recovered_Germany.txt');

# ones in days to be considered and zero otherwise
mask_virus     =[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
mask_antibodies=[0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]




execution_de =  []

def model(x):
    '''
    Params description:
    
    x[0]=> Ap0
    x[1]=> Apm0
    x[2]=> Thn0
    x[3]=> The0
    x[4]=> Tkn0
    x[5]=> Tkn0
    x[6]=> Tke0
    x[7]=> B0
    x[8]=> Ps0
    x[9]=> Pl0
    x[10]=> Bm0
    x[11]=> A0  
    x[12]=> pi_v 
    x[13]=> c_v1
    x[14]=> c_v2
    x[15]=> delta
    x[16]=> k_v1
    x[17]=> k_v2
    x[18]=> alpha_Ap
    x[19]=> beta_Ap
    x[20]=> k_ap1
    x[21]=> k_ap2
    x[22]=>delta_Apm
    x[23]=>alpha_Tn
    x[24]=>pi_T
    x[25]=>k_te1
    x[26]=>delta_te
    x[27]=>alpha_B
    x[28]=>pi_B1
    x[29]=>pi_B2
    x[30]=>beta_S
    x[31]=>beta_L
    x[32]=>beta_Bm
    x[33]=>delta_S
    x[34]=>delta_L
    x[35]=>gamma_M
    x[36]=>k_bm1
    x[37]=>k_bm2
    x[38]=>pi_AS
    x[39]=>pi_AL
    x[40]=>delta_A_G
    x[41]=>delta_A_M
    x[42]=>c11
    x[43]=>c12
    x[44]=>c13
    x[45]=>c14
    x[46]=>Ap0 => homeostasis
    x[47]=>Thn0 => homeostasis
    x[48]=>Tkn0 => homeostasis
    x[49]=> B0  => homeostasis
    '''

    ts=45

    P0 = (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11])
    model_args = (x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40], x[41], x[42], x[43], x[44], x[45], x[46], x[47], x[48], x[49])

    Ps = odeint(immune_response, P0, ts, args=(model_args), full_output=True, printmessg=True) 

    V=Ps[:,0] # virus
    A_m=Ps[:,11] # antibody
    A_g=Ps[:,12] # antibody
    
    #Viremia error 
    #virus represents experimental data, V the numerical one 
    virus_aux = np.multiply(virus, mask_virus)
    V_aux = np.multiply(V[:opt_last], mask_virus[first_day:opt_last])
    erro_V = np.linalg.norm(virus_aux[first_day:opt_last]-V_aux, np.inf)/np.linalg.norm(virus_aux[first_day:opt_last], np.inf)
    
    #antibody G
    antibody_g_aux = np.multiply(antibody_g, mask_antibodies)
    IgG_aux = np.multiply(A_g[:opt_last], mask_antibodies[first_day:opt_last])
    erro_IgG = np.linalg.norm(antibody_g_aux[first_day:opt_last]-IgG_aux, np.inf)/np.linalg.norm(antibody_g_aux[first_day:opt_last], np.inf)

    #antibody M
    antibody_m_aux = np.multiply(antibody_m, mask_antibodies)
    IgM_aux = np.multiply(A_m[:opt_last], mask_antibodies[first_day:opt_last])
    erro_IgM = np.linalg.norm(antibody_m_aux[first_day:opt_last]-IgM_aux, np.inf)/np.linalg.norm(antibody_m_aux[first_day:opt_last], np.inf)

    
    weight = 0.5
    erro = weight*erro_IgG + weight*erro_IgM + erro_V


    return [erro, V, A_m, A_g, ts, erro_V, erro_IgM, erro_IgG]

def model_adj(x):
    result = model(x)
    return result[0]

if __name__ == "__main__":    
        	
    opt_last =  45 # os ultimos opt_last dias não serão contados para o ajuste se o valor for negativo, coloque o tamanho do vetor (len(infected)) caso queira considerar todos
    
    #for use in P(t) = c0*exp(c1*t)
    c0,c1 = exp_mmq(dia[:opt_last], virus[:opt_last])
    
    #define os bounds para cada um dos parâmetros
    bounds = [(1,999), (1.0e6,1.0e7), (0.0, 1.0), (1.0e5,1.0e7),(0.0,1.0),(1.0e4,1.0e6),(1.0e3,9.0e5),(0.0,1.0),(1.0e3,9.9e5),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0), (1.1, 1.9), (2.1, 2.9), (0.1, 0.9), (0.0001, 0.0009), (1.0E-06, 1.0E-05), (1.0E-03, 1.0E-02), (1.0E-01, 1.0E-00), (0.1, 0.8), (10.0, 90.0), (1.0E-01, 1.0E-00), (1.0E-04, 1.0E-03), (1.0E-05, 1.0E-04), (1.0E-08, 1.0E-07), (0.1,0.9), (1.0E+00,1.0E+01), (1.0E-06,1.0E-05), (1.0E-8,1.0E-9), (0.0001,0.0009), (1.0E-06,1.0E-05), (1.0E-06,1.0E-05), (1.0,9.0), (1.0E-04,1.0E-03), (1.0E-04,1.0E-03), (1.0E-5,1.0E-4),(2000.0,3000.0), (0.001,0.009), (0.0001,0.0009), (0.01,0.09), (0.01,0.09), (1.0E-04,1.0E-03), (1.0E-07,1.0E-06), (1.0E-08,1.0E-07), (0.1,0.9),(1.0E6,1.0E7), (1.0E6,1.0E7), (1.0E5,1.0E6), (1.0E1,1.9E1)]

    #chama a evolução diferencial que o result contém o melhor individuo
    result = differential_evolution(model_adj, bounds, strategy='best1bin', popsize=20)
    print('Params order: ')
    print ('pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1, k_ap2, delta_Apm, alpha_Tn, pi_T, k_te1, delta_te, alpha_B, pi_B1, pi_B2, beta_S, beta_L, beta_Bm, delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS, pi_AL, delta_A_G, delta_A_M, c11, c12, c13, c14, Ap0, Thn0, Tkn0, B0')
    print(result.x)
    #saving the best offspring...
    np.savetxt('params_simple.txt',result.x)
    
    #saving the samples for UQ
    np.savetxt('execution_de_100_ge.txt',execution_de)
   
    erro, V, A_m, A_g, ts, erro_V, erro_IgM, erro_IgG = model(result.x)
    print("RELATIVE ERROR")
    print("Erro viremia: ", erro_V)
    print("Erro IgM: ", erro_IgM)
    print("Erro IgG: ", erro_IgG)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.set_size_inches(12, 25)
   
   
    ax1.set_title('Viremia and Antibodies')
    
    #Plot active infected cases
    ax1.plot(ts, log10(V), label='Viremia model', linewidth=4)
    ax1.plot(ts, virus[first_day:], label='data', linewidth=4)
    ax1.set_xlabel('day')
    ax1.set_ylabel('log 10(copies/ml)')    
    ax1.legend()
    ax1.grid()
        
    #Plot death cases 
    ax2.plot(ts, log2(A_g), label='IgG model', linewidth=4)
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

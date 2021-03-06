#!/usr/bin/python

import sys
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
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
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami



#################################LEITURA DE DADOS############################################

#from COVID-19_all
    #######   Viremia  log10 copias/ml ##########
dadosViremiaLog10 = pd.read_csv('../../data/Viral_load.csv',',')
dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 
dadosCitocinaObitos = pd.read_csv('../../data/IL6_non-survivors_19.csv',',')
dadosCitocinaSobreviventes = pd.read_csv('../../data/IL6_survivors_19.csv',',')
dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM.csv',',')
antibody_g = dadosAnticorposLog2_avg['IgG']
antibody_m = dadosAnticorposLog2_avg['IgM']

dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM.csv',',')
antibody_g = dadosAnticorposLog2_avg['IgG']
antibody_m = dadosAnticorposLog2_avg['IgM']

#TEMPO
dias_de_simulação = 35
t=range(dias_de_simulação)



def eval_model(x):
 
    """
    x: model parameters
    x[0]:  V0
    x[1]: pi_v
    x[2]: pi_c_apm
    x[3]: pi_c_i 
    x[4]: pi_c_tke
    x[5]: beta_apm (achei k_apm)
    x[6]: beta_tk (achei k_tk)
    x[7]:  k_v1
    x[8]:  k_v2
    x[9]:  k_v3
    """

	#CONDICOES INICIAIS FROM COVID-19_ALL
    V0 = 2.18808824e+00
    Ap0 = 1.0e6
    Apm0 = 0.0
    Ai0=0
    C0=0
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

    #vetor que salva todas as condicoes iniciais
    P0 = [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G,Ai0,C0]


	######################PARAMETROS DO MODELO######################################
    pi_v = x[0]
    c_v1 = x[1]#2.63
    c_v2 = x[2]#0.60
    k_v1 = x[3]#3.5e-3
    k_v2 = x[4]#9.5e-5
    alpha_Ap = 1.87E-06*0.4 #remover
    beta_Ap = x[5]#2.00E-03
    c_ap1 = x[6] 
    c_ap2 = x[7]

    delta_Apm = x[8]#8.14910996e+00 
    alpha_Tn = 2.17E-04 #=>remover
    beta_tk = x[9]#1.431849023090428446e-05
    pi_tk = x[10]#1.0E-08 
    delta_tk = x[11]#0.0003
    alpha_B = x[12]#3.578236584371140339e+02
    pi_B1 = x[13]#8.979145365768647095e-05
    pi_B2 = x[14]#1.27E-8

    beta_ps = x[15]#6.0e-6
    beta_pl = x[16]#5.0e-6
    beta_Bm = x[17]#1.0E-06
    delta_S = x[18]#2.5
    delta_L = x[19]#0.35
    gamma_M = x[20]#(1.95E-06)*500.0
    k_bm1 = x[21]#1.0e-5      
    k_bm2 = x[22]#2500.0 
    pi_AS = x[23]#0.087
    pi_AL = x[24]#0.001
    delta_ag = x[25]#0.07
    delta_am = x[26]#0.07
    alpha_th = x[27]#2.17E-04
    beta_th = x[28]#1.8e-5#1.0E-07
    pi_th = x[29]#1.0E-08  
    delta_th = x[30]#0.3
    Ap0 = x[31]#1.0e6
    Thn0 = x[32]#1.0e6
    Tkn0 = x[33]#5.0e5
    B0 = x[34]#2.5E5


    pi_c_apm = x[35]#7.43773673e-01
    pi_c_i = x[36]#1.97895565e-02
    pi_c_tke = x[37]#0.04730172
    delta_c = x[38]#8.26307952e+00
    beta_apm = x[39]#5.36139617e-01
    k_v3 = x[40]#3.08059068e-03
    beta_tke = x[41]#2.10152618e-01
    
    model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, c_ap1, c_ap2,
    delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk, alpha_B, pi_B1, pi_B2, 
    beta_ps, beta_pl, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke)

	

    #resolve o modelo
    y=integrate.odeint(immune_response_v3, P0, t, args=(model_args))

    #ordem de retorno 
    #V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, C = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]
    
    
   


    return y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]

if __name__ == "__main__":


    # uso
    '''if(len(sys.argv) != 6):
        print("Usage: sirmodel [arquivo_samples] [caso] [opt_forecast] [path_output] [output extension]")
        sys.exit(0)'''

    #arqpar = sys.argv[1]
    #caso = sys.argv[2]
    #opt_last = 0#int(sys.argv[3]);
    #opt_forecast = int(sys.argv[3]); # days of forecast
    #out_path = sys.argv[4]#'./output/20_noft_nodelay/'
    #ext = sys.argv[5]#'./output/20_noft_nodelay/'

    '''print('Arquivo de Samples: ', arqpar)
    print('Caso: ', caso)
    print('Dias de previsão: ', opt_forecast)
    print('Nome do output: ', out_path)
    print('Extensao do output: ', ext)'''

    # options for simulation
    
    
    
    
    opt_uq = False
    opt_loocv = False
    opt_save_evals = True
    
    opt_sobol_salib = True
    opt_sobol_pce = False


    
   

    #teste
    #V0 = cp.Uniform(2.406897064,1.969279416)
    
    pi_v = 0.1955
    c_v1 = 2.63
    c_v2 = 0.60
    k_v1 = 3.5e-3
    k_v2 = 9.5e-5
    alpha_Ap = 1.87E-06*0.4
    beta_Ap = 2.00E-03
    c_ap1 = 0.8  
    c_ap2 = 40.0

    delta_Apm = 8.14910996e+00 
    alpha_Tn =2.17E-04 
    beta_tk = 1.431849023090428446e-05
    pi_tk = 1.0E-08 
    delta_tk = 0.0003
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
    beta_th = 1.8e-5#1.0E-07
    pi_th = 1.0E-08  
    delta_th = 0.3
    Ap0 = 1.0e6
    Thn0 = 1.0e6
    Tkn0 = 5.0e5
    B0 = 2.5E5


    pi_c_apm = 7.43773673e-01
    pi_c_i = 1.97895565e-02
    pi_c_tke = 0.04730172
    delta_c = 8.26307952e+00
    beta_apm = 5.36139617e-01
    k_v3 = 3.08059068e-03
    beta_tke = 2.10152618e-01
    
    label_param = ("pi_v", "c_v1", "c_v2", "k_v1", "k_v2", 
        "beta_Ap", "c_ap1", "c_ap2", "delta_Apm", "beta_tk",
        "pi_tk", "delta_tk", "alpha_B","pi_B1", "pi_B2", 
        "beta_ps", "beta_pl", "beta_Bm","delta_S", "delta_L", "gamma_M", "k_bm1", "k_bm2", "pi_AS",
        "pi_AL", "delta_ag", "delta_am", "alpha_th", "beta_th", "pi_th", "delta_th", "Ap0", "Thn0", "Tkn0", "B0",  
        "pi_c_apm", "pi_c_i","pi_c_tke","delta_c", "beta_apm", "k_v3", "beta_tke")
    
    
    if opt_uq:
        min_bound = 0.9
        max_bound = 1.1
        
        
        model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, beta_Ap, c_ap1, c_ap2,
        delta_Apm, beta_tk, pi_tk, delta_tk, alpha_B, pi_B1, pi_B2, 
        beta_ps, beta_pl, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
        pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0,  
        pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke)
        
        for i in range(len(model_args)):
            vpdf = cp.Uniform(model_args[i]*min_bound,model_args[i]*max_bound)
            if i==0:
                dist = cp.J(vpdf)
            else:
                dist = cp.J(dist, vpdf)
        
        '''
        pdf_pi_v = cp.Uniform(pi_v*min_bound,pi_v*max_bound)
        pdf_c_v1 = cp.Uniform(c_v1*min_bound,c_v1*max_bound)
        pdf_c_v2 = cp.Uniform(c_v2*min_bound,c_v2*max_bound)
        pdf_k_v1 = cp.Uniform(k_v1*min_bound,k_v1*max_bound)
        pdf_k_v2 = cp.Uniform(k_v2*min_bound,k_v2*max_bound) 
        pdf_alpha_Ap = cp.Uniform(alpha_Ap*min_bound,alpha_Ap*max_bound) 
        pdf_beta_Ap = cp.Uniform(beta_Ap*min_bound,beta_Ap*max_bound) 
        pdf_c_ap1 = cp.Uniform(c_ap1*min_bound,c_ap1*max_bound) 
        pdf_c_ap2 = cp.Uniform(c_ap2*min_bound,c_ap2*max_bound) 
        pdf_delta_Apm = cp.Uniform(delta_Apm*min_bound,delta_Apm*max_bound)
        pdf_alpha_Tn = cp.Uniform(alpha_Tn*min_bound,alpha_Tn*max_bound)
        pdf_beta_tk = cp.Uniform(beta_tk*min_bound,beta_tk*max_bound)
        pdf_pi_tk = cp.Uniform(pi_tk*min_bound,pi_tk*max_bound)
        pdf_delta_tk = cp.Uniform(delta_tk*min_bound,delta_tk*max_bound)
        pdf_alpha_B = cp.Uniform(alpha_B*min_bound,alpha_B*max_bound)
        
        dist = cp.J(pdf_pi_v, pdf_c_v1,pdf_c_v2, pdf_k_v1,pdf_k_v2,pdf_alpha_Ap, 
            pdf_beta_Ap, pdf_c_ap1, pdf_c_ap2, pdf_delta_Apm, pdf_alpha_Tn,
            pdf_beta_tk, pdf_pi_tk, pdf_delta_tk)
        dist = cp.J(dist, pdf_alpha_B)
        '''
        #---------------------------------MATRIZ COVARIANCIA, DISTRIBUICAO-----------------------------------------------

        '''m = np.loadtxt(arqpar, comments='#');

        Cov_matrix = np.cov(m[:, 3:13].T)
        mean_vector = np.mean(m[:,3:13], axis=0)
        std_vector = np.std(m[:,3:13], axis=0)
        print("mean = ", mean_vector)
        print("std = ", std_vector)

        
        dist = cp.MvNormal(mean_vector, Cov_matrix)'''

       
       
        
        npar = len(dist)

        #grau da aproximacao polinomial
        degpc = 2
        ns = 10000#3*P(degpc,npar)
        print("number of input parameters %d" % npar)
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
        k = 0
        lpts = range(ns)
        samp = samples.T
        print("evaluating samples: ")
        for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        #for s in samples.T:
            s = samp[k]
            k = k+1
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
        
        '''
        #Chaos Poly
        
        # create pce
        pce = cp.orth_ttr(degpc, dist)
        # compute surrogate models (sm_)
        sm_virus  = cp.fit_regression(pce, samples, evals_virus)
        sm_c  = cp.fit_regression(pce, samples, evals_C)
        sm_am  = cp.fit_regression(pce, samples, evals_AM)
        sm_ag  = cp.fit_regression(pce, samples, evals_AG)
        '''
            
        #-----------------------------------------------------------------------------------------------------------------------# 
        #
        # Plot data
        #
        plt.style.use('../estilo/PlotStyle.mplstyle')
        plt.close('all')

        output_path = './output_monte_carlo/'
        ext = '.png'
        itvl = 6
        caso = 'caso.'
        

        
        
        #saving the evals
        if opt_save_evals:
            np.savetxt(output_path+caso+'_evals_virus.txt',evals_virus)
            np.savetxt(output_path+caso+'_evals_Ap.txt',evals_Ap)
            np.savetxt(output_path+caso+'_evals_Apm.txt',evals_Apm)
            np.savetxt(output_path+caso+'_evals_Thn.txt',evals_Thn)
            np.savetxt(output_path+caso+'_evals_The.txt',evals_The)
            np.savetxt(output_path+caso+'_evals_Tkn.txt',evals_Tkn)
            np.savetxt(output_path+caso+'_evals_Tke.txt',evals_Tke)
            np.savetxt(output_path+caso+'_evals_B.txt',evals_B)
            np.savetxt(output_path+caso+'_evals_Ps.txt',evals_Ps)
            np.savetxt(output_path+caso+'_evals_Pl.txt',evals_Pl)
            np.savetxt(output_path+caso+'_evals_Bm.txt',evals_Bm)
            np.savetxt(output_path+caso+'_evals_AM.txt',evals_AM)
            np.savetxt(output_path+caso+'_evals_AG.txt',evals_AG)
            np.savetxt(output_path+caso+'_evals_I.txt',evals_I)
            np.savetxt(output_path+caso+'_evals_C.txt',evals_C)


        #mean_virus= plot_confidence_interval(plt, t, evals_virus, 'red', 'Modelo: Virus')
        plt.figure('Viremia')
        #dadosViremiaLog10.plot.scatter(x='Day',y='Viral_load',color='m',label='Dados experimentais')
        plt.plot(t, dadosViremiaLog10['Viral_load'], 'o', label='data', linewidth=4)
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('dia')
        plt.ylabel('população')
        plt.legend(loc='upper left', prop={'size':13})
        plt.grid(True)
        plt.title("Solution Virus")
        mean_virus= plot_confidence_interval(plt, t, evals_virus, 'red', 'Modelo: Virus')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path+caso+'_output_Virus'+ext)
        plt.show()
        

        #-------------------------ANTICORPOS--------------------------------------
        
        plt.figure('Anticorpos')
        plt.xlim(0.0,dias_de_simulação)
        plt.legend(loc='upper left', prop={'size':13})
        plt.grid(True)
        plt.title("Solution Anticorpos")
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.ylim(0.0,8.0)
        #obs:esta sem log
        plt.plot(t, antibody_m, 'o', label='IgM data', linewidth=4)
        plt.plot(t, antibody_g, 'o', label='IgG data', linewidth=4)
        plt.xlabel('Tempo pós-infecção (dias)')
        plt.ylabel('Anticorpos - sem log 2 S/CO')
        mean_AM = plot_confidence_interval(plt, t, evals_AM, 'orange', 'Modelo: A_M')
        mean_AG = plot_confidence_interval(plt, t, evals_AG, 'red', 'Modelo: A_G')
        plt.savefig(output_path+'Anticorpos.pdf',bbox_inches='tight',dpi = 300)
        
        
        #-------------------------AP--------------------------------------
        
        plt.figure('Ap')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Apresentadores')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Ap = plot_confidence_interval(plt, t, evals_Ap, 'green', 'Modelo: Ap')
        plt.savefig(output_path+'Ap.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------APM---------------------------------------------
       
        plt.figure('Apm')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Apresentadores Maduras')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Apm = plot_confidence_interval(plt, t, evals_Apm, 'blue', 'Modelo: Apm')
        plt.savefig(output_path+'Apm.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------Thn---------------------------------------------
        
        plt.figure('Thn')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Th Naive')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Thn = plot_confidence_interval(plt, t, evals_Thn, 'black', 'Modelo: Thn')
        plt.savefig(output_path+'Thn.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------The---------------------------------------------
        
        plt.figure('The')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Th Efetora')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_The = plot_confidence_interval(plt, t, evals_The, 'purple', 'Modelo: The')
        plt.savefig(output_path+'The.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------Tkn---------------------------------------------    
        
        plt.figure('Tkn')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Tk Naive')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Tkn = plot_confidence_interval(plt, t, evals_Tkn, 'orange', 'Modelo: Tkn')
        plt.savefig(output_path+'Tkn.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------Tke---------------------------------------------
       
        plt.figure('Tke')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Tk Efetora')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Tke = plot_confidence_interval(plt, t, evals_Tke, 'red', 'Modelo: Tke')
        plt.savefig(output_path+'Tke.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------B---------------------------------------------
        
        plt.figure('B')
        plt.xlim(0.0,dias_de_simulação)
        plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('B')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_B = plot_confidence_interval(plt, t, evals_B, 'green', 'Modelo: B')
        plt.savefig(output_path+'B.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #np.savetxt(output_path+'b.txt',y[:,7])
        
        #-------------------------Ps---------------------------------------------
        
        plt.figure('Ps')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Ps')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Ps = plot_confidence_interval(plt, t, evals_Ps, 'blue', 'Modelo: Ps')
        plt.savefig(output_path+'Ps.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------Pl---------------------------------------------
        
        plt.figure('Pl')
        plt.xlim(0.0,dias_de_simulação)
        plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Pl')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Pl = plot_confidence_interval(plt, t, evals_Pl, 'black', 'Modelo: Pl')
        plt.savefig(output_path+'Pl.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------Bm---------------------------------------------
        
        plt.figure('Bm')
        plt.xlim(0.0,dias_de_simulação)
        plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('Bm')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_Bm = plot_confidence_interval(plt, t, evals_Bm, 'purple', 'Modelo: Bm')
        plt.savefig(output_path+'Bm.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------AM---------------------------------------------
        
        plt.figure('A_M')
        plt.xlim(0.0,dias_de_simulação)
        plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('A_M')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_AM = plot_confidence_interval(plt, t, evals_AM, 'orange', 'Modelo: A_M')
        plt.savefig(output_path+'A_M.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------AG---------------------------------------------
       
        plt.figure('A_G')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('A_G')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_AG = plot_confidence_interval(plt, t, evals_AG, 'red', 'Modelo: A_G')
        plt.savefig(output_path+'A_G.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------I---------------------------------------------
        
        plt.figure('Ai')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('APC infectada')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_I = plot_confidence_interval(plt, t, evals_I, 'green', 'Modelo: I')
        plt.savefig(output_path+'Ai.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        #-------------------------C SOBREVIVENTES---------------------------------------------
        
        plt.figure('C')
        #dadosCitocinaSobreviventes.plot.scatter(x='Day',y='IL6(pg/mL)',color='g',label='Dados experimentais(Sobreviventes)')
        plt.plot(dadosCitocinaSobreviventes['Day']+5, dadosCitocinaSobreviventes['IL6(pg/mL)'], 'o', label='data', linewidth=4)
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('C')
        plt.legend(loc='upper left', prop={'size':13})
        plt.xticks(rotation=45)
        plt.tight_layout()
        mean_C = plot_confidence_interval(plt, t, evals_C, 'blue', 'Modelo: C')
        plt.savefig(output_path+'C.pdf',bbox_inches='tight',dpi = 300)
        plt.show()
        
        print("vim ate aqui")
        
        
        #-------------------------C OBITOS---------------------------------------------
        '''
        plt.figure('C')
        dadosCitocinaObitos.plot.scatter(x='Day',y='IL6(pg/mL)',color='y',label='Dados experimentais(Obito)')
        plt.xlim(0.0,dias_de_simulação)
        #plt.ylim(0.0,8.0)
        plt.plot(t,y[:,14],label='C',linewidth=1.5, linestyle="-")
        plt.xlabel('Tempo pós-vacinação (dias)')
        plt.ylabel('C')
        plt.legend()
        plt.savefig(output_path+'C2.pdf',bbox_inches='tight',dpi = 300)
        '''
   

    
    # Sensitivity analysis

    #-------------------------------------------SOBOL---------------------------------------------#
    if(opt_sobol_pce):
        fig, ax = plt.subplots(2, 2)
        
        plt.figure();
        plt.title("V")
        plot_sensitivity(plt, t, sm_virus, dist, label_param)
        plt.legend(loc=(1.04,0),prop={'size': 6})
        plt.savefig('output_sens_virus.pdf')
        
        plt.figure();
        plt.title("C")
        plot_sensitivity(plt, t, sm_c, dist, label_param)
        plt.legend(loc=(1.04,0),prop={'size': 6})
        plt.savefig('output_sens_c.pdf')
        
        plt.figure();
        plt.title("IgG")
        plot_sensitivity(plt, t, sm_ag, dist, label_param)
        plt.legend(loc=(1.04,0), prop={'size': 6})
        plt.savefig('output_sens_igg.pdf')
        
        plt.figure();
        plt.title("IgM")
        plot_sensitivity(plt, t, sm_am, dist, label_param)
        plt.legend(loc=(1.04,0), prop={'size': 6})
        plt.savefig('output_sens_igm.pdf')
    
    if(opt_sobol_salib):
        min_bound = 0.9
        max_bound = 1.1
    
        model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, beta_Ap, c_ap1, c_ap2,
            delta_Apm, beta_tk, pi_tk, delta_tk, alpha_B, pi_B1, pi_B2, 
            beta_ps, beta_pl, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
            pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0,  
            pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke)
        vbounds = []
        for i in range(len(model_args)):
            vbounds.append([model_args[i]*min_bound, model_args[i]*max_bound])
        
        # Define the model inputs
        problem = {
            'num_vars': 42,
            'names': ['pi_v', 'c_v1', 'c_v2', 'k_v1', 'k_v2', 'beta_Ap', 'c_ap1', 'c_ap2', 
                'delta_Apm', 'beta_tk','pi_tk', 'delta_tk', 'alpha_B','pi_B1', 'pi_B2', 
                'beta_ps', 'beta_pl', 'beta_Bm','delta_S', 'delta_L', 'gamma_M', 'k_bm1',
                'k_bm2', 'pi_AS', 'pi_AL', 'delta_ag', 'delta_am', 'alpha_th', 'beta_th', 
                'pi_th', 'delta_th', 'Ap0', 'Thn0', 'Tkn0', 'B0',  'pi_c_apm', 'pi_c_i',
                'pi_c_tke','delta_c', 'beta_apm', 'k_v3', 'beta_tke'],
            'bounds': vbounds
        }

        # Generate samples
        nsobol = 10000
        param_values = saltelli.sample(problem, nsobol, calc_second_order=False)

        # Run model (example)
        #Y = Ishigami.evaluate(param_values)
        k = 0
        nexec = np.shape(param_values)[0]
        #print("SHAPE ", np.shape(param_values)[0] , np.shape(param_values)[1] )
        lpts = range( nexec )
        sm_v = []
        sm_ag = []
        sm_am = []
        sm_c = []
        print("evaluating samples for SA: ")
        for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        #for s in samples.T:
            s = param_values[k,:] #samp[k]
            s = np.array(s)
            k = k+1
            virus,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = eval_model(s)
            
            #ignora 1o passo de tempo porque é a condição inicial e não tem variacia, levando a um NaN
            sm_v.append(virus[1:]) 
            sm_ag.append(A_G[1:])  
            sm_am.append(A_M[1:]) 
            sm_c.append(C[1:]) 
        
        sm_v = np.array(sm_v)
        sm_ag = np.array(sm_ag)
        sm_am = np.array(sm_am)
        sm_c = np.array(sm_c)

        #print( "SHAPE SM_S", np.shape(sm_s) )
        #Y = Ishigami.evaluate(param_values)
        nsteps = np.shape(sm_v)[1]
        Si_v = np.zeros((nsteps,42))
        Si_ag = np.zeros((nsteps,42))
        Si_am = np.zeros((nsteps,42))
        Si_c = np.zeros((nsteps,42))

        # Perform analysis
        for i in range(nsteps):
            sob_v = sobol.analyze(problem, sm_v[:,i], calc_second_order=False, print_to_console=False)
            sob_ag = sobol.analyze(problem, sm_ag[:,i], calc_second_order=False, print_to_console=False)
            sob_am = sobol.analyze(problem, sm_am[:,i], calc_second_order=False, print_to_console=False)
            sob_c = sobol.analyze(problem, sm_c[:,i], calc_second_order=False, print_to_console=False)

            Si_v[i,:] = sob_v['S1']
            Si_ag[i,:] = sob_ag['S1']
            Si_am[i,:] = sob_am['S1']
            Si_c[i,:] = sob_c['S1']

        print("salvando arquivos Sobol")
        if(opt_save_evals):
            np.savetxt('ouput_sobol_v.txt',Si_v)
            np.savetxt('ouput_sobol_ag.txt',Si_ag)
            np.savetxt('ouput_sobol_am.txt',Si_am)
            np.savetxt('ouput_sobol_c.txt',Si_c)
        
        plt.figure();
        plt.title("V")
        plot_sensitivity_mc(plt, t[1:], Si_v.T, label_param)
        plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
        plt.tight_layout()
        plt.savefig('output_sens_virus.pdf')
        
        plt.figure();
        plt.title("IgG")
        plot_sensitivity_mc(plt, t[1:], Si_ag.T, label_param)
        plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
        plt.tight_layout()
        plt.savefig('output_sens_igg.pdf')
        
        plt.figure();
        plt.title("IgM")
        plot_sensitivity_mc(plt, t[1:], Si_am.T, label_param)
        plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
        plt.tight_layout()
        plt.savefig('output_sens_igm.pdf')
        
        plt.figure();
        plt.title("C")
        plot_sensitivity_mc(plt, t[1:], Si_c.T, label_param)
        plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
        plt.tight_layout()
        plt.savefig('output_sens_c.pdf')






    #
    # Cross validation 
    #
    if(opt_loocv):
        loo("S", sm_s, ns, npar, evals_s, samples, dist)
        loo("I", sm_i, ns, npar, evals_i, samples, dist)
        loo("R", sm_r, ns, npar, evals_r, samples, dist)
        loo("M", sm_m, ns, npar, evals_m, samples, dist)
    
    
    print('done')
# Fim

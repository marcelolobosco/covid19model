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
#from SALib.sample import saltelli
#from SALib.analyze import sobol
#from SALib.test_functions import Ishigami



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



def eval_model(pi_v, data):
 
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
    #pi_v = 0.1955
    c_v1 = 2.63
    c_v2 = 0.60
    k_v1 = 3.5e-3
    k_v2 = 9.5e-5
    alpha_Ap = 1.87E-06*0.4
    beta_Ap = 2.00E-03
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

    beta_S = 6.0e-6
    beta_L = 5.0e-6
    beta_Bm = 1.0E-06
    delta_S = 2.5
    delta_L = 0.35
    gamma_M = (1.95E-06)*500.0
    k_bm1 = 1.0e-5      
    k_bm2 = 2500.0 
    pi_AS = 0.087
    pi_AL = 0.001
    delta_A_G = 0.07
    delta_A_M = 0.07
    c11 = 2.17E-04
    c12 = 1.8e-5#1.0E-07
    c13 = 1.0E-08  
    c14 = 0.3
    Ap0 = 1.0e6
    Thn0 = 1.0e6
    Tkn0 = 5.0e5
    B0 = 2.5E5


    pi_c_apm = 7.43773673e-01
    pi_c_i = 1.97895565e-02
    pi_c_tke = 0.04730172
    delta_c = 8.26307952e+00
    k_apm = 5.36139617e-01
    k_v3 = 3.08059068e-03
    k_tk = 2.10152618e-01
    
    model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1, k_ap2,
    delta_Apm, alpha_Tn, pi_T, k_te1, delta_te, alpha_B, pi_B1, pi_B2, 
    beta_S, beta_L, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_A_G, delta_A_M, c11, c12, c13, c14, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, k_apm, k_v3, k_tk)

	#------------------------------------------------------------------------------------------------------------------#
    
    #c0,c1 = define_ft(casos, dia); ?

    # ordem do x  ['amax','r','ti','tf', 'e','theta', 'tau_1','tau_2', 'tau_3', 'm']
    #ordem na fun  amax,r,ti,tf,m,e,r1,r2):
    #if opt_ft:
        #sirargs = ( x[0]/data['total_population'],x[1],x[2],x[3],x[9],x[4], 1.0/(x[6]+x[7]), 1.0/(x[6] + x[8]), data['first_day_exp'], c0,c1 )
    #else:
	    #sirargs = ( x[0]/data['total_population'],x[1],x[2],x[3],x[9],0.0, 1.0/(x[6]+x[7]), 1.0/(x[6] + x[8]), data['first_day_exp'], c0,c1 )

    #resolve o modelo
    y,d=integrate.odeint(immune_response_v3, P0, t, args=(model_args), full_output=1, printmessg=True)

    #ordem de retorno 
    #V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, C = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]
    
    
    #("a_{max}", "r", "t_i", "t_f", "m","e", "\tau1", "\tau2", "\tau_3", "h", "c")
    '''t1 = x[6]
    t2 = x[7]
    t3 = x[8]
    h  = param_h #x[9]
    c  = param_c #x[10]

    if (opt_delay):
        In = np.zeros(len(I));
        Rn = np.zeros(len(R));
    
        ii = find_nearest(ts, t1)
        if (ts[ii]-t1) < 0.0: ii = ii+1;
        for i in range(0, ii):
            In[i] = infected[first_day]
            Rn[i] = recovery[first_day]
        for i in range(ii, len(In)):
            j = find_nearest(ts, ts[i]-t1);
            In[i] = subnot*I[j]
            Rn[i] = subnot*R[j]
    else:    
        In = subnot*I
        Rn = subnot*R
    
    C = In+Rn+M'''
    
    # calculado o numero de leitos
    '''IH = np.zeros(len(ts))
    IC = np.zeros(len(ts))
   
    ii = find_nearest(ts, subnot*t1)
    for i in range(ii, len(In)):
        j = find_nearest(ts, ts[i]-subnot*t1)
        IH[i] = h*In[j]
        IC[i] = c*In[j]
    #Leitos = IG - OIG - RIG'''



    return y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]
    #return IgM,IgG,Viremia,Citocina
    #return S,In,Rn,M,C, IH,IC

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
    '''opt_delay = False
    opt_ft = False
    opt_leito = False
    opt_sobol = False
    opt_loocv = False
    opt_recovered = False
    opt_save_evals = True
    opt_flex = False'''

    # ajeita os dados para o caso
    data = None
    
    

    '''if(caso == 'Brazil'):
        data = data_brazil
        opt_leito = False
        casos = np.loadtxt('./data/active_world.txt');
        dia = np.array(range(0,len(casos)));
    elif(caso == 'MG'):
        data = data_mg
        opt_leito = False
        casos = data_brazil['infected'][35:]
        dia = np.array(range(0,len(casos)));
    elif(caso == 'SJDR'):
        data = data_sjdr
        opt_leito = False
        casos = np.loadtxt('./data/active_world.txt');
        dia = np.array(range(0,len(casos)));
    elif(caso == 'JF'):
        data = data_jf
        opt_leito = False
        casos = np.loadtxt('./data/active_world.txt');
        dia = np.array(range(0,len(casos)));
    else:
        print(f"Coutry {caso} not found or not available")
        sys.exit(0)'''
    '''elif(caso == 'MG'):
        data = data_australia
        opt_leito = False
        casos = np.loadtxt('./data/active_world.txt');
        dia = np.array(range(0,len(casos)));
    elif(caso == 'JF'):
        data = data_germany
        opt_leito = False
        casos = np.loadtxt('./data/active_world.txt');
        dia = np.array(range(0,len(casos)));
    elif(caso == 'SJDR'):
        data = data_nz
        opt_leito = False
        casos = np.loadtxt('./data/active_world.txt');
        dia = np.array(range(0,len(casos)));
    '''
    
    
    '''
    elif(caso == 'MG'):
        data = data_mg
        opt_leito = False
        casos = data_brazil['infected'][35:]
        dia = np.array(range(0,len(casos)));
    elif(caso == 'JF'):
        data = data_jf
        opt_leito = True
        casos = data_mg['infected']
        dia = np.array(range(0,len(casos)));
    elif(caso == 'Varginha'):
        data = data_vga
        opt_leito = False
    '''
    '''if opt_last == 0:
        c0,c1 = define_ft(casos, dia);
    else:
        c0,c1 = define_ft(casos[:-opt_last], dia[:-opt_last]);'''


    
    # label_w = ['amax','r','ti','tf', 'e','theta', 'tau_1','tau_2', 'tau_3', 'm']
    #c = 0.2703802
    #h = 2*c
    #cov = 0.1

    #teste
    #V0 = cp.Uniform(2.406897064,1.969279416)
    pi_v = cp.Uniform(0.166175,0.224825)
    distribution = cp.J(pi_v)
    
    #---------------------------------MATRIZ COVARIANCIA, DISTRIBUICAO-----------------------------------------------

    '''m = np.loadtxt(arqpar, comments='#');

    Cov_matrix = np.cov(m[:, 3:13].T)
    mean_vector = np.mean(m[:,3:13], axis=0)
    std_vector = np.std(m[:,3:13], axis=0)
    print("mean = ", mean_vector)
    print("std = ", std_vector)

    
    dist = cp.MvNormal(mean_vector, Cov_matrix)'''

    #pode ser removido por enquanto... depois olhamos o caso da italia
    #param_h    = h# cp.Uniform(h-cov*h,h+cov*h) #calculo de leitos
    #param_c    = c# cp.Uniform(c-cov*c,c+cov*c)

   
    #label_param = ("$a_{max}$", "r", "$t_i$", "$t_f$", "e", "$\theta$", "$\tau_1$", "$\tau_2$", "$\tau_3$", "m")
    npar = len(distribution)

    #grau da aproximacao polinomial
    degpc = 2
    ns = 1000#3*P(degpc,npar)
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
    
    samples = distribution.sample(ns,"L")
    k = 0
    lpts = range(ns)
    samp = samples.T
    print("evaluating samples: ")
    for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    #for s in samples.T:
        s = samp[k]
        k = k+1
        virus,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = eval_model(s,data)
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

    output_path = './output_monte_carlo/'
    ext = '.png'
    itvl = 6
    caso = 'caso.'
    # tempo para plotar
    #ts = range(len(infected) - first_day + opt_forecast)
    #ti = range(len(infected[first_day:]))

    '''if(recovery is not None):
        tr = range(len(recovery[first_day:]))
    td = range(len(deaths[first_day:]))'''
    
    ### PARA PLOTAR AS ESTIMATIVAS
    '''tam = len(infected) - first_day + opt_forecast
    # ultima data com 15 dias previsao
    #now = dt.date(2020, 5, 1)
    initial_date = dt.date(2020, 1, 22) + dt.timedelta(days=first_day)
    days = [initial_date]
    for i in range(tam-1):
        initial_date += dt.timedelta(days=+1)
        days.append(initial_date)
    
    ### PARA OS DADOS
    tam2 = len(infected[first_day:])
    days2 = days[:tam2]'''

    
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
   

    '''
    # Sensitivity analysis

    #-------------------------------------------SOBOL---------------------------------------------#
    if(opt_sobol):
        # Define the model inputs
        problem = {
            'num_vars': 10,
            'names': ['amax', 'r', 'ti', 'tf', 'e', 'sub', 't1', 't2', 't3', 'm'],
            'bounds': [[mean_vector[0]-3*std_vector[0], mean_vector[0]+3*std_vector[0]],
                       [mean_vector[1]-3*std_vector[1], mean_vector[1]+3*std_vector[1]],
                       [mean_vector[2]-3*std_vector[2], mean_vector[2]+3*std_vector[2]],
                       [mean_vector[3]-3*std_vector[3], mean_vector[3]+3*std_vector[3]],
                       [mean_vector[4]-3*std_vector[4], mean_vector[4]+3*std_vector[4]],
                       [mean_vector[5]-3*std_vector[5], mean_vector[5]+3*std_vector[5]],
                       [mean_vector[6]-3*std_vector[6], mean_vector[6]+3*std_vector[6]],
                       [mean_vector[7]-3*std_vector[7], mean_vector[7]+3*std_vector[7]],
                       [mean_vector[8]-3*std_vector[8], mean_vector[8]+3*std_vector[8]],
                       [mean_vector[9]-3*std_vector[9], mean_vector[9]+3*std_vector[9]]
                      ]
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
        sm_s = []
        sm_i = []
        sm_r = []
        sm_m = []
        print("evaluating samples for SA: ")
        for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        #for s in samples.T:
            s = param_values[k,:] #samp[k]
            s = np.array(s)
            k = k+1
            suc,inf,rec,mor,conf, ih,ic = eval_model(s, data)
            sm_s.append(suc) # s
            sm_i.append(conf) # i --- TO-DO conf soh para o Brazil
            sm_r.append(rec) # r
            sm_m.append(mor) # m
        
        sm_s = np.array(sm_s)
        sm_i = np.array(sm_i)
        sm_r = np.array(sm_r)
        sm_m = np.array(sm_m)

        #print( "SHAPE SM_S", np.shape(sm_s) )
        #Y = Ishigami.evaluate(param_values)
        nsteps = np.shape(sm_s)[1]
        Si_s = np.zeros((nsteps,10))
        Si_i = np.zeros((nsteps,10))
        Si_r = np.zeros((nsteps,10))
        Si_m = np.zeros((nsteps,10))

        # Perform analysis
        for i in range(nsteps):
            sob_s = sobol.analyze(problem, sm_s[:,i], calc_second_order=False, print_to_console=False)
            sob_i = sobol.analyze(problem, sm_i[:,i], calc_second_order=False, print_to_console=False)
            sob_r = sobol.analyze(problem, sm_r[:,i], calc_second_order=False, print_to_console=False)
            sob_m = sobol.analyze(problem, sm_m[:,i], calc_second_order=False, print_to_console=False)

            Si_s[i,:] = sob_s['S1']
            Si_i[i,:] = sob_i['S1']
            Si_r[i,:] = sob_r['S1']
            Si_m[i,:] = sob_m['S1']
                
            # Si_i['S1'], Si_r['S1'], Si_m['S1'])

        print("salvando arquivos Sobol")
        np.savetxt('sobol_'+caso+'_s.txt',Si_s)
        np.savetxt('sobol_'+caso+'_i.txt',Si_i)
        np.savetxt('sobol_'+caso+'_r.txt',Si_r)
        np.savetxt('sobol_'+caso+'_m.txt',Si_m)


    #
    # Cross validation 
    #
    if(opt_loocv):
        loo("S", sm_s, ns, npar, evals_s, samples, dist)
        loo("I", sm_i, ns, npar, evals_i, samples, dist)
        loo("R", sm_r, ns, npar, evals_r, samples, dist)
        loo("M", sm_m, ns, npar, evals_m, samples, dist)
    
    
    #printing the estimation values in the last day
    
    
    print('Estimativa para o último dia em '+caso)
    print('\tMedia - DP \t Media \t Media + DP')
    print_mean_std(evals_c, 'Conf.');
    print_mean_std(evals_i, 'Ativos');
    print_mean_std(evals_m, 'Óbitos');
    print_mean_std(evals_r, 'Recup.');
    
    print('\tMedia \t Intervalo de confiança')
    print_mean_ci(evals_c, 'Conf.');
    print_mean_ci(evals_i, 'Ativos');
    print_mean_ci(evals_m, 'Óbitos');
    print_mean_ci(evals_r, 'Recup.');
    
    print('done')
    '''
# Fim

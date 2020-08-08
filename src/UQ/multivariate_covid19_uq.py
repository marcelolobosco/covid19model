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
'''
#data_korea['first_day'] = len(data_korea['infected']) - 45

# global data for brazil
data_brazil = {}
data_brazil['first_day'] = 125 #first day with more then 100 cases
data_brazil['first_day_exp'] = 51 #first day with more then 100 cases
data_brazil['total_population'] = 209.3e6
data_brazil['deaths'] = np.loadtxt('data/death_Brazil.txt')
data_brazil['recovery'] = np.loadtxt('data/recovered_Brazil.txt')
data_brazil['infected'] = np.loadtxt('data/active_Brazil.txt')
data_brazil['confirmed'] = np.loadtxt('data/confirmed_Brazil.txt')
data_brazil['simple_care'] = None
data_brazil['intensive_care'] = None



# global data for Minas Gerais
data_mg = {}
data_mg['first_day'] = 46
data_mg['first_day_exp'] = 12 # diferença entre 26/02 (1o de BR) e 08/03 (1o MG)
data_mg['total_population'] = 21168791 # Fonte https://cidades.ibge.gov.br/brasil/mg/panorama
data_mg['deaths'] = np.loadtxt('data/death_MG.txt')
data_mg['recovery'] = np.loadtxt('data/recovered_MG.txt') 
data_mg['infected'] = np.loadtxt('data/active_MG.txt')
data_mg['confirmed'] = np.loadtxt('data/confirmed_MG.txt')
data_mg['simple_care'] = None
data_mg['intensive_care'] = None

# global data for Sao Joao Del Rei
data_sjdr = {}
data_sjdr['first_day'] = 85
data_sjdr['first_day_exp'] = 6 # diferença entre 08/03 (1o de MG) e 14/03 (1o JF)
data_sjdr['total_population'] = 90082 # Fonte https://cidades.ibge.gov.br/brasil/mg/juiz-de-fora/panorama
data_sjdr['deaths'] = np.loadtxt('data/death_SJ.txt')
data_sjdr['recovery'] = np.loadtxt('data/recovered_SJ.txt')
data_sjdr['infected'] = np.loadtxt('data/active_SJ.txt')
data_sjdr['confirmed'] = np.loadtxt('data/confirmed_SJ.txt')
data_sjdr['simple_care'] = None
data_sjdr['intensive_care'] = None

# global data for Juiz de Fora
data_jf = {}
data_jf['first_day'] = 53
data_jf['first_day_exp'] = 6 # diferença entre 08/03 (1o de MG) e 14/03 (1o JF)
data_jf['total_population'] = 568873 # Fonte https://cidades.ibge.gov.br/brasil/mg/juiz-de-fora/panorama
data_jf['deaths'] = np.loadtxt('data/death_JF.txt')
data_jf['recovery'] = np.loadtxt('data/recovered_JF.txt')
data_jf['infected'] = np.loadtxt('data/active_JF.txt')
data_jf['confirmed'] = np.loadtxt('data/confirmed_JF.txt')
data_jf['simple_care'] = None
data_jf['intensive_care'] = None

'''
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
dias_de_simulação = 35
t=range(dias_de_simulação)



def eval_model(V0, data):
    
    """
    x: model parameters
    x[0]:  a_max
    x[1]:  r
    x[2]:  ti
    x[3]:  delta
    x[4]:  e
    x[5]:  theta
    x[6]:  tau1
    x[7]:  tau2
    x[8]:  tau3
    x[9]:  m
    """

    """
    x: model parameters
    x[0]:  V0
    x[1]: 
    x[2]:  
    x[3]:  
    x[4]:  
    x[5]:  
    x[6]:  
    x[7]:  
    x[8]:  
    x[9]:  
    """

    # dados
    '''first_day = data['first_day']
    deaths = data['deaths']
    recovery = data['recovery']
    infected = data['infected']
    confirmed = data['confirmed']'''

    '''#from COVID-19_all
    #######   Viremia  log10 copias/ml ##########
	dadosViremiaLog10 = pd.read_csv('../../data/Viral_load.csv',',')
	dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 
	dadosCitocinaObitos = pd.read_csv('../../data/IL6_non-survivors_19.csv',',')
	dadosCitocinaSobreviventes = pd.read_csv('../../data/IL6_survivors_19.csv',',')
	dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM.csv',',')
	antibody_g = dadosAnticorposLog2_avg['IgG']
	antibody_m = dadosAnticorposLog2_avg['IgM']'''

    # tempo
    #ts = range(len(infected) - first_day + opt_forecast)
    dias_de_simulação = 35
    t=range(dias_de_simulação)

    # condicao inicial 
    #['amax','r','ti','tf', 'e','theta', 'tau_1','tau_2', 'tau_3', 'm']
    '''subnot = x[5]
    c_aux = infected[first_day]/subnot+recovery[first_day]/subnot+deaths[first_day]
    S0 = data['total_population'] - c_aux
    I0 = infected[first_day]/subnot
    R0 = recovery[first_day]/subnot
    M0 = deaths[first_day]
    P0 = [S0,I0,R0,M0]'''

	#CONDICOES INICIAIS FROM COVID-19_ALL
	#V0 = 2.18808824e+00
    #V0 = x[0]
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
    pi_v = 0.1955
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

    # resolve o modelo
    #Ps = odeint(SIR_PP_V4, P0, ts, args=sirargs)
    y,d=integrate.odeint(immune_response_v3, P0, t, args=(model_args), full_output=1, printmessg=True)

    #ordem de retorno 
    #V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, C = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13]

    # recupera resultados 
    '''S = Ps[:,0] # suscetiveis
    I = Ps[:,1] # infectados
    R = Ps[:,2] # recuperados
    M = Ps[:,3] # obitos, mortes'''
    
    
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

    return y[:,0]
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
    
    '''argumento1 = sys.argv[1]

    print("Primeiro argumento: " ,argumento1)'''

    '''print('Arquivo de Samples: ', arqpar)
    print('Caso: ', caso)
    print('Dias de previsão: ', opt_forecast)
    print('Nome do output: ', out_path)
    print('Extensao do output: ', ext)'''

    # options for simulation
    opt_delay = False
    opt_ft = False
    opt_leito = False
    opt_sobol = False
    opt_loocv = False
    opt_recovered = False
    opt_save_evals = True


    opt_flex = False
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

    V0 = cp.Uniform(2.406897064,1.969279416)
    distribution = cp.J(V0)
    nodes, weigths = cp.generate_quadrature(order=3, dist=distribution, rule="Gaussian")

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
    #npar = len(distribution)

    # grau da aproximacao polinomial
    #degpc = 2
    #ns = 10000#3*P(degpc,npar)
    #print("number of input parameters %d" % npar)
    #print("polynomial degree %d" % degpc)
    #print("min number of samples %d" % ns)

    # create samples and evaluate them
    #samples = distribution.sample(ns,"L")
    
    evals_virus = []
    
   # evals_IgM = []
   # evals_IgG = []
   # evals_Viremia = []
   # evals_Citocina = []
    '''evals_s = []
    evals_i = []
    evals_r = []
    evals_m = []
    evals_c = []
    evals_ih = []
    evals_ic = []'''


    
    #k = 0
    #lpts = range(ns)
    #samp = samples.T
    #print("evaluating samples: ")
    #for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    for node in nodes.T:
        virus = eval_model(node[01], data)
        evals_virus.append(virus)
     
        
        
        #s = samp[k]
        #k = k+1'''

    '''
    if (opt_flex):
        suc,inf,rec,mor,conf,ih,ic = eval_model_flex(s, data)
    else:
            suc,inf,rec,mor,conf,ih,ic = eval_model(s, data)'''

        #igm,igg,viremia,citocina = eval_model(s, data)
        #evals_IgM.append(igm)
        #evals_IgG.append(igg)
        #evals_Viremia.append(viremia)
        #evals_Citocina.append(citocina)    
    '''evals_s.append(suc) # s
        evals_i.append(inf) # i
        evals_r.append(rec) # r
        evals_m.append(mor) # m
        evals_c.append(conf)# c
        evals_ih.append(ih) # ih
        evals_ic.append(ic) # ih'''

    '''evals_s = np.array(evals_s)
    evals_i = np.array(evals_i)
    evals_r = np.array(evals_r)
    evals_m = np.array(evals_m)
    evals_c = np.array(evals_c)
    evals_ih = np.array(evals_ih)
    evals_ic = np.array(evals_ic)'''

    
    evals_virus = np.array(evals_virus)
    #evals_IgM = np.array(evals_IgM)
    #evals_IgG = np.array(evals_IgG)
    #evals_Viremia = np.array(evals_Viremia)
    #evals_Citocina = np.array(evals_Citocina)


   
    
    #
    # Plot data
    #
    plt.style.use('estilo/PlotStyle.mplstyle')
    plt.close('all')

    # dados para plotar
    '''first_day = data['first_day']
    deaths = data['deaths']
    recovery = data['recovery']
    infected = data['infected']
    confirmed = data['confirmed']'''


    '''if(opt_leito):
        simple_care = data['simple_care']
        intensive_care = data['intensive_care']'''

    
    out_path = './output/'
    ext = '.png'
    itvl = 6
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
        #np.savetxt(out_path+caso+'_evals_IgM.txt',evals_IgM)
        #np.savetxt(out_path+caso+'_evals_IgG.txt',evals_IgG)
        #np.savetxt(out_path+caso+'_evals_Viremia.txt',evals_Viremia)
        #np.savetxt(out_path+caso+'_evals_Citocina.txt',evals_Citocina)

    # infectados e recuperados
    '''fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=itvl))'''
    
    ###mean_i = plot_mean_std(ax, ts, evals_i, 'r', 'modelo: I $\pm$ std')
    #mean_i = plot_mean_std(ax, days, evals_i, 'r', 'Model: Active Infected $\pm$ std')
    #mean_i = plot_confidence_interval(ax, days, evals_i, 'r', 'Modelo: Casos Ativos 95IC')
    #np.savetxt(out_path+caso+'_evals_i.txt',evals_i) 

    
        
    
    '''
       mean_IgM = plot_confidence_interval(plt, t, evals_IgM, 'r', 'Modelo: testando IgM')
       plt.grid(True)
       plt.xticks(rotation=45)
       plt.title("Solution IgM")
       plt.legend(loc='best')
       plt.tight_layout()
       plt.savefig(out_path+'_output_IgM'+ext)
       plt.show()'''
    
    
    
    plt.figure('Viremia')
    #dadosViremiaLog10.plot.scatter(x='Day',y='Viral_load',color='m',label='Dados experimentais')
    plt.plot(t, dadosViremiaLog10['Viral_load'], 'o', label='data', linewidth=4)
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    mean_virus =plot_confidence_interval_poly(plt, t, evals_virus, distribution, "Virus", "red")
    plt.grid(True)
    plt.title("Solution Virus")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path+'_output_Virus'+ext)
    plt.show()
    
    
    #mean_IgG = plot_confidence_interval(ax, days, evals_i, 'r', 'Modelo: Casos Ativos 95IC')
    #mean_Viremia = plot_confidence_interval(ax, days, evals_i, 'r', 'Modelo: Casos Ativos 95IC')
    #mean_Citocina = plot_confidence_interval(ax, days, evals_i, 'r', 'Modelo: Casos Ativos 95IC')


    '''if (opt_recovered):
        mean_r = plot_mean_std(ax, ts, evals_r, 'g', 'Modelo: Recuperado 95IC')
        plt.plot(days2, recovery[first_day:], 'o', color="darkgreen", label='Dados: Recuperado')'''
    
    ###plt.plot(ti, infected[first_day:], 'o', color="darkred", label='Dados: Infectados Ativos')
    #plt.plot(days2, infected[first_day:], 'o', color="darkred", label='Data: Active Infected')
    '''if opt_last == 0:
        plt.plot(days2, infected[first_day:], 'o', color="darkred", label='Dados: Casos Ativos')
    else:	
        plt.plot(days2[:-opt_last], infected[first_day:-opt_last], 'o', color="darkred", label='Dados: Casos Ativos')
        plt.plot(days2[-opt_last:], infected[-opt_last:], '*', color="darkred")


    ax.set_xlabel('dia')
    ax.set_ylabel('população')
    #plt.gcf().autofmt_xdate()
    ax.legend(loc='upper left', prop={'size':13})
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path+caso+'_output_I'+ext)'''
    

    # plot recovery
    '''fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=itvl))
    ###plot_mean_std(ax, ts, evals_m, 'grey', 'modelo: D $\pm$ std')
    
    #plot_mean_std(ax, days, evals_m, 'grey', 'Model: Deaths $\pm$ std')
    plot_confidence_interval(ax, days, evals_r, 'green', 'Modelo: Recuperados 95IC')
    if opt_last ==0:
        plt.plot(days2, recovery[first_day:], 'o', color="darkgreen", label='Dados: Recuperados')
    else:
        plt.plot(days2[:-opt_last], recovery[first_day:-opt_last], 'o', color="darkgreen", label='Dados: Recuperados')
        plt.plot(days2[-opt_last:], recovery[-opt_last:], '*', color="darkgreen")

    ax.set_xlabel('dia')
    ax.set_ylabel('população')
    ax.legend(loc='upper left', prop={'size':13})
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path+caso+'_output_R'+ext)'''


    # plot mortes
    '''fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=itvl))'''
    ###plot_mean_std(ax, ts, evals_m, 'grey', 'modelo: D $\pm$ std')
    
    #plot_mean_std(ax, days, evals_m, 'grey', 'Model: Deaths $\pm$ std')
    '''plot_confidence_interval(ax, days, evals_m, 'grey', 'Modelo: Óbtos 95IC')
    if opt_last ==0:
        plt.plot(days2, deaths[first_day:], 'o', color="black", label='Dados: Óbtos')
    else:
        plt.plot(days2[:-opt_last], deaths[first_day:-opt_last], 'o', color="black", label='Dados: Óbtos')
        plt.plot(days2[-opt_last:], deaths[-opt_last:], '*', color="black")

    ax.set_xlabel('dia')
    ax.set_ylabel('população')
    ax.legend(loc='upper left', prop={'size':13})
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path+caso+'_output_M'+ext)'''
    
    # plot confirmados
    '''fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=itvl))
    ###plot_mean_std(ax, ts, evals_c, 'blue', 'modelo: C $\pm$ std')
    #plot_mean_std(ax, days, evals_c, 'blue', 'Model: Infected Confirmed $\pm$ std')
    plot_confidence_interval(ax, days, evals_c, 'blue', 'Modelo: Casos Totais 95IC')
    #np.savetxt(out_path+caso+'_evals_c.txt',evals_c)
    ###plt.plot(td, confirmed[first_day:], 'o', color="darkblue", label='Dados:  Infectados Confirmados')
    if opt_last == 0:
        plt.plot(days2, confirmed[first_day:], 'o', color="darkblue", label='Dados: Casos Totais')
    else:
        plt.plot(days2[:-opt_last], confirmed[first_day:-opt_last], 'o', color="darkblue", label='Dados: Casos Totais')
        plt.plot(days2[-opt_last:], confirmed[-opt_last:], '*', color="darkblue")
    ax.set_xlabel('dia')
    ax.set_ylabel('população')
    ax.legend(loc='upper left', prop={'size':13})
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path+caso+'_output_C'+ext)'''

    # plot leitos
    '''if(opt_leito):
        #tih = range(len(simple_care[first_day:]))
        #tic = range(len(intensive_care[first_day:]))
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
        #mean_ih = plot_mean_std(ax, ts, evals_ih, 'purple', 'model: IH $\pm$ std')
        mean_ic = plot_mean_std(ax, days, evals_ic, 'brown',  'Modelo: Ocupação de Leitos UTI')
        np.savetxt('evals_ic_atual.txt',evals_ic) 
        limit_cti = np.ones(len(ts))
        limit_cti *=37
        plt.plot(days,limit_cti, '--', color='purple', label='Leitos UTI total')
        #plt.plot(tih, simple_care[first_day:], 'o', color="darkmagenta", label='dados: IH')
        #plt.plot(tic, intensive_care[first_day:],'o', color="saddlebrown",label='dados: IC')
        ax.set_xlabel('data')
        ax.set_ylabel('Leitos UTI')
        plt.gcf().autofmt_xdate()
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_path+caso+'_output_ih_ic'+ext)'''

    '''
    # Sensitivity analysis
    #
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

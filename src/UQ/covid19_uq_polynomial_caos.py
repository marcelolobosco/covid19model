#!/usr/bin/python

import sys
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd


# barra de progresso =)
# sudo pip install tqdm
from tqdm import tqdm
#from tqdm import tqdm_gui

# Bibliotecas proprias

from uqtools import *
from NovoModelo import *


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

def plot_confidence_interval_poly(ax,time,poly,dist,textlabel,linecolor):
	mean = cp.E(poly,dist) #agora eu deixo de colocar mean e sdev no codigo la em baixo?
	sdev = cp.Std(poly,dist)
	perc_min = 5.0
	perc_max = 95.0
	x = dist.sample(1000, "H") 
	values = [cp.poly.caller.call(poly,[k]) for k in x.T]
	percentile = np.percentile(values, [perc_min, perc_max], axis=0)
	ax.plot(time, mean, lw=2, color=linecolor, label =textlabel)
	ax.fill_between(time, percentile[0,:], percentile[1,:], alpha= 0.5, color=linecolor)
	return mean


#*
def eval_model(k_v3, data):
    
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
    x[1]: pi_v
    x[2]: pi_c_apm 
    x[3]: pi_c_tke
    x[4]:  
    x[5]:  
    x[6]:  
    x[7]:  
    x[8]:  
    x[9]:  
    """

	#CONDICOES INICIAIS FROM COVID-19_ALL
    V0 = 2.18808824e+00
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
#*

    pi_c_apm = 7.43773673e-01
    pi_c_i = 1.97895565e-02
    pi_c_tke = 0.04730172
    delta_c = 8.26307952e+00
    k_apm = 5.36139617e-01
    #k_v3 = 3.08059068e-03
    k_tk = 2.10152618e-01
    
    model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1, k_ap2,
    delta_Apm, alpha_Tn, pi_T, k_te1, delta_te, alpha_B, pi_B1, pi_B2, 
    beta_S, beta_L, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_A_G, delta_A_M, c11, c12, c13, c14, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, k_apm, k_v3, k_tk)

	#--------------------------------------------------------------------------
    
    # resolve o modelo
    #Ps = odeint(SIR_PP_V4, P0, ts, args=sirargs)
    y,d=integrate.odeint(immune_response_v3, P0, t, args=(model_args), full_output=1, printmessg=True)

    #ordem de retorno 
    #V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13]

    # recupera resultados 
    '''S = Ps[:,0] # suscetiveis
    I = Ps[:,1] # infectados
    R = Ps[:,2] # recuperados
    M = Ps[:,3] # obitos, mortes'''
    
    
    
    return y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]
    #return IgM,IgG,Viremia,Citocina
    #return S,In,Rn,M,C, IH,IC

if __name__ == "__main__":


    # ajeita os dados para o caso
    data = None
    

#------------------PARAMETROS PARA UQ ----------------------------------------  
    #V0 = cp.Uniform(2.406897064,1.969279416) (10percent)
    #V0 = cp.Uniform(1.09404412,3.5) (50percentile)
    #pi_v = cp.Uniform(0.166175,0.224825) #(15percent)
    #pi_v = cp.Uniform(0.1564,0.2346) #(20percent)
    #pi_v = cp.Uniform(0.146625,0.244375) #(25percent)
    #pi_v = cp.Uniform(0.09775,0.29325) #(50percent)
    #pi_c_apm = cp.Uniform(0.6693963057,0.8181510403) #(10percent)
    #pi_c_apm = cp.Uniform(0.55783025475,0.92971709125) #(25percent)
    #pi_c_i = cp.Uniform(0.01781060085,0.02176851215) #(10percent)
    #pi_c_i = cp.Uniform(0.01484216738,0.02473694562) #(25percent)
    #pi_c_tke = cp.Uniform(0.04730170,0.052031892) #(10percent) #*
    #pi_c_tke = cp.Uniform(0.03547629,0.05912715) #(25percent)
    #k_apm = cp.Uniform(0.4825256553,0.5897535787) #(10percent)
    #k_apm = cp.Uniform(0.40210471275,0.67017452125) #(25percent)
    #k_tk = cp.Uniform(0.1891373562,0.2311678798) #(10percent)
    #k_tk = cp.Uniform(0.1576144635,0.2626907725) #(25percent)
    #k_v1 = cp.Uniform(0.00315,0.00385) #(10percent) 0.004375
    #k_v1 = cp.Uniform(0.002625,0.004375) #(25percent) 
    #k_v2 = cp.Uniform(0.0000855,0.0001045) #(10percent)
    #k_v2 = cp.Uniform(0.00007125,0.00011875) #(25percent)
    #k_v3 = cp.Uniform(0.00277253162,0.00338864974) #(10percent)
    k_v3 = cp.Uniform(0.00231044301,0.00385073835) #(25percent)
    
    
#-----------------CHAOSPY-----------------------------------------------------   
    distribution = cp.J(k_v3) #*
    nodes, weigths = cp.generate_quadrature(order=3, dist=distribution, rule="Gaussian")
    #npar = len(dist)
    # grau da aproximacao polinomial
    #degpc = 2
    #ns = 10000#3*P(degpc,npar)
    #print("number of input parameters %d" % npar)
    #print("polynomial degree %d" % degpc)
    #print("min number of samples %d" % ns)

    # create samples and evaluate them
    #samples = dist.sample(ns,"L")
#--------------------EVALS----------------------------------------------------  
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
    
    #retorno da chamada de imune_response_v3
    #V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C
    
   # evals_IgM = []
   # evals_IgG = []
   # evals_Viremia = []
   # evals_Citocina = []
   

#---------------------METODO MONTE CARLO--------------------------------------

    ''' 
    k = 0
    lpts = range(ns)
    samp = samples.T
    print("evaluating samples: ")
    for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

    #for s in samples.T:
        s = samp[k]
        k = k+1
        #virus = eval_model(s,data)
        #evals_virus.append(virus)
        citocina = eval_model(s,data)
        evals_citocina.append(citocina)'''

    #for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    
#--------------METODO ANTERIOR------------------------------------------------    
    for node in nodes.T:
        virus,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = eval_model(node[0],data)
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
        #citocina = eval_model(node[0])
        #evals_citocina.append(citocina)
#-----------------------------------------------------------------------------    
    
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
    
    #evals_citocina = np.array(evals_citocina)
    #evals_IgM = np.array(evals_IgM)
    #evals_IgG = np.array(evals_IgG)
    #evals_Viremia = np.array(evals_Viremia)
    #evals_Citocina = np.array(evals_Citocina)
#-----------------------------------------------------------------------------

    #para salvar
    ext = '.pdf'
    output_path = './output_teste/'
    caso = 'teste25%k_v3' #*

#--------------POLYNOMIALS E MODEL APROX--------------------------------------
    polynomials = cp.orth_ttr(order=3, dist= distribution)
    model_approx_Virus= cp.fit_quadrature(polynomials, nodes, weigths, evals_virus)
    
    plt.plot(t, dadosViremiaLog10['Viral_load'], 'o', label='data', linewidth=4)
    #plot_confidence_interval(plt, t, evals_virus, 'r', 'Modelo: Virus')
    plot_confidence_interval_poly(plt, t, model_approx_Virus, distribution, "Virus", "red")
    plt.grid(True)
    plt.title("Solution Virus")
    plt.legend(loc='best')
    plt.savefig(output_path+caso+'Virus'+ext)
    plt.show()
    
    
    #polynomials = cp.orth_ttr(order=3, dist= distribution)
    #model_approx_Citocina= cp.fit_quadrature(polynomials, nodes, weigths, evals_citocina)
    '''plt.plot(dadosCitocinaSobreviventes['Day']+5, dadosCitocinaSobreviventes['IL6(pg/mL)'], 'o', label='data', linewidth=4)
    plt.xlim(0.0,dias_de_simulação)
    plot_confidence_interval(plt, t, evals_citocina, 'r', 'Modelo: Citocina')
    #plot_confidence_interval_poly(plt, t, model_approx_Citocina, distribution, "Citocina", "orange")
    plt.grid(True)
    plt.title("Solution Citocina")
    plt.legend(loc='best')
    plt.savefig(out_path+caso+'Citocina_pi_c_tke'+ext,bbox_inches='tight',dpi = 300)
    plt.show()'''
    #-------------------------ANTICORPOS--------------------------------------
    model_approx_AM= cp.fit_quadrature(polynomials, nodes, weigths, evals_AM)
    model_approx_AG= cp.fit_quadrature(polynomials, nodes, weigths, evals_AG)
    plt.figure('Anticorpos')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,np.log2(y[:,11]+1),label='IgM Modelo',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_AM, distribution, "A_M Modelo", "blue")
    plt.plot(t, antibody_m, 'o', label='IgM data', linewidth=4)
    #plt.plot(t,np.log2(y[:,12]+1),label='IgG Modelo',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_AG, distribution, "A_G Modelo", "green")
    plt.plot(t, antibody_g, 'o', label='IgG data', linewidth=4)
    plt.xlabel('Tempo pós-infecção (dias)')
    plt.ylabel('Anticorpos - sem log 2 S/CO')
    plt.legend()
    plt.savefig(output_path+'Anticorpos.pdf',bbox_inches='tight',dpi = 300)
    
    
    #-------------------------AP--------------------------------------
    model_approx_Ap= cp.fit_quadrature(polynomials, nodes, weigths, evals_Ap)
    plt.figure('Ap')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,1],label='Ap',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Ap, distribution, "Ap", "green")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Apresentadores')
    plt.legend()
    plt.savefig(output_path+'Ap.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------APM---------------------------------------------
    model_approx_Apm= cp.fit_quadrature(polynomials, nodes, weigths, evals_Apm)
    plt.figure('Apm')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,2],label='Apm',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Apm, distribution, "Apm", "blue")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Apresentadores Maduras')
    plt.legend()
    plt.savefig(output_path+'Apm.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------Thn---------------------------------------------
    model_approx_Thn= cp.fit_quadrature(polynomials, nodes, weigths, evals_Thn)
    plt.figure('Thn')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,3],label='Thn',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Thn, distribution, "Thn", "orange")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Th Naive')
    plt.legend()
    plt.savefig(output_path+'Thn.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------The---------------------------------------------
    model_approx_The= cp.fit_quadrature(polynomials, nodes, weigths, evals_The)
    plt.figure('The')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,4],label='The',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_The, distribution, "The", "black")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Th Efetora')
    plt.legend()
    plt.savefig(output_path+'The.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------Tkn---------------------------------------------    
    model_approx_Tkn= cp.fit_quadrature(polynomials, nodes, weigths, evals_Tkn)
    plt.figure('Tkn')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,5],label='Tkn',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Tkn, distribution, "Tkn", "purple")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Tk Naive')
    plt.legend()
    plt.savefig(output_path+'Tkn.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------Tke---------------------------------------------
    model_approx_Tke= cp.fit_quadrature(polynomials, nodes, weigths, evals_Tke)
    plt.figure('Tke')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,6],label='Tke',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Tke, distribution, "Tke", "blue")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Tk Efetora')
    plt.legend()
    plt.savefig(output_path+'Tke.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------B---------------------------------------------
    model_approx_B= cp.fit_quadrature(polynomials, nodes, weigths, evals_B)
    plt.figure('B')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,7],label='B',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_B, distribution, "B", "red")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('B')
    plt.legend()
    plt.savefig(output_path+'B.pdf',bbox_inches='tight',dpi = 300)
    
    #np.savetxt(output_path+'b.txt',y[:,7])
    
    #-------------------------Ps---------------------------------------------
    model_approx_Ps= cp.fit_quadrature(polynomials, nodes, weigths, evals_Ps)
    plt.figure('Ps')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,8],label='Ps',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Ps, distribution, "Ps", "orange")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Ps')
    plt.legend()
    plt.savefig(output_path+'Ps.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------Pl---------------------------------------------
    model_approx_Pl= cp.fit_quadrature(polynomials, nodes, weigths, evals_Pl)
    plt.figure('Pl')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,9],label='Pl',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Pl, distribution, "Pl", "green")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Pl')
    plt.legend()
    plt.savefig(output_path+'Pl.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------Bm---------------------------------------------
    model_approx_Bm= cp.fit_quadrature(polynomials, nodes, weigths, evals_Bm)
    plt.figure('Bm')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,10],label='Bm',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_Bm, distribution, "Bm", "purple")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Bm')
    plt.legend()
    plt.savefig(output_path+'Bm.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------AM---------------------------------------------
    #model_approx_AM= cp.fit_quadrature(polynomials, nodes, weigths, evals_AM)
    plt.figure('A_M')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
   # plt.plot(t,y[:,11],label='Ai',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_AM, distribution, "A_M", "red")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('A_M')
    plt.legend()
    plt.savefig(output_path+'A_M.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------AG---------------------------------------------
    #model_approx_AG= cp.fit_quadrature(polynomials, nodes, weigths, evals_AG)
    plt.figure('A_G')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,12],label='Ai',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_AG, distribution, "A_G", "black")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('A_G')
    plt.legend()
    plt.savefig(output_path+'A_G.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------I---------------------------------------------
    model_approx_I= cp.fit_quadrature(polynomials, nodes, weigths, evals_I)
    plt.figure('Ai')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,13],label='Ai',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_I, distribution, "Ai", "blue")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('APC infectada')
    plt.legend()
    plt.savefig(output_path+'Ai.pdf',bbox_inches='tight',dpi = 300)
    
    #-------------------------C SOBREVIVENTES---------------------------------------------
    model_approx_C= cp.fit_quadrature(polynomials, nodes, weigths, evals_C)
    plt.figure('C')
    #dadosCitocinaSobreviventes.plot.scatter(x='Day',y='IL6(pg/mL)',color='g',label='Dados experimentais(Sobreviventes)')
    plt.plot(dadosCitocinaSobreviventes['Day']+5, dadosCitocinaSobreviventes['IL6(pg/mL)'], 'o', label='data', linewidth=4)
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    #plt.plot(t,y[:,14],label='C',linewidth=1.5, linestyle="-")
    plot_confidence_interval_poly(plt, t, model_approx_C, distribution, "C", "orange")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('C')
    plt.legend()
    plt.savefig(output_path+'C.pdf',bbox_inches='tight',dpi = 300)
    '''
    #-------------------------C OBITOS---------------------------------------------
    plt.figure('C')
    dadosCitocinaObitos.plot.scatter(x='Day',y='IL6(pg/mL)',color='y',label='Dados experimentais(Obito)')
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.plot(t,y[:,14],label='C',linewidth=1.5, linestyle="-")
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('C')
    plt.legend()
    plt.savefig(output_path+'C2.pdf',bbox_inches='tight',dpi = 300)'''
    
    print("vim ate aqui")



   
# Fim

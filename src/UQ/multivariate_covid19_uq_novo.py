#Importação de Bibliotecas
import sys
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm_gui
from uqtools import *
from NovoModelo import *

#################################LEITURA DE DADOS#############################
"""
    Reading the data from (colocar fonte aqui)

    :param dadosViremiaLog10: dataframe
    :param dadosAnticorposLog2: dataframe
    :param dadosCitocinaObitos: dataframe
    :param dadosCitocinaSobreviventes: dataframe
    :param dadosAnticorposLog2_avg: dataframe
    :param antibody_g: dataframe
    :param antibody_m: dataframe
    :param dadosAnticorposLog2_avg: dataframe

    """
#from COVID-19_all
#######   Viremia  log10 copias/ml ##########

dadosViremiaLog10 = pd.read_csv('../../data/Viral_load.csv',',')
dadosAnticorposLog2 = pd.read_csv('../../data/IgG_IgM_21_1b.csv',',') 
dadosCitocinaObitos = pd.read_csv('../../data/IL6_non-survivors_19.csv',',')
dadosCitocinaSobreviventes = pd.read_csv('../../data/IL6_survivors_19.csv',',')
dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM.csv',',')
antibody_g = dadosAnticorposLog2_avg['IgG']
antibody_m = dadosAnticorposLog2_avg['IgM']
dadosIL6= pd.read_csv('../../data/IL6_storm_ajuste.csv',',')
dadosAnticorposLog2_avg = pd.read_csv('../../data/IgG_IgM.csv',',')
antibody_g = dadosAnticorposLog2_avg['IgG']
antibody_m = dadosAnticorposLog2_avg['IgM']

#Runtime
dias_de_simulação = 35
t=range(dias_de_simulação)

##############################################################################

def plot_confidence_interval_log(ax, time, evals, linecolor, textlabel):
    """
    Compute and plot statistical moments (mean,std) within a 95% confidence interval with log data

    :return mean:
    """
    mean = np.mean(evals, axis=0)
    sdev = np.std(evals, axis=0)
    perc_min = 2.5
    perc_max = 97.5
    percentile = np.percentile(evals, [perc_min, perc_max], axis=0)
    ax.plot(time, np.log2(mean+1), lw=2, color=linecolor, label=textlabel)
    ax.fill_between(time, np.log2(percentile[0,:]+1) , np.log2(percentile[1,:]+1), alpha=0.5, color=linecolor)
    return mean


def eval_model(x):

    """
    Initializes the initial conditions of the model. Solves initial value problem, using variable values at t = 0, 
    to find its time evolution to t> 0. The odeint function takes care of this directly, s
    olving PVIs for ODE systems.
    
    :return V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, C = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]:
    """

    """
    Initial conditions from COVID-19_ALL

    :param V0: float
        initial virus condition
    :param Ap0: float
        immature APC's initial condition
    :param Apm0: float
        mature APC's initial condition
    :param Ai0: float
        Ai 
    :param C0: float
        C0 cytokines initial condition
    :param Thn0: float
        TDC4 + naive cells initial condition
    :param The0: float
        TDC4 + effector cells initial condition
    :param Tkn0: float
        TDC8 + naive cells initial condition
    :param Tke0: float
        TDC8 + effector cells initial condition
    :param B0: float
        memory B cells initial condition
    :param Ps0: float
        short-lived plasma cells initial condition
    :param Pl0: float
        long-lived plasma cells initial condition
    :param A0_M : float
        antibodies IgM initial condition
    :param A0_G : float
        antibodies IgG initial condition

    """
    V0 = 2.18808824e+00
    Ap0 = 1.0e6
    Apm0 = 0.0
    Ai0= 0.0
    C0= 0.0
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

    #Vector that saves all initial conditions
    P0 = [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0_M,A0_G,Ai0,C0]


	###################### MODEL PARAMETERS ######################################
    """
    Those with 'fit' are the parameters evaluated in the UQ.
    z is the vector that stores non-survivors
    x is the vector that stores survivors

    """    
    pi_v = x[0] #fit
    c_v1 = 2.63
    c_v2 = 0.60
    k_v1 = 3.5e-3
    k_v2 = 9.5e-5
    alpha_Ap = 1.87E-06*0.4 #remove
    beta_Ap = x[1] #fit
    c_ap1 = x[2] #fit
    c_ap2 = x[3] #fit

    delta_Apm = x[4] #fit
    alpha_Tn = 2.17E-04 #=>remove
    beta_tk = x[5] #fit
    pi_tk = 1.0E-08
    delta_tk = 0.0003
    alpha_B =3.578236584371140339e+02
    pi_B1 = 8.979145365768647095e-05
    pi_B2 = 1.27E-8

    beta_ps = x[6] #fit
    beta_pl = x[7] #fit
    beta_Bm = 1.0E-06
    delta_S = x[8] #fit
    delta_L = x[9] #fit
    gamma_M = (1.95E-06)*500.0
    k_bm1 = 1.0e-5    
    k_bm2 = 2500.0 
    pi_AS = x[10] #fit
    pi_AL = x[11] #fit
    delta_ag = 0.07
    delta_am = 0.07
    alpha_th = x[12] #fit
    beta_th = x[13] #fit
    pi_th = 1.0E-08
    delta_th= 0.3
    Ap0 = 1.0e6 
    Thn0 = 1.0e6 
    Tkn0 = 5.0e5 
    B0 = 2.5E5


    pi_c_apm = 7.43773673e-01
    pi_c_i = 1.97895565e-02
    pi_c_tke = x[14] #fit
    delta_c = x[15] #fit
    beta_apm = x[16] #fit
    k_v3 = 3.08059068e-03 
    beta_tke = x[17] #fit
    
 
    #UQ assessed non-survivors
    '''
    beta_apm = 1.48569967 
    beta_tk = 0.10171796
    
    #vivo 
    beta_apm = x[16] #fit
    beta_tk = x[17] #fit
    
    '''

    #Vector filled with model parameters
    model_args = (pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, c_ap1, c_ap2,
    delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk, alpha_B, pi_B1, pi_B2, 
    beta_ps, beta_pl, beta_Bm,delta_S, delta_L, gamma_M, k_bm1, k_bm2, pi_AS,
    pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0,  
    pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke)
    

    #Solves the model using odeint function
    y,d=integrate.odeint(immune_response_v3, P0, t, args=(model_args), full_output=1, printmessg=True)

    #Return order
    #V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, C = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]
    

    return y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7], y[:,8], y[:,9], y[:,10], y[:,11], y[:,12], y[:,13], y[:,14]

if __name__ == "__main__":

    #Output Settings
    output_path = './output_monte_carlo/'
    ext = '.png'
    itvl = 6
    caso = 'caso.'

    
    #To execute the survivors
    arqpar = "./cluster/execution_de_survivor.txt"

    #To execute the non-survivors
    #arqpar2 = "./cluster/execution_de_non_survivor.txt"
    
    print('Surviving Samples file: ', arqpar)
    print('Output name:', output_path)
    print('Output extension:', ext)

    #Fixes the data for the case
    data = None

    
    label_param = ("pi_v", "c_v1", "c_v2", "k_v1", "k_v2", 
        "beta_Ap", "c_ap1", "c_ap2", "delta_Apm", "beta_tk",
        "pi_tk", "delta_tk", "alpha_B","pi_B1", "pi_B2", 
        "beta_ps", "beta_pl", "beta_Bm","delta_S", "delta_L", "gamma_M", "k_bm1", "k_bm2", "pi_AS",
        "pi_AL", "delta_ag", "delta_am", "alpha_th", "beta_th", "pi_th", "delta_th", "Ap0", "Thn0", "Tkn0", "B0",  
        "pi_c_apm", "pi_c_i","pi_c_tke","delta_c", "beta_apm", "k_v3", "beta_tke")


    label_covid = ['pi_v', 'beta_Ap', 'c_ap1', 'c_ap2', 'delta_Apm', 'beta_tk', 'beta_ps', 'beta_pl', 'delta_S',
    'delta_L', 'pi_AS', 'pi_AL', 'alpha_th', 'beta_th', 'pi_c_tke', 'delta_c', 'beta_apm', 'beta_tke']


    #--------------------For survivors---------------------#
    m = np.loadtxt(arqpar, comments='#')
    M= m[:,:-1]
    
    #--------------------For non-survivors---------------------#
    #m2 = np.loadtxt(arqpar2, comments='#')
    #M2= m2[:,:-1]
    #Cov_matrix_2 = np.cov(M2.T)
    #mean_vector_2 = np.mean(M2, axis=0)
    #std_vector_2 = np.std(M2, axis=0)
    #print("mean = ", mean_vector)
    #print("std = ", std_vector)
    #dist2 = cp.MvNormal(mean_vector_2, Cov_matrix_2)
    
    '''
    It sets up a covariancy matrix, then extracts the mean and standard deviation.
    '''

    #-------------------only 1 distribution---------------------#

    Cov_matrix = np.cov(M.T)
    mean_vector = np.mean(M, axis=0)
    std_vector = np.std(M, axis=0)
    print("mean = ", mean_vector)
    print("std = ", std_vector)

    dist = cp.MvNormal(mean_vector, Cov_matrix)
    
    #-------------------only 1 distribution---------------#
    #-------------------***********----------------------#
   
    
    '''
    And then, the degree of polynomial approximation and the number of samples
    '''
    
    #Number of parameters
    npar = len(dist)
   
    #Degree of polynomial approximation
    degpc = 2
    ns = 10000#3*P(degpc,npar)
    print("number of input parameters two txts  %d" % npar)
   
    print("polynomial degree %d" % degpc)
    print("min number of samples %d" % ns)

    #Create samples and evaluate them
    #samples = distribution.sample(ns,"L")
    #-----------------------------------------------------------------------------------------------------------------------#

    #----------------------------------------------------EVALS--------------------------------------------------------------#
    '''
    Creation of arrays that store the evals of each population
    '''
    
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

    '''
    The samples are evaluated within the loop and added to their respective arrays:
    '''

    for i in tqdm(lpts,bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
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
        
    #-----------------------------------------------------------------------------------------------------------------------# 
    #Plot data
    plt.style.use('estilo/PlotStyle.mplstyle')
    plt.close('all')


        
    #Saving the evals
    #if opt_save_evals:
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
    #To copy if necessary
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
    
    
    #-------------------------VIRUS--------------------------------------
    
  
    plt.figure('Viremia')
    plt.plot(t, dadosViremiaLog10['Viral_load'], 'o', label='data', linewidth=4)
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Post-vaccination time (days)')
    plt.ylabel('Virus')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.title("Solution Virus")
    mean_virus= plot_confidence_interval(plt, t, evals_virus, 'red', 'Modelo: Virus')
    #plot_mean_std(plt, t, evals_virus, 'blue', 'Model: Virus std')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path+caso+'_output_Virus'+ext)
    #plt.show()
    

    #-------------------------ANTIBODIES--------------------------------------
    
    plt.figure('Antibodies')
    plt.xlim(0.0,dias_de_simulação)
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.title("Solution Antibodies.")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.plot(t, np.log2(antibody_m+1), 'o', label='IgM data', linewidth=4)
    plt.plot(t, np.log2(antibody_g+1), 'o', label='IgG data', linewidth=4)
    plt.xlabel('Post-vaccination time (days)')
    plt.ylabel('Antibodies - log 2 S/CO')
    mean_AM = plot_confidence_interval_log(plt, t, evals_AM, 'orange', 'Model: A_M')
    mean_AG = plot_confidence_interval_log(plt, t, evals_AG, 'red', 'Model: A_G')
    plt.savefig(output_path+'Antibodies'+ext,bbox_inches='tight',dpi = 300)
    
    
    #-------------------------AP--------------------------------------
    
    plt.figure('Ap')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Apresentadores')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Ap = plot_confidence_interval(plt, t, evals_Ap, 'green', 'Modelo: Ap')
    plt.savefig(output_path+'Ap'+ext,bbox_inches='tight',dpi = 300)

    
    #-------------------------APM---------------------------------------------
   
    plt.figure('Apm')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Apresentadores Maduras')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Apm = plot_confidence_interval(plt, t, evals_Apm, 'blue', 'Modelo: Apm')
    plt.savefig(output_path+'Apm'+ext,bbox_inches='tight',dpi = 300)
    
    
    #-------------------------Thn---------------------------------------------
    
    plt.figure('Thn')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Th Naive')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Thn = plot_confidence_interval(plt, t, evals_Thn, 'black', 'Modelo: Thn')
    plt.savefig(output_path+'Thn'+ext,bbox_inches='tight',dpi = 300)
    
    #-------------------------The---------------------------------------------
    
    plt.figure('The')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Th Efetora')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_The = plot_confidence_interval(plt, t, evals_The, 'purple', 'Modelo: The')
    plt.savefig(output_path+'The'+ext,bbox_inches='tight',dpi = 300)
    
    
    #-------------------------Tkn---------------------------------------------    
    
    plt.figure('Tkn')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Tk Naive')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Tkn = plot_confidence_interval(plt, t, evals_Tkn, 'orange', 'Modelo: Tkn')
    plt.savefig(output_path+'Tkn'+ext,bbox_inches='tight',dpi = 300)
    
    
    #-------------------------Tke---------------------------------------------
   
    plt.figure('Tke')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Tk Efetora')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Tke = plot_confidence_interval(plt, t, evals_Tke, 'red', 'Modelo: Tke')
    plt.savefig(output_path+'Tke'+ext,bbox_inches='tight',dpi = 300)
    
    #-------------------------B---------------------------------------------
    
    plt.figure('B')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('B')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_B = plot_confidence_interval(plt, t, evals_B, 'green', 'Modelo: B')
    plt.savefig(output_path+'B'+ext,bbox_inches='tight',dpi = 300)
    
    #np.savetxt(output_path+'b.txt',y[:,7])
    
    #-------------------------Ps---------------------------------------------
    
    plt.figure('Ps')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Ps')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Ps = plot_confidence_interval(plt, t, evals_Ps, 'blue', 'Modelo: Ps')
    plt.savefig(output_path+'Ps'+ext,bbox_inches='tight',dpi = 300)
    
    #-------------------------Pl---------------------------------------------
    
    plt.figure('Pl')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Pl')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Pl = plot_confidence_interval(plt, t, evals_Pl, 'black', 'Modelo: Pl')
    plt.savefig(output_path+'Pl'+ext,bbox_inches='tight',dpi = 300)
    
    #-------------------------Bm---------------------------------------------
    
    plt.figure('Bm')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('Bm')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_Bm = plot_confidence_interval(plt, t, evals_Bm, 'purple', 'Modelo: Bm')
    plt.savefig(output_path+'Bm'+ext,bbox_inches='tight',dpi = 300)
    
    #-------------------------AM---------------------------------------------
    
    plt.figure('A_M')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('A_M')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_AM = plot_confidence_interval(plt, t, evals_AM, 'orange', 'Modelo: A_M')
    plt.savefig(output_path+'A_M'+ext,bbox_inches='tight',dpi = 300)

    
    #-------------------------AG---------------------------------------------
   
    plt.figure('A_G')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('A_G')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_AG = plot_confidence_interval(plt, t, evals_AG, 'red', 'Modelo: A_G')
    plt.savefig(output_path+'A_G'+ext,bbox_inches='tight',dpi = 300)
   
    
    #-------------------------I---------------------------------------------
    
    plt.figure('Ai')
    plt.xlim(0.0,dias_de_simulação)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('APC infectada')
    plt.legend(loc='upper left', prop={'size':13})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_I = plot_confidence_interval(plt, t, evals_I, 'green', 'Modelo: I')
    plt.savefig(output_path+'Ai'+ext,bbox_inches='tight',dpi = 300)
    
    #-------------------------C SURVIVORS---------------------------------------------
    
    
    plt.figure('C')
    plt.plot(dadosCitocinaSobreviventes['Day']+5, dadosCitocinaSobreviventes['IL6(pg/mL)'], 'o', label='data', linewidth=4)
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('C')
    plt.title("Solution Citocina")
    plt.legend(loc='upper left', prop={'size':13})
    plt.xticks(rotation=45)
    plt.tight_layout()
    mean_C = plot_confidence_interval(plt, t, evals_C, 'blue', 'Modelo: C')
    plt.savefig(output_path+'C'+ext,bbox_inches='tight',dpi = 300)
    plt.show()
    
    print("end code")
    
    #-------------------------C NON-SURVIVORS---------------------------------------------
    
    '''
    plt.figure('C')
    #dadosCitocinaObitos.plot.scatter(x='Day',y='IL6(pg/mL)',color='y',label='Dados experimentais(Obito)')
    plt.plot(dadosIL6['Day']+5, dadosIL6['IL6(pg/mL)'], 'o', label='data', linewidth=4)
    plt.xlim(0.0,dias_de_simulação)
    #plt.ylim(0.0,8.0)
    plt.xlabel('Tempo pós-vacinação (dias)')
    plt.ylabel('C')
    plt.legend()
    mean_C = plot_confidence_interval(plt, t, evals_C, 'blue', 'Modelo: C')
    plt.savefig(output_path+'C2.png',bbox_inches='tight',dpi = 300)
    #plt.show()
    '''

    
#End

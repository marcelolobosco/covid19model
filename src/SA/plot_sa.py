import sys
import numpy as np
import matplotlib.pyplot as plt


# Bibliotecas proprias

from uqtools import *
from NovoModelo import *

#TEMPO
dias_de_simulação = 35
t=range(dias_de_simulação)


if __name__ == "__main__":

    label_param = ("pi_v", "c_v1", "c_v2", "k_v1", "k_v2", 
        "beta_Ap", "c_ap1", "c_ap2", "delta_Apm", "beta_tk",
        "pi_tk", "delta_tk", "alpha_B","pi_B1", "pi_B2", 
        "beta_ps", "beta_pl", "beta_Bm","delta_S", "delta_L", "gamma_M", "k_bm1", "k_bm2", "pi_AS",
        "pi_AL", "delta_ag", "delta_am", "alpha_th", "beta_th", "pi_th", "delta_th", "Ap0", "Thn0", "Tkn0", "B0",  
        "pi_c_apm", "pi_c_i","pi_c_tke","delta_c", "beta_apm", "k_v3", "beta_tke")


    Si_v = np.loadtxt('sobol_v.txt')
    Si_ag = np.loadtxt('sobol_ag.txt')
    Si_am = np.loadtxt('sobol_am.txt')
    Si_c = np.loadtxt('sobol_c.txt')
    
    #plot_highest_sensitivity_mc(plt, t[1:], Si_v[1:].T, label_param)
    
    plt.figure();
    plt.title("V")
    plot_highest_sensitivity_mc(plt, t[1:], Si_v[1:].T, label_param)
    plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
    plt.tight_layout()
    plt.savefig('output_sens_virus.pdf')
    
    plt.figure();
    plt.title("IgG")
    plot_highest_sensitivity_mc(plt, t[1:], Si_ag[1:].T, label_param)
    plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
    plt.tight_layout()
    plt.savefig('output_sens_igg.pdf')
    
    plt.figure();
    plt.title("IgM")
    plot_highest_sensitivity_mc(plt, t[1:], Si_am[1:].T, label_param)
    plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
    plt.tight_layout()
    plt.savefig('output_sens_igm.pdf')
    
    plt.figure();
    plt.title("C")
    plot_highest_sensitivity_mc(plt, t[1:], Si_c[1:].T, label_param)
    plt.legend(loc=(1.04,0),prop={'size': 8}, ncol=2)
    plt.tight_layout()
    plt.savefig('output_sens_c.pdf')


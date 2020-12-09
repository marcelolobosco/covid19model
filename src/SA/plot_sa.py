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
        "pi_AL", "delta_ag", "delta_am", "alpha_th", "beta_th", "pi_th", "delta_th",  
        "pi_c_apm", "pi_c_i","pi_c_tke","delta_c", "beta_apm", "k_v3", "beta_tke", "alpha_Ap", "alpha_Tn")


    Si_v = np.loadtxt('./output_sobol_v.txt')
    Si_ag = np.loadtxt('./output_sobol_ag.txt')
    Si_am = np.loadtxt('./output_sobol_am.txt')
    Si_c = np.loadtxt('./output_sobol_c.txt')
    
    #plot_highest_sensitivity_mc(plt, t[1:], Si_v[1:].T, label_param)
    cols=1
    num=10
    plt.style.use('estilo/PlotStyle.mplstyle')
    plt.close('all')
    
    plt.figure();
    plt.title("Virus")
    plot_highest_sensitivity_mc(plt, t[1:], Si_v.T, label_param, num)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",prop={'size': 13}, ncol=cols)
    plt.grid(True)
    plt.tight_layout()    
    plt.savefig('output_sens_virus.pdf',bbox_inches='tight',dpi = 300)
    
    plt.figure();
    plt.title("IgG")
    plot_highest_sensitivity_mc(plt, t[1:], Si_ag.T, label_param, num)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",prop={'size': 13}, ncol=cols)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_sens_igg.pdf',bbox_inches='tight',dpi = 300)
    
    plt.figure();
    plt.title("IgM")
    plot_highest_sensitivity_mc(plt, t[1:], Si_am.T, label_param, num)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",prop={'size': 13}, ncol=cols)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_sens_igm.pdf',bbox_inches='tight',dpi = 300)
    
    plt.figure();
    plt.title("Cytokines")
    plot_highest_sensitivity_mc(plt, t[1:], Si_c.T, label_param, num)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",prop={'size': 13}, ncol=cols)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_sens_c.pdf',bbox_inches='tight',dpi = 300)


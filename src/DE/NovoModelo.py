#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:48:27 2018

@author: carlabonin
"""
from scipy import integrate
import numpy as np

def immune_response_v4 (P, t, pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap,
                     c_ap1, c_ap2, delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk,
                     alpha_B, pi_B1, pi_B2, beta_ps, beta_pl, beta_Bm,
                     delta_S, delta_L, gamma_bm, k_bm1, k_bm2, pi_AS,
                     pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0, 
                     pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke):
    """
    Simple Model
    """
    V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14]

    
####################################################################################
#Equações
####################################################################################
    dV_dt = pi_v*V - k_v1*V*A_M - k_v1*V*A_G - k_v2*V*Tke - k_v3*V*Apm #- ((c_v1*V)/(c_v2+V))
    dAp_dt = C*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) #alpha_Ap*(1+C)*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V))
    dApm_dt = beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) - delta_Apm*Apm - beta_apm * Apm * V
    dI_dt = beta_apm * Apm * V + beta_tke * Tke * V - delta_Apm*I
    dThn_dt = alpha_th*(Thn0 - Thn) - beta_th*Apm*Thn 
    dThe_dt = beta_th*Apm*Thn + pi_th*Apm*The - delta_th*The
    dTkn_dt = (C)*(Tkn0 - Tkn) - beta_tk*(C+1)*Apm*Tkn #alpha_Tn*(1+C)*(Tkn0 - Tkn) - beta_tk*Apm*Tkn
    dTke_dt = beta_tk*(C+1)*Apm*Tkn + pi_tk*Apm*Tke - delta_tk*Tke - beta_tke * Tke * V
    dB_dt = alpha_B*(B0 - B) + pi_B1*V*B + pi_B2*The*B - beta_ps*Apm*B - beta_pl*The*B - beta_Bm*The*B
    dPs_dt = beta_ps*Apm*B - delta_S*Ps
    dPl_dt = beta_pl*The*B - delta_L*Pl + gamma_bm*Bm 
    dBm_dt = beta_Bm*The*B + k_bm1*Bm*(1 - Bm/(k_bm2)) - gamma_bm*Bm
    dA_M_dt = pi_AS*Ps - delta_am*A_M # anticorpos vida curta
    dA_G_dt = pi_AL*Pl - delta_ag*A_G #memória imune
    dC_dt = pi_c_apm *V*Apm + pi_c_i*I + pi_c_tke*V*Tke - delta_c * C
    
    return [dV_dt, dAp_dt, dApm_dt, dThn_dt, dThe_dt, dTkn_dt, dTke_dt, dB_dt, dPs_dt, dPl_dt, dBm_dt, dA_M_dt, dA_G_dt, dI_dt, dC_dt]
    

def immune_response_v3 (P, t, pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap,
                     c_ap1, c_ap2, delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk,
                     alpha_B, pi_B1, pi_B2, beta_ps, beta_pl, beta_Bm,
                     delta_S, delta_L, gamma_bm, k_bm1, k_bm2, pi_AS,
                     pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0, 
                     pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke):
    """
    Simple Model
    """
    V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14]

    
####################################################################################
#Equações
####################################################################################
    dV_dt = pi_v*V - k_v1*V*A_M - k_v1*V*A_G - k_v2*V*Tke - k_v3*V*Apm #- ((c_v1*V)/(c_v2+V))
    dAp_dt = C*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) #alpha_Ap*(1+C)*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V))
    dApm_dt = beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) - beta_apm * Apm * V - delta_Apm*Apm
    dI_dt = beta_apm * Apm * V + beta_tke * Tke * V - delta_Apm*I
    dThn_dt = alpha_th*(Thn0 - Thn) - beta_th*Apm*Thn 
    dThe_dt = beta_th*Apm*Thn + pi_th*Apm*The - delta_th*The
    dTkn_dt = (C)*(Tkn0 - Tkn) - beta_tk*(C+1)*Apm*Tkn #alpha_Tn*(1+C)*(Tkn0 - Tkn) - beta_tk*Apm*Tkn
    dTke_dt = beta_tk*(C+1)*Apm*Tkn + pi_tk*Apm*Tke - beta_tke * Tke * V - delta_tk*Tke
    dB_dt = alpha_B*(B0 - B) + pi_B1*V*B + pi_B2*The*B - beta_ps*Apm*B - beta_pl*The*B - beta_Bm*The*B
    dPs_dt = beta_ps*Apm*B - delta_S*Ps
    dPl_dt = beta_pl*The*B - delta_L*Pl + gamma_bm*Bm 
    dBm_dt = beta_Bm*The*B + k_bm1*Bm*(1 - Bm/(k_bm2)) - gamma_bm*Bm
    dIgM_dt = pi_AS*Ps - delta_am*A_M # anticorpos vida curta
    dIgG_dt = pi_AL*Pl - delta_ag*A_G #memória imune
    dC_dt = pi_c_apm * Apm + pi_c_i * I + pi_c_tke * Tke - delta_c * C
    
    return [dV_dt, dAp_dt, dApm_dt, dThn_dt, dThe_dt, dTkn_dt, dTke_dt, dB_dt, dPs_dt, dPl_dt, dBm_dt, dIgM_dt, dIgG_dt, dI_dt, dC_dt]

  
def immune_response_v2 (P, t, pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap,
                     c_ap1, c_ap2, delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk,
                     alpha_B, pi_B1, pi_B2, beta_ps, beta_pl, beta_Bm,
                     delta_S, delta_L, gamma_bm, k_bm1, k_bm2, pi_AS,
                     pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0, 
                     pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3, beta_tke):
    """
    Simple Model
    """
    V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, I ,C = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14]

    
####################################################################################
#Equações
####################################################################################
    dV_dt = pi_v*V - k_v1*V*A_M - k_v1*V*A_G - k_v2*V*Tke - k_v3*V*Apm #- ((c_v1*V)/(c_v2+V))
    dAp_dt = C*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) #alpha_Ap*(1+C)*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V))
    dApm_dt = beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) - delta_Apm*Apm - beta_apm * Apm * V
    dI_dt = beta_apm * Apm * V + beta_tke * Tke * V - delta_Apm*I
    dThn_dt = alpha_th*(Thn0 - Thn) - beta_th*Apm*Thn 
    dThe_dt = beta_th*Apm*Thn + pi_th*Apm*The - delta_th*The
    dTkn_dt = (C)*(Tkn0 - Tkn) - beta_tk*Apm*Tkn #alpha_Tn*(1+C)*(Tkn0 - Tkn) - beta_tk*Apm*Tkn
    dTke_dt = beta_tk*Apm*Tkn + pi_tk*Apm*Tke - delta_tk*Tke - beta_tke * Tke * V
    dB_dt = alpha_B*(B0 - B) + pi_B1*V*B + pi_B2*The*B - beta_ps*Apm*B - beta_pl*The*B - beta_Bm*The*B
    dPs_dt = beta_ps*Apm*B - delta_S*Ps
    dPl_dt = beta_pl*The*B - delta_L*Pl + gamma_bm*Bm 
    dBm_dt = beta_Bm*The*B + k_bm1*Bm*(1 - Bm/(k_bm2)) - gamma_bm*Bm
    dA_M_dt = pi_AS*Ps - delta_am*A_M # anticorpos vida curta
    dA_G_dt = pi_AL*Pl - delta_ag*A_G #memória imune
    dC_dt = pi_c_apm * Apm + pi_c_i * I + pi_c_tke * Tke - delta_c * C
    
    return [dV_dt, dAp_dt, dApm_dt, dThn_dt, dThe_dt, dTkn_dt, dTke_dt, dB_dt, dPs_dt, dPl_dt, dBm_dt, dA_M_dt, dA_G_dt, dI_dt, dC_dt]



def immune_response (P, t, pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap,
                     c_ap1, c_ap2, delta_Apm, alpha_Tn, beta_tk, pi_tk, delta_tk,
                     alpha_B, pi_B1, pi_B2, beta_ps, beta_pl, beta_Bm,
                     delta_S, delta_L, gamma_bm, k_bm1, k_bm2, pi_AS,
                     pi_AL, delta_ag, delta_am, alpha_th, beta_th, pi_th, delta_th, Ap0, Thn0, Tkn0, B0, 
                     pi_c_apm, pi_c_i,pi_c_tke,delta_c, beta_apm, k_v3):
    """
    Simple Model
    """
    V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G, Ai,C = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14]

    
####################################################################################
#Equações
####################################################################################
    dV_dt = pi_v*V - k_v1*V*A_M - k_v1*V*A_G - k_v2*V*Tke - k_v3*V*Apm #- ((c_v1*V)/(c_v2+V))
    dAp_dt = C*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) #alpha_Ap*(1+C)*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V))
    dApm_dt = beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) - delta_Apm*Apm - beta_apm * Apm * V
    dAi_dt = beta_apm * Apm * V - delta_Apm*Ai
    dThn_dt = alpha_th*(Thn0 - Thn) - beta_th*Apm*Thn 
    dThe_dt = beta_th*Apm*Thn + pi_th*Apm*The - delta_th*The
    dTkn_dt = (C)*(Tkn0 - Tkn) - beta_tk*Apm*Tkn #alpha_Tn*(1+C)*(Tkn0 - Tkn) - beta_tk*Apm*Tkn
    dTke_dt = beta_tk*Apm*Tkn + pi_tk*Apm*Tke - delta_tk*Tke
    dB_dt = alpha_B*(B0 - B) + pi_B1*V*B + pi_B2*The*B - beta_ps*Apm*B - beta_pl*The*B - beta_Bm*The*B
    dPs_dt = beta_ps*Apm*B - delta_S*Ps
    dPl_dt = beta_pl*The*B - delta_L*Pl + gamma_bm*Bm 
    dBm_dt = beta_Bm*The*B + k_bm1*Bm*(1 - Bm/(k_bm2)) - gamma_bm*Bm
    dA_M_dt = pi_AS*Ps - delta_am*A_M # anticorpos vida curta
    dA_G_dt = pi_AL*Pl - delta_ag*A_G #memória imune
    dC_dt = pi_c_apm * Apm + pi_c_i * Ai + pi_c_tke * Tke - delta_c * C
    
    return [dV_dt, dAp_dt, dApm_dt, dThn_dt, dThe_dt, dTkn_dt, dTke_dt, dB_dt, dPs_dt, dPl_dt, dBm_dt, dA_M_dt, dA_G_dt, dAi_dt, dC_dt]


####################################################################################
#Resolvendo o sistema de EDOs
####################################################################################   
def immune_response_old (P, t, pi_v=1.25, c_v1=2.63, c_v2=0.60, k_v1=0.000120532191*0.4, k_v2=1.87E-06*0.4, alpha_Ap=2.50E-03, beta_Ap=5.5e-01,
                     c_ap1=0.8, c_ap2=40.0, delta_Apm=5.38E-01, alpha_Tn=2.17E-04, beta_tk=1.0E-05, pi_tk=1.0E-08, delta_tk=0.1*0.003,
                     alpha_B=6.0E+00, pi_B1=4.826E-06, pi_B2=1.27E-10*100.0, beta_ps=0.000672, beta_pl=5.61E-06, beta_Bm=1.0E-06,
                     delta_S=2.0, delta_L=(2.22E-04)*1.8*0.6, gamma_bm=(1.95E-06)*500.0, k_bm1=1.0e-5, k_bm2=2500.0, pi_AS=0.002,
                     pi_AL=0.00068, delta_ag=0.01, delta_am=0.05, alpha_th = 2.17E-04, beta_th = 1.0E-07, pi_th = 1E-08, delta_th = 0.22, Ap0 = 1.0e6, Thn0 = (1.0e6), Tkn0 = 5.0e5, B0 = (1.0e3)*250.0):
    """
    Simple Model
    """
    V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12]

    
####################################################################################
#Equações
####################################################################################
    dV_dt = pi_v*V - (c_v1*V)/(c_v2+V) - k_v1*V*A_M - k_v1*V*A_G - k_v2*V*Tke
    dAp_dt = alpha_Ap*(Ap0 - Ap) - beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V))
    dApm_dt = beta_Ap*Ap*(c_ap1*(V)/(c_ap2 + V)) - delta_Apm*Apm
    dThn_dt = alpha_th*(Thn0 - Thn) - beta_th*Apm*Thn 
    dThe_dt = beta_th*Apm*Thn + pi_th*Apm*The - delta_th*The
    dTkn_dt = alpha_Tn*(Tkn0 - Tkn) - beta_tk*Apm*Tkn
    dTke_dt = beta_tk*Apm*Tkn + pi_tk*Apm*Tke - delta_tk*Tke
    dB_dt = alpha_B*(B0 - B) + pi_B1*V*B + pi_B2*The*B - beta_ps*Apm*B - beta_pl*The*B - beta_Bm*The*B
    dPs_dt = beta_ps*Apm*B - delta_S*Ps
    dPl_dt = beta_pl*The*B - delta_L*Pl + gamma_bm*Bm 
    dBm_dt = beta_Bm*The*B + k_bm1*Bm*(1 - Bm/(k_bm2)) - gamma_bm*Bm
    dA_M_dt = pi_AS*Ps - delta_am*A_M 
    dA_G_dt = pi_AL*Pl - delta_ag*A_G
    
    return [dV_dt, dAp_dt, dApm_dt, dThn_dt, dThe_dt, dTkn_dt, dTke_dt, dB_dt, dPs_dt, dPl_dt, dBm_dt, dA_M_dt, dA_G_dt]

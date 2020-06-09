#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:48:27 2018

@author: carlabonin
"""
from scipy import integrate
import numpy as np
    
####################################################################################
#Resolvendo o sistema de EDOs
####################################################################################   
def immune_response (P, t, pi_v=1.25, c_v1=2.63, c_v2=0.60, k_v1=0.000120532191*0.4, k_v2=1.87E-06*0.4, alpha_Ap=2.50E-03, beta_Ap=5.5e-01,
                     k_ap1=0.8, k_ap2=40.0, delta_Apm=5.38E-01, alpha_Tn=2.17E-04, pi_T=1.0E-05, k_te1=1.0E-08, delta_te=0.1*0.003,
                     alpha_B=6.0E+00, pi_B1=4.826E-06, pi_B2=1.27E-10*100.0, beta_S=0.000672, beta_L=5.61E-06, beta_Bm=1.0E-06,
                     delta_S=2.0, delta_L=(2.22E-04)*1.8*0.6, gamma_M=(1.95E-06)*500.0, k_bm1=1.0e-5, k_bm2=2500.0, pi_AS=0.002,
                     pi_AL=0.00068, delta_A_G=0.01, delta_A_M=0.05, c11 = 2.17E-04, c12 = 1.0E-07, c13 = 1E-08, c14 = 0.22, Ap0 = 1.0e6, Thn0 = (1.0e6), Tkn0 = 5.0e5, 
                     B0 =  (1.0e3)*250.0):
    """
    Simple Model
    """
    V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A_M, A_G = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12]

    
####################################################################################
#Equações
####################################################################################
    dV_dt = pi_v*V - (c_v1*V)/(c_v2+V) - k_v1*V*A_M - k_v1*V*A_G - k_v2*V*Tke
    dAp_dt = alpha_Ap*(Ap0 - Ap) - beta_Ap*Ap*(k_ap1*(V)/(k_ap2 + V))
    dApm_dt = beta_Ap*Ap*(k_ap1*(V)/(k_ap2 + V)) - delta_Apm*Apm
    dThn_dt = c11*(Thn0 - Thn) - c12*Apm*Thn 
    dThe_dt = c12*Apm*Thn + c13*Apm*The - c14*The
    dTkn_dt = alpha_Tn*(Tkn0 - Tkn) - pi_T*Apm*Tkn
    dTke_dt = pi_T*Apm*Tkn + k_te1*Apm*Tke - delta_te*Tke
    dB_dt = alpha_B*(B0 - B) + pi_B1*V*B + pi_B2*The*B - beta_S*Apm*B - beta_L*The*B - beta_Bm*The*B
    dPs_dt = beta_S*Apm*B - delta_S*Ps
    dPl_dt = beta_L*The*B - delta_L*Pl + gamma_M*Bm 
    dBm_dt = beta_Bm*The*B + k_bm1*Bm*(1 - Bm/(k_bm2)) - gamma_M*Bm
    dA_M_dt = pi_AS*Ps - delta_A_M*A_M 
    dA_G_dt = pi_AL*Pl - delta_A_G*A_G
    
    return [dV_dt, dAp_dt, dApm_dt, dThn_dt, dThe_dt, dTkn_dt, dTke_dt, dB_dt, dPs_dt, dPl_dt, dBm_dt, dA_M_dt, dA_G_dt]

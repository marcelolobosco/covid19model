#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:48:27 2018

@author: carlabonin
"""
from scipy import integrate
import numpy as np

####################################################################################
#Parâmetros
#####################################################################################
pi_v  	=	1.25
c_v1       =     2.63
c_v2       =     0.60
k_v1  	=	0.000120532191*0.4
k_v2  	=	1.87E-06*0.4
alpha_Ap  	=	2.50E-03
beta_Ap  	=    5.5e-01
k_ap1      =    0.8
k_ap2      =    40.0
delta_Apm  	=	5.38E-01
alpha_Tn  	=	2.17E-04
pi_T  	=	1.0E-05
k_te1      =     1.0E-08
delta_te   =     0.1*0.003
alpha_B  	=	6.0E+00
pi_B1  	=	4.826E-06
pi_B2  	=	1.27E-10*100.0
beta_S  	=	0.000672
beta_L  	=	5.61E-06
beta_Bm  	=	1.0E-06
delta_S  	=	2.0
delta_L  	=	(2.22E-04)*1.8*0.6
gamma_M  	=	(1.95E-06)*500.0
k_bm1      =     1.0e-5
k_bm2      =     2500.0
pi_AS  	=	0.002
pi_AL  	=	0.00068
delta_A    =     0.04
c11 =  2.17E-04
c12 = 1.0E-07
c13 = 1E-08
c14 = 0.22

#######  CONDIÇÕES INICIAIS PARA AJUSTE DE UNIDADES
V0 = 10 #  infectious dose # 27476.0 IU/0.5ml  ---->  5 IU/ml diluído em 5500ml de sangue do corpo  ----> 
###  x1.91 = 9.55 PFU/ml no corpo = 0.98log10 PFU/ml no corpo   ------> 3.89 log10 copias/ml no corpos (*) = 7728.0 copias/ml
###(*)  log10 PFU/ml = [0.974 x log10 copias/ml] - 2.807
Ap0 = 1.0e6
Apm0 = 0.0
Thn0 = (1.0e6)#*0.5
The0 = 0.0  #### Convertendo de ul para ml
#Tkn0 = 5.0e5
Tkn0 = (1.0e3)*500.0#(1.0e3)*500.0
Tke0 = 0.0
B0 =  (1.0e3)*250.0#125000.0#
Ps0 = 0.0
Pl0 = 0.0
Bm0 = 0.0
A0 = 0.0  

    
####################################################################################
#Equações
####################################################################################
#y[0]
def V(y, t): 
    return  pi_v*y[0] - (c_v1*y[0])/(c_v2+y[0]) - k_v1*y[0]*y[11] - k_v2*y[0]*y[6]
    
#y[1]    
def Ap(y,t): 
#   k3 = 10.0 
#   aux = ((k_ap1*(y[0]-k3))/(k_ap2 + y[0]- k3))
#   if aux<0.0:
#       aux = 0.0
#   return alpha_Ap*(Ap0 - y[1]) - beta_Ap*y[1]*aux
   return alpha_Ap*(Ap0 - y[1]) - beta_Ap*y[1]*(k_ap1*(y[0])/(k_ap2 + y[0]))
   #return alpha_Ap*(Ap0 - y[1]) - beta_Ap*y[1]*(k_ap1 + np.tanh(y[0]-k_ap2))
  
 
#y[2]
def Apm(y,t): 
#   k3 = 10.0
#   aux = ((k_ap1*(y[0]-k3))/(k_ap2 + y[0]- k3))
#   if aux<0.0:
#       aux = 0.0
#   return  beta_Ap*y[1]*aux - delta_Apm*y[2] 
    return beta_Ap*y[1]*(k_ap1*(y[0])/(k_ap2 + y[0])) - delta_Apm*y[2] 
#    return beta_Ap*y[1]*(k_ap1 + np.tanh(y[0]-k_ap2)) - delta_Apm*y[2] 

#y[3]
def Thn(y,t):
   return  c11*(Thn0 - y[3]) - c12*y[2]*y[3]
    
#y[4]
def The(y,t):
   return  c12*y[2]*y[3] + c13*y[2]*y[4] - c14*y[4] 
 
#y[5]    
def Tkn(y,t):
    return  alpha_Tn*(Tkn0 - y[5]) - pi_T*y[2]*y[5] 
                      
#y[6]
def Tke(y,t):
    return  pi_T*y[2]*y[5] + k_te1*y[2]*y[6] - delta_te*y[6]

#y[7]  
def B(y,t):
    return alpha_B*(B0 - y[7]) + pi_B1*y[0]*y[7] + pi_B2*y[4]*y[7] - beta_S*y[2]*y[7] - beta_L*y[4]*y[7] - beta_Bm*y[4]*y[7]
                    
#y[8]   
def Ps(y,t):
   return beta_S*y[2]*y[7] - delta_S*y[8] 
    
#y[9]    
def Pl(y,t):
    return beta_L*y[4]*y[7] - delta_L*y[9] + gamma_M*y[10] 
   
#y[10]
def Bm(y,t):
    return beta_Bm*y[4]*y[7] + k_bm1*y[10]*(1 - y[10]/(k_bm2)) - gamma_M*y[10]
    
#y[11]    
def A(y,t):
    return pi_AS*y[8] + pi_AL*y[9] - delta_A*y[11]
  

    
    
####################################################################################
#Resolvendo o sistema de EDOs
####################################################################################    
def f(y,t):
    return V(y,t),Ap(y,t),Apm(y,t),Thn(y,t),The(y,t),Tkn(y,t),Tke(y,t),B(y,t),Ps(y,t),Pl(y,t),Bm(y,t),A(y,t)

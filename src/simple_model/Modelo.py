#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:48:27 2018

@author: carlabonin
"""
from scipy import integrate
import numpy as np

#####################################################################################
##Parâmetros
#####################################################################################
#pi_v  	=	1.36*0.5#*1.25
#c_v1       =     50.0#2.63
#c_v2       =     0.60
#k_v1  	=	0.000120532191*0.4
#k_v2  	=	1.87E-06*0.4
#alpha_Ap  	=	2.50E-03
#beta_Ap  	=    5.5e-01
#k_ap1      =    0.8
#k_ap2      =    40.0
#delta_Apm  	=	5.38E-01
#alpha_Tn  	=	7.0E-04#2.17E-04
#pi_T  	=	1.0E-05
#k_te1      =     1.0E-08
#delta_te   =     0.1*0.003
#alpha_B  	=	6.0E+00
#pi_B1  	=	4.826E-06
#pi_B2  	=	1.27E-10*100.0
#beta_S  	=	0.000672
#beta_L  	=	1.61E-06
#beta_Bm  	=	5.0E-07#1.0E-06
#delta_S  	=	2.0
#delta_L  	=	(2.22E-04)*1.8*0.6
#gamma_M  	=	1.95E-04
#k_bm1      =     1.0e-2#1.0e-5
#k_bm2      =     5000.0#2500.0
#pi_AS  	=	0.002
#pi_AL  	=	0.00068
#delta_A    =     0.04
#c11 =  2.17E-03#2.17E-04
#c12 = 5.0E-05#1.0E-05
#c13 = 5.0E-08#1E-08
#c14 = 0.22


#######  Imunomoduladores #########
#beta_S  	=	0.000672*0.2
#beta_L  	=	(1.61E-06)*0.2
#beta_Bm  	=	(1.0E-06)*0.2

#########   teste Criancas
#beta_L  	=	(1.61E-06)*0.01
#delta_L  	=	(2.22E-04)*4.0
#beta_S  	=	0.000672*0.4


##################################    Resultados 08 Abril  ###################################
####################################################################################
#Parâmetros
#####################################################################################
pi_v  	=	1.36*0.5#*1.25
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

########## Reducao de Bm  ########

#####  Imunomoduladores #########
#beta_S  	=	0.000672*0.5
#beta_L  	=	(5.61E-06)*0.5
#beta_Bm  	=	(1.0E-06)*0.5
###### OU ########
#alpha_B  	=	(6.0E+00)*0.5


#####   teste Criancas
#beta_L  	=	(5.61E-06)*0.3
#delta_L  	=     ((2.22E-04)*1.8*0.6)*2.0


#######  CONDIÇÕES INICIAIS PARA AJUSTE DE UNIDADES
V0 = 726.12# 3.64#14.00#75.06#269.05##7728.0# 27476.0 IU/0.5ml  ---->  5 IU/ml diluído em 5500ml de sangue do corpo  ----> 
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
A0 = 150.0  

    
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
    
t3=np.linspace(0,5000,20000)     
def rodaModelo(Npi_v,Nc_v1,Nc_v2,Nk_v1,Nk_v2,Nalpha_Ap,Nbeta_Ap,Nk_ap1,Nk_ap2,Ndelta_Apm,Nalpha_Tn,Npi_T,Nk_te1,Ndelta_te,Nalpha_B,Npi_B1,Npi_B2,Nbeta_S,Nbeta_L,Nbeta_Bm,Ndelta_S,Ndelta_L,Ngamma_M,Nk_bm1,Nk_bm2,Npi_AS,Npi_AL,Ndelta_A,Nc11,Nc12,Nc13,Nc14):
    global pi_v  	
    global c_v1       
    global c_v2       
    global k_v1  	
    global k_v2  	
    global alpha_Ap  	
    global beta_Ap  	
    global k_ap1      
    global k_ap2      
    global delta_Apm  	
    global alpha_Tn  	
    global pi_T  	
    global k_te1      
    global delta_te   
    global alpha_B  	
    global pi_B1  	
    global pi_B2  	
    global beta_S  	
    global beta_L  	
    global beta_Bm  	
    global delta_S  	
    global delta_L  	
    global gamma_M  	
    global k_bm1      
    global k_bm2      
    global pi_AS  	
    global pi_AL  	
    global delta_A    
    global c11 
    global c12 
    global c13 
    global c14 


        
    pi_v = Npi_v
    c_v1 =  Nc_v1    
    c_v2  =  Nc_v2   
    k_v1  	= Nk_v1
    k_v2  	= Nk_v2
    alpha_Ap= Nalpha_Ap 	
    beta_Ap  =	Nbeta_Ap
    k_ap1     = Nk_ap1
    k_ap2      = Nk_ap2
    delta_Apm  	= Ndelta_Apm
    alpha_Tn  	= Nalpha_Tn
    pi_T  	= Npi_T
    k_te1 = Nk_te1
    delta_te =  Ndelta_te
    alpha_B  =	Nalpha_B
    pi_B1  	= Npi_B1
    pi_B2  	= Npi_B2
    beta_S  = Nbeta_S	
    beta_L  = Nbeta_L	
    beta_Bm  =	Nbeta_Bm
    delta_S  =	Ndelta_S
    delta_L  =	Ndelta_L
    gamma_M  =	Ngamma_M
    k_bm1     = Nk_bm1
    k_bm2      = Nk_bm2
    pi_AS  	= Npi_AS
    pi_AL  	= Npi_AL
    delta_A =   Ndelta_A 
    c11 = Nc11
    c12 = Nc12
    c13 = Nc13
    c14 = Nc14

#    
#    global V0 
#    global Ap0 
#    global Apm0 
#    global Thn0 
#    global The0 
#    global Tkn0 
#    global Tke0 
#    global B0 
#    global Ps0 
#    global Pl0
#    global Bm0 
#    global A0 
#    
#    
#    V0 =  NV0
#    Ap0 = NAp0
#    Apm0 = NApm0
#    Thn0 = NThn0
#    The0 =NThe0
#    Tkn0 = NTkn0
#    Tke0 = NTke0
#    B0 = NB0
#    Ps0 = NPs0
#    Pl0 = NPl0
#    Bm0 = NBm0
#    A0 = NA0
    
       
    y,d=integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0], t3, full_output=True, printmessg=True)
    return y



    
def rodaModeloCompleto(Npi_v,Nc_v1,Nc_v2,Nk_v1,Nk_v2,Nalpha_Ap,Nbeta_Ap,Nk_ap1,Nk_ap2,Ndelta_Apm,Nalpha_Tn,Npi_T,Nk_te1,Ndelta_te,Nalpha_B,Npi_B1,Npi_B2,Nbeta_S,Nbeta_L,Nbeta_Bm,Ndelta_S,Ndelta_L,Ngamma_M,Nk_bm1,Nk_bm2,Npi_AS,Npi_AL,Ndelta_A,Nc11,Nc12,Nc13,Nc14,NV0,NAp0,NApm0,NThn0,NThe0,NTkn0,NTke0,NB0,NPs0,NPl0,NBm0,NA0,Nt3):
    global pi_v  	
    global c_v1       
    global c_v2       
    global k_v1  	
    global k_v2  	
    global alpha_Ap  	
    global beta_Ap  	
    global k_ap1      
    global k_ap2      
    global delta_Apm  	
    global alpha_Tn  	
    global pi_T  	
    global k_te1      
    global delta_te   
    global alpha_B  	
    global pi_B1  	
    global pi_B2  	
    global beta_S  	
    global beta_L  	
    global beta_Bm  	
    global delta_S  	
    global delta_L  	
    global gamma_M  	
    global k_bm1      
    global k_bm2      
    global pi_AS  	
    global pi_AL  	
    global delta_A    
    global c11 
    global c12 
    global c13 
    global c14 


        
    pi_v = Npi_v
    c_v1 =  Nc_v1    
    c_v2  =  Nc_v2   
    k_v1  	= Nk_v1
    k_v2  	= Nk_v2
    alpha_Ap= Nalpha_Ap 	
    beta_Ap  =	Nbeta_Ap
    k_ap1     = Nk_ap1
    k_ap2      = Nk_ap2
    delta_Apm  	= Ndelta_Apm
    alpha_Tn  	= Nalpha_Tn
    pi_T  	= Npi_T
    k_te1 = Nk_te1
    delta_te =  Ndelta_te
    alpha_B  =	Nalpha_B
    pi_B1  	= Npi_B1
    pi_B2  	= Npi_B2
    beta_S  = Nbeta_S	
    beta_L  = Nbeta_L	
    beta_Bm  =	Nbeta_Bm
    delta_S  =	Ndelta_S
    delta_L  =	Ndelta_L
    gamma_M  =	Ngamma_M
    k_bm1     = Nk_bm1
    k_bm2      = Nk_bm2
    pi_AS  	= Npi_AS
    pi_AL  	= Npi_AL
    delta_A =   Ndelta_A 
    c11 = Nc11
    c12 = Nc12
    c13 = Nc13
    c14 = Nc14

    
    global V0 
    global Ap0 
    global Apm0 
    global Thn0 
    global The0 
    global Tkn0 
    global Tke0 
    global B0 
    global Ps0 
    global Pl0
    global Bm0 
    global A0 
    
    
    V0 =  NV0
    Ap0 = NAp0
    Apm0 = NApm0
    Thn0 = NThn0
    The0 =NThe0
    Tkn0 = NTkn0
    Tke0 = NTke0
    B0 = NB0
    Ps0 = NPs0
    Pl0 = NPl0
    Bm0 = NBm0
    A0 = NA0
    
       
    y,d=integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0], Nt3, full_output=True, printmessg=True)
    return y    
    
def rodaModelo1(Npi_v,Nk_v1,Nk_v2,Nbeta_Ap,Npi_B1,Npi_AS,Npi_AL):	
    
    global pi_v  	
#    global c_v1       
#    global c_v2       
    global k_v1  	
    global k_v2  	 	
    global beta_Ap  
#    global k_ap1      
#    global k_ap2  
    global pi_B1 
    global pi_AS  	
    global pi_AL  
    
    
    pi_v   = pi_v*Npi_v
    #c_v1   = c_v1*Nc_v1    
    #c_v2   = c_v2*Nc_v2   
    k_v1  	= k_v1*Nk_v1
    k_v2  	= k_v2*Nk_v2	
    beta_Ap  =	beta_Ap*Nbeta_Ap
    #k_ap1  = k_ap1*Nk_ap1
    #k_ap2  = k_ap2*Nk_ap2
    pi_B1  	= pi_B1*Npi_B1
    pi_AS  	= pi_AS*Npi_AS
    pi_AL  	= pi_AL*Npi_AL
       
    y,d=integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0], t3, full_output=True, printmessg=True)
    return y   
    
def rodaModelo2(Npi_v,Nc_v1,Nc_v2,Nk_v1,Nk_v2,Nalpha_Ap,Nbeta_Ap,Nk_ap1,Nk_ap2,Ndelta_Apm,Nalpha_Tn,Npi_T,Nk_te1,Ndelta_te,Nalpha_B,Npi_B1,Npi_B2,Nbeta_S,Nbeta_L,Nbeta_Bm,Ndelta_S,Ndelta_L,Ngamma_M,Nk_bm1,Nk_bm2,Npi_AS,Npi_AL,Ndelta_A,Nc11,Nc12,Nc13,Nc14,NV0,NAp0,NApm0,NThn0,NThe0,NTkn0,NTke0,NB0,NPs0,NPl0,NBm0,NA0,Nt3):
    global pi_v  	
    global c_v1       
    global c_v2       
    global k_v1  	
    global k_v2  	
    global alpha_Ap  	
    global beta_Ap  	
    global k_ap1      
    global k_ap2      
    global delta_Apm  	
    global alpha_Tn  	
    global pi_T  	
    global k_te1      
    global delta_te   
    global alpha_B  	
    global pi_B1  	
    global pi_B2  	
    global beta_S  	
    global beta_L  	
    global beta_Bm  	
    global delta_S  	
    global delta_L  	
    global gamma_M  	
    global k_bm1      
    global k_bm2      
    global pi_AS  	
    global pi_AL  	
    global delta_A    
    global c11 
    global c12 
    global c13 
    global c14 


        
    pi_v = Npi_v
    c_v1 =  Nc_v1    
    c_v2  =  Nc_v2   
    k_v1  	= Nk_v1
    k_v2  	= Nk_v2
    alpha_Ap= Nalpha_Ap 	
    beta_Ap  =	Nbeta_Ap
    k_ap1     = Nk_ap1
    k_ap2      = Nk_ap2
    delta_Apm  	= Ndelta_Apm
    alpha_Tn  	= Nalpha_Tn
    pi_T  	= Npi_T
    k_te1 = Nk_te1
    delta_te =  Ndelta_te
    alpha_B  =	Nalpha_B
    pi_B1  	= Npi_B1
    pi_B2  	= Npi_B2
    beta_S  = Nbeta_S	
    beta_L  = Nbeta_L	
    beta_Bm  =	Nbeta_Bm
    delta_S  =	Ndelta_S
    delta_L  =	Ndelta_L
    gamma_M  =	Ngamma_M
    k_bm1     = Nk_bm1
    k_bm2      = Nk_bm2
    pi_AS  	= Npi_AS
    pi_AL  	= Npi_AL
    delta_A =   Ndelta_A 
    c11 = Nc11
    c12 = Nc12
    c13 = Nc13
    c14 = Nc14

    
    global V0 
    global Ap0 
    global Apm0 
    global Thn0 
    global The0 
    global Tkn0 
    global Tke0 
    global B0 
    global Ps0 
    global Pl0
    global Bm0 
    global A0 
    
    
    V0 =  NV0
    Ap0 = NAp0
    Apm0 = NApm0
    Thn0 = NThn0
    The0 =NThe0
    Tkn0 = NTkn0
    Tke0 = NTke0
    B0 = NB0
    Ps0 = NPs0
    Pl0 = NPl0
    Bm0 = NBm0
    A0 = NA0
    
       
    y,d=integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0], Nt3, full_output=True, printmessg=True)
    return y
    
def rodaModeloCondIniciais(NAp0,NApm0,NThn0,NThe0,NTkn0,NTke0,NB0,NPs0,NPl0,NBm0,NA0,Nt3):
    global Ap0 
    global Apm0 
    global Thn0 
    global The0 
    global Tkn0 
    global Tke0 
    global B0 
    global Ps0 
    global Pl0
    global Bm0 
    global A0 
    
    
    Ap0 = NAp0
    Apm0 = NApm0
    Thn0 = NThn0
    The0 =NThe0
    Tkn0 = NTkn0
    Tke0 = NTke0
    B0 = NB0
    Ps0 = NPs0
    Pl0 = NPl0
    Bm0 = NBm0
    A0 = NA0
    
       
    y,d=integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0], Nt3, full_output=True, printmessg=True)
    return y        
    

#!/usr/bin/python
import numpy as np
from expmmq import *
c0 = None
c1 = None

def define_ft(casos, dia):
# considerando dia0 como 14/03 (primeiro caso em JF). 
# ref: https://www.worldometers.info/coronavirus/coronavirus-cases/#total-cases
#casos = np.loadtxt('./data/active_world.txt');
#dia = np.array(range(0,len(casos)));

    c0,c1 = exp_mmq(dia, casos)
    return c0,c1;

def SIR(P, t):
    """
    Modelo SIR classico
    """
    S,I,R = P[0],P[1],P[2]        
    beta = 0.0005
    mu = 0.00001
    gamma = 0.1

    dSdt = -beta*S*I
    dIdt = beta*S*I - gamma*I
    dRdt = gamma*I
    
    return [dSdt, dIdt, dRdt]

def SIR_PP(P,t,a,b,c,d,e):
    """
    Modelo SIR modificado v1
    """

    S,I,R = P[0],P[1],P[2]

    ft = e*c0*np.exp(c1*t)
    alfa = a*b
    beta = c*d
    gamma = 1.0-beta
    
    dSdt = -alfa*S*I
    dIdt = alfa*S*I + ft - beta*I - gamma*I
    dRdt = gamma*I
    
    return [dSdt, dIdt, dRdt]

def SIR_PP_V2(P,t,amax,r,ti,tf,c,d,e,r1,r2):
    """
    Modelo SIR modificado v2
    ( x[0],x[1],x[2],x[3],x[4],x[5],x[6], 1./(x[7]+x[8]), 1./(x[7] + x[9]) )
    """
    #
    S,I,R = P[0], P[1], P[2]

    # parametros
    ft = e*c0*np.exp(c1*t)
    if t<ti:
        alfa=amax
    elif ti <= t and t<=tf:
        alfa=amax*(1-r)/(ti-tf)*(t-ti) + amax
    else:
        alfa=amax*r
    beta1 = c*d*r1
    beta2 = (1.0-c*d)*r2

    dSdt = -alfa*S*I
    dIdt = alfa*S*I + ft- beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt]   

def SIR_PP_V3(P,t,amax,r,ti,tf,m,e,r1,r2):
    """
    Modelo final da NT01
    """
    S,I,R,M = P[0], P[1], P[2], P[3]

    # parametros
    ft = e*c0*np.exp(c1*t)
    if t<ti:
        alfa=amax
    elif ti <= t and t<=tf:
        alfa=amax*(1-r)/(ti-tf)*(t-ti) + amax
    else:
        alfa=amax*r

    beta1 = m*r1
    beta2 = (1.0-m)*r2

    dSdt = -alfa*S*I
    dIdt = alfa*S*I + ft- beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt]  


def SIR_PP_V4(P,t,amax,r,ti,delta,m,e,r1,r2, first_day, c0, c1):
    """
    Modelo final da NT02
    """
    S,I,R,M = P[0], P[1], P[2], P[3]

    # parametros
    t_exp = t + first_day; #para alinhar o 1 dia dos dados da exponencial com o 1 dia do experimento
    ft = e*c0*np.exp(c1*t_exp)
    if t<ti:
        alfa=amax
    elif ti <= t and t<=ti+delta:
        alfa=amax*(1-r)/(-delta)*(t-ti) + amax
    else:
        alfa=amax*r

    beta1 = m*r1
    beta2 = (1.0-m)*r2

    dSdt = -alfa*S*I
    dIdt = alfa*S*I + ft- beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt]  
    
def SIR_PP_V5(P,t,amax,rf1,rf2,rf3,m,e,r1,r2, first_day, c0, c1):
    """
    Modelo V5
    """
    S,I,R,M = P[0], P[1], P[2], P[3]
    
    delta_d = 20;
    
    if t<=delta_d:
        alfa = t*(rf1-1)/delta_d + 1
    elif t<=2*delta_d:
        alfa = (t-delta_d)*(rf2-rf1)/delta_d + rf1       
    elif t<=3*delta_d:
        alfa = (t-2*delta_d)*(rf3-rf2)/delta_d + rf2 
    else:
        alfa = rf3
	
    alfa *=amax
	
	# parametros
    t_exp = t + first_day; #para alinhar o 1 dia dos dados da exponencial com o 1 dia do experimento
    ft = e*c0*np.exp(c1*t_exp)
		
    beta1 = m*r1
    beta2 = (1.0-m)*r2

    dSdt = -alfa*S*I
    dIdt = alfa*S*I + ft- beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt] 
    
def SIR_PP_V4_no_ft(P,t,amax,r,ti,delta,m,r1,r2, first_day, c0, c1):
    """
    Modelo final da NT02
    """
    S,I,R,M = P[0], P[1], P[2], P[3]

    # parametros
    t_exp = t + first_day; #para alinhar o 1 dia dos dados da exponencial com o 1 dia do experimento
    if t<ti:
        alfa=amax
    elif ti <= t and t<=ti+delta:
        alfa=amax*(1-r)/(-delta)*(t-ti) + amax
    else:
        alfa=amax*r

    beta1 = m*r1
    beta2 = (1.0-m)*r2

    dSdt = -alfa*S*I
    dIdt = alfa*S*I - beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt]  


def SIR_PP_Flex(P,t,amax,r,ti,delta,m,e,r1,r2, first_day, c0, c1, r_mult, shift, last_day):
    S,I,R,M = P[0], P[1], P[2], P[3]
    


    # parametros
    t_exp = t + first_day; #para alinhar o 1 dia dos dados da exponencial com o 1 dia do experimento
    ft = e*c0*np.exp(c1*t_exp)
    if t<ti:
        alfa=amax
    elif ti <= t and t<=min(ti+delta, last_day):
        alfa=amax*(1-r)/(-delta)*(t-ti) + amax
    elif t <= last_day:
        alfa=amax*r
    elif t > last_day and t <=last_day+shift:
        alfa=amax*r*(r_mult-1)/shift*(t-last_day) + amax*r
    else:
        alfa=amax*r*r_mult
        
    beta1 = m*r1
    beta2 = (1.0-m)*r2

    dSdt = -alfa*S*I
    #dSdt = beta1*S*I/P[0]
    dIdt = alfa*S*I + ft- beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt] 
    
def SIR_PP_Flex2(P,t,amax,r,ti,delta,m,e,r1,r2, first_day, c0, c1, r_mult, shift, last_day, delta_e):
    S,I,R,M = P[0], P[1], P[2], P[3]
    


    # parametros
    t_exp = t + first_day; #para alinhar o 1 dia dos dados da exponencial com o 1 dia do experimento
    
    #implementing e(t)
    if t <= last_day:
        aux_e = 1
    elif (t-last_day <= delta_e): 
        aux_e = (last_day - t)/delta_e + 1
    else: 
        aux_e = 0;
		
    ft = aux_e*e*c0*np.exp(c1*t_exp)
    
    if t<ti:
        alfa=amax
    elif ti <= t and t<=min(ti+delta, last_day):
        alfa=amax*(1-r)/(delta)*(t-ti) + amax
    elif t <= last_day:
        alfa=amax*r
    else:
        alfa=amax*r*(r_mult-1)/shift*(t-last_day) + amax*r
        
    beta1 = m*r1
    beta2 = (1.0-m)*r2

    dSdt = -alfa*S*I
    #dSdt = beta1*S*I/P[0]
    dIdt = alfa*S*I + ft- beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt] 

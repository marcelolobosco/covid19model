import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import integrate
from scipy.integrate import ode
from math import log as ln
from math import log10 as log
T = 250
t_points = 5000
t=np.linspace(0,T,t_points)
def model():
    A0 = 0.0
    AC0 = 0
    Ap0 = 1E06
    Apm0 = 0.0
    B0 = 2.5E05
    Bm0 = 0.0
    C0 = 0
    L0 = 1E6
    Ldead0 = 0
    Linfec0 = 0
    Pl0 = 0.0
    Ps0 = 0.0
    The0 = 0.0
    Thn0 = 1E06
    Tke0 = 0.0
    Tkn0 = 5E05
    Treg0 = 0
    V0 = 500
    alpha_Ap = 2.50E-03
    alpha_B = 6.0E+00
    alpha_Thn = 2.17E-04
    alpha_Tkn = 2.17E-04
    betaC_Apm = 1E-06*0
    betaC_Linfec = 1E-06*0
    betaC_The = 1E-06*0
    betaC_Tke = 1E-06*0
    beta_Ap = 5.5e-04
    beta_Bm = 1.0E-06
    beta_L = 5.61E-06
    beta_S = 0.000672
    beta_Thn = 1.0E-07
    beta_Tkn = 1.0E-05
    c_v1 = 2.63
    c_v2 = 0.60
    dl = 1E-04
    i = 1E-04
    k_ap1 = 0.8
    k_ap2 = 40.0
    m_A = 0.04
    m_Bm = 0.000975
    m_C = 0.05
    m_L = 0.00023976
    m_S = 2.0
    m_apm = 5.38E-01
    m_the = 0.22
    m_tke = 0.0003
    pi_AL = 0.00068
    pi_AS = 0.002
    pldead = 1E-07
    factor = 10
    plinfec = 1E-05/factor
    pv_a = 4.82E-05/factor
    pv_apm = 1E-05/factor
    pv_tke = 7.48E-05/factor
    r1_B = 4.826E-06
    r2_B = 1.27E-8
    r_bm1 = 1.0e-5
    r_bm2 = 2500.0
    r_the = 1E-08
    r_tke = 1.0E-08
    rv = 1
    rrv = 1000
    s_ap = 1E-06*0
    s_thn = 1E-06*0
    s_tkn = 1E-06*0

    def V(u, t):
        repV = rv*u[0]
        deathLinfec_V = dl*u[14]
        deathLinfec_releaseV = rrv*deathLinfec_V
        phagV_A = pv_a*u[0]*u[12]/(1 + u[0])
        phagV_Innate = (c_v1*u[0]/(1 + c_v2*u[0]))//(1 + u[0])
        #phagV_Innate = 0
        phagV_Tke = pv_tke*u[0]*u[7]/(1 + u[0])
        phagV_apm = pv_apm*u[0]*u[2]/(1 + u[0])
        infecL_V = i*u[13]*u[0]
        return repV + deathLinfec_releaseV - phagV_Innate - phagV_apm - phagV_A - phagV_Tke - infecL_V

    def Ap(u, t):
        act_Ap = beta_Ap*u[1]*(k_ap1*(u[0] + u[16])/(k_ap2 + (u[0] + u[16])))
        homeos_Ap = alpha_Ap*(Ap0 - u[1])
        mig_Ap = s_ap*u[16]*u[1]
        return homeos_Ap + mig_Ap - act_Ap

    def Apm(u, t):
        act_Ap = beta_Ap*u[1]*(k_ap1*(u[0] + u[16])/(k_ap2 + (u[0] + u[16])))
        death_Apm = m_apm*u[2]        
        return act_Ap - death_Apm

    def Thn(u, t):
        act_thn = beta_Thn*u[2]*u[3]
        homeos_Thn = alpha_Thn*(Thn0 - u[3])
        mig_Thn = s_thn*u[16]*u[3]
        return homeos_Thn + mig_Thn - act_thn

    def The(u, t):
        act_thn = beta_Thn*u[2]*u[3]
        phagV_apm = pv_apm*u[0]*u[2]/(1 + u[0])
        rep_the = r_the*phagV_apm*u[4]
        death_The = m_the*u[4]
        return act_thn + rep_the - death_The

    def Treg(u, t):
        return 0

    def Tkn(u, t):
        homeos_Tkn = alpha_Tkn*(Tkn0 - u[6])
        act_Tkn = beta_Tkn*u[2]*u[6]
        mig_Tkn = s_tkn*u[16]*u[6]
        return homeos_Tkn + mig_Tkn - act_Tkn

    def Tke(u, t):
        phagV_Tke = pv_tke*u[0]*u[7]
        act_Tkn = beta_Tkn*u[2]*u[6]
        rep_Tke = r_tke*phagV_Tke*u[7]
        death_Tke = m_tke*u[7]
        return act_Tkn + rep_Tke - death_Tke

    def B(u, t):
        rep_B = (r1_B*u[0] + r2_B*u[4])*u[8]    
        homeos_B = alpha_B*(B0 - u[8])
        diff_B = (beta_S*u[2] + beta_L*u[4] + beta_Bm*u[4])*u[8]
        return homeos_B + rep_B - diff_B

    def Ps(u, t):
        death_Ps = m_S*u[9]
        return beta_S*u[2]*u[8] - death_Ps

    def Pl(u, t):
        death_Pl = m_L*u[10]    
        return beta_L*u[4]*u[8] - death_Pl

    def Bm(u, t):
        rep_Bm = r_bm1*u[11]*(1 - u[11]/r_bm2)
        death_Bm = m_Bm*u[11]
        return beta_Bm*u[4]*u[8] + rep_Bm - death_Bm

    def A(u, t):
        prodA_Pl = pi_AL*u[10]
        prodA_Ps = pi_AS*u[9]
        death_A = m_A*u[12]
        return prodA_Ps + prodA_Pl - death_A

    def L(u, t):
        infecL_V = i*u[13]*u[0]
        return - infecL_V

    def Linfec(u, t):
        phagLinfec_Tke = plinfec*u[14]*u[7]
        infecL_V = i*u[13]*u[0]
        deathLinfec_V = dl*u[14]
        return infecL_V - deathLinfec_V - phagLinfec_Tke

    def Ldead(u, t):
        phagLdead_M = pldead*u[15]*(u[2] + u[1])
        deathLinfec_V = dl*u[14]
        return deathLinfec_V - phagLdead_M

    def C(u, t):
        phagV_apm = pv_apm*u[0]*u[2]/(1 + u[0])
        prodC = betaC_Linfec*u[14] + betaC_Apm*phagV_apm + betaC_The*u[4] + betaC_Tke*u[7]
        death_C = m_C*u[16]
        return prodC - death_C

    def AC(u, t):
        return 0

    def f(u,t):
        return V(u,t),Ap(u,t),Apm(u,t),Thn(u,t),The(u,t),Treg(u,t),Tkn(u,t),Tke(u,t),B(u,t),Ps(u,t),Pl(u,t),Bm(u,t),A(u,t),L(u,t),Linfec(u,t),Ldead(u,t),C(u,t),AC(u,t)

    u,d = integrate.odeint(f, [V0,Ap0,Apm0,Thn0,The0,Treg0,Tkn0,Tke0,B0,Ps0,Pl0,Bm0,A0,L0,Linfec0,Ldead0,C0,AC0], t, full_output=1, rtol=10e-6)

    return u[:,0],u[:,1],u[:,2],u[:,3],u[:,4],u[:,5],u[:,6],u[:,7],u[:,8],u[:,9],u[:,10],u[:,11],u[:,12],u[:,13],u[:,14],u[:,15],u[:,16],u[:,17]

results = {}
results['V'], results['Ap'], results['Apm'], results['Thn'], results['The'], results['Treg'], results['Tkn'], results['Tke'], results['B'], results['Ps'], results['Pl'], results['Bm'], \
results['A'], results['L'], results['Linfec'], results['Ldead'], results['C'], results['AC'] = model()
def savetxt(filename, t, M):
    fd = open(filename, "w")
    cont = 0
    for i in t:
        fd.write(str(M[cont]) + "\n")
        cont = cont + 1
    fd.close()
cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=180)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
color_index = 0
colorVal = scalarMap.to_rgba(color_index)
for k in results.keys():
    savetxt(str(k) + '.txt', t, results[k])
    fig = plt.figure(figsize=(6,4))
    plt.plot(t, results[k], label=str(k), color=colorVal)
    plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('concentration (1/mm³)')
    plt.legend()
    fig.savefig(str(k) + '.svg', format='svg', bbox_inches='tight')
    #fig.savefig(str(k) + '.jpg', format='jpg', bbox_inches='tight', fontsize=16)
    color_index += 10
    colorVal = scalarMap.to_rgba(color_index)
    
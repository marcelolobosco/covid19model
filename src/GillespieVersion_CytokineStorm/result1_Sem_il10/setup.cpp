// constants. c stands for creation, d for destruction, r for arn, p for protein
const double
	tmax = 50.0,
	dt = 0.1, // time step on the averages
    Ap_max = 1E06,
    Thn_max = 1E06,
    Tkn_max = 5E05,
    B_max = 2.5E05,
	alpha_Ap = 2.50E-03,
    alpha_B = 6, //6.0E+00,
    alpha_Thn = 2.17E-04,
    alpha_Treg = 0,
    alpha_Tkn = 2.17E-04*10,

    beta_Ap = 5.5e-01,
    beta_Bm = 1.0E-06,
    beta_Pl = 5.61E-06,
    beta_Ps = 0.000672,
    beta_Thn = 1.0E-07,
    beta_Thn_Treg = 1.0E-06,
    beta_Tkn = 1.0E-05,

    /*betaC_Apm = 1.0E-04,
    betaC_The = 1.0E-05,
    betaC_Tke = 1.0E-05, 
    betaC_B = 1.0E-05,
    betaC_Dead = 1.0E-05*0,    
    alphaC = 1, 
    alphaC_Apm = 0.1, 
    alphaC_The = 0.1,
    alphaC_Tke = 0.1,
    alphaC_B = 0.1,
    alphaC_IgM = 0.1,
    alphaC_IgG = 0.1,*/    
    betaTnf_Apm = 0.1,
    betaIl6_Apm = 0.1,
    betaIl6_The = 0.1,
    betaIl6_Tke = 0.1,
    betaIl6_B = 0.1,
    atnf_tnf = 0.1,
    ail6_tnf = 0.2,
    atnf_il6 = 0.1,
    ail10_il6 = 0.1, 
    ail6_il6 = 0.1,
    atnf_il10 = 0.1,
    ail10_il10 = 0.1,
    ail6_il10 = 0.1,

    ail6_The = 0.1, 
    ail6_Tke = 0.1, 
    ail6_B = 0.1, 
    ail6_IgM = 0.1, 
    ail6_IgG = 0.1, 

    c_v1 = 2.63, //1.5
    c_v2 = 0.60,    
    gamma_M = 9.75E-04,
    
    keqC = 50,
    k_ap1 = 0.8,
    k_ap2 = 40.0,
    k_bm1 = 1.0E-05,
    k_bm2 = 2500, 
    m_IgM = 0.8,
    m_IgG = 0.02,
    m_Bm = 0.000975,    
    m_Pl = 0.00023976,
    m_Ps = 2.0,
    m_apm = 0.538*2,//5.38E-01*10,
    m_the = 0.22*2,
    m_treg = 0.22*2,
    m_tke = 0.3,
    m_Tnf = 1,  //?
    m_Il6 = 0.33, //?
    m_Il10 = 1, //?
    Mtnf = 20,
    Mil6 = 200,
    Mil10 = 1,
    pi_AL = 0.00068,
    pi_AS = 0.002,
    pv_tke = 1E-05,//7.48E-07  #1.57E-03
    pv_a = 4.82E-05, //4.82E-05 #4.875E-04
    
    r1_B = 4.826E-06,
    r2_B = 1.27E-8,
    r_bm1 = 1E-05,
    r_bm2 = 2500.0,
    r_the = 1E-08,
    r_tke = 1.0E-08, //1E-08,
    rv = 1.3;//0.68;
const int n_runs = 5;

/*
# pi_v, c_v1, c_v2, k_v1, k_v2, alpha_Ap, beta_Ap, k_ap1,
    # k_ap2, delta_Apm, alpha_Tn, pi_T, k_te1, delta_te, alpha_B, pi_B1,   
    # pi_B2, beta_S, beta_L, beta_Bm, delta_S, delta_L, gamma_M, k_bm1, 
    # k_bm2, pi_AS, pi_AL, delta_A_G, delta_A_M, c11, c12, c13, 
    # c14, Ap0, Thn0, Tkn0, B0
    model_args = (x[1], 2.63, 0.60, x[2], x[3], 2.50E-03, 5.5e-01, 0.8, 
    40.0, x[4], 2.17E-04, x[5], x[6], 0.0003, x[7], 4.826E-06, 
    1.27E-8, 0.000672, 5.61E-06, 1.0E-06, 2.0, 2.3976E-04, 9.75E-04, 1.0e-5, 
    2500.0, 0.002, 0.00068, x[8], x[9], 2.17E-04, x[10], x[11], 
    0.22, 1.0e6, 1.0e6, 5.0e5, 2.5E5)
*/

std::random_device rddevice; // collects a seed from a random device
std::mt19937 mersenne_engine {rddevice()}; // Mersenne Twister
std::uniform_real_distribution<double> distribution (0.0,1.0);

namespace gillespie {

	double step(int* V, int* Ap, int *Apm, int *Thn, int *The, int *Tkn, int *Tke, int *B, \
    int *Ps, int *Pl, int *Bm, int *IgM, int *IgG, int *Tnf, int *Il6, int *Il10){
		double st,x,tau;

        /*double cytokines_infl_Apm = (1 + alphaC_Apm*(*C));
        double cytokines_infl_The = (1 + alphaC_The*(*C));
        double cytokines_infl_Tke = (1 + alphaC_Tke*(*C));
        double cytokines_infl_B = (1 + alphaC_B*(*C));
        double cytokines_infl_IgM = (1 + alphaC_IgM*(*C));
        double cytokines_infl_IgG = (1 + alphaC_IgG*(*C));*/

        double ytnf = betaTnf_Apm*(*Apm) + atnf_tnf*(*Tnf) + ail6_tnf*(*Il6);
        double pTnf = Mtnf/(1 + exp(-ytnf));
        double dTnf = m_Tnf*(*Tnf);

        double yil6 = - atnf_il6*(*Tnf) + ail10_il6*(*Il10) + ail6_il6*(*Il6) + \
        betaIl6_Apm*(*Apm) + betaIl6_The*(*The) + betaIl6_Tke*(*Tke) + betaIl6_B*(*B);
        double pIl6 = Mil6/(1 + exp(-yil6));
        double dIl6 = m_Il6*(*Il6);

        double il6_eff_The = 1 + ail6_The*(*Il6); //Mil6_The/(1 + exp(-ail6_The*(*Il6)))
        double il6_eff_Tke = 1 + ail6_Tke*(*Il6);
        double il6_eff_B = 1 + ail6_B*(*Il6);
        double il6_eff_IgM = 1 + ail6_IgM*(*Il6);
        double il6_eff_IgG = 1 + ail6_IgG*(*Il6);

        double yil10 = atnf_il10*(*Tnf) + ail10_il10*(*Il10) + ail6_il10*(*Il6);
        double pIl10 = Mil10/(1 + exp(-yil10))*0;
        double dIl10 = m_Il10*(*Il10)*0;

		double reV = rv*(*V);        
        double pVI = (c_v1*(*V)/(c_v2 + (*V)));
        double hAp = alpha_Ap*(Ap_max - (*Ap)); 
        double act_Ap = beta_Ap*(*Ap)*(k_ap1*(*V)/(k_ap2 + (*V)));
        double apmD = m_apm*(*Apm);
        double hThn = alpha_Thn*(Thn_max - (*Thn)); 
        double dfthe = beta_Thn*(*Apm)*(*Thn)*il6_eff_The;
        double rpthe = r_the*(*Apm)*(*The)*il6_eff_The;
        double dthe = m_the*(*The);
        double hTkn = alpha_Tkn*(Tkn_max - (*Tkn));
        double aTkn = beta_Tkn*(*Apm)*(*Tkn)*il6_eff_Tke;
        double rpTke = r_tke*(*Apm)*(*Tke)*il6_eff_Tke;
        double dTke = m_tke*(*Tke);   
        double pVTke = pv_tke*(*V)*(*Tke);
        double rpB = r2_B*(*The)*(*B)*il6_eff_B; 
        double hB = alpha_B*(1)*(B_max - (*B)); 
        double dfB_Ps = beta_Ps*(*Apm)*(*B)*il6_eff_B;
        double dPs = m_Ps*(*Ps);        
        double pIgM = pi_AS*(*Ps)*il6_eff_IgM; //produção de IgM
        double dA = m_IgM*(*IgM); //IgM 
        double pVA = pv_a*(*V)*(*IgM); //pVIgM

        double dfB_Pl = beta_Pl*(*The)*(*B)*il6_eff_B;
        double dfB_Bm = beta_Bm*(*The)*(*B)*il6_eff_B;
        double dPl = m_Pl*(*Pl);
        double memPl = gamma_M*(*Bm);
        double rBm = k_bm1*(*Bm)*(1 - (*Bm)/(k_bm2));
        double pIgG = pi_AL*(*Pl)*il6_eff_IgG;
        double pVIgG = pv_a*(*V)*(*IgG); //neutralização do vírus 
        double dIgG = m_IgG*(*IgG);

		st = reV + pVI + hAp + act_Ap + apmD + hThn + dfthe + rpthe + dthe + hTkn + aTkn \
        + rpTke + dTke + pVTke + rpB + hB + dfB_Ps + dPs + pIgM + dA + pVA + dfB_Pl + dfB_Bm \
        + dPl + memPl + rBm + pIgG + pVIgG + dIgG + pTnf + dTnf + pIl6 + dIl6 + pIl10 + dIl10; 
		x = distribution(mersenne_engine);
		tau = -std::log(distribution(mersenne_engine))/st;

		if(x < (reV)/st){
			++(*V);
		} else if(x < (reV+pVI)/st){
			--(*V);
		} else if(x < (reV+pVI+hAp)/st){
			++(*Ap);
		} 
        else if(x < (reV+pVI+hAp+act_Ap)/st){
			--(*Ap);
            ++(*Apm);
		} 
        else if(x < (reV+pVI+hAp+act_Ap+apmD)/st){
			--(*Apm);
            //++(*Dead);
		} 
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn)/st){
			++(*Thn);            
		} 
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe)/st){
			--(*Thn); 
            ++(*The);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe)/st){			
            ++(*The);
		} 
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe)/st){
			--(*The);
            //++(*Dead);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn)/st){
			++(*Tkn);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn)/st){
			--(*Tkn);
            ++(*Tke);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke)/st){			
            ++(*Tke);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke)/st){
			--(*Tke); 
            //++(*Dead);           
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke)/st){
			--(*V);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        )/st){
			++(*B);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB)/st){
			++(*B);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps)/st){
			--(*B);
            ++(*Ps);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs)/st){			
            --(*Ps);
            //++(*Dead);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM)/st){			
            ++(*IgM);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA)/st){
            --(*IgM);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA)/st){
            --(*V);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl)/st){
            --(*B);
            ++(*Pl);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm)/st){
            --(*B);
            ++(*Bm);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl)/st){
            --(*Pl);            
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl)/st){
            --(*Bm);
            ++(*Pl);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm)/st){
            ++(*Bm);            
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG)/st){
            ++(*IgG);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG)/st){
            --(*V);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG)/st){         
            --(*IgG);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pTnf)/st){
            ++(*Tnf);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pTnf+dTnf)/st){
            --(*Tnf);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pTnf+dTnf+pIl6)/st){
            ++(*Il6);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pTnf+dTnf+pIl6\
        +dIl6)/st){
            --(*Il6);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pTnf+dTnf+pIl6\
        +dIl6+pIl10)/st){
            ++(*Il10);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pTnf+dTnf+pIl6\
        +dIl6+pIl10+dIl10)/st){
            --(*Il10);
		}
		return tau;
	}

	void delta(int* V, int* Ap, int *Apm, int *Thn, int *The, int *Tkn, int *Tke, int *B, \
    int *Ps, int *Pl, int *Bm, int *IgM, int *IgG, int *Tnf, int *Il6, int *Il10, double time_delta){
		double time = 0.0;
		while(time < time_delta){
			time += step(V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,IgM,IgG,Tnf,Il6,Il10);
		}
	}

}

template <typename T, typename Iterator>
double mean(Iterator begin, Iterator end, T coll_size){
	return (double)(std::accumulate(begin,end,0.0))/coll_size;
}

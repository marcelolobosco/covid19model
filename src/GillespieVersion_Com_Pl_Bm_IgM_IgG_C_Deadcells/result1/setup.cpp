// constants. c stands for creation, d for destruction, r for arn, p for protein
const double
	tmax = 60.0,
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

    betaC_Apm = 1.0E-02,
    betaC_The = 1.0E-03,
    betaC_Tke = 1.0E-04, 
    betaC_B = 1.0E-04,
    betaC_Dead = 1.0E-07,
    alphaC = 0.5, //0.5    
    alphaC_The = 1,
    alphaC_Tke = 5,
    alphaC_B = 1,

    c_v1 = 1,
    c_v2 = 0.60,    
    gamma_M = 9.75E-04,
    
    keqC = 1000,
    k_ap1 = 0.8,
    k_ap2 = 40.0,
    k_bm1 = 1.0E-05,
    k_bm2 = 2500, 
    m_A = 0.5,
    m_IgG = 0.1,
    m_Bm = 0.000975,    
    m_Pl = 0.00023976,
    m_Ps = 2.0,
    m_apm = 0.538*3,//5.38E-01*10,
    m_the = 0.22,
    m_treg = 0.22,
    m_tke = 0.3*3,
    m_C = 7,
    pi_AL = 0.00068,
    pi_AS = 0.002,
    pv_tke = 1.5E-03,//7.48E-07  #1.57E-03
    pv_a = 1E-06, //4.82E-05 #4.875E-04
    
    r1_B = 4.826E-06,
    r2_B = 1.27E-8,
    r_bm1 = 1E-05,
    r_bm2 = 2500.0,
    r_the = 1E-08,
    r_tke = 5.0E-07, //1E-08,
    rv = 1.3;//0.68;
const int n_runs = 10;

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
    int *Ps, int *Pl, int *Bm, int *IgM, int *IgG, int *C, int *Dead){
		double st,x,tau;

        double cytokines_infl = (1 + (*C)/(keqC + alphaC*(*C)));

		double reV = rv*(*V);        
        double pVI = (c_v1*(*V)/(c_v2 + (*V)));
        double hAp = alpha_Ap*(1)*(Ap_max - (*Ap));
        double act_Ap = beta_Ap*(*Ap)*(k_ap1*(*V)/(k_ap2 + (*V)));
        double apmD = m_apm*(*Apm);
        double hThn = alpha_Thn*(1)*(Thn_max - (*Thn));
        double dfthe = beta_Thn*(*Apm)*(*Thn)*alphaC_The*cytokines_infl;
        double rpthe = r_the*(*Apm)*(*The)*alphaC_The*cytokines_infl;
        double dthe = m_the*(*The);
        double hTkn = alpha_Tkn*(1)*(Tkn_max - (*Tkn));
        double aTkn = beta_Tkn*(*Apm)*(*Tkn)*alphaC_Tke*cytokines_infl;
        double rpTke = r_tke*(*Apm)*(*Tke)*alphaC_Tke*cytokines_infl;
        double dTke = m_tke*(*Tke);   
        double pVTke = pv_tke*(*V)*(*Tke);
        double rpB = r2_B*(*The)*(*B)*alphaC_B*cytokines_infl;
        double hB = alpha_B*(1)*(B_max - (*B)); 
        double dfB_Ps = beta_Ps*(*Apm)*(*B)*alphaC_B*cytokines_infl;
        double dPs = m_Ps*(*Ps);        
        double pIgM = pi_AS*(*Ps)*cytokines_infl; //IgM
        double dA = m_A*(*IgM); //IgM 
        double pVA = pv_a*(*V)*(*IgM); //pVIgM

        double dfB_Pl = beta_Pl*(*The)*(*B)*cytokines_infl;
        double dfB_Bm = beta_Bm*(*The)*(*B);
        double dPl = m_Pl*(*Pl);
        double memPl = gamma_M*(*Bm);
        double rBm = k_bm1*(*Bm)*(1 - (*Bm)/(k_bm2));
        double pIgG = pi_AL*(*Pl)*cytokines_infl; //produção de IgG
        double pVIgG = pv_a*(*V)*(*IgG); //neutralização do vírus 
        double dIgG = m_IgG*(*IgG);
        double pC_Apm = betaC_Apm*(*Apm);//*(*V);
        double pC_The = betaC_The*(*The);//*(*V);
        double pC_Tke = betaC_Tke*(*Tke);//*(*V);
        double pC_B = betaC_B*(*B);//*(*V);
        double dC = m_C*(*C);

        double pC_Dead = betaC_Dead*(*Dead);        

		st = reV + pVI + hAp + act_Ap + apmD + hThn + dfthe + rpthe + dthe + hTkn + aTkn \
        + rpTke + dTke + pVTke + rpB + hB + dfB_Ps + dPs + pIgM + dA + pVA + dfB_Pl + dfB_Bm \
        + dPl + memPl + rBm + pIgG + pVIgG + dIgG + pC_Apm + pC_The + pC_Tke + pC_B + dC + 
        pC_Dead;
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
            ++(*Dead);
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
            ++(*Dead);           
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
            ++(*Dead);
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
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pC_Apm)/st){
            ++(*C);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pC_Apm+pC_The)/st){
            ++(*C);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pC_Apm+pC_The+pC_Tke)/st){
            ++(*C);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pC_Apm+pC_The+pC_Tke \
        +pC_B)/st){
            ++(*C);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pC_Apm+pC_The+pC_Tke \
        +pC_B+dC)/st){
            --(*C);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+pIgM+dA+pVA+dfB_Pl+dfB_Bm+dPl+memPl+rBm+pIgG+pVIgG+dIgG+pC_Apm+pC_The+pC_Tke \
        +pC_B+dC+pC_Dead)/st){
            ++(*C);
		}
		return tau;
	}

	void delta(int* V, int* Ap, int *Apm, int *Thn, int *The, int *Tkn, int *Tke, int *B, \
    int *Ps, int *Pl, int *Bm, int *IgM, int *IgG, int *C, int *Dead, double time_delta){
		double time = 0.0;
		while(time < time_delta){
			time += step(V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,IgM,IgG,C,Dead);
		}
	}

}

template <typename T, typename Iterator>
double mean(Iterator begin, Iterator end, T coll_size){
	return (double)(std::accumulate(begin,end,0.0))/coll_size;
}

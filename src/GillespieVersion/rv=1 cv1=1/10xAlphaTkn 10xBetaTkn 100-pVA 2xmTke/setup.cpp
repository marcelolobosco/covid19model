// constants. c stands for creation, d for destruction, r for arn, p for protein
const double
	tmax = 40.0,
	dt = 0.1, // time step on the averages
    Ap_max = 1E06,
    Thn_max = 1E06,
    Tkn_max = 5E05,
    B_max = 2.5E05,
	alpha_Ap = 2.50E-03,
    alpha_B = 6.0E+00,
    alpha_Thn = 2.17E-04,
    alpha_Treg = 0,
    alpha_Tkn = 2.17E-04*10,

    beta_Ap = 5.5e-01,
    beta_Bm = 1.0E-06,
    beta_Pl = 5.61E-06,
    beta_Ps = 0.000672,
    beta_Thn = 1.0E-07,
    beta_Thn_Treg = 1.0E-06,
    beta_Tkn = 1.0E-05*10,

    c_v1 = 1.,
    c_v2 = 0.60,    
    
    k_ap1 = 0.8,
    k_ap2 = 40.0,
    m_A = 0.04,
    m_Bm = 0.000975,    
    m_Pl = 0.00023976,
    m_Ps = 2.0,
    m_apm = 5.38E-01,  
    m_the = 0.22,
    m_treg = 0.22,
    m_tke = 0.3*2,
    pi_AL = 0.00068,
    pi_AS = 0.002,
    pldead = 1E-05,
    plinfec = 1.57E-03,
    pv_tke = 1.57E-03, //7.48E-07 #1.57E-03
    pv_a = 4.87E-04/100, //4.82E-05 #4.875E-04
    pv_apm = 1E-04,

    r1_B = 4.826E-06,
    r2_B = 1.27E-8,
    r_bm1 = 1E-05,
    r_bm2 = 2500.0,
    r_the = 1E-08,
    r_tke = 1E-08,
    rv = 1;
const int n_runs = 10;

std::random_device random_device; // collects a seed from a random device
std::mt19937 mersenne_engine {random_device()}; // Mersenne Twister
std::uniform_real_distribution<double> distribution (0.0,1.0);

namespace gillespie {

	double step(int* V, int* Ap, int *Apm, int *Thn, int *The, int *Tkn, int *Tke, int *B, int *Ps, int *A){
		double st,x,tau;
		
		double reV = rv*(*V);        
        double pVI = (c_v1*(*V)/(1 + c_v2*(*V)));
        double hAp = alpha_Ap*(1)*(Ap_max - (*Ap));
        double act_Ap = beta_Ap*(*Ap)*(k_ap1*(*V)/(k_ap2 + (*V)));
        double apmD = m_apm*(*Apm);
        double hThn = alpha_Thn*(1)*(Thn_max - (*Thn));
        double dfthe = beta_Thn*(*Apm)*(*Thn);        
        double rpthe = r_the*(*Apm)*(*The);
        double dthe = m_the*(*The);
        double hTkn = alpha_Tkn*(1)*(Tkn_max - (*Tkn));
        double aTkn = beta_Tkn*(*Apm)*(*Tkn);        
        double rpTke = r_tke*(*Apm)*(*Tke);
        double dTke = m_tke*(*Tke);   
        double pVTke = pv_tke*(*V)*(*Tke);     
        double rpB = r2_B*(*The)*(*B);        
        double hB = alpha_B*(1)*(B_max - (*B)); 
        double dfB_Ps = beta_Ps*(*Apm)*(*B); 
        double dPs = m_Ps*(*Ps);        
        double prodPs = pi_AS*(*Ps);
        double dA = m_A*(*A);
        double pVA = pv_a*(*V)*(*A);

		st = reV + pVI + hAp + act_Ap + apmD + hThn + dfthe + rpthe + dthe + hTkn + aTkn \
        + rpTke + dTke + pVTke + rpB + hB + dfB_Ps + dPs + prodPs + dA + pVA; 
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
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+prodPs)/st){			
            ++(*A);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+prodPs+dA)/st){
            --(*A);
		}
        else if(x < (reV+pVI+hAp+act_Ap+apmD+hThn+dfthe+rpthe+dthe+hTkn+aTkn+rpTke+dTke+pVTke+rpB \
        +hB+dfB_Ps+dPs+prodPs+dA+pVA)/st){
            --(*V);
		}
		return tau;
	}

	void delta(int* V, int* Ap, int *Apm, int *Thn, int *The, int *Tkn, int *Tke, int *B, int *Ps, int *A, double time_delta){
		double time = 0.0;
		while(time < time_delta){
			time += step(V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,A);
		}
	}

}

template <typename T, typename Iterator>
double mean(Iterator begin, Iterator end, T coll_size){
	return (double)(std::accumulate(begin,end,0.0))/coll_size;
}

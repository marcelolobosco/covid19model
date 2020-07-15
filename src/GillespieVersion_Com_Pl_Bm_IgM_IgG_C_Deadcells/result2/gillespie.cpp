#include <iostream> // print things, cout
#include <fstream> // save to files, ofstream
#include <array>
#include <random>
#include <string> // unused for now, stoi
#include <cmath> // log
#include <algorithm> // accumulate, fill
#include <sstream>
#include "setup.cpp"

using namespace std; 

int main(){
	const int time_size = (int)(tmax/dt)+1;
	std::array<double,time_size> t_means,v_means,ap_means,apm_means,thn_means,the_means, \
	tkn_means,tke_means,b_means,ps_means,pl_means,bm_means,igm_means,igg_means,c_means;
	v_means[0] = 0.;
	ap_means[0] = 0.; 
	apm_means[0] = 0.;
	thn_means[0] = 0.;
	the_means[0] = 0.;
	tkn_means[0] = 0.;
	tke_means[0] = 0.;
	b_means[0] = 0.;
	ps_means[0] = 0.;
	pl_means[0] = 0.;
	bm_means[0] = 0.;
	igm_means[0] = 0.;
	igg_means[0] = 0.;
	c_means[0] = 0.;

	std::iota(t_means.begin(),t_means.end(),0.0);
	std::transform(t_means.begin(),t_means.end(),t_means.begin(),[=](double d){return d*dt;});

	// each of this array will contain the result of an individual cells
	std::array<int,n_runs> V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,IgM,IgG,C,Dead;
	std::fill(V.begin(),V.end(),1000);
	std::fill(Ap.begin(),Ap.end(),1.0E06);
	std::fill(Apm.begin(),Apm.end(),0);
	std::fill(Thn.begin(),Thn.end(),0);//1.E06);
	std::fill(The.begin(),The.end(),0);
	std::fill(Tkn.begin(),Tkn.end(),0);//5.E05);
	std::fill(Tke.begin(),Tke.end(),0);
	std::fill(B.begin(),B.end(),0);//2.5E05);
	std::fill(Ps.begin(),Ps.end(),0);
	std::fill(Pl.begin(),Pl.end(),0);
	std::fill(Bm.begin(),Bm.end(),0);
	std::fill(IgM.begin(),IgM.end(),0);
	std::fill(IgG.begin(),IgG.end(),0);
	std::fill(C.begin(),C.end(),0);
	std::fill(Dead.begin(),Dead.end(),0);

	int r,p;
	std::ofstream Vfiles[n_runs], Apfiles[n_runs], Apmfiles[n_runs], \
	Thnfiles[n_runs], Thefiles[n_runs], Tknfiles[n_runs], Tkefiles[n_runs], \
	Bfiles[n_runs], Psfiles[n_runs], Plfiles[n_runs], Bmfiles[n_runs], IgMfiles[n_runs],
	IgGfiles[n_runs], Cfiles[n_runs], Deadfiles[n_runs];
	for(int j=0; j<n_runs; j++){
		stringstream virusfilename, apfilename, apmfilename, thnfilename, thefilename, \
		tknfilename, tkefilename, bfilename, psfilename, plfilename, bmfilename, igmfilename, 
		iggfilename, cfilename, deadfilename;
		virusfilename << "V" << j << ".dat";
		apfilename << "Ap" << j << ".dat";
		apmfilename << "Apm" << j << ".dat";
		thnfilename << "Thn" << j << ".dat";
		thefilename << "The" << j << ".dat";
		tknfilename << "Tkn" << j << ".dat";
		tkefilename << "Tke" << j << ".dat";
		bfilename << "B" << j << ".dat";
		psfilename << "Ps" << j << ".dat";
		plfilename << "Pl" << j << ".dat";
		bmfilename << "Bm" << j << ".dat";
		igmfilename << "IgM" << j << ".dat";
		iggfilename << "IgG" << j << ".dat";
		cfilename << "C" << j << ".dat";
		deadfilename << "Dead" << j << ".dat";
		Vfiles[j].open(virusfilename.str());
		Apfiles[j].open(apfilename.str());
		Apmfiles[j].open(apmfilename.str());
		Thnfiles[j].open(thnfilename.str());
		Thefiles[j].open(thefilename.str());
		Tknfiles[j].open(tknfilename.str());
		Tkefiles[j].open(tkefilename.str());
		Bfiles[j].open(bfilename.str());
		Psfiles[j].open(psfilename.str());
		Plfiles[j].open(plfilename.str());
		Bmfiles[j].open(bmfilename.str());
		IgMfiles[j].open(igmfilename.str());
		IgGfiles[j].open(iggfilename.str());
		Cfiles[j].open(cfilename.str());
		Deadfiles[j].open(deadfilename.str());
	}

	for(int j=0; j<n_runs; j++){
		Vfiles[j] << V[j] << endl; 
		Apfiles[j] << Ap[j] << endl;
		Apmfiles[j] << Apm[j] << endl;
		Thnfiles[j] << Thn[j] << endl;
		Thefiles[j] << The[j] << endl;
		Tknfiles[j] << Tkn[j] << endl;
		Tkefiles[j] << Tke[j] << endl;
		Bfiles[j] << B[j] << endl;
		Psfiles[j] << Ps[j] << endl;
		Plfiles[j] << Pl[j] << endl;
		Bmfiles[j] << Bm[j] << endl;
		IgMfiles[j] << IgM[j] << endl;
		IgGfiles[j] << IgG[j] << endl;
		Cfiles[j] << C[j] << endl; 
		Deadfiles[j] << Dead[j] << endl;
	}

	v_means[0] = mean(V.begin(),V.end(),n_runs);
	ap_means[0] = mean(Ap.begin(),Ap.end(),n_runs);
	apm_means[0] = mean(Apm.begin(),Apm.end(),n_runs);
	thn_means[0] = mean(Thn.begin(),Thn.end(),n_runs);
	the_means[0] = mean(The.begin(),The.end(),n_runs);
	tkn_means[0] = mean(Tkn.begin(),Tkn.end(),n_runs);
	tke_means[0] = mean(Tke.begin(),Tke.end(),n_runs);
	b_means[0] = mean(B.begin(),B.end(),n_runs);
	ps_means[0] = mean(Ps.begin(),Ps.end(),n_runs);
	pl_means[0] = mean(Pl.begin(),Pl.end(),n_runs);
	bm_means[0] = mean(Bm.begin(),Bm.end(),n_runs);
	igm_means[0] = mean(IgM.begin(),IgM.end(),n_runs);
	igg_means[0] = mean(IgG.begin(),IgG.end(),n_runs);

	for(int i=1; i<time_size; i++){
		#pragma omp parallel
		{
			#pragma omp for //schedule(dynamic)
			for(int j=0; j<n_runs; j++){
				gillespie::delta(&V[j],&Ap[j],&Apm[j],&Thn[j],&The[j],&Tkn[j],&Tke[j], \
				&B[j], &Ps[j], &Pl[j], &Bm[j], &IgM[j], &IgG[j], &C[j], &Dead[j], dt);
				Vfiles[j] << V[j] << endl; 
				Apfiles[j] << Ap[j] << endl;
				Apmfiles[j] << Apm[j] << endl;
				Thnfiles[j] << Thn[j] << endl;
				Thefiles[j] << The[j] << endl;
				Tknfiles[j] << Tkn[j] << endl;
				Tkefiles[j] << Tke[j] << endl;
				Bfiles[j] << B[j] << endl;
				Psfiles[j] << Ps[j] << endl;
				Plfiles[j] << Pl[j] << endl;
				Bmfiles[j] << Bm[j] << endl;
				IgMfiles[j] << IgM[j] << endl;
				IgGfiles[j] << IgG[j] << endl;
				Cfiles[j] << C[j] << endl; 
				Deadfiles[j] << Dead[j] << endl;
			}
		}		
		v_means[i] = mean(V.begin(),V.end(),n_runs);
		ap_means[i] = mean(Ap.begin(),Ap.end(),n_runs);
		apm_means[i] = mean(Apm.begin(),Apm.end(),n_runs);
		thn_means[i] = mean(Thn.begin(),Thn.end(),n_runs);
		the_means[i] = mean(The.begin(),The.end(),n_runs);
		tkn_means[i] = mean(Tkn.begin(),Tkn.end(),n_runs);
		tke_means[i] = mean(Tke.begin(),Tke.end(),n_runs);
		b_means[i] = mean(B.begin(),B.end(),n_runs);
		ps_means[i] = mean(Ps.begin(),Ps.end(),n_runs);
		pl_means[i] = mean(Pl.begin(),Pl.end(),n_runs);
		bm_means[i] = mean(Bm.begin(),Bm.end(),n_runs);
		igm_means[i] = mean(IgM.begin(),IgM.end(),n_runs);
		igg_means[i] = mean(IgG.begin(),IgG.end(),n_runs);
		std::cout << "\r" << (int)t_means[i] << "/" << (int)tmax << std::flush;
	}

	for(int j=0; j<n_runs; j++){
		Vfiles[j].close();
		Apfiles[j].close();
		Apmfiles[j].close();
		Thnfiles[j].close();
		Thefiles[j].close();
		Tknfiles[j].close();
		Tkefiles[j].close();
		Bfiles[j].close();
		Psfiles[j].close();
		Plfiles[j].close();
		Bmfiles[j].close();
		IgMfiles[j].close();
		IgGfiles[j].close();
		Cfiles[j].close();
		Deadfiles[j].close();
	}

	std::ofstream output_v("V_mean.dat");
	std::ofstream output_ap("Ap_mean.dat");
	std::ofstream output_apm("Apm_mean.dat");
	std::ofstream output_the("The_mean.dat");
	std::ofstream output_tke("Tke_mean.dat");
	std::ofstream output_ps("Ps_mean.dat");
	std::ofstream output_pl("Pl_mean.dat");
	std::ofstream output_bm("Bm_mean.dat");
	std::ofstream output_igm("IgM_mean.dat");
	std::ofstream output_igg("IgG_mean.dat");
	std::ofstream output_c("C_mean.dat");
	for(int i=0; i<time_size; i++){
		output_v << t_means[i] << "\t" << v_means[i] << "\n";
		output_ap << t_means[i] << "\t" << ap_means[i] << "\n";
		output_apm << t_means[i] << "\t" << apm_means[i] << "\n";
		output_the << t_means[i] << "\t" << the_means[i] << "\n";
		output_tke << t_means[i] << "\t" << tke_means[i] << "\n";
		output_ps << t_means[i] << "\t" << ps_means[i] << "\n";
		output_pl << t_means[i] << "\t" << pl_means[i] << "\n";
		output_bm << t_means[i] << "\t" << bm_means[i] << "\n";
		output_igm << t_means[i] << "\t" << igm_means[i] << "\n";
		output_igg << t_means[i] << "\t" << igg_means[i] << "\n";
		output_c << t_means[i] << "\t" << c_means[i] << "\n";
	}
	
	std::cout << "\nFnished" << std::endl;
	return 0;
}

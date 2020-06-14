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
	tkn_means,tke_means,b_means,ps_means,a_means;
	v_means[0] = 0.;
	ap_means[0] = 0.; 
	apm_means[0] = 0.;
	thn_means[0] = 0.;
	the_means[0] = 0.;
	tkn_means[0] = 0.;
	tke_means[0] = 0.;
	b_means[0] = 0.;
	ps_means[0] = 0.;
	a_means[0] = 0.;

	std::iota(t_means.begin(),t_means.end(),0.0);
	std::transform(t_means.begin(),t_means.end(),t_means.begin(),[=](double d){return d*dt;});

	// each of this array will contain the result of an individual cells
	std::array<int,n_runs> V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,A;	
	std::fill(V.begin(),V.end(),500);
	std::fill(Ap.begin(),Ap.end(),1E06);
	std::fill(Apm.begin(),Apm.end(),0);
	std::fill(Thn.begin(),Thn.end(),0);
	std::fill(The.begin(),The.end(),0);
	std::fill(Tkn.begin(),Tkn.end(),0);
	std::fill(Tke.begin(),Tke.end(),0);
	std::fill(B.begin(),B.end(),0);
	std::fill(Ps.begin(),Ps.end(),0);
	std::fill(A.begin(),A.end(),0);

	int r,p;
	std::ofstream Vfiles[n_runs], Apfiles[n_runs], Apmfiles[n_runs], \
	Thnfiles[n_runs], Thefiles[n_runs], Tknfiles[n_runs], Tkefiles[n_runs], \
	Bfiles[n_runs], Psfiles[n_runs], Afiles[n_runs];
	for(int j=0; j<n_runs; j++){
		stringstream virusfilename, apfilename, apmfilename, thnfilename, thefilename, \
		tknfilename, tkefilename, bfilename, psfilename, afilename;
		virusfilename << "V" << j << ".dat";
		apfilename << "Ap" << j << ".dat";
		apmfilename << "Apm" << j << ".dat";
		thnfilename << "Thn" << j << ".dat";
		thefilename << "The" << j << ".dat";
		tknfilename << "Tkn" << j << ".dat";
		tkefilename << "Tke" << j << ".dat";
		bfilename << "B" << j << ".dat";
		psfilename << "Ps" << j << ".dat";
		afilename << "A" << j << ".dat";
		Vfiles[j].open(virusfilename.str());
		Apfiles[j].open(apfilename.str());
		Apmfiles[j].open(apmfilename.str());
		Thnfiles[j].open(thnfilename.str());
		Thefiles[j].open(thefilename.str());
		Tknfiles[j].open(tknfilename.str());
		Tkefiles[j].open(tkefilename.str());
		Bfiles[j].open(bfilename.str());
		Psfiles[j].open(psfilename.str());
		Afiles[j].open(afilename.str());
	}

	for(int i=1; i<time_size; i++){
		#pragma omp parallel
		{
			#pragma omp for //schedule(dynamic)
			for(int j=0; j<n_runs; j++){
				gillespie::delta(&V[j],&Ap[j],&Apm[j],&Thn[j],&The[j],&Tkn[j],&Tke[j], \
				&B[j], &Ps[j], &A[j], dt);
				Vfiles[j] << V[j] << endl; 
				Apfiles[j] << Ap[j] << endl;
				Apmfiles[j] << Apm[j] << endl;
				Thnfiles[j] << Thn[j] << endl;
				Thefiles[j] << The[j] << endl;
				Tknfiles[j] << Tkn[j] << endl;
				Tkefiles[j] << Tke[j] << endl;
				Bfiles[j] << B[j] << endl;
				Psfiles[j] << Ps[j] << endl;
				Afiles[j] << A[j] << endl;
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
		a_means[i] = mean(A.begin(),A.end(),n_runs);
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
		Afiles[j].close();
	}

	std::ofstream output_v("V_mean.dat");
	std::ofstream output_ap("Ap_mean.dat");
	std::ofstream output_apm("Apm_mean.dat");
	std::ofstream output_the("The_mean.dat");
	std::ofstream output_tke("Tke_mean.dat");
	std::ofstream output_ps("Ps_mean.dat");
	std::ofstream output_a("A_mean.dat");
	for(int i=0; i<time_size; i++){
		output_v << t_means[i] << "\t" << v_means[i] << "\n";
		output_ap << t_means[i] << "\t" << ap_means[i] << "\n";
		output_apm << t_means[i] << "\t" << apm_means[i] << "\n";
		output_the << t_means[i] << "\t" << the_means[i] << "\n";
		output_tke << t_means[i] << "\t" << tke_means[i] << "\n";
		output_ps << t_means[i] << "\t" << ps_means[i] << "\n";
		output_a << t_means[i] << "\t" << a_means[i] << "\n";
	}
	
	std::cout << "\nFnished" << std::endl;
	return 0;
}

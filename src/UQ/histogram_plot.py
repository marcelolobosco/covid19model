from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import sys

#global settings
ext = '.png'
output_path = './'



if __name__ == "__main__":
	
	if(len(sys.argv) != 2):
		print("Usage: histogram_plot [arquivo_sampes]")
		sys.exit(0)
	arqsamples = sys.argv[1]
	
	opt_scatter = True
	opt_hitogram = False;
	
	
	m = np.loadtxt(arqsamples, comments='#');
	
	
	num_rows, num_cols = m.shape
	print(num_rows, num_cols)

	m=m[m[:,0].argsort()]
	m=m[:25000]
	num_rows, num_cols = m.shape
	print(num_rows, num_cols)    
	m = m[:,1:]

	colors = np.array([0.0, 0.0, 0.0])
	area = np.pi

	labels = ('v0', 'pi_c_apm', 'pi_c_i', 'pi_c_tke', 'delta_c', 'beta_apm',
        'k_v3','beta_tke','pi_v','k_v1','k_v2','beta_Ap')
	vsize = len(labels)
	
	if (opt_scatter):
		# scatterplot
		fig, ax = plt.subplots(vsize, vsize, figsize=(60,60))

		for i in range(vsize):
			for j in range (vsize):
				if (j<i):
					ax[i][j].set_ylim(m[:,j].min()*0.9,m[:,j].max()*1.1)
					ax[i][j].set_xlim(m[:,i].min()*0.9,m[:,i].max()*1.1)
					ax[i][j].scatter(m[:,i], m[:,j],c='black', s=area, alpha=0.3)
					ax[i][j].set_xlabel(labels[i])
					ax[i][j].set_ylabel(labels[j])
				elif(i==j): 			
					ax[i][j].hist(m[:,i], bins=20, density=True, color='gray')
					ax[i][j].set_xlabel(labels[i])
					print(labels[i]+"\t{:.1e}".format(m[:,i].mean())+"\t{:.1e}".format(np.std(m[:,i])))
				else:    
					ax[i][j].set_visible(False)                

		plt.savefig(output_path+'scatter'+ext)


	if (opt_hitogram):
		# 1 figure for the histogram of each parameter
		for i in range(vsize):
			print(f"saving histogram for {labels[i]}")
			plt.figure(i)
			sb.distplot(m[:,i], hist=False, norm_hist=False, )
			plt.tick_params(labelsize=14)
			plt.xlabel(labels[i],fontsize=16)
			plt.ylabel('Density',fontsize=16)
			hname = 'fig_hist_' + labels[i] +ext
			plt.savefig(output_path+hname)

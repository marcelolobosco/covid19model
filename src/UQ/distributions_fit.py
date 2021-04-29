import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
import pandas as pd
import sys

plt.style.use('ggplot')


def fit_scipy_distributions(array, bins, data_name, plot_hist = True, save_hist=False, plot_best_fit = True, plot_all_fits = False):
    """
    Fits a range of Scipy's distributions (see scipy.stats) against an array-like input.
    Returns the sum of squared error (SSE) between the fits and the actual distribution.
    Can also choose to plot the array's histogram along with the computed fits.
    N.B. Modify the "CHANGE IF REQUIRED" comments!
    
    Input: array - array-like input
           bins - number of bins wanted for the histogram
           plot_hist - boolean, whether you want to show the histogram
           plot_best_fit - boolean, whether you want to overlay the plot of the best fitting distribution
           plot_all_fits - boolean, whether you want to overlay ALL the fits (can be messy!)
    
    Returns: results - dataframe with SSE and distribution name, in ascending order (i.e. best fit first)
             best_name - string with the name of the best fitting distribution
             best_params - list with the parameters of the best fitting distribution.
    """
    
    if plot_best_fit or plot_all_fits:
        assert plot_hist, "plot_hist must be True if setting plot_best_fit or plot_all_fits to True"
    
    # Returns un-normalised (i.e. counts) histogram
    y, x = np.histogram(np.array(array), bins=bins)
    
    # Some details about the histogram
    bin_width = x[1]-x[0]
    N = len(array)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0 # go from bin edges to bin middles
    
    # selection of available distributions
    # CHANGE THIS IF REQUIRED
    DISTRIBUTIONS = [st.norm,st.lognorm,st.uniform]

    if plot_hist:
        fig, ax = plt.subplots()
        h = ax.hist(np.array(array), bins = bins, color = 'w')

    # loop through the distributions and store the sum of squared errors
    # so we know which one eventually will have the best fit
    sses = []
    for dist in tqdm(DISTRIBUTIONS):
        name = dist.__class__.__name__[:-4]

        params = dist.fit(np.array(array))
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
        pdf_scaled = pdf * bin_width * N # to go from pdf back to counts need to un-normalise the pdf

        sse = np.sum((y - pdf_scaled)**2)
        sses.append([sse, name])

        # Not strictly necessary to plot, but pretty patterns
        if plot_all_fits:
            ax.plot(x_mid, pdf_scaled, label = name+' - SSE ='+"{:10.4f}".format(sse))
    
    if plot_all_fits:
        plt.legend(loc=1)

    # CHANGE THIS IF REQUIRED
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')

    # Things to return - df of SSE and distribution name, the best distribution and its parameters
    results = pd.DataFrame(sses, columns = ['SSE','distribution']).sort_values(by='SSE') 
    best_name = results.iloc[0]['distribution']
    best_dist = getattr(st, best_name)
    best_params = best_dist.fit(np.array(array))
    
    if plot_best_fit:
        new_x = np.linspace(x_mid[0] - (bin_width * 2), x_mid[-1] + (bin_width * 2), 1000)
        best_pdf = best_dist.pdf(new_x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        best_pdf_scaled = best_pdf * bin_width * N
        ax.plot(new_x, best_pdf_scaled, label = best_name)
        plt.legend(loc=1)

    if save_hist:
        plt.savefig('./output_'+data_name+'.png')
    
    if plot_hist:
        plt.show()
    
    return results, best_name, best_params


if __name__ == "__main__":
	
    if(len(sys.argv) != 2):
        print("Usage: distibution_fit.py [arquivo_sampes]")
        sys.exit(0)
    arqsamples = sys.argv[1]
    
    m = np.loadtxt(arqsamples, comments='#');
	
	
    num_rows, num_cols = m.shape
    print(num_rows, num_cols)
    
    #remove 1st column (error)
    m = m[:,1:]
    
    labels = ('v0', 'pi_c_apm', 'pi_c_i', 'pi_c_tke', 'delta_c', 'beta_apm',
        'k_v3','beta_tke','pi_v','k_v1','k_v2','beta_Ap')
    vsize = len(labels)
    dists = []
    for i in range(vsize):
        sses, best_name, best_params = fit_scipy_distributions(m[:,i], bins=50, data_name=labels[i], save_hist=True, plot_best_fit=True, plot_all_fits=False)
        print('Param: '+labels[i])
        print('SSEs:\n',sses)
        print('Best distribution name: ',best_name)
        print('Best distribution params: ', best_params)
        print(' ')
        dists.append([labels[i], best_name, best_params])
#print(dists)
df = pd.DataFrame(dists, columns = ['Param Name', 'Distribution Name', 'Distribution Params'])
print(df)

vet = df['Distribution Params'][0]
print(vet[0])
df.to_pickle(arqsamples+'_dists.pkl')

import numpy as np
import chaospy as cp
from math import factorial as fat
# barra de progresso =)
# sudo pip install tqdm
from tqdm import tqdm
from tqdm import tqdm_gui

def P(order,degree):
    return int((fat(degree+order)/(fat(degree)*fat(order))))

def shift_samples(s, ss, k):
    j = 0
    for i in range(len(s)):
        if i != k:
            ss[j] = s[i]
            j += 1

def shift_evals(s, ss, k):
    j = 0
    for i in range(len(s)):
        if i != k:
            ss[j][:] = s[i][:]
            j += 1     

def loo(textlabel, poly_exp, nsamp, nparam, qoi_evals, samples, dist):
    """
    Leave one out cross validation test
    """
    print("----------------------------------------------------------------")
    print("LOO for variable " + textlabel)
    print("----------------------------------------------------------------")

    subs_samples = np.empty((nparam, nsamp - 1))
    qoi_samples = np.empty((nsamp - 1, np.shape(qoi_evals)[1]))
    qoi_dev = np.empty((nsamp, np.shape(qoi_evals)[1]))
    qoi_delta = np.empty((nsamp, np.shape(qoi_evals)[1]))

    for i in range(nsamp):
        # preenche as amostras de teste
        shift_samples(samples.T, subs_samples.T, i)
        # preenche as amostras das QoI de teste
        shift_evals(qoi_evals, qoi_samples, i)
        # "cria" os polinomios de "teste"
        subs_poly_qoi = cp.fit_regression(poly_exp, subs_samples, qoi_samples)
        # avalia os polinomios de "teste"
        y_qoi_s = cp.poly.caller.call(subs_poly_qoi, samples.T[i])
        # calcula os desvios padrão
        qoi_dev[i] = cp.Std(subs_poly_qoi, dist)
        # calcula os deltas
        qoi_delta[i] = abs(abs(y_qoi_s) - abs(qoi_evals[i]))
        if i == int(ns/4):
            print('25% ____________________')
        if i == int(ns/2):
            print('50% ____________________')
        if i == int(3*ns/4):
            print('75% ____________________')
    
    # calcula q2
    qoi_q2 = 1.0 - np.mean(qoi_delta, axis=0) / np.mean(qoi_dev, axis=0)
    
    # informacoes
    print("Q2 for each time step:")
    print(qoi_q2)
    print(f"Min Q2: {np.min(qoi_q2)}")
    print(f"Mean Q2: {np.mean(qoi_q2)}")
    print(f"Max Q2: {np.max(qoi_q2)}")
    print("----------------------------------------------------------------")

def plot_sensitivity(ax, time, poly, dist, label_param):
    """
    Compute and plot Sobol indices
    """
    print("Computing main sobol index...")
    main_sobol_s = cp.Sens_m(poly, dist)

    vcolor = ('red', 'green', 'orange', 'blue', 'yellow',
              'purple', 'cyan', 'brown', 'gray', 'deeppink',
              'firebrick', 'sienna')
    vline = ('solid', 'dashed', 'dashdot', 'dotted')
    k = 0
    l = 0
    
    for sm in main_sobol_s:
        ax.plot(time, sm, color=vcolor[k%len(vcolor)],ls=vline[l], lw=2, label=label_param[k])
        k +=1
        if k%len(vcolor)==0:
            l+=1;
        
    ax.xlabel('tempo (dias)')
    ax.ylabel('indices Sobol')

def plot_sensitivity_mc(ax, time, main_sobol_s, label_param):
    """
    Compute and plot Sobol indices
    """

    vcolor = ('red', 'green', 'orange', 'blue', 'yellow',
              'purple', 'cyan', 'brown', 'gray', 'deeppink',
              'firebrick', 'sienna')
    vline = ('solid', 'dashed', 'dashdot', 'dotted')
    k = 0
    l = 0
    
    for sm in main_sobol_s:
        ax.plot(time, sm, color=vcolor[k%len(vcolor)],ls=vline[l], lw=2, label=label_param[k])
        k +=1
        if k%len(vcolor)==0:
            l+=1;
        
    ax.xlabel('tempo (dias)')
    ax.ylabel('indices Sobol')


def plot_mean_std_poly(ax, time, poly, dist, linecolor, textlabel):
    """
    Compute and plot statistical moments (mean,std)
    """
    mean = cp.E(poly, dist)
    sdev = cp.Std(poly, dist)
    ax.plot(time, mean, lw=2, color=linecolor, label=textlabel)
    ax.fill_between(time, (mean-sdev), (mean+sdev), alpha=0.5, color=linecolor)
    return mean

def plot_mean_std(ax, time, evals, linecolor, textlabel):
    """
    Compute and plot statistical moments (mean,std)
    """
    mean = np.mean(evals, axis=0)
    sdev = np.std(evals, axis=0)
    ax.plot(time, mean, lw=2, color=linecolor, label=textlabel)
    ax.fill_between(time, (mean-sdev), (mean+sdev), alpha=0.5, color=linecolor)
    return mean
    
def plot_confidence_interval(ax, time, evals, linecolor, textlabel):
    """
    Compute and plot statistical moments (mean,std)
    """
    mean = np.mean(evals, axis=0)
    sdev = np.std(evals, axis=0)
    perc_min = 2.5
    perc_max = 97.5
    percentile = np.percentile(evals, [perc_min, perc_max], axis=0)
    ax.plot(time, mean, lw=2, color=linecolor, label=textlabel)
    ax.fill_between(time, percentile[0,:], percentile[1,:], alpha=0.5, color=linecolor)
    return mean

def plot_confidence_interval_poly(ax, time, poly, dist, textlabel, linecolor):
    """
    Compute and plot prediction interval
    """
    mean = cp.E(poly, dist)
    sdev = cp.Std(poly, dist)
    perc_min = 5.0
    perc_max = 95.0
    x = dist.sample(1000, "H")
    values = [cp.poly.caller.call(poly, k) for k in x.T]
    percentile = np.percentile(values, [perc_min, perc_max], axis=0)
    ax.plot(time, mean, lw=2, color=linecolor, label=textlabel)
    ax.fill_between(time, percentile[0,:], percentile[1,:], alpha=0.5, color=linecolor)
    return mean        

def plot_histogram(poly, dist, title, ax, fig, fname):
    qoi_dist = cp.QoI_Dist(poly, dist)
    #print 'Skewness ', title, ' ', cp.Skew(poly, dist)
    #print 'Kurtosis ', title, ' ', cp.Kurt(poly, dist)
    x = dist.sample(1000, "L")
    values = [cp.poly.caller.call(poly, i) for i in x.T]
    percentile = np.percentile(values, [2.5, 97.5])

    num_bins = 40
    n, bins, patches = ax.hist(values, num_bins, density=True, facecolor='blue', alpha=0.4)
    #ax.plot(x, values, 'black')
    mean = np.mean(values)
    std = np.std(values)
    ##ax.plot(bins, qoi_dist.pdf(bins), 'r')
    #kde = stats.gaussian_kde(values)
    ##ax.plot(bins, qoi_dist.pdf(bins), 'r')
    np.savetxt(title.replace('$','').replace('/','')+'_pdf.txt', np.array([bins, qoi_dist.pdf(bins)]).T)
    #ax.axvline(x=mean, ls='-', color='gray' )
    #ax.axvline(x=mean + std, ls='--', color='gray' )
    #ax.axvline(x=mean - std, ls='--', color='gray' )
    #ax.axvline(x=percentile[0], ls=':', color='gray' )
    #ax.axvline(x=percentile[1], ls=':', color='gray' )
    np.savetxt(title.replace('$','').replace('/','')+'_perc.txt', np.array([mean, mean+std, mean-std, percentile[0], percentile[1]]) )
    ax.set_title(title)
    #plt.show()
    #fig.tight_layout()
    #fig.savefig(fname)

    return ax

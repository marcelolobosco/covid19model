import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import sys


if __name__ == "__main__":
    # uso
    if(len(sys.argv) != 3):
        print("Usage: sirmodel [arquivo_samples] [case]")
        sys.exit(0)

    arqpar = sys.argv[1]
    caso = sys.argv[2]

    m = np.loadtxt(arqpar, comments='#');


    num_rows, num_cols = m.shape
    print(num_rows, num_cols)

    m=m[m[:,0].argsort()]
    m=m[:25000]
    num_rows, num_cols = m.shape
    print(num_rows, num_cols)    
    m = m[:,1:]


    #labelsFile = ('b','$r$','$t_i$','$\\Delta$','$e$','$\\theta$','$\\tau_1$','$\\tau_2$','$\\tau_3$','m')
    label_covid = ('$V_0$', '$\\pi_{c_{apm}}$', '$\\pi_{c_i}$', '$\\pi_{c_{tke}}$', '$\\delta_c$', '$\\beta_{apm}$',
            '$k_{v3}$','$\\beta_{tke}$','$\\pi_v$','$k_{v1}$','$k_{v2}$','$\\beta_{Ap}$')
    corrcof_m = np.corrcoef(m.T)
    '''
    for j in range (10):
        print(labelsFile[j]+'\t', end = '')
    print('')

    for i in range(10):
        print(labelsFile[j]+'\t', end = '')
        for j in range (10):
            print("{:.2},\t".format(corrcof_m[i][j]), end = '')
        print('')
    '''
    #np.savetxt('cov_br.txt',np.cov(m[:, 3:13].T))

    df = pd.DataFrame(data=m, columns=label_covid)


    plt.figure(figsize = (10,7))

    plt.title(caso)
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True, linewidths=1)
    plt.savefig('corrMatrix_'+caso+'.pdf')

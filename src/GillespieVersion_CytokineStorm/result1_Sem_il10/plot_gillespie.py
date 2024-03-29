#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys 
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

def getArrayFromFile(filename):
    a = []
    with open(filename) as afile:
        a = afile.readlines()
    a = [content.strip() for content in a]
    a = [float(i) for i in a]
    return a

cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=100)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
color_index = 0
colorVal = scalarMap.to_rgba(color_index)

v0 = getArrayFromFile('V0.dat')
t = []
dt = 0.1 
v = 0
for i in range(0,len(v0)):
    t.append(v)    
    v += dt 

#colors = iter( (np.linspace(0, 1, 10)))
nruns = int(str(sys.argv[1]))
plt.figure("Sars-Cov-2")
for k in range(0,nruns):
    #plt.plot(t, np.log2(getArrayFromFile("V"+str(k)+".dat")), label="Sars-Cov-2", color=colorVal)
    plt.plot(t, getArrayFromFile("V"+str(k)+".dat"), label="Sars-Cov-2", color=colorVal)
    #plt.yscale('log2')
    plt.xlabel('time (days)')
    #plt.ylabel('log2 (viremia)')
    plt.ylabel('viremia')
    #plt.legend()    
    plt.title("Sars-Cov-2")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('V.pdf', bbox_inches='tight', dpi=300) 
plt.savefig('V.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Immature APC")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Ap"+str(k)+".dat"), label="Immature APC", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Immature APC")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Ap.pdf', bbox_inches='tight', dpi=300)    
plt.savefig('Ap.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Mature APC")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Apm"+str(k)+".dat"), label="Mature APC", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Mature APC")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Apm.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Apm.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Naive T CD4")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Thn"+str(k)+".dat"), label="Naive T CD4", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Naive T CD4")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Thn.pdf',bbox_inches='tight',dpi=300)
plt.savefig('Thn.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Effector T CD4")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("The"+str(k)+".dat"), label="Effector T CD4", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Effector T CD4")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('The.pdf', bbox_inches='tight',dpi=300)
plt.savefig('The.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Naive T CD8")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Tkn"+str(k)+".dat"), label="Naive T CD8", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Naive T CD8")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Tkn.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Tkn.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Effector T CD8")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Tke"+str(k)+".dat"), label="Effector T CD8", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Effector T CD8")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Tke.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Tke.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("B cell")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("B"+str(k)+".dat"), label="B cell", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("B cell")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('B.pdf', bbox_inches='tight', dpi=300)
plt.savefig('B.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Short-lived Plasma cells")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Ps"+str(k)+".dat"), label="Short-lived Plasma cells", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Short-lived Plasma cells")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Ps.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Ps.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("Long-lived Plasma cells")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Pl"+str(k)+".dat"), label="Long-lived Plasma cells", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("Long-lived Plasma cells")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Pl.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Pl.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("B memory cell")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Bm"+str(k)+".dat"), label="B memory cell", color=colorVal)
    #plt.yscale('log')
    plt.xlabel('time (days)')
    plt.ylabel('cells number')
    #plt.legend()    
    plt.title("B memory cell")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Bm.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Bm.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("IgM")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("IgM"+str(k)+".dat"), label="IgM", color=colorVal)    
    plt.xlabel('time (days)')
    #plt.ylabel('log2 (antibody level)')
    plt.ylabel('antibody level')
    #plt.legend()    
    plt.title("IgM")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('IgM.pdf', bbox_inches='tight', dpi=300)
plt.savefig('IgM.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("IgG")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("IgG"+str(k)+".dat"), label="IgG", color=colorVal)    
    plt.xlabel('time (days)')
    #plt.ylabel('log2 (antibody level)')
    plt.ylabel('antibody level')    
    plt.title("IgG")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('IgG.pdf', bbox_inches='tight', dpi=300)
plt.savefig('IgG.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure(r"Tnf-$\alpha$")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Tnf"+str(k)+".dat"), label=r"Tnf-$\alpha$", color=colorVal)    
    plt.xlabel('time (days)')
    plt.ylabel('pg/ml')        
    plt.title(r"Tnf-$\alpha$")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Tnf-alpha.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Tnf-alpha.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("IL-6")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Il6"+str(k)+".dat"), label="IL-6", color=colorVal)    
    plt.xlabel('time (days)')
    plt.ylabel('pg/ml')    
    plt.title("IL-6")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Il6.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Il6.svg', format='svg', bbox_inches='tight')

color_index = 0
colorVal = scalarMap.to_rgba(color_index)
plt.figure("IL-10")
for k in range(0,nruns):
    plt.plot(t, getArrayFromFile("Il10"+str(k)+".dat"), label="IL-10", color=colorVal)    
    plt.xlabel('time (days)')
    plt.ylabel('pg/ml')    
    plt.title("IL-10")
    color_index += 20
    colorVal = scalarMap.to_rgba(color_index)
plt.savefig('Il10.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Il10.svg', format='svg', bbox_inches='tight')
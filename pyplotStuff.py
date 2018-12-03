#!/usr/bin/python
'''Ploting results'''
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

f = open("input.in", "r")
header = f.readline()
header = f.readline()
header = f.readline()
header = f.readline()
line   = f.readline()
line   = line.strip()
columns = line.split()
nx = int(columns[0])

fn = 'temp'
gfname = './Results/'+fn+'Geom.dat'
tgfname = './Results/'+fn+'TargetGeom.dat'
flfname = './Results/'+fn+'Flow.dat'
fcfname = './Results/'+fn+'FlowConv.dat'
optcfname = './Results/'+fn+'OptConv.dat'
optcfname2 = './Results/'+fn+'OptConv2.dat'
optcfname3 = './Results/'+fn+'OptConv3.dat'
optcfname4 = './Results/'+fn+'OptConv4.dat'
optcfname5 = './Results/'+fn+'OptConv5.dat'
optcfnameAH = './Results/'+fn+'OptConvAH.dat'
optcfnameEH = './Results/'+fn+'OptConvEH.dat'

opttfname = './Results/'+fn+'OptTime.dat'
opttfname2 = './Results/'+fn+'OptTime2.dat'
opttfname3 = './Results/'+fn+'OptTime3.dat'
opttfname4 = './Results/'+fn+'OptTime4.dat'
opttfname5 = './Results/'+fn+'OptTime5.dat'
opttfnameAH = './Results/'+fn+'OptTimeAH.dat'
opttfnameEH = './Results/'+fn+'OptTimeEH.dat'

#GMRESname = './Results/'+fn+'GMRESconv.dat'
CGname = './Results/'+fn+'CGconv.dat'

ptname = 'targetP.dat'

# Read shape results
data = np.loadtxt(gfname)
lbound = 0
x = data[lbound:lbound+nx]
xhalf = np.append(x-(1.0/nx)/2.0, 1.0)
lbound += nx
S = data[lbound:lbound+nx+1]
lbound += nx+1

# Read shape results
data = np.loadtxt(tgfname)
lbound = 0
xtar = data[lbound:lbound+nx]
xhalftar = np.append(x-(1.0/nx)/2.0, 1.0)
lbound += nx
Star = data[lbound:lbound+nx+1]
lbound += nx+1

# Read shape results
data = np.loadtxt(flfname)
lbound = 0
pres = data[lbound:lbound+nx]
lbound += nx
rho  = data[lbound:lbound+nx]
lbound += nx
Mach = data[lbound:lbound+nx]
lbound += nx

# Read convergence information
data = np.loadtxt(fcfname)
lbound = 0
conv = data[lbound:len(data)]

# Read target pressure
data = np.loadtxt(ptname)
nx = int(data[0])

targetp=data[1:nx+1]

# Read Optimization Convergence Residual
data = np.loadtxt(optcfname)
data2 = np.loadtxt(optcfname2)
data3 = np.loadtxt(optcfname3)
data4 = np.loadtxt(optcfname4)
data5 = np.loadtxt(optcfname5)
dataAH = np.loadtxt(optcfnameAH)
dataEH = np.loadtxt(optcfnameEH)
lbound = 0
optconv = data[lbound:len(data)]
optconv2 = data2[lbound:len(data2)]
optconv3 = data3[lbound:len(data3)]
optconv4 = data4[lbound:len(data4)]
optconv5 = data5[lbound:len(data5)]
optconvAH = dataAH[lbound:len(dataAH)]
optconvEH = dataEH[lbound:len(dataEH)]
# Read Optimization Time
data = np.loadtxt(opttfname)
data2 = np.loadtxt(opttfname2)
data3 = np.loadtxt(opttfname3)
data4 = np.loadtxt(opttfname4)
data5 = np.loadtxt(opttfname5)
dataAH = np.loadtxt(opttfnameAH)
dataEH = np.loadtxt(opttfnameEH)
lbound = 0
opttime = data[lbound:len(data)]
opttime2 = data2[lbound:len(data2)]
opttime3 = data3[lbound:len(data3)]
opttime4 = data4[lbound:len(data4)]
opttime5 = data5[lbound:len(data5)]
opttimeAH = dataAH[lbound:len(dataAH)]
opttimeEH = dataEH[lbound:len(dataEH)]

# Read GMRES Convergence Information
#data = np.loadtxt(GMRESname)
#lbound = 0
#res = data[lbound:len(data)]

# Read CG convergence Information
#data = np.loadtxt(CGname)
#lbound = 0
#res = data[lbound:len(data)]


#adname = './Results/'+fn+'Adjoint.dat'
## Read Adjoint Results
#data = np.loadtxt(adname)
#adj1=data[0*nx+1:1*nx + 1]
#adj2=data[1*nx+1:2*nx + 1]
#adj3=data[2*nx+1:3*nx + 1]


pp=PdfPages('Figures.pdf')

# Plot Channel
plt.figure()
plt.title('Channel Shape')
cshape1 = plt.plot(xhalf,S,'-ob',markerfacecolor='None', markeredgecolor='b')
cshape2 = plt.plot(xhalftar,Star,'xr',markerfacecolor='None', markeredgecolor='r')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.5)
pp.savefig()

# Plot Pressure and Target Pressure
plt.figure()
plt.title('Pressure Distribution')
PressureCurve = plt.plot(x,pres,'-ob',markerfacecolor='None', markeredgecolor='b', label='Current')
TargetPCurve  = plt.plot(x,targetp,'xr',markerfacecolor='None', markeredgecolor='r', label='Target')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.5)
plt.legend(loc='upper right')
pp.savefig()


# Plot Mach Distribution
plt.figure()
plt.title('Mach Distribution')
nxMach=len(Mach)
adj1Curve = plt.plot(range(nxMach),Mach,'-ob',markerfacecolor='None', markeredgecolor='b')
plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.5)
pp.savefig()

# Plot Convergence
plt.figure()
plt.title('Flow Convergence')
nIt=len(conv)
#convCurve = plt.loglog(range(nIt),conv, '-x')
convCurve = plt.semilogy(range(nIt),conv, '-x')
plt.grid(b=True, which='both', color='black', linestyle='-',alpha=0.5)
pp.savefig()

# Plot Opt Convergence
plt.figure()
plt.title('Optimization Convergence')
nIt=len(optconv)
nIt2=len(optconv2)
nIt3=len(optconv3)
nIt5=len(optconv5)
nIt4=len(optconv4)
nItAH=len(optconvAH)
nItEH=len(optconvEH)
convCurve = plt.semilogy(range(nIt),optconv, '-x', color = 'r', label='0.5e-1 truncNewton')
convCurve2 = plt.semilogy(range(nIt2),optconv2, '-x', color = 'b', label='1e-2 truncNewton' )
convCurve3 = plt.semilogy(range(nIt3),optconv3, '-x', color = 'g', label='1e-3 trunctNewton' )
convCurve4 = plt.semilogy(range(nIt4),optconv4, '-x', color = 'm', label='0.8e-1 truncNewton' )
# convCurve5 = plt.semilogy(range(nIt5),optconv5, '-x', color = 'k', label='exactNewton' )
convCurve6 = plt.semilogy(range(nItAH),optconvAH, '-x', color = 'c', label='ApproxH' )
#convCurve7 = plt.semilogy(range(nItEH),optconvEH, '-x', color = 'k', label='ExactH' )
plt.grid(b=True, which='both', color='black', linestyle='-',alpha=0.5)
plt.legend(loc='upper right', fontsize = 'x-small')
pp.savefig()

plt.figure()
plt.title('Optimization time Convergence')
convCurve = plt.semilogy(opttime,optconv, '-x', color = 'r', label='0.5e-1 truncNewton')
convCurve2 = plt.semilogy(opttime2,optconv2, '-x', color = 'b', label='1e-2 truncNewton')
convCurve3 = plt.semilogy(opttime3,optconv3, '-x', color = 'g', label='1e-3 truncNewton')
convCurve4 = plt.semilogy(opttime4,optconv4, '-x', color = 'm', label='0.8e-1 truncNewton')
#convCurve5 = plt.semilogy(opttime5,optconv5, '-x', color = 'k', label='exactNewton')
convCurveAH = plt.semilogy(opttimeAH,optconvAH, '-x', color = 'c', label='ApproxH')
#convCurveEH = plt.semilogy(opttimeEH,optconvEH, '-x', color = 'k', label='ExactH')

plt.grid(b=True, which='both', color='black', linestyle='-',alpha=0.5)
plt.legend(loc='upper right',fontsize = 'x-small')
pp.savefig()

#Plot GMRES convergence
#plt.figure()
#plt.title('GMRES convergence')
#nIt = len(res)
#GRMEScurve = plt.semilogy(range(nIt),res, '-x')
#plt.grid(b=True, which='both', color='black', linestyle='-',alpha=0.5)
#pp.savefig()

#Plot CG convergence
#plt.figure()
#plt.title('CG convergence')
#nIt = len(res)
#CGcurve = plt.semilogy(range(nIt),res, '-x')
#plt.grid(b=True, which='both', color='black', linestyle='-',alpha=0.5)
#pp.savefig()


## Plot Adjoint Distribution
#plt.figure()
#plt.title('Adjoint Distribution')
#adj1Curve = plt.plot(x,adj1,'-ob',markerfacecolor='None', markeredgecolor='b',label='Adj 1')
#plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.5)
#pp.savefig()
#
## Plot Adjoint Distribution
#plt.figure()
#plt.title('Adjoint Distribution')
#adj2Curve = plt.plot(x,adj2,'-or',markerfacecolor='None', markeredgecolor='r',label='Adj 1')
#plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.5)
#pp.savefig()
#
## Plot Adjoint Distribution
#plt.figure()
#plt.title('Adjoint Distribution')
#adj3Curve = plt.plot(x,adj3,'-og',markerfacecolor='None', markeredgecolor='g',label='Adj 3')
#plt.grid(b=True, which='major', color='black', linestyle='-',alpha=0.5)
#pp.savefig()


# Close PDF
pp.close()


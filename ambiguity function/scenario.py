import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as constant
from scipy.signal import chirp 
from mpl_toolkits.mplot3d import Axes3D
from ambiguity_function import ambgfun
from chirp_train import chirp_pulse_train
from rect_pulse_train import rect_pulse_train
import os

def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Define the base directory to search
base_dir = '/Volumes/CRI/SST-Thesis/codici /Radar_simulator_toolkit/Ambfun/scenario'

# Find the files
data_file_1 = find_file('Satellite-CHEBELLO-Sensor-Sensor1-To-Satellite-SENTINEL-6_46984_pos-vel.csv', base_dir)
time_access_file_1 = find_file('Satellite-CHEBELLO-Sensor-Sensor1-To-Satellite-SENTINEL-6_46984_time_access.csv', base_dir)
data_file_2 = find_file('Satellite-CHEBELLO-To-Satellite-GALILEO_10_40890_pos-vel.csv', base_dir)
time_access_file_2 = find_file('Satellite-CHEBELLO-To-Satellite-GALILEO_10_40890_time_access.csv', base_dir)

#constant 
c = 299792458 #m/s
K = constant.Boltzmann

# radar parameter 
P = 4e3 # W
f = 1e9 #Ghz 
wawelength = c/f

case = 2

if case == 1 : 
    # data sentinel 6 46984 
    data = np.genfromtxt(data_file_1, delimiter=",", skip_header=1) 
    time_access = np.genfromtxt(time_access_file_1,skip_header=1)
elif case == 2:
    # data Galileo 10 40890
    data = np.genfromtxt(data_file_2, delimiter=",", skip_header=1) 
    time_access = np.genfromtxt(time_access_file_2,delimiter=",",skip_header=1)


# data extrapolation
t_s = data[:,0]
xr = data[:,1] 
yr = data[:,2]
zr = data[:,3]
r = np.array([xr,yr,zr])
vxr = data[:,4]
vyr = data[:,5]
vzr = data[:,6]
v_r = np.array([vxr,vyr,vzr])
norm_r = np.zeros(len(xr))
vr = np.zeros(len(xr))



for i in range(len(xr)):
    norm_r[i] = np.linalg.norm([xr[i],yr[i],zr[i]])
    vr[i] = (np.dot(r[:,i],v_r[:,i])/norm_r[i]) * 1e3 # m/s 
    
V_max = np.max(vr)

fd = 2*vr/wawelength
if True: 
    # doppler estimation 
    plt.figure()
    plt.plot(t_s,fd)
    plt.grid()
    plt.xlabel('minute past')
    plt.ylabel('Doppler frequency Hz')
    plt.title('Doppler freq shift')
    plt.show()

###### Doppler Ambiguity
PRF_min = 2*np.max(fd) # Questa PRF comporta di non avere ambiguità in doppler 

###### Range Ambiguity
R_max = 100e3
PRI_min = 2*R_max/c # Questa PRI comporta di non avere ambiguità in range
PRF_max = 1/PRI_min

# range resolution 
delta_r = 15e3 #Km
B_min = c/(2*delta_r) # Banda che permette di avere questa risoluzione 
tau_res = 2*delta_r/c # risoluzione in tempo 

# Doppler Resolution 
delta_v = 100 #m/s 
T_obs = wawelength/(2*delta_v) # tempo di osservazione per avere questa risoluzione 
f_res = delta_v/wawelength # risoluzione in frequenza

print('PRF minima per evitare ambiguità in doppler',PRF_min*1e-3,'Khz')
print('PRF massima per evitare ambiguità in range',1/PRI_min*1e-3,'Khz')
print('Banda per avere una risoluzione di 15 Km',B_min*1e-6,'Mhz')
print('Tempo di osservazione per avere una risoluzione di 100 m/s',T_obs,'s')

# parametri del segnale, usiamo un chirp train 
PRF = 100e3
PRI = 1/PRF
fs = 1000000
tau = 5e-6
B = 500e6
f0 = -B/2
f1 = B/2
beta = (f1-f0)/tau
t = np.arange(0,tau,1/fs)
n_impulses = 5

A = 8*R_max*V_max/(c*wawelength)


#x_s,t = chirp_pulse_train(f0,f1,tau,PRF,fs,n_impulses) # chirp train
x_s,t = rect_pulse_train(tau,PRF,fs,n_impulses) # rect train

if np.max(t)>T_obs:
    print('tempo del segnala maggiore a quello desiderato')
    

if False:
    plt.figure()
    plt.plot(t*1e6,np.real(x_s))
    plt.xlabel('time (us)')
    plt.ylabel('Amplitude')
    plt.title('Signal')
    plt.grid()
    plt.show()
    
    
if A < 1: 
    print('Posso scegliere una PRF che implica il fatto di non essere ambigui in range e in doppler')
else: 
    print('Non posso scegliere una PRF che permetta di non essere ambigui in range e in doppler')

A,td,f =ambgfun(x_s,fs,PRF)
f = f/1e3
td = (td*1e6)*n_impulses

a = [0,0]
b = [0,1/(n_impulses*PRI)]

plt.contourf(td, f, A)
plt.xlabel('Time Delay (us)')
plt.ylabel('Doppler Frequency (kHz)')
plt.axhline(y=(-f_res/2)/1e3, color='r', linestyle='--')
plt.axvline(x=(-tau_res/2)*1e6, color='r', linestyle='--')
plt.axhline(y=(f_res/2)/1e3, color='r', linestyle='--')
plt.axvline(x=(tau_res/2)*1e6, color='r', linestyle='--')
plt.axvline(x=(n_impulses*(1/PRF)*1e6), color='red', linestyle='--')
#plt.plot(a,b,color='yellow',linestyle='--')
plt.title('Ambiguity Function')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid((td*n_impulses), f)
ax.plot_surface(X, Y, A, cmap='viridis', linewidth=2)
ax.set_xlabel('Time Delay (us)')
ax.set_ylabel('Doppler Frequency (kHz)')
ax.set_zlabel('Ambiguity Function')
plt.title('Ambiguity Function')
plt.show()

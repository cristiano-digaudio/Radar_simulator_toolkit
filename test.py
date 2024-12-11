import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as constant
from rect_pulse_train import rect_pulse_train, rect_pulse_train_stagg_prf
from chirp_train import chirp_pulse_train
from mpl_toolkits.mplot3d import Axes3D
from ambiguity_function import ambgfun
import matplotlib.patches as patches


# definizione del segnale 
fd_max = 40e3
PRF = [1.5e3,2e3]
PRF_nstg = 10e3
count = [1,3] 
fs = 1000000
tau = 5e-6
B = 100000
f0 = -B/2
f1 = B/2
beta = (f1-f0)/tau
t = np.arange(0,tau,1/fs)
n_impulses = 10

# definizione dei parametri del segnale utilizzando le risoluzioni 
if True: 
    delta_r = 15e3 # risoluzione in range 
    B = 2*delta_r # banda del chirp
    tau = 1/B # durata del rect



#x_s,t = chirp_pulse_train(f0,f1,tau,PRF,fs,n_impulses) # chirp train 
x_s,t = rect_pulse_train(tau,PRF_nstg,fs,n_impulses) # rect train 
#x_s,t = rect_pulse_train_stagg_prf(tau,PRF,count,fs,n_impulses) # rect train stagg PRF


if True:
    plt.figure()
    plt.plot(t*1e6,np.real(x_s))
    plt.xlabel('time (us)')
    plt.ylabel('Amplitude')
    plt.title('Signal')
    plt.grid()
    plt.show()


A,td,f =ambgfun(x_s,fs,PRF)
f = f/1e3
td = td*1e6

mu = B / tau  # Pendenza (Hz/s)
x = np.linspace(-80, 80, 1000)  # Delay (us)

y = mu * x * 1e-6 / 1e3  # Doppler Frequency (kHz)

tau_max = 2*delta_r/constant.c


fig,ax = plt.subplots()
plt.contour(td, f, A)
#plt.plot(x, y, 'r-', linewidth=2, label='Slope B/tau')
plt.xlabel('Time Delay (us)')
plt.ylabel('Doppler Frequency (kHz)')
plt.title('Ambiguity Function')
rect = patches.Rectangle((0, (-1/(2*t[-1]))/1e3), t[-1]*1e6, (1/(t[-1]))/1e3, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.axvline(x=tau_max*1e6, color='r', linestyle='--', label='Range Resolution')
plt.grid()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(td, f)
ax.plot_surface(X, Y, A, cmap='viridis', linewidth=2)
ax.set_xlabel('Time Delay (us)')
ax.set_ylabel('Doppler Frequency (kHz)')
ax.set_zlabel('Ambiguity Function')
plt.title('Ambiguity Function')
plt.show()

import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(z=A, x=td, y=f, colorscale='Viridis')])
fig.update_layout(title='3D Ambiguity Function', scene=dict(
                    xaxis_title='Time Delay (us)',
                    yaxis_title='Doppler Frequency (kHz)',
                    zaxis_title='Ambiguity Function'))
#fig.show()



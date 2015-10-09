import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

#Parte 1

#Funciones para usar Runge Kutta de orden 3.

def f(y, x):
    '''
    Función vectorial usada para correr método de integración.
    Funciones y=Y(s) y x=dY/ds son los argumentos de esta función.
    '''
    return x, -y-mu*(y**2-1)*x

def get_k1(y_n, x_n, h, f):
    f_eval = f(y_n, x_n)
    return h * f_eval[0], h * f_eval[1]

def get_k2(y_n, x_n, h, f):
    k1 = get_k1(y_n, x_n, h, f)
    f_eval = f(y_n + k1[0]/2., x_n + k1[1]/2.)
    return h * f_eval[0], h * f_eval[1]

def get_k3(y_n, x_n, h,f):
    k1=get_k1(y_n, x_n, h, f)
    k2=get_k2(y_n, x_n, h, f)
    f_eval=f(y_n-k1[0]-2*k2[0],x_n -k1[1]-2*k2[0])
    return h*f_eval[0],h*f_eval[1]

def rk3_step(y_n, x_n, h, f):
    k1 = get_k1(y_n, x_n, h, f)
    k2 = get_k2(y_n, x_n, h, f)
    k3 = get_k1(y_n, x_n, h, f)
    y_n1 = y_n + (1/6.)*(k1[0] + k2[0] + k3[0])
    x_n1 = x_n + (1/6.)*(k1[1] + k2[1] + k3[1])
    return y_n1, x_n1


#Constantes, paso de tiempo y número de pasos e inicializamos los vectores que
#queremos calcular

mu = 1.425
N_steps = 40000

H = 20*np.pi/N_steps

Y = np.zeros(N_steps)
X = np.zeros(N_steps)

#Aplicamos el método Runge Kutta 3

#Primero para las condiciones iniciales 1)

Y[0] = 0.1
X[0] = 0

for i in range(1, N_steps):
    Y[i], X[i] = rk3_step(Y[i-1], X[i-1], H, f)

#Resultados gráficos.

s = [H * i for i in range(N_steps)] #Variable de la función y=y(s)

plt.figure(1)
plt.figure(1).clf()
plt.plot(s,Y, color='b')
plt.title("Grafico $y(s)$ v/s $s$ para condiciones iniciales (1)")
plt.xlabel('$s$',fontsize=14)
plt.ylabel('$y(s)$',fontsize=14)
plt.savefig("fig1.png")
plt.show()

plt.figure(2)
plt.figure(2).clf()
plt.plot(Y, X, color='r')
plt.title("Grafico $dy/ds$ v/s $y(s)$ para condiciones iniciales (1) ")
plt.xlabel('$y(s)$')
plt.ylabel('$dy/ds$')
plt.savefig("fig2.png")
plt.show()

#Para condiciones iniciales 2)

Y[0] = 4
X[0] = 0

for i in range(1,N_steps):
    Y[i],X[i]= rk3_step(Y[i-1], X[i-1], H, f)

#Resultados gráficos.

plt.figure(3)
plt.figure(3).clf()
plt.plot(s,Y, color='b')
plt.title("Grafico $y(s)$ v/s $s$ para condiciones iniciales (2)")
plt.xlabel('$s$',fontsize=14)
plt.ylabel('$y(s)$',fontsize=14)
plt.savefig("fig3.png")
plt.show()

plt.figure(4)
plt.figure(4).clf()
plt.plot(Y, X, color='r')
plt.title("Grafico $dy/ds$ v/s $y(s)$ para condiciones iniciales (2) ")
plt.xlabel('$y(s)$',fontsize=14)
plt.ylabel('$dy/ds$', fontsize=14)
plt.savefig("fig4.png")
plt.show()


#Pregunta 2

#Sistema de Lorentz.

def lorenz(t,v):
    return [sigma*(v[1]-v[0]), v[0]*(rho-v[2])-v[1],v[0]*v[1]-beta*v[2]]

#Constantes, paso de tiempo y número de pasos e inicializamos los vectores que
#queremos calcular, con condiciones iniciales a elegir.

sigma = 10
beta  = 8/3.
rho   = 28

N_steps2 = 10000
t_values = np.linspace(1e-3, 10* np.pi, N_steps2)

x     = np.zeros(N_steps2)
y     = np.zeros(N_steps2)
z     = np.zeros(N_steps2)

v0    = [1,1,1]  #Modificables

#Integrador

r = ode(lorenz)
r.set_integrator('dopri5')
r.set_initial_value(v0)

#Integración usando 'dopri15'.

for i in range(len(t_values)):
    r.integrate(t_values[i])
    x[i], y[i] , z[i]= r.y

#Resultados gráficos.

fig = plt.figure(5)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

plt.title("$ \ Atractor \ de \ Lorenz$",fontsize=20)

ax.plot(x,y,z,color='g')
ax.set_xlabel('$x(t)$',fontsize=14)
ax.set_ylabel('$y(t)$',fontsize=14)
ax.set_zlabel('$z(t)$',fontsize=14)
ax.tick_params(axis='x', which='major', labelsize=10, top='off')
ax.tick_params(axis='y', which='major', labelsize=10, top='off')
ax.tick_params(axis='z', which='major', labelsize=10, top='off')
plt.savefig("fig5.png")
plt.show()

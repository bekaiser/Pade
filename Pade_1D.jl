# 4th-order Pade scheme test
# Bryan Kaiser
# 5/17/16

using DataArrays
using PyPlot
using PyCall

# =============================================================================
# domain

L = 3000.0 # km, domain size
Lcenter = 0.0 # x value @ the center of the grid
N = 1000 # series length (must be at least even)
dx = L/(float(N)-1) # km, uniform grid spacing
Fx = 1.0/dx # 1/km, spatial sampling rate
x = collect(0:dx:dx*(N-1))-(L/2.0-Lcenter) # km, centered uniform grid

# =============================================================================
# test signals

test = 2 # enter 1 for sin, 2 for cos, 3 for Gaussian

if test == 1 # test signal: sin(kx) (for slip BCs only)
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
Fs = 1.0/lambda # 1/km, signal frequency (analogy to hz)
Nycriterion = Fs/Fx # Nyquist criterion, must be less than 0.5
#@show(Nycriterion)
ks = 2.0*pi/lambda # rad/km
psi = A*sin(ks*x) # km/s
vA = A*ks*cos(ks*x) 
d2psi = -A*ks^2.0*sin(ks*x) 

elseif test == 2 # test signal: cos(kx) (for no-slip BCs)
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
Fs = 1.0/lambda # 1/km, signal frequency (analogy to hz)
Nycriterion = Fs/Fx # Nyquist criterion, must be less than 0.5
#@show(Nycriterion)
ks = 2.0*pi/lambda # rad/km
psi = A*cos(ks*x) # km/s
vA = -A*ks*sin(ks*x) 
d2psi = -A*ks^2.0*cos(ks*x)

elseif test == 3 # test signal: Gaussian(x) 
sigma = L/10.0
X = x-Lcenter
psi = exp(-X.^2.0/(2.0*sigma^2.0)) # test signal equivalent to gaussmf(x,[L/10 0]);
vA = X.*psi.*(-sigma^(-2.0)) # test signal derivative (Gaussian)
d2psi = psi.*(X.^2.0-sigma^2.0)./sigma^4.0 

end # test signal choice

# signal plot 
plot(x, psi, "b") 
xlabel("x (km)")
title("signal psi")
show()
#readline()


# =============================================================================
# dpsi/dx solution by 4th order Pade finite difference scheme

bc = 1 # enter 1 for no-slip, 2 for slip boundary conditions

# (linear) finite difference matrix operator
A = zeros(N,N)
upvals = diagind(A,1)
midvals = diagind(A)
lowvals = diagind(A,-1)
A[upvals] = dx/3.0
A[midvals] = 4*dx/3.0
A[lowvals] = dx/3.0
b = zeros(N) # grid point vector of u

if bc == 1 # no-slip (dpsi/dx=v=0 at x=[0,L])
#= test with cos(kx) because its derivative at x=[0,L] is zero =#  
b[2:N-1] = psi[3:N]-psi[1:N-2] # b[1]=b[N]=0
A[1,1:2] = [1 0] # dpsi/dx = 0 boundary condition at x=0
A[N,N-1:N] = [0 1] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = L 

elseif bc == 2 # free-slip (dpsi/dx=v at x=[0,L] by forward diff) 
b[2:N-1] = psi[3:N]-psi[1:N-2] # b[1]=b[N]=0
A[1,1:2] = [dx 0] # dpsi/dx = 0 boundary condition at x=0
A[N,N-1:N] = [0 dx] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = L
# 5th order forward / backward finite differences: 
b[1:1] = [-137.0/60.0 5.0 -5.0 10.0/3.0 -5.0/4.0 1.0/5.0]*psi[1:6] 
b[N:N] = [ -1.0/5.0 5.0/4.0 -10.0/3.0 5.0 -5.0 137.0/60.0]*psi[N-5:N] 

end # bc choice

# compute first derivative: A*(dpsi/dx) = b, solve for dpsi/dx
#dpsidx = inv(A)*b
v = A\b
#norm(A*x-b) # check residual

# =============================================================================
# plots

# first derivative solution plot 
plot(x, v, "r",x,vA,"b") 
xlabel("x (km)")
#legend('v','v_A')
title("4th-order Pade")
show()
#readline()

# first derivative error plot 
semilogy(x, abs(v-vA),"k") 
xlabel("x (km)")
title("dpsi/dx error, 4th-order Pade")
show()
#readline()



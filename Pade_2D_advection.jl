# 4th-order Pade scheme for computing 1st order derivatives: advection test
# Bryan Kaiser
# 1/21/16

using DataArrays
using PyPlot
using PyCall
@pyimport numpy as np
@pyimport pylab as py

## ============================================================================
# domain

Lx = 3000.0 # km, domain size
Ly = Lx # km
Lxcenter = 0.0 # x value @ the center of the grid
Lycenter = 0.0 # y value @ the center of the grid
N = 2^8 # series length (must be at least even)
dx = Lx/(float(N)-1) # km, uniform longitudinal grid spacing
dy = Ly/(float(N)-1) # km, uniform latitudinal grid spacing
Fx = 1.0/dx # 1/km, longitudinal spatial sampling rate
Fy = 1.0/dy # 1/km, latitudinal spatial sampling rate
x = collect(0:dx:dx*(N-1))-(Lx/2.0-Lxcenter) # km, centered uniform grid 
y = collect(0:dy:dy*(N-1))-(Ly/2.0-Lycenter) # km, centered uniform grid 
X,Y = np.meshgrid(x,y) 

# boundary conditions:
bc = 2 # 1 for no-slip, 2 for free-slip

## ============================================================================
# test signals

test = 2

if test == 1 # 2D Gaussian => no advection!
psi0 = 5.0 # km/s
sigma = Lx/10.0
psi = exp(-((X-Lxcenter).^2+(Y-Lycenter).^2)./(2.0*sigma^2)).*psi0 # streamfunction
dxpsiA = -(X-Lxcenter).*psi.*(sigma^(-2)) 
dypsiA = -(Y-Lycenter).*psi.*(sigma^(-2)) 
nabla_psiA = dxpsiA+dypsiA # sum d/dx + d/dy
d2xpsiA = psi.*((X-Lxcenter).^2.*(sigma^(-4))-sigma^(-2))
d2ypsiA = psi.*((Y-Lycenter).^2.*(sigma^(-4))-sigma^(-2))
nabla2_psiA = d2xpsiA+d2ypsiA # sum d^2/dx^2 + d^2/dy^2
qA = nabla2_psiA # sum d^2/dx^2 + d^2/dy^2
uA = -dypsiA
vA = dxpsiA
dqdxA = psi.*((X-Lxcenter).*4.0/sigma^4-((X-Lxcenter).*(Y-Lycenter).^2)./sigma^6-((X-Lxcenter).^3)./sigma^6)
dqdyA = psi.*((Y-Lycenter).*4.0/sigma^4-((Y-Lycenter).*(X-Lxcenter).^2)./sigma^6-((Y-Lycenter).^3)./sigma^6)
JA = dxpsiA.*dqdyA-dypsiA.*dqdxA
@show(sum(sum(JA))) # non-divergent flow check

elseif test == 2 # 2D asymmetric Gaussian exp(-((x-a)^2+(y-b))/(2*c^2)) 
psi0 = 1000.0 # km/s
sigmax = Lx/10.0
sigmay = Ly/60.0
psi = exp(-((X-Lxcenter).^2)./(2.0*sigmax^2)-((Y-Lycenter).^2)./(2.0*sigmay^2) ).*psi0 # streamfunction
dxpsiA = -psi.*(X-Lxcenter)./sigmax^2
dypsiA = -psi.*(Y-Lycenter)./sigmay^2
nabla_psiA =  dxpsiA+dypsiA # sum d/dx + d/dy
d2xpsiA = psi.*(((X-Lxcenter).^2)./sigmax^4-1/sigmax^2)
d2ypsiA = psi.*(((Y-Lycenter).^2)./sigmay^4-1/sigmay^2)
nabla2_psiA = d2xpsiA+d2ypsiA # sum d^2/dx^2 + d^2/dy^2
qA = nabla2_psiA  # sum d^2/dx^2 + d^2/dy^2
uA = -dypsiA
vA = dxpsiA
dqdxA = psi.*((X-Lxcenter)./(sigmax^2*sigmay^2)+(X-Lxcenter).*(3.0/sigmax^4)-((X-Lxcenter).*(Y-Lycenter).^2)./(sigmax^2*sigmay^4)-((X-Lxcenter).^3)./sigmax^6)
dqdyA = psi.*((Y-Lycenter)./(sigmax^2*sigmay^2)+(Y-Lycenter).*(3.0/sigmay^4)-((Y-Lycenter).*(X-Lxcenter).^2)./(sigmax^4*sigmay^2)-((Y-Lycenter).^3)./sigmay^6)
JA = dxpsiA.*dqdyA-dypsiA.*dqdxA
@show(maximum(JA))

elseif test == 3 # the appropriate BCs for this IC are ...?
psiA = -((Y.^3.0)./3.0-Y+(X.^3.0)./3.0-X).*psi0  # streamfunction
dxpsiA = (X.^2.0-1.0).*psi0
dypsiA = (Y.^2.0-1.0).*psi0
#u = (1.0-Y.^2.0).*psi0
#v = (X.^2.0-1.0).*psi0
qA = -((X+Y).*2.0).*psi0
jacobianA = -((X.^2.0-1.0)-(Y.^2.0-1.0)).*(2.0*psi0^2.0)

end # test signal choice

CP1 = py.contourf(X,Y,psi,200,cmap="Spectral")
xlabel("x (km)")
ylabel("y (km)")
title("psi, signal")
py.colorbar(CP1)
py.show()

CP1 = py.contourf(X,Y,uA,200,cmap="RdBu")
xlabel("x (km)")
ylabel("y (km)")
title("u, signal")
py.colorbar(CP1)
py.show()

CP1 = py.contourf(X,Y,vA,200,cmap="RdBu")
xlabel("x (km)")
ylabel("y (km)")
title("v, signal")
py.colorbar(CP1)
py.show()

CP1 = py.contourf(X,Y,JA,200,cmap="PuOr")
xlabel("x (km)")
ylabel("y (km)")
title("J(psi,q), signal")
py.colorbar(CP1)
py.show()


## ============================================================================
# dpsi/dx solution by 4th order Pade finite difference scheme

dxpsi = zeros(N,N) # first derivative

Ax = zeros(N,N) # finite difference matrix operator for a single grid vector
upvals = diagind(Ax,1)
midvals = diagind(Ax)
lowvals = diagind(Ax,-1)
Ax[upvals] = dx/3.0
Ax[midvals] = 4.0*dx/3.0
Ax[lowvals] = dx/3.0
if bc == 1 # no-slip 
	#Ax[1,1:2] = [dx 0] # dpsi/dx = 0 boundary condition at x=0
	#Ax[N,N-1:N] = [0 dx] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = Lx 
elseif bc == 2 # free-slip 
	Ax[1,1:2] = [dx 0] # dpsi/dx = 0 boundary condition at x=0
	Ax[N,N-1:N] = [0 dx] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = L
end

bx = zeros(N) # x grid point vector of differentiated variable 
for yrow = 2:N-1 # dpsi/dx row from y=dy to y=dy*(N-2)
	if bc == 1 #= no-slip =#
		bx[2:N-1] = psi[yrow,3:N]-psi[yrow,1:N-2] # bx[i=1]=bx[i=N]=0
	elseif bc == 2 #= free-slip =#
		bx[2:N-1] = psi[yrow,3:N]-psi[yrow,1:N-2] # b[1]=b[N]=0
		# 5th-order backward difference:
		bx[1:1] = [-137.0/60.0 5.0 -5.0 10.0/3.0 -5.0/4.0 1.0/5.0]*psi[yrow,1:6]' 
		# 5th-order backward difference:
		bx[N:N] = [ -1.0/5.0 5.0/4.0 -10.0/3.0 5.0 -5.0 137.0/60.0]*psi[yrow,N-5:N]' 
	end # bc choice
	# compute first derivative: A*(dpsi/dx) = b, solve for dpsi/dx
	dxpsi[yrow,1:N] = Ax\bx
	bx = zeros(N)
end # yid loop, dpsidx

dxpsi[1,1:N] = zeros(1,N) # no normal flow, bottom of domain y=0, v=0
dxpsi[N,1:N] = zeros(1,N) # no normal flow, top of domain y=L, v=0
v = dxpsi # meridonal velocity

# meridonal velocity
CP3 = py.contourf(X,Y,v,100,cmap="RdBu")
xlabel("x (km)")
ylabel("y (km)")
title("v, computed")
py.colorbar(CP3)
py.show()

# meridonal velocity, computational error 
CP3 = py.contourf(X,Y,abs(vA-v),100,cmap="gray")
xlabel("x (km)")
ylabel("y (km)")
title("v, computational error")
py.colorbar(CP3)
py.show()


## ============================================================================
# dpsi/dy solution by 4th order Pade finite difference scheme

dypsi = zeros(N,N) # first derivative

Ay = zeros(N,N) # finite difference matrix operator for a single grid vector
upvals = diagind(Ay,1)
midvals = diagind(Ay)
lowvals = diagind(Ay,-1)
Ay[upvals] = dy/3.0
Ay[midvals] = 4*dy/3.0
Ay[lowvals] = dy/3.0
if bc == 1 #= no-slip =#
	Ay[1,1:2] = [dy 0] # dpsi/dx = 0 boundary condition at x=0
	Ay[N,N-1:N] = [0 dy] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = Lx 
elseif bc == 2 #= free-slip =#
	Ay[1,1:2] = [dy 0] # dpsi/dx = 0 boundary condition at x=0
	Ay[N,N-1:N] = [0 dy] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = L
end

by = zeros(N) # x grid point vector of differentiated variable 
for xrow = 2:N-1 # dpsi/dx row from y=dy to y=dy*(N-2)
	if bc == 1 #= no-slip =#
		by[2:N-1] = psi[3:N,xrow]-psi[1:N-2,xrow] # bx[i=1]=bx[i=N]=0
	elseif bc == 2 #= free-slip =#
		by[2:N-1] = psi[3:N,xrow]-psi[1:N-2,xrow] # b[1]=b[N]=0
		# 5th-order backward difference:
		by[1:1] = [-137.0/60.0 5.0 -5.0 10.0/3.0 -5.0/4.0 1.0/5.0]*psi[1:6,xrow] 
		# 5th-order forward difference:
		by[N:N] = [ -1.0/5.0 5.0/4.0 -10.0/3.0 5.0 -5.0 137.0/60.0]*psi[N-5:N,xrow] 	
	end # bc choice
	# compute first derivative: A*(dpsi/dx) = b, solve for dpsi/dx
	dypsi[1:N,xrow] = Ay\by
	by = zeros(N)
end # yid loop, dpsidx

dypsi[1,1:N] = zeros(1,N) # no normal flow, bottom of domain y=0, v=0
dypsi[N,1:N] = zeros(1,N) # no normal flow, top of domain y=L, v=0
u = -dypsi # zonal velocity

# zonal velocity 
CP3 = py.contourf(X,Y,u,100,cmap="RdBu")
xlabel("x (km)")
ylabel("y (km)")
title("u, computed")
py.colorbar(CP3)
py.show()

# zonal velocity, computational error 
CP3 = py.contourf(X,Y,abs(uA-u),100,cmap="gray")
xlabel("x (km)")
ylabel("y (km)")
title("u, computational error")
py.colorbar(CP3)
py.show()


## ============================================================================
# advection function

function advection(q,u,v) # 4th-order Pade scheme derivatives (free-slip)
	
	# zonal derivatives
	dxq = zeros(N,N) 
	Ax = zeros(N,N) # finite difference matrix operator for a single grid vector
	upvals = diagind(Ax,1)
	midvals = diagind(Ax)
	lowvals = diagind(Ax,-1)
	Ax[upvals] = dx/3.0
	Ax[midvals] = 4.0*dx/3.0
	Ax[lowvals] = dx/3.0
	Ax[1,1:2] = [dx 0] # dpsi/dx = 0 boundary condition at x=0
	Ax[N,N-1:N] = [0 dx] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = L
	bx = zeros(N) # x grid point vector of differentiated variable 
	for yrow = 2:N-1 # dpsi/dx row from y=dy to y=dy*(N-2)
		bx[2:N-1] = q[yrow,3:N]-q[yrow,1:N-2] # b[1]=b[N]=0
		bx[1:1] = [-137.0/60.0 5.0 -5.0 10.0/3.0 -5.0/4.0 1.0/5.0]*psi[yrow,1:6]' # 5th FD
		bx[N:N] = [ -1.0/5.0 5.0/4.0 -10.0/3.0 5.0 -5.0 137.0/60.0]*psi[yrow,N-5:N]' # 5th FD
		# compute first derivative: A*(dpsi/dx) = b, solve for dpsi/dx
		dxq[yrow,1:N] = Ax\bx
		bx = zeros(N)
	end # yid loop, dpsidx
	dxq[1,1:N] = zeros(1,N) # no normal flow, bottom of domain y=0, v=0
	dxq[N,1:N] = zeros(1,N) # no normal flow, top of domain y=L, v=0

	# meridonal derivatives
	dyq = zeros(N,N) 
	Ay = zeros(N,N) # finite difference matrix operator for a single grid vector
	upvals = diagind(Ay,1)
	midvals = diagind(Ay)
	lowvals = diagind(Ay,-1)
	Ay[upvals] = dy/3.0
	Ay[midvals] = 4*dy/3.0
	Ay[lowvals] = dy/3.0
	Ay[1,1:2] = [dy 0] # dpsi/dx = 0 boundary condition at x=0
	Ay[N,N-1:N] = [0 dy] # dpsi/dx = 0 boundary condition at x = dx*(N-1) = L
	by = zeros(N) # x grid point vector of differentiated variable 
	for xrow = 2:N-1 # dpsi/dx row from y=dy to y=dy*(N-2)
		by[2:N-1] = q[3:N,xrow]-q[1:N-2,xrow] # b[1]=b[N]=0
		by[1:1] = [-137.0/60.0 5.0 -5.0 10.0/3.0 -5.0/4.0 1.0/5.0]*psi[1:6,xrow] # 5th FD
		by[N:N] = [ -1.0/5.0 5.0/4.0 -10.0/3.0 5.0 -5.0 137.0/60.0]*psi[N-5:N,xrow] # 5th FD
		# compute first derivative: A*(dpsi/dx) = b, solve for dpsi/dx
		dyq[1:N,xrow] = Ay\by
		by = zeros(N)
	end # yid loop, dpsidx
	dyq[1,1:N] = zeros(1,N) # no normal flow, bottom of domain y=0, v=0
	dyq[N,1:N] = zeros(1,N) # no normal flow, top of domain y=L, v=0

	J = u.*dxq+v.*dyq
	return J
end

J = advection(qA,u,v)

# zonal velocity 
CP3 = py.contourf(X,Y,J,100,cmap="PuOr")
xlabel("x (km)")
ylabel("y (km)")
title("J(psi,q), computed")
py.colorbar(CP3)
py.show()

# zonal velocity, computational error 
CP3 = py.contourf(X,Y,abs(JA-J),100,cmap="gray")
xlabel("x (km)")
ylabel("y (km)")
title("J(psi,q), computational error")
py.colorbar(CP3)
py.show()

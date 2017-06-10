% Fourier Analysis, Discretization Error 
% Bryan Kaiser
% 10/29/15

close all; clear all; clc

%------------------------------------------------------------------------------
% set up 

% wavelength and length parameters 
N = 2^11;
L = 1; % wavelength to be resolved
kL = [0:(pi/(N-1)):pi].*L; % 2*pi/lambda*L = [0:pi]
% k is a spectrum of wavenumbers that can be 
% resolved within the wavelength L. 
lambda = (kL.^(-1)).*(2*pi); % lambda = [inf:2L]
np = fliplr(lambda); % number of discrete points 
% within [0:L]. np = lambda' / L = 2*pi / kL
% np = [2:inf]

%------------------------------------------------------------------------------

% (non-dimensional) modified wavenumbers k*L
k2 = sin(kL)./L; % 2nd-order central difference
k4 = (8*sin(kL)-sin(2*kL))./(6*L); % 4th-order central difference
k4P= (3*sin(kL))./((2+cos(kL)).*L); % 4th-order Pade scheme
k5f = (exp(kL.*1i).*5-exp(kL.*2i).*5+exp(kL.*3i).*(10/3)-...
    exp(kL.*4i).*(5/4)+exp(kL.*5i)./5-137/60).*(-1i/L); % 5th-order forward difference
k6P = (sin(kL).*14/9+sin(kL.*2)./18)./(1+cos(kL).*(2/3)); % 6th-order Pade scheme
k6f = (exp(kL.*1i).*6-exp(kL.*2i).*(15/2)+...
    exp(kL.*3i).*(20/3) -exp(kL.*4i).*(15/4)+...
    exp(kL.*5i).*(6/5)-exp(kL.*6i)./6-49/20).*(-1i/L); % 6th-order forward difference

% error 
err2 = abs(kL-k2)./abs(kL);
err4 = abs(kL-k4)./abs(kL);
err4P = abs(kL-k4P)./abs(kL);
err5fr = abs(kL-real(k5f))./abs(kL);
err5fi = abs(kL-imag(k5f))./abs(kL);
err6P = abs(kL-k6P)./abs(kL);
err6fr = abs(kL-real(k6f))./abs(kL);
err6fi = abs(kL-imag(k6f))./abs(kL);

%------------------------------------------------------------------------------
% plots

% exact non-dim wavenumber (kdx) vs. resolved non-dim wavenumber
figure; set(gca,'FontSize',12);
plot(kL,k2,'Color',[0.9 0 0],'LineWidth',2); hold on % 'LineStyle','--'
plot(kL,k4,'Color',[0.6 0 0],'LineWidth',2); hold on
plot(kL,k4P,'Color',[0 0 0.9],'LineWidth',2); hold on
plot(kL,abs(k5f),'Color',[0 0.9 0],'LineWidth',2); hold on
plot(kL,k6P,'Color',[0.5 0 0.5],'LineWidth',2); hold on
plot(kL,abs(k6f),'Color',[0 0.6 0],'LineWidth',2); hold on
plot(kL,kL,'k','LineWidth',2)
xlabel('$$\kappa\Lambda$$','interpreter','latex','FontSize',20) 
ylabel('$${\kappa}^{,}\Lambda$$','interpreter','latex','FontSize',20) 
legend('CD2','CD4','P4','FD5','P6','FD6',2); axis([0,pi,0,pi])

% resolved wavenumber percent difference
figure; set(gca,'FontSize',12);
loglog(kL,err2,'Color',[0.9 0 0],'LineWidth',2); hold on
loglog(kL,err4,'Color',[0.6 0 0],'LineWidth',2); hold on
loglog(kL,err4P,'Color',[0 0 0.9],'LineWidth',2); hold on
loglog(kL,err5fr,'Color',[0 0.9 0],'LineWidth',2); hold on
loglog(kL,err5fi,'Color',[0 0.9 0],'LineWidth',2,'LineStyle','--'); hold on
loglog(kL,err6P,'Color',[0.5 0 0.5],'LineWidth',2);
loglog(kL,err6fr,'Color',[0 0.6 0],'LineWidth',2); hold on
loglog(kL,err6fi,'Color',[0 0.6 0],'LineWidth',2,'LineStyle','--'); 
xlabel('$$\kappa\Lambda$$','interpreter','latex','FontSize',20) 
ylabel('$${\%}\hspace{2mm}error$$','interpreter','latex','FontSize',20) 
axis([0,pi,1E-16,1E2]); grid on
legend('CD2','CD4','P4','Re(FD5)','Im(FD5)','P6','Re(FD6)','Im(FD6)',4);

% number of points per resolved wavelength
figure; set(gca,'FontSize',12);
semilogx(np,fliplr(k2),'Color',[0.9 0 0],'LineWidth',2); hold on
semilogx(np,fliplr(k4),'Color',[0.6 0 0],'LineWidth',2); hold on
semilogx(np,fliplr(k4P),'Color',[0 0 0.9],'LineWidth',2); hold on
semilogx(np,fliplr(real(k5f)),'Color',[0 0.9 0],'LineWidth',2); hold on
semilogx(np,fliplr(imag(k5f)),'Color',[0 0.9 0],'LineWidth',2,'LineStyle','--'); hold on
semilogx(np,fliplr(k6P),'Color',[0.5 0 0.5],'LineWidth',2); hold on
semilogx(np,fliplr(real(k6f)),'Color',[0 0.6 0],'LineWidth',2); hold on
semilogx(np,fliplr(imag(k6f)),'Color',[0 0.6 0],'LineWidth',2,'LineStyle','--'); hold on
semilogx(np,fliplr(kL),'k','LineWidth',2)
xlabel('$$N={\lambda}/{\Lambda}$$','interpreter','latex','FontSize',20) 
ylabel('$${\kappa^{,}}{\Lambda}$$','interpreter','latex','FontSize',20) 
legend('CD2','CD4','P4','Re(FD5)','Im(FD5)','P6','Re(FD6)','Im(FD6)');
axis([2,200,0,pi]); grid on

% USING THE 5th ORDER FORWARD DIFF (FREE-SLIP), 
% FOR N=1024 AND L=3000km, 
% 9 GRID POINTS TIMES dx=L/N 
% MEANS THAT THE SMALLEST WAVE FULLY 
% RESOLVED IS ~26km. 

% A MORE CONSERVATIVE ESTIMATE FOR N=2048, L=3000km,
% IS 10 GRID POINTS S0 15km WAVES ARE FULLY 
% RESOLVED.

% IN THE INTERIOR (FREE-SLIP) OR NO-SLIP
% THE PADE SCHEME IS 7 GRID POINTS
% SO FOR N=1024, L=3000km RESOLVED
% WAVELENGTH IS ~20km.

% WHEN kL IS 0.12 AND LARGER, THE 5th
% FORWARD DIFFERENCE ERROR IS LARGEST
% ERROR.

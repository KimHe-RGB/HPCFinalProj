%% play around different solvers and show animation
clear all

t0 = 0; tfinal = 240;

% time and space discretization grid
C = 1;
ht = 0.005; 
L = 60;
Nx = 129; 
dx = L / (Nx-1);
xs = reshape((-(Nx-1)/2:(Nx-1)/2)*dx, [Nx,1]);

init_cond_f = @(x, t) sin(pi/20*x);
Heat_kernel = @(xs, t, C) exp(-xs.^2/(4*C*t)) / sqrt(4*pi*C*t);

U0 = reshape(init_cond_f(xs, 0), [Nx,1]);
% U0 = reshape([zeros((Nx-1)/2,1); 1; zeros((Nx-1)/2,1)], [Nx,1]); % Delta function

[uu,tt] = EE_FD(U0, dx, [t0, tfinal], ht, C);
[uu2,tt2] = RK2_FD(U0, dx, [t0, tfinal], ht, C);

% true analytical solution is the heat kernel
real_U = Heat_kernel(xs,tt(1),C);

figure(2); clf; 
hold on;
plt0 = plot(xs, U0, ':k', DisplayName="$$U_0$$");
plt1 = plot(xs, uu(:,1), '--r',DisplayName="coarse Forward Euler");
plt2 = plot(xs, uu2(:,1), '-.b', DisplayName="fine RK4");
plt3 = plot(xs, real_U, ':g', DisplayName="Real");
scatter(xs((Nx+1)/2),U0((Nx+1)/2));
xlim([-L/2, L/2])
ylim([-1, 1])
grid on
figure(2); legend([plt0 plt1 plt2 plt3], interpreter="latex", Location="southwest");
title(sprintf('t = %7.5f',tt(1)),'fontsize',18), drawnow
set(gcf,'doublebuffer','on')
disp('press <return> to begin'), pause  % wait for user input

for tn = 1:length(tt)
    if(mod(tn-1, 100) == 0 || tn==length(tt))
        set(plt1, 'ydata', uu(:,tn))
        set(plt2, 'ydata', uu2(:,tn))
        set(plt3, 'ydata', Heat_kernel(xs,tt(tn),C));
        title(sprintf('Nx = %d, h = %4.6f, t = %7.5f',Nx,ht,tt(tn)),'fontsize',18), drawnow
    end
end
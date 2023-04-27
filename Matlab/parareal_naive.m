% parareal algorithm:
% G: propagation operator
% F: a more accurate approximation operator

% discretization
% U_{n}^{k} = u(n*hx, k*ht)

% U_{n+1}^{k+1} = 
%  G(t_{n-1}, t_n, U_{n+1}^{k+1})
%  - G(t_{n-1}, t_n, U_n^k)
%  + F(t_{n-1}, t_n, U_n^k)  
% --- (2.2)

% for k -> infty 
clear all

t0 = 0; tfinal = 100;
% partition time domain into N equispaced subintervals
N = 20; T_size = (tfinal-t0)/N;
T_intervals = zeros(2, N);
for i = 1:N
    T_intervals(:,i) = [t0+(i-1)*T_size; t0+i*T_size];
end

% compute initial conditions U_0, U_1, U_2, ... , U_N, from t = t0, t1,
% ..., tfinal
% we use a inexpensive operator, say Forward Euler
phi_sol_f = @(x, t) 1 / 2 * sech(1/2*(x - t)).^2;

% coarse time and space discretization grid
C = 0.1;
ht = 0.005; 
L = 60;
Nx = 64; 
hx = L / Nx;
xs = reshape((0:Nx-1)*hx, [Nx,1]) - L/2;
% U0 = reshape(phi_sol_f(xs, 0), [Nx,1]);
U0 = reshape([zeros(Nx/2-1,1); 1; 1; zeros(Nx/2-1,1)], [Nx,1]); % "block function"

[uu,tt] = forward_euler_FD(U0, xs, hx, [t0, tfinal], ht, C);
[uu2,tt2] = RK2_FD(U0, xs, hx, [t0, tfinal], ht, C);

figure(2); clf; 
hold on;
plt0 = plot(xs, U0, '-.k');
plt1 = plot(xs, uu(:,1), '--r');
plt2 = plot(xs, uu2(:,1), ':b');
xlim([-L/2, L/2])
ylim([-0.5, 1])
grid on
figure(2); legend([plt0 plt1 plt2], ["y0", "yt1", "yt2"], interpreter="latex", Location="southwest");
title(sprintf('t = %7.5f',tt(1)),'fontsize',18), drawnow

set(gcf,'doublebuffer','on')
disp('press <return> to begin'), pause  % wait for user input

for tn = 1:length(tt)
    if(mod(tn-1, 100) == 0 || tn==length(tt))
        set(plt1, 'ydata', uu(:,tn))
        set(plt2, 'ydata', uu2(:,tn))
        title(sprintf('Nx = %d, h = %4.6f, t = %7.5f',Nx,ht,tt(tn)),'fontsize',18), drawnow
    end
end

% for each subintervals we compute with a relatively expensive operator,
for interval = T_intervals
    % interval : [0; Tsize], [Tsize, 2Tsize], ...
    t0 = interval(1);
    tfinal = interval(2);

end

% solve the heat equation using finite difference discretization, with forward euler method
function [uu,tt] = forward_euler_FD(u0, xs, hx, tspan, ht, C)
    Nx = length(xs);
    t0 = tspan(1); tfinal = tspan(2);
    
    % forward euler method
    e = ones(Nx,1);
    A = spdiags([e -2*e e],-1:1,Nx,Nx) * C / hx^2;
    t = t0;
    u = u0;
    
    i=1;    t_steps = ceil((tfinal - t0) / ht) + 1;
    tt = zeros(1,t_steps); uu = zeros(Nx,t_steps);
    done = t >= tfinal;
    while ~done
        tt(i) = t; uu(:,i) = u; i=i+1;
        if(t + ht >= tfinal)
            ht = tfinal - t;
            done = true;
        end
        u = u + ht * A * u;
        % fix the boundary condition u(t, x=0) and u(t, x=L)
        %  u = [u0(1); u(2:end-1); u0(end)];
        t = t + ht;
    end
end

function [uu,tt] = RK2_FD(u0, xs, hx, tspan, ht, C)
    Nx = length(xs);
    t0 = tspan(1); tfinal = tspan(2);
    
    % finite difference recurrence
    e = ones(Nx,1);
    A = spdiags([e -2*e e],-1:1,Nx,Nx) * C / hx^2;
    F = @(ht, u) ht * A * u;

    t = t0;
    u = u0;
    
    i=1;    t_steps = ceil((tfinal - t0) / ht) + 1;
    tt = zeros(1,t_steps); uu = zeros(Nx,t_steps);
    done = t >= tfinal;
    while ~done
        tt(i) = t; uu(:,i) = u; i=i+1;
        if(t + ht >= tfinal)
            ht = tfinal - t;
            done = true;
        end
        % u_{n+1/2} = u_{n} + F(t_n, t_n+0.5*ht, u_n)
        u_half = u + F(ht/2, u);
        % u_{n+1} = u_{n} + F(t_n, t_n+ht, u_{n+1/2})
        u = u + F(ht, u_half);
        
        % fix the boundary condition u(t, x=0) and u(t, x=L)
        %  u = [u0(1); u(2:end-1); u0(end)];
        t = t + ht;
    end
end
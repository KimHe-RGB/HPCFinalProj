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



%% parareal
clear all

t0 = 0; tfinal = 30;
% partition time domain into N equispaced subintervals
N = 20; 
interval_size = (tfinal-t0)/N;
intervals = zeros(2, N);

for i = 1:N
    intervals(:,i) = [t0+(i-1)*interval_size; t0+i*interval_size];
end

% compute initial conditions U_0, U_1, U_2, ... , U_N, 
% from t = t0, t1, ..., tfinal 
% for each subinterval, using inexpensive coarse operator to compute the
% initial value for it.
interval_dt = 1e-5;                   % time grid for subinterval

overall_dt = 1e-2;                    % time grid for coarse 
interval_initVals = zeros(1, N);   

C = 1;
L = 60;
Nx = 128;
dx = L / Nx;
xs = reshape((0:Nx-1)*dx, [Nx,1]) - L/2;

init_cond_f2 = @(x, t) sin(pi/20 * x);
U0 = reshape(init_cond_f2(xs, 0), [Nx,1]);

[uu,tt] = forward_euler_FD(U0, xs, dx, [t0, tfinal], overall_dt, C);


% for each subintervals we compute with a relatively expensive operator,
for interval = intervals
    % interval : [0; Tsize], [Tsize, 2Tsize], ...
    t0 = interval(1); tfinal = interval(2);
    [uu2,tt2] = RK4_FD(U0, xs, dx, [t0, tfinal], interval_dt, C);

end

err_norm1_2 = zeros(size(t_sizes));
err_norm2_2 = zeros(size(t_sizes));
for i = 1:length(t_sizes)
    t_size = t_sizes(i);
    t1 = t_size;
    y1 = phi_sol_f(Nx, t1);

    % run both methods with 1 and 1/2 time step size
    [uu,tt] = forward_euler_FD(U0, xs, dx, [t0, tfinal], ht, C);
    [uu2,tt2] = RK4_FD(U0, xs, dx, [t0, tfinal], ht, C);

    % since N is big enough its ok not to interpolate
    e1_M = phi_1 - y0;
    e2_M = phi_2 - y0;
    ee_M = phi_1 - phi_2;
    err_norm1_2(i) = norm(e1_M, 2)*dx;
    err_norm2_2(i) = norm(e2_M, 2)*dx;
end

%% plot
t_sizes_f = t_sizes.^2;
xticksGrid = [1e-3, 1e-2];
yticksGrid = [1e-6, 1e-4, 1e-2];
figure(1); clf; hold;
figure(1); plt1=plot(t_sizes, err_norm1_2, "b-o", DisplayName="$$\|E_{RK1}\|_1$$");
figure(1); plt2=plot(t_sizes, err_norm2_2, "r--o", DisplayName="$$\|E_{RK2}\|_1$$");
figure(1); plt3=plot(t_sizes, t_sizes.^2,  "k-.o", DisplayName="$$\Delta t^2$$");
xticks(xticksGrid)
yticks(yticksGrid)
xline(xticksGrid,'--',Color=[0.7,0.7,0.7])
yline(yticksGrid,'--',Color=[0.7,0.7,0.7])
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'XDir','reverse')
xlabel("Timestep size $$\Delta t$$", Interpreter="latex");
ylabel("Error $$E$$", Interpreter="latex");
legend([plt1,plt2,plt3], Interpreter="latex");
title("Error of PDE solver in Time domain, $$L_1$$", Interpreter="latex");





function [uu,tt] = Crank_Nicholson(u0, xs, hx, tspan, ht, C)
    Nx = length(xs);
    t0 = tspan(1); tfinal = tspan(2);
    
    % finite difference recurrence
    e = ones(Nx,1);
    A = spdiags([e -2*e e], -1:1, Nx, Nx) * C / hx^2;
    
    F = @(ht, u) ht * A * u;
    R = spdiags([e  2*e e], -1:1, Nx, Nx) * C / hx^2;


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

        % fix the boundary condition u(t, x=0) and u(t, x=L)
        u = [u0(1); u(2:end-1); u0(end)];
        t = t + ht;
    end
end
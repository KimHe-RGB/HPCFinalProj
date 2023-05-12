% this solver assumes Dirichlet BC at left end (x=0) 
% and Neumann BC at x = 1
clear all
u_sol_f = @(x,t) exp(-pi^2*t/4) * cos(pi*x/2);
lbry_f = @(t) exp(-pi^2*t/4);               % left boundary condition
rbry_f = @(t) -pi/2*exp(-pi^2*t/4);         % right boundary condition (neumann)
leftType = "Dirichlet";
rightType = "Neumann";
diffusivity_f = @(x) 2 + cos(pi.*x);      % diffusivity function
% forcing term
force_f = @(x,t) pi^2/2*exp(-pi^2*t/4)*( ...
    -sin(pi*x).*sin(pi/2*x) ...
    + cos(pi/2*x)/2 ...
    + cos(pi*x).*cos(pi/2*x)/2);

init_cond_f = @(x) cos(pi*x/2);         % init condition function=u_sol(x,0)


%% double inhomogenous Dirichlet BC
clear all
u_sol_f = @(x,t) exp(-pi^2*t/4) * sin(pi*x);
diffusivity_f = @(x) 0.*x + 1; % const diffusivity == 1
lbry_f = @(t) 0;               % left BC (Dirichlet)
rbry_f = @(t) 0;               % right BC (Dirichlet)
leftType = "Dirichlet";
rightType = "Dirichlet";
init_cond_f = @(x) sin(pi*x);         % init condition function=u_sol(x,0)
force_f = @(x,t) 3/4*pi^2*exp(-pi^2*t/4)*sin(pi*x);

%% homogenous Dirichlet + inhomogenous Dirichlet BC
clear all
u_sol_f = @(x,t) exp(-pi^2*t/4) * cos(pi*x/2);
diffusivity_f = @(x) 0.*x + 1;              % const diffusivity == 1
lbry_f = @(t) exp(-pi^2*t/4);               % left BC (Dirichlet)
rbry_f = @(t) 0;                            % right BC (Dirichlet)
leftType = "Dirichlet";
rightType = "Dirichlet";
init_cond_f = @(x) cos(pi*x/2);         % init condition function=u_sol(x,0)
force_f = @(x,t) 0.*x;


%% Visualize the solution
CFL = 1/1.6;
range = 4:5;
hxs = 2.^(-range);
hts = hxs.^2 * CFL;

hx = hxs(2);
ht = hts(2);

x0 = 0; xfinal = 1; 
t0 = 0; tfinal = 10; 

u = Heat1D_EE_solver([x0 hx xfinal],[t0 ht tfinal], ...
    init_cond_f, diffusivity_f, force_f,...
    lbry_f, rbry_f, ...
    leftType, rightType);

xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);
real_u = u_sol_f(xs, tfinal);

figure(1); 
hold on;
plot(xs, u, 'b', DisplayName="Approx");
plot(xs, real_u, 'r--', DisplayName="Real");
figure(1); legend(interpreter="latex", Location="southwest");
xlabel("x",Interpreter="latex");
ylabel("u(x,t)",Interpreter="latex");
title('Parabolic solution using Implicit Euler')

%% multiple grid sizes & spatial convergence order
CFL = 1/6;
range = -8:1:-1;
hxs = 2.^(range);
hts = hxs.^2 * CFL;

x0 = 0; xfinal = 1; 
t0 = 0; tfinal = 2; 
err_1s = zeros(length(hxs),1);
err_2s = zeros(length(hxs),1);
err_infs = zeros(length(hxs),1);

for i = 1:length(hxs)
    hx = hxs(i); ht = hts(i);
    % The grid I am using is x_{.5}, x_{1.5},..., x_{N-.5}
    xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);

    % Solve with explicit solver
    u = Heat1D_RK3_solver([x0 hx xfinal],[t0 ht tfinal], ...
        init_cond_f, diffusivity_f, force_f,...
        lbry_f, rbry_f, ...
        leftType, rightType);
    real_u = u_sol_f(xs,tfinal);
    
    M = 1024;
    u_int = reshape(interpft(u,M), [M,1]); 
    real_u_int = reshape(interpft(real_u,M), [M,1]);

    err_1s(i) = norm(u_int - real_u_int, 1);
    err_2s(i) = norm(u_int - real_u_int, 2);
    err_infs(i) = norm(u_int - real_u_int, "inf");
end

% error convergence order
figure(2); clf; 
hold on;
plot(hxs, err_1s, 'r-o', DisplayName="$$L_1$$ error")
plot(hxs, err_2s, 'g-o', DisplayName="$$L_2$$ error")
plot(hxs, err_infs, 'b-o', DisplayName="$$L_\infty$$ error")
plot(hxs, hxs.^2, 'k--o', DisplayName="$$O(h^2)$$");
figure(2); legend(interpreter="latex", Location="southwest");
grid on
xlabel("Grid size $$\Delta x$$",Interpreter="latex");
ylabel("Error", Interpreter="latex");
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'XDir','reverse')
title('Spatial Error convergence, EE')

%% part 2: multiple time step sizes & temporal convergence of Explicit methods
CFL = 1/6;
range = -8:1:-1;
hxs = 2.^(range);
hts = hxs.^2 * CFL;

x0 = 0; xfinal = 1; 
t0 = 0; tfinal = 0.25; 
err_1s = zeros(length(hxs),1);
err_2s = zeros(length(hxs),1);
err_infs = zeros(length(hxs),1);

for i = 1:length(hts)
    hx = hxs(i); ht = hts(i);
    % The grid I am using is x_{.5}, x_{1.5},..., x_{N-.5}
    xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);

    % Solve with explicit solver
    u = Heat1D_EE_solver([x0 hx xfinal],[t0 ht tfinal], ...
        init_cond_f, diffusivity_f, force_f,...
        lbry_f, rbry_f, ...
        leftType, rightType);
    real_u = u_sol_f(xs,tfinal);
    
    M = 1024;
    u_int = reshape(interpft(u,M), [M,1]); 
    real_u_int = reshape(interpft(real_u,M), [M,1]);

    err_1s(i) = norm(u_int - real_u_int, 1);
    err_2s(i) = norm(u_int - real_u_int, 2);
    err_infs(i) = norm(u_int - real_u_int, "inf");
end

% error convergence order
figure(2); clf; 
hold on;
plot(hts, err_1s, 'r-o', DisplayName="$$L_1$$ error")
plot(hts, err_2s, 'g-o', DisplayName="$$L_2$$ error")
plot(hts, err_infs, 'b-o', DisplayName="$$L_\infty$$ error")
plot(hts, 40*hts.^2, 'k--o', DisplayName="$$O(\Delta t^2)$$");
plot(hts, 10*hts, '--o', DisplayName="$$O(\Delta t)$$",Color=[.5 .5 .5]);

figure(2); legend(interpreter="latex", Location="southwest");
grid on
xlabel("Time step size $$\Delta t$$",Interpreter="latex");
ylabel("Error", Interpreter="latex");
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'XDir','reverse')
title('Temporal Error convergence, EE')

%% part 2: multiple time step sizes & temporal convergence of implicit methods
range = -10:1:-3;
hts = 2.^range;
hxs = 2.^range;

x0 = 0; xfinal = 1; 
t0 = 0; tfinal = 1; 
err_1s = zeros(length(hxs),1);
err_2s = zeros(length(hxs),1);
err_infs = zeros(length(hxs),1);

for i = 1:length(hts)
    hx = hxs(i); ht = hts(i);
    % The grid I am using is x_{.5}, x_{1.5},..., x_{N-.5}
    xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);

    % Solve with explicit solver
    u = Heat1D_CN_solver([x0 hx xfinal],[t0 ht tfinal], ...
        init_cond_f, diffusivity_f, force_f,...
        lbry_f, rbry_f, ...
        leftType, rightType);
    real_u = u_sol_f(xs,tfinal);
    
    M = 1024;
    u_int = reshape(interpft(u,M), [M,1]); 
    real_u_int = reshape(interpft(real_u,M), [M,1]);

    err_1s(i) = norm(u_int - real_u_int, 1)*ht;
    err_2s(i) = norm(u_int - real_u_int, 2)*ht;
    err_infs(i) = norm(u_int - real_u_int, "inf")*ht;
end

% error convergence order
figure(2); clf; 
hold on;
plot(hts, err_1s, 'r-o', DisplayName="$$L_1$$ error")
plot(hts, err_2s, 'g-o', DisplayName="$$L_2$$ error")
plot(hts, err_infs, 'b-o', DisplayName="$$L_\infty$$ error")
plot(hts, 40*hts.^2, 'k--o', DisplayName="$$O(\Delta t^2)$$");
plot(hts, 10*hts, '--o', DisplayName="$$O(\Delta t)$$",Color=[.5 .5 .5]);

figure(2); legend(interpreter="latex", Location="southwest");
grid on
xlabel("Time step size $$\Delta t$$",Interpreter="latex");
ylabel("Error", Interpreter="latex");
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'XDir','reverse')
title('Temporal Error convergence, CN')

% this solver assumes Dirichlet BC at left end (x=0) 
% and Neumann BC at x = 1
clear all
u_sol_f = @(x,t) exp(-pi^2*t/4) * cos(pi*x/2);
lbry_f = @(t) exp(-pi^2*t/4);               % left boundary condition
rbry_f = @(t) -pi/2*exp(-pi^2*t/4);         % right boundary condition (neumann)
diffusivity_f = @(x) 2 + cos(pi.*x);      % diffusivity function
% forcing term
force_f = @(x,t) pi^2/2*exp(-pi^2*t/4)*( ...
    -sin(pi*x).*sin(pi/2*x) ...
    + cos(pi/2*x)/2 ...
    + cos(pi*x).*cos(pi/2*x)/2);

init_cond_f = @(x) cos(pi*x/2);         % init condition function=u_sol(x,0)

%% Visualize the solution
CFL = 1/6;
range = 4:5;
hxs = 2.^(-range);
hts = hxs.^2 * CFL;

hx = hxs(2);
ht = hts(2);

x0 = 0; xfinal = 1; 
t0 = 0; tfinal = 1; 

u = Heat1D_RK3_solver([x0 hx xfinal],[t0 ht tfinal], ...
    init_cond_f, diffusivity_f, force_f,...
    lbry_f, rbry_f, ...
    "Dirichlet", "Neumann");

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



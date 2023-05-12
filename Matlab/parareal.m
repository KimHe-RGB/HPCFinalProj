%% this solver assumes Dirichlet BC at x=0 and Neumann BC at x = 1
% We want our true solution to be slowly diffusing away
% so that for very long time step the solution does not 
% hit machine epsilon
clear all;
% expecting to evaluate in a very long time
Total_T = 2^8; % 65536

u_sol_f = @(x,t) exp(-pi^2*t/Total_T) * cos(pi*x/2);
lbry_f = @(t) exp(-pi^2*t/Total_T);               % left boundary condition
% rbry_f = @(t) -pi/2*exp(-pi^2*t/Total_T);         % right bdry cond for xfinal = 1,3,5,7 
rbry_f = @(t) 0;                                    % right bdry cond for xfinal = 2,4,6,8..
leftType = "Dirichlet";
rightType = "Neumann";
diffusivity_f = @(x) 2 + cos(pi.*x);      % diffusivity function
% forcing term
force_f = @(x,t) ... 
    pi^2/2*exp(-pi^2*t/Total_T) ...
       *(-sin(pi*x).*sin(pi/2*x) ...
         +cos(pi/2*x) ...
         +cos(pi*x).*cos(pi/2*x)/2) ... 
    + pi^2/Total_T*exp(-pi^2*t/Total_T)*cos(pi/2*x).*cos(pi*x);

init_cond_f = @(x) cos(pi*x/2);         % init condition function=u_sol(x,0)

%% visualize the solution at a short timespan
CFL = 1/6;
hxs = 2.^-5;
hts = hxs.^2 * CFL;

hx = hxs;
ht = hts;

x0 = 0; xfinal = 4; 
t0 = 0; tfinal = 100; 

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



%%
% Total T = 2^16
% cores = 2^4       ==>     T interval / core ~ 2^12
%
% --- Initialize cost ---
% coarse time grid size ~ 2^0
% number of coarse time grids = 2^16
% ---------------------- 
% 
% target fine time grid size ~ 2^-5 ==> MOL is O(ht^2) + O(hx^2), so hx ~
% ht to maximize accuracy ==> fine space grid size ~ 2^-5, target error O(2^-10)
% 
% number of time step iterations = Total T / fine time size = 2^21
% number of time step per core ~ 2^17
% 
% every T interval conatins 2^12 coarse time grids, so every 2^12 coarse
% time grids, we record the coarse approximation as an initial condition 

% we know that due to 
% split T into several trunks; each trunk will parallelize later 
T0 = 0;
core_num = 16;
interval_size = (Total_T - T0)/core_num;
intervals = zeros(2, core_num);
for i = 1:core_num
    intervals(:,i) = [t0+(i-1)*interval_size; t0+i*interval_size];
end

% Decide the boundary conditions at x = 0 and x = 4
x0 = 0; xfinal = 4;
% evaluate on a coarse time grid with large time step
% the number of coarse time grids is multiple of interval_num
coarse_hx = 2^1;         % coarse_hx = 2
CFL = 2^-3;
coarse_ht = hxs^2 * CFL; % coarse_ht = 2^-1;
t0 = T0; tfinal = Total_T;
u_coarse = Heat1D_EE_solver([x0 coarse_hx xfinal],[t0 coarse_ht tfinal], ...
    init_cond_f, diffusivity_f, force_f,...
    lbry_f, rbry_f, ...
    leftType, rightType);

% fine time grid size is much smaller than coarse, and should work on a
% much smaller time range. We should also ensure that the coarse time grid
% is a subset of fine time grid.
% error of CN ~ O(ht^2) + O(hx^2), to maximize we let ht = hx
fine_ht = 2^-5;
finx_hx = 2^-5;

% from the result of coarse grid, 
init_cond_fs = zeros(2, core_num);

% This part goes parallel
parfor i = 1:core_num
    ts = intervals(:,i);
    t0 = ts(1); tfinal = ts(2);
    disp(tfinal);
        
end




























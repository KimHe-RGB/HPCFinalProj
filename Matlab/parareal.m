% % We want our true solution to be slowly diffusing away
% % so that for very long time step the solution does not 
% % hit machine epsilon
% However this is probably not a good idea because it make the system of
% ODEs stiff to solve, which in long T make Crank-Nicolson unstable 
% (CN error will oscillate and will not decrease, and accumulate across time) 
% 
% clear all;
% % expecting to evaluate in a very long time
% Total_T = 2^10;
% 
% u_sol_f = @(x,t) exp(-pi^2*t/Total_T) * cos(pi*x/2);
% % lbry_f = @(t) exp(-pi^2*t/Total_T);               % left boundary condition
% % leftType = "Dirichlet";
% % rbry_f = @(t) -exp(-pi^2*t/Total_T);
% % rightType = "Dirichlet";
% lbry_f = @(t) 0; 
% leftType = "Neumann";
% rbry_f = @(t) 0;
% rightType = "Neumann";
% 
% diffusivity_f = @(x) 2 + cos(pi.*x);      % diffusivity function
% % forcing term
% force_f = @(x,t) ... 
%     pi^2/4*exp(-pi^2*t/Total_T) ...
%        *(-2*sin(pi*x).*sin(pi/2*x) ...
%          +2*cos(pi/2*x) ...
%          +cos(pi*x).*cos(pi/2*x)) ... 
%     - pi^2/Total_T*exp(-pi^2*t/Total_T)*cos(pi/2*x);
% 
% init_cond_f = @(x) cos(pi*x/2);         % init condition function=u_sol(x,0)
% 
% x0 = 0; xfinal = 2; 
%%
clear all
Total_T = 64;

u_sol_f = @(x,t) exp(-pi^2*t/4) * cos(pi*x/2);
% lbry_f = @(t) exp(-pi^2*t/4);               % left boundary condition (dirichlet)
% leftType = "Dirichlet";
lbry_f = @(t) 0;                            % left boundary condition (neumann)
leftType = "Neumann";
rbry_f = @(t) -pi/2*exp(-pi^2*t/4);         % right boundary condition (neumann)
rightType = "Neumann";

diffusivity_f = @(x) 2 + cos(pi.*x);      % diffusivity function

% forcing term
force_f = @(x,t) pi^2/2*exp(-pi^2*t/4)*( ...
    -sin(pi*x).*sin(pi/2*x) ...
    + cos(pi/2*x)/2 ...
    + cos(pi*x).*cos(pi/2*x)/2);

init_cond_f = @(x) cos(pi*x/2);         % init condition function=u_sol(x,0)

x0 = 0; xfinal = 1; 
%% for comparison, directly solve the system with the fine operator
t0 = 0; tfinal = Total_T; 
hx = 2^-4;
ht = hx;
xs = x0+hx/2:hx:xfinal-hx/2; 
init_u = init_cond_f(xs);
coarse_solver_u = Heat1D_IE_solver_coarse([x0 hx xfinal],[t0 ht tfinal], ...
    init_u, diffusivity_f, force_f,...
    lbry_f, rbry_f, ...
    leftType, rightType);
tic
fine_solver_u = Heat1D_CN_solver_fine([x0 hx xfinal],[t0 ht tfinal], ...
    init_u, diffusivity_f, force_f,...
    lbry_f, rbry_f, ...
    leftType, rightType);
real_u = u_sol_f(xs, tfinal);
toc

M = 1024;
finer_hx = (xfinal-x0)/M;
coarse_solver_uInt = interpft(coarse_solver_u,M);
fine_solver_uInt = interpft(fine_solver_u,M);
real_uInt = interpft(real_u,M);
finer_xs = x0+finer_hx/2:finer_hx:xfinal-finer_hx/2; 

figure(2); 
hold on;
plot(xs, coarse_solver_u, 'b-', DisplayName="Coarse Solver");
plot(xs, fine_solver_u, 'r-', DisplayName="Fine Solver");
plot(xs, real_u, 'g--', DisplayName="Real");
figure(2); legend(interpreter="latex", Location="southwest");
xlabel("x",Interpreter="latex");
ylabel("u(x,t)",Interpreter="latex");

error_coarse = norm(reshape(real_uInt,size(coarse_solver_uInt)) - coarse_solver_uInt, 2)
error_fine = norm(reshape(real_uInt,size(fine_solver_uInt)) - fine_solver_uInt, 2)
%%
% target fine time grid size ~ 2^-5 ==> MOL is O(ht^2) + O(hx^2), so hx ~
% ht to maximize accuracy ==> fine space grid size ~ 2^-5
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
    intervals(:,i) = [T0+(i-1)*interval_size; T0+i*interval_size];
end

% fine time grid size is much smaller than coarse, and should work on a
% much smaller time range. We should also ensure that the coarse time grid
% is a subset of fine time grid.
% error of CN ~ O(ht^2) + O(hx^2), to maximize we let ht = hx
fine_ht = 2^-5;
fine_hx = fine_ht;
coarse_ht = 1;          % coarse ht is much larger than fine ht
coarse_hx = fine_hx;    % make this consistent with the fine grid

% the grid that we want to evaluate the PDE, 
% here we use a staggered grid
fine_xs = x0+fine_hx/2:fine_hx:xfinal-fine_hx/2; 
Nxs = length(fine_xs);

%%
% already have U^0_0 = G^0_0 = init_cond(t=0)
% initialization U^0_{n+1} = G(t_n, t_n, U^0_n) from n = 0,...core_num-1
U = zeros(core_num+1,Nxs);
U(1,:) = init_cond_f(fine_xs); 
F = zeros(core_num+1, Nxs);
F(1,:) = init_cond_f(fine_xs);
G = zeros(core_num+1, Nxs);
G(1,:) = init_cond_f(fine_xs);
for i = 1:core_num
    t0 = T0+(i-1)*interval_size; tfinal = T0+i*interval_size;
    % coarse approximation from i -> i+1
    G_coarse = Heat1D_IE_solver_coarse( ...
        [x0 coarse_hx xfinal],[t0 coarse_ht tfinal], U(i,:), ...
        diffusivity_f, force_f,...
        lbry_f, rbry_f, ...
        leftType, rightType);
    % initial value for i+1 is the ending value of i
    U(i+1,:) = G_coarse;
    G(i+1,:) = G_coarse;
end

% parareal iteration
parareal_iters = 16;
for iter = 1:parareal_iters
    % from the result of coarse grid, compute the fine grid
    % initialization of F^0_n+1 = F(t_n. t_n+1, G^0_n) from n = 0,...core_num-1
    % This part goes parallel
    for i = 1:core_num
        t0_1 = T0+(i-1)*interval_size; tfinal_1 = T0+i*interval_size;
        F_fine = Heat1D_CN_solver_fine( ...
            [x0 fine_hx xfinal],[t0_1 fine_ht tfinal_1], U(i,:), ...
            diffusivity_f, force_f,...
            lbry_f, rbry_f, ...
            leftType, rightType);
        F(i+1,:) = F_fine;
    end
    % coarse update
    for j = 2:core_num+1
        t0 = T0+(j-1)*interval_size; tfinal = T0+j*interval_size;
        % new approximation from i -> i+1
        G_coarse_new = Heat1D_IE_solver_coarse( ...
            [x0 coarse_hx xfinal],[t0 coarse_ht tfinal], U(j-1,:), ...
            diffusivity_f, force_f,...
            lbry_f, rbry_f, ...
            leftType, rightType);
        % Compute the correction
        % Add the correction to the approximate solution on the coarse grid
        G_coarse_new = reshape(G_coarse_new, [1, Nxs]);
        U(j,:) = G_coarse_new - G(j,:) + F(j,:);
        G(j,:) = G_coarse_new;
    end
end

parareal_u = U(end,:);
real_u = u_sol_f(fine_xs, Total_T);

% Plot and visualize the solution
figure(2); plot(fine_xs, parareal_u, "-.", DisplayName="Parareal");

%% Compute the parareal error over x grid, and compare to the CN solver 
% M = 2^8;
% finer_hx = (xfinal-x0)/M;
% finer_xs = x0+finer_hx/2:finer_hx:xfinal-finer_hx/2; 
% 
% parareal_uInt = interpft(parareal_u,M);
% real_uInt = interpft(real_u,M);
% fine_solver_uInt = interpft(fine_solver_u,M);
% parareal_error = abs(parareal_uInt - reshape(real_uInt,size(parareal_uInt)));
% CN_error = abs(fine_solver_uInt - reshape(real_uInt,size(fine_solver_uInt)));
parareal_error = abs(parareal_u - real_u);


figure(3); 
hold on;
plot(fine_xs, parareal_error, DisplayName=sprintf("Parareal, %d Iter", parareal_iters));
figure(3); legend(interpreter="latex", Location="southwest");
xlabel("x",Interpreter="latex");
ylabel("Error",Interpreter="latex");
set(gca,'YScale','log')
title("Error in L2 norm");
%%
CN_error = abs(fine_solver_u - reshape(real_u,size(fine_solver_u)));
figure(3); plot(fine_xs, CN_error, 'k-', DisplayName="CN");


%% adapted from the Implicit Euler Solver
% served as coarse solver
function u = Heat1D_IE_solver_coarse(xspan, tspan, ...
    icu, df, F,...
    lbry, rbry, leftBCType, rightBCType)

    t0 = tspan(1); ht = tspan(2); tfinal = tspan(3);
    x0 = xspan(1); hx = xspan(2); xfinal = xspan(end);    
    % the grid that we want to evaluate the PDE, 
    % here we use a staggered grid
    xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);
        
    % init loop
    t = t0; i = 0; 
    done = t >= tfinal;
    
    % initial condition 
    assert(length(icu) == Nxs);
    u = reshape(icu, [Nxs, 1]);

    % spatial discretization
    [L, r] = Spatial_Operator_Heat1D(xspan,df, leftBCType, rightBCType);
    
    % Linear system to solve for Implicit Euler
    I = eye(Nxs); 
    R = @(t) [lbry(t); zeros(Nxs-2,1); rbry(t)] .* r;
    RHS = @(t) reshape(-F(xs, t), [Nxs,1]) + R(t);
    Fn = @(u, t, ht) (I - ht*L) \ (u - ht*RHS(t));
    
    while ~done
        i = i + 1;
        if(t + ht >= tfinal)
            ht = tfinal - t;
            done = true;
        end
        % One Step Method (Backward/Implicit Euler)
        u = Fn(u, t, ht);
        t = t + ht;
    end
end

%% Solve 1d Heat Equation u_t = (d(x)u_x)_x + f
% handle Dirichlet BCs or/and Neumann BCs
% 
% PARAMETERS:
% xspan = [x0, grid_size, xfinal]
% F : forcing function as a function of x and t
% icf: initial condition function of x
% df: thermo diffusity function of space x 
% F: forcing function of x and t
% lbry, rbry: left or right boundary condition function of t
% leftBCType, rightBCType: Neumann or Dirichlet

% Crank-Nicolson Method
% O(h^2), O(dt)
% unconditionally stable
function u = Heat1D_CN_solver_fine(xspan, tspan, ...
    icu, df, F,...
    lbry, rbry, leftBCType, rightBCType)

    t0 = tspan(1); ht = tspan(2); tfinal = tspan(3);
    x0 = xspan(1); hx = xspan(2); xfinal = xspan(end);    
    % the grid that we want to evaluate the PDE, 
    % here we use a staggered grid
    xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);
        
    % init loop
    t = t0; i = 1;
    done = t >= tfinal;

    % initial condition
    assert(length(icu) == Nxs);
    u = reshape(icu, [Nxs, 1]);

    % spatial discretization
    [L, r] = Spatial_Operator_Heat1D(xspan,df, leftBCType, rightBCType);

    % Linear system to solve for Crank-Nicolson
    I = eye(Nxs); 
    R = @(t) [lbry(t); zeros(Nxs-2,1); rbry(t)] .* r;
    RHS = @(t) reshape(-F(xs, t), [Nxs,1]) + R(t);
    Fn = @(u,t,ht) (I-ht*L/2) \ ((I+ht*L/2)*u - ht*RHS(t+ht/2));

    while ~done
        i = i + 1;
        if(t + ht >= tfinal)
            ht = tfinal - t;
            done = true;
        end
        % One Step Method
        u = Fn(u, t, ht);
        t = t + ht;
    end
end





























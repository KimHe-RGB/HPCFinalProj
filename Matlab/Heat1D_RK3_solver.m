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

% RK3 method
% to be tested
function u = Heat1D_RK3_solver(xspan, tspan, ...
    icf, df, F,...
    lbry, rbry, leftBCType, rightBCType)

    t0 = tspan(1); ht = tspan(2); tfinal = tspan(3);
    x0 = xspan(1); hx = xspan(2); xfinal = xspan(end);    
    % the grid that we want to evaluate the PDE, 
    % here we use a staggered grid
    xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);
        
    % init loop
    t = t0; i = 1;
    u = reshape(icf(xs), [Nxs, 1]);

    done = t >= tfinal;

    [L, r] = Spatial_Operator_Heat1D(xspan,df, leftBCType, rightBCType);

    
    R = @(t) [lbry(t); zeros(Nxs-2,1); rbry(t)] .* r;
    RHS = @(t) reshape(F(xs, t), [Nxs,1]) - R(t);
    
    % RHS of the ODE: u_t = u_xx + f = L*u + RHS(t)
    Fn = @(u, t) L*u + RHS(t);

    while ~done
        i = i + 1;
        if(t + ht >= tfinal)
            ht = tfinal - t;
            done = true;
        end
        % 0   | 0   0   0
        % 1/2 |1/2  0   0 
        % 1   |-1   2   0
        %      1/6 2/3 1/6
        % RK3 time stepping
        k1 = Fn(u, t);
        k2 = Fn(u + ht*k1 /2, t + ht/2);
        k3 = Fn(u - ht*k1 + ht*k2*2, t + ht);
        u = u + ht*(k1 + k2 *4 + k3)/6;
        t = t + ht;
    end
end

function res = RK4(ht, u0, f)

    k1 = f(u0, t);
    k2 = f(u0 + ht*k1, t+ht);
    k3 = f(u0 + ht*k1/4 + ht*k2/4, t+ht);

    res = u0 + (k1 + k2 + 4*k3)*ht/6;

end
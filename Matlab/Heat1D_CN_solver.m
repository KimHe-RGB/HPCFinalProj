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
function u = Heat1D_CN_solver(xspan, tspan, ...
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

    % Linear system to solve for Implicit Euler
    I = eye(Nxs); 

    R = @(t) [lbry(t); zeros(Nxs-2,1); rbry(t)] .* r;
    RHS = @(t) reshape(-F(xs, t), [Nxs,1]) + R(t);
    Fn = @(u, t, ht) (I + ht*L)*u - ht*RHS(t);

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


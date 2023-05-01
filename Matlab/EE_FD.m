% solve the heat equation using finite difference discretization, with forward euler method
% explicit Euler
function [uu,tt] = EE_FD(u0, hx, tspan, ht, C)
    Nx = length(u0);
    t0 = tspan(1); tfinal = tspan(2);
    
    % forward euler method
    e = ones(Nx,1);
    A = spdiags([e -2*e e],-1:1,Nx,Nx) * C / hx^2;
    t = t0;
    u = u0;
    
    i=1;    t_steps = ceil((tfinal - t0) / ht)+1;
    tt = zeros(1,t_steps); uu = zeros(Nx,t_steps);
    tt(i) = t; uu(:,i) = u;
    done = t >= tfinal;
    while ~done
        i=i+1;
        if(t + ht >= tfinal)
            ht = tfinal - t;
            done = true;
        end
        u = u + ht * A * u;
        % fix the boundary condition u(t, x=0) and u(t, x=L)
        u = [u0(1); u(2:end-1); u0(end)];
        t = t + ht;

        tt(i) = t; uu(:,i) = u;
    end
end

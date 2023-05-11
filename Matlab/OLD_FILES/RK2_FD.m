function [uu,tt] = RK2_FD(u0, hx, tspan, ht, C)
    Nx = length(u0);
    t0 = tspan(1); tfinal = tspan(2);
    
    % finite difference recurrence
    e = ones(Nx,1);
    A = spdiags([e -2*e e],-1:1,Nx,Nx) * C / hx^2;
    F = @(u) A * u;

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
        % u_{n+1/2} = u_{n} + F(t_n, t_n+0.5*ht, u_n)
        u_1 = u + ht/2*F(u);
        % u_{n+1} = u_{n} + F(t_n, t_n+ht, u_{n+1/2})
        u = u + ht*F(u_1);
        
        % fix the boundary condition u(t, x=0) and u(t, x=L)
        u = [u0(1); u(2:end-1); u0(end)];
        t = t + ht;

        tt(i) = t; uu(:,i) = u;
    end
end

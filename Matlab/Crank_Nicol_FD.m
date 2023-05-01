function [uu,tt] = Crank_Nicol_FD(u0, hx, tspan, ht, C)
    Nx = length(u0);
    t0 = tspan(1); tfinal = tspan(2);
    
    % finite difference recurrence
    b = (-C/2/hx^2)*ones(Nx,1);     % Super diagonal on LHS
    c = b;                          % Subdiagonal on LHS
    a = (1/ht)*ones(Nx,1) - (b+c);  % Main Diagonal on LHS
    at = (1/ht + b + c);            % Coefficient of u_i^k on RHS
    
    a(1) = 1; a(end) = 1; 
    % Fix coefficients of boundary nodes
    F = spdiags([b,a,c],-1:1,Nx,Nx);
    F(1,2) = 0; F(end, end-1) = 0;
    
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
        %         d = - [0; c(2:end-1).*u(1:end-2); 0] ...
        %             + [u0(1); at(2:end-1).*u(2:end-1); u0(end)] ...
        %             - [0; b(2:end-1).*u(3:end); 0];
        d = [u0(1);
            - c(2:end)*u(1:end-2)
            + at(2:end-1).*u(2:end-1)
            + b(2:end-1).*u(3:end);
            u0(end)];
        u = F\d;
        t = t + ht;

        tt(i) = t; uu(:,i) = u;
    end
end

%% Finite difference method to solve 1d Heat 
% u_t = C*u_xx + f(x,t)
C = 1;
N = 128;
L = 60;
hx = L / N;

t0 = 0; tfinal = 60;
ht = 0.01;

% initial value u(t=0, x)
phi_sol_f = @(x, t) 1 / 2 * sech(1/2*(x - t)).^2;
xs = reshape((0:N-1)*hx, [N,1]) - L/2;
u0 = phi_sol_f(xs, 0);
u0 = reshape(u0, [N,1]);

% figure(1); clf; hold;
% figure(1); plot(xs, u0, "-o", DisplayName="u(0,x)");
% xlabel("1d space $$x$$", Interpreter="latex");
% ylabel("$$u(x,t)$$", Interpreter="latex");
% legend(Interpreter="latex");
% title("1d Heat equation", Interpreter="latex");
[uu, tt] = finite_diff(u0, xs, hx, [t0 tfinal], ht, C);

figure(2); clf; 
hold on;
plt0 = plot(xs, u0, '-.k');
plt1 = plot(xs, uu(:,1), '-r');
xlim([-L/2, L/2])
ylim([-0.5, 1])
grid on
figure(2); legend([plt0 plt1], ["y0", "yt"], interpreter="latex", Location="southwest");
title(sprintf('t = %7.5f',tt(1)),'fontsize',18), drawnow

set(gcf,'doublebuffer','on')
disp('press <return> to begin'), pause  % wait for user input

for tn = 1:length(tt)
    if(mod(tn-1, 100) == 0 || tn==length(tt))
        set(plt1, 'ydata', uu(:,tn))
        title(sprintf('N = %d, h = %4.6f, t = %7.5f',N,ht,tt(tn)),'fontsize',18), drawnow
    end
end

function [uu,tt] = finite_diff(u0, xs, hx, tspan, ht, C)
    N = length(xs);
    t0 = tspan(1); tfinal = tspan(2);
    
    e = ones(N,1);
    A = spdiags([e -2*e e],-1:1,N,N) * C / hx^2;
    t = t0;
    u = u0;
    
    i=1;    t_steps = ceil((tfinal - t0) / ht) + 1;
    tt = zeros(1,t_steps); uu = zeros(N,t_steps);
    done = t >= tfinal;
    while ~done
        tt(i) = t; uu(:,i) = u; i=i+1;
        if(t + ht >= tfinal)
            ht = tfinal - t;
            done = true;
        end
        u = u + ht * A * u;
        % fix the boundary condition u(t, x=0) and u(t, x=L)
        %  u = [u0(1); u(2:end-1); u0(end)];
        t = t + ht;
    end
end



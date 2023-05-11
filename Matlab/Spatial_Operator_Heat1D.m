% RETURN:
% L : an Nx x Nx discretize operator
% r : the corrector term for boundary conditions, function of time
function [L, r] = Spatial_Operator_Heat1D(xspan, df, leftBCType, rightBCType)
    % grid
    x0 = xspan(1); hx = xspan(2); xfinal = xspan(end);    
    % staggered grid, interior pts
    xs = x0+hx/2:hx:xfinal-hx/2; Nxs = length(xs);
    % regular grid
    xr = x0:hx:xfinal;  
    
    % diffusity function evaled on regular grid
    dxr = df(xr);
    
    % Second-order Spatial discretization operator for (d(x)u_x)_x
    diag1 = [dxr(1:end)];                    % higher subdiagonal
    diag2 = [-dxr(1:end-1)-dxr(2:end),0];   % diagonal
    diag3 = [dxr(2:end),0];                 % lower subdiag
    L = spdiags([diag3; diag2; diag1]',-1:1,Nxs,Nxs);

    %  BC modification to the RHS
    % modify L and add inhomogenous vector to match specified  
    if (leftBCType == "Neumann")
        L_corrector = -dxr(1)/hx;

        L(1,1) = -dxr(2);
        L(1,2) = dxr(2);
    
    elseif (leftBCType == "Dirichlet") 
        L_corrector = -2*dxr(1)/hx^2;

        L(1,1) = -2*dxr(1) - dxr(2);
        L(1,2) = dxr(2);
    else
        error("Invalid left boundary type")
    end
    
    if (rightBCType == "Neumann")
        R_corrector = -dxr(end)/hx;

        L(end,end) = -dxr(end-1);
        L(end,end-1) = dxr(end-1);
    elseif (rightBCType == "Dirichlet") 
        R_corrector = -2*dxr(end)/hx^2;

        L(end,end) = -2*dxr(end) - dxr(end-1);
        L(end,end-1) = dxr(end-1);
    else
        error("Invalid left boundary type")
    end

    L = L ./ hx^2;
    r = [  L_corrector;
            zeros(Nxs-2,1);
            R_corrector];
end
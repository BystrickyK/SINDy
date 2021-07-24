function dxdt = lorenz(t,X,u,params)
%Input form (t, X, u, parameters)
    
    u = u(t);
    u1 = u(1);
    u2 = u(2);
    u3 = u(3);
    
    x = X(1);
    y = X(2);
    z = X(3);
    
    dxdt = [params.sigma*(y-x) + u1,...
        x*(params.rho - z) - y + sin(u2),...
        x*y - params.beta*z + u1*x]';
    
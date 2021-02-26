%% Initialize
clear all;
close all;
clc;

params = [0.2, 0.2, 0.1,...             % pendulum weights
            0.5, 0.25, 0.1,...           % penndulum moments of inertia around CoMs
            0.15, 0.125, 0.05,...          % pendulum distances from joints to CoMs
            0.3, 0.25, 0.1,...           % pendulum lengths
            0.001, 0.001, 0.001];     % joint damping coefficients
        
odefun = @(t, X)triplePendCart(t, X, 0, params);

%% Solve
x0 = [0.5, pi, -pi, pi*31/32, 0, 0, 0, 0]; % initial conditions
tspan = [0 60]; % time span

% Solve the system of ODEs
sol = ode15s(odefun, tspan, x0);

results = array2table([sol.x', sol.y']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'phi3', 'Ds', 'Dphi1','Dphi2', 'Dphi3'};

%% Plot
stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
grid on
%% Initialize
clear all;
close all;
clc;

l_1 = 0.3; l_2 = 0.3; l_3 = 0.3;
a_1 = 0.1; a_2 = 0.1; a_3 = 0.1;
m_1 = 3; m_2 = 3; m_3 = 3;
I_1 = 0.3*m_1*a_1^2; I_2 = 0.3*m_2*a_2^2; I_3 = 0.3*m_3*a_3^2;
b_1 = 0.01; b_2 = 0.005; b_3 = 0.002;

params = [m_1, m_2, m_3,...             % pendulum weights
            I_1, I_2, I_3,...           % penndulum moments of inertia around CoMs
            a_1, a_2, a_3,...          % pendulum distances from joints to CoMs
            l_1, l_2, l_3,...           % pendulum lengths
            b_1, b_2, b_3];     % joint damping coefficients
        
odefun = @(t, X)triplePendCart(t, X, 0, params);

%% Solve
x0 = [0.5, pi, -pi, pi*99/100, 0, 0, 0, 0]; % initial conditions
tspan = [0 30]; % time span

% Solve the system of ODEs
sol = ode15s(odefun, tspan, x0);

x = logspace(0, 1.47, 1350) - 1;
y = deval(sol, x);
results = array2table([x', y']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'phi3', 'Ds', 'Dphi1','Dphi2', 'Dphi3'};

%% Plot
stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
grid on

%% Animation
q = [results.s, results.phi1, results.phi2, results.phi3];

P_c = @(q) [q(1); 0];        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))];         % Pendulum 1

P_2 = @(q) [l_2*sin(q(3)) + l_1*sin(q(2)) + q(1);
    -l_2*cos(q(3)) - l_1*cos(q(2))];         % Pendulum 2

P_3 = @(q) [l_3*sin(q(4)) + l_2*sin(q(3)) + l_1*sin(q(2)) + q(1);
    -l_3*cos(q(4)) - l_2*cos(q(3)) - l_1*cos(q(2))];         % Pendulum 3


h = figure();
plot(0,0)
hold on
grid on
xlim([-0.5 1.5])
ylim([-1 1])
for k = 1:length(q)
    title(x(k))
    
    pc = P_c(q(k, :));
    p1 = P_1(q(k, :));
    p2 = P_2(q(k, :));
    p3 = P_3(q(k, :));
    
    cla
    plot([pc(1), p1(1)], [pc(2), p1(2)], 'kO-', 'LineWidth', 3)
    plot([p1(1), p2(1)], [p1(2), p2(2)], 'rO-', 'LineWidth', 3)
    plot([p2(1), p3(1)], [p2(2), p3(2)], 'bO-', 'LineWidth', 3)
%     drawnow
    
    frames(k) = getframe(gcf);
end

writerObj = VideoWriter('triplePendulumCart.avi');
writerObj.FrameRate = 30;

open(writerObj);
for k = 1:length(frames)
   frame = frames(k);
   writeVideo(writerObj, frame);
end
close(writerObj);
    
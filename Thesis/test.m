%% Initialize
clear all;
close all;
clc;

l_1 = 0.3; l_2 = 0.3; l_3 = 0.3;
a_1 = 0.1; a_2 = 0.1; a_3 = 0.1;
m_1 = 3; m_2 = 3; m_3 = 3; m_c = 3;
I_1 = 0.3*m_1*a_1^2; I_2 = 0.3*m_2*a_2^2; I_3 = 0.3*m_3*a_3^2;
b_1 = 0.0; b_2 = 0.00; b_3 = 0.00;

params = [m_1, m_2, m_3,...             % pendulum weights
            I_1, I_2, I_3,...           % penndulum moments of inertia around CoMs
            a_1, a_2, a_3,...          % pendulum distances from joints to CoMs
            l_1, l_2, l_3,...           % pendulum lengths
            b_1, b_2, b_3];     % joint damping coefficients
        
odefun = @(t, X)triplePendCart(t, X, 0, params);

%% Solve
x0 = [0.5, pi, -pi, pi*99/100, 0, 0, 0, 0]; % initial conditions
tspan = [0 20]; % time span

% Solve the system of ODEs
sol = ode15s(odefun, tspan, x0);

x = (logspace(0, 0.6, 1200) - 1) * 6.6;
% x = linspace(0, 30, 600);
y = deval(sol, x);
results = array2table([x', y']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'phi3', 'Ds', 'Dphi1','Dphi2', 'Dphi3'};

%% Plot
% stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
% grid on

%% Define important coordinates and their derivatives
q = [results.s, results.phi1, results.phi2, results.phi3];
dq = [results.Ds, results.Dphi1, results.Dphi2, results.Dphi3];

P_c = @(q) [q(1); 0];        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))];         % Pendulum 1

P_2 = @(q) [l_2*sin(q(3)) + l_1*sin(q(2)) + q(1);
    -l_2*cos(q(3)) - l_1*cos(q(2))];         % Pendulum 2

P_3 = @(q) [l_3*sin(q(4)) + l_2*sin(q(3)) + l_1*sin(q(2)) + q(1);
    -l_3*cos(q(4)) - l_2*cos(q(3)) - l_1*cos(q(2))];         % Pendulum 3

dP_c = @(q, dq) [dq(1); 0];

dP_1 = @(q,dq) [dq(1) + a_1*cos(q(2))*dq(2);
                a_1*sin(q(2))*dq(2)];

dP_2 = @(q,dq) [dq(1) + a_2*cos(q(3))*dq(3) + l_1*cos(q(2))*dq(2);
                a_2*sin(q(3))*dq(3) + l_1*sin(q(2))*dq(2)];
            
            
dP_3 = @(q,dq) [dq(1) + a_3*cos(q(4))*dq(4) + l_2*cos(q(3))*dq(3) + l_1*cos(q(2))*dq(2);
                a_3*sin(q(4))*dq(4) + l_2*sin(q(3))*dq(3) + l_1*sin(q(2))*dq(2)];

%% Calculate kinetic, potential and total energies

T = @(q,dq) (0.5 * m_1 * dP_1(q,dq)' * dP_1(q,dq) + ...
    0.5 * m_2 * dP_2(q,dq)' * dP_2(q,dq) + ...
    0.5 * m_3 * dP_3(q,dq)' * dP_3(q,dq) + ...
    0.5 * I_1 * dq(2)^2 + ...
    0.5 * I_2 * dq(3)^2 + ...
    0.5 * I_3 * dq(4)^2 + ...
    0.5 * m_c * dP_c(q,dq)' * dP_c(q,dq));  % Kinetic energy


g = 9.0665;      % Gravitational acceleration
V = @(q) (g * m_1 * P_1(q)'*[0;1] + ...
    g * m_2 * P_2(q)'*[0;1] + ...
    g * m_3 * P_3(q)'*[0;1]);

KE = [];    % Kinetic
PE = [];    % Potential
TE = [];    % Total
for k = 1:length(q)
    KE(k) = T(q(k, :),dq(k, :));
    PE(k) = V(q(k, :));
    TE(k) = KE(k)+PE(k);
end


%% Animation
h = figure();
a1 = subplot(1,2,1);
hold on
grid on
xlim([-0.5 1.5])
ylim([-1 1])
axis manual

a2 = subplot(3,2,2);
xlim([0 30])
ylim([min(KE)-10, max(KE)+10])
ylabel("Kinetic")
l2 = animatedline(a2);

a3 = subplot(3,2,4);
xlim([0 30])
ylim([min(PE)-10, max(PE)+10])
ylabel("Potential")
l3 = animatedline(a3);

a4 = subplot(3,2,6);
xlim([0 30])
ylim([min(TE)-10, max(TE)+10])
ylabel("Total")
l4 = animatedline(a4);

for k = 1:length(q)
    cla(a1)
    
    pc = P_c(q(k, :));
    p1 = P_1(q(k, :));
    p2 = P_2(q(k, :));
    p3 = P_3(q(k, :));
    
    title(a1, x(k))
    plot(a1, [pc(1), p1(1)], [pc(2), p1(2)], 'kO-', 'LineWidth', 3)
    plot(a1, [p1(1), p2(1)], [p1(2), p2(2)], 'rO-', 'LineWidth', 3)
    plot(a1, [p2(1), p3(1)], [p2(2), p3(2)], 'bO-', 'LineWidth', 3)

    addpoints(l2, x(k), KE(k))
    addpoints(l3, x(k), PE(k))
    addpoints(l4, x(k), TE(k))
    
    drawnow
    frames(k) = getframe(gcf);
    
end

writerObj = VideoWriter('triplePendulumCart_tmp.avi');
writerObj.FrameRate = 30;

open(writerObj);
for k = 1:length(frames)
   frame = frames(k);
   writeVideo(writerObj, frame);
end
close(writerObj);
    
%% Initialize
disp("Initializing...")
clear all;
close all;
clc;

l_1 = 0.3; l_2 = 0.3; l_3 = 0.3;
a_1 = 0.15; a_2 = 0.15; a_3 = 0.15;
m_1 = 4; m_2 = 3; m_3 = 3; m_c = 1;
I_1 = 0.9*m_1*a_1^2; I_2 = 0.9*m_2*a_2^2; I_3 = 0.9*m_3*a_3^2;
b_1 = 0.07; b_2 = 0.05; b_3 = 0.03;

params = [m_1, m_2, m_3,...             % pendulum weights
            I_1, I_2, I_3,...           % penndulum moments of inertia around CoMs
            a_1, a_2, a_3,...          % pendulum distances from joints to CoMs
            l_1, l_2, l_3,...           % pendulum lengths
            b_1, b_2, b_3];     % joint damping coefficients
        
        
%% Create external input signal
disp("Creating input signal...")
t_end = 12;     % simulation time
f_s = 100;     % sampling
f_c = 5;        % cutoff
u_a = 20;      % input power

% Random walk
    [filt_b, filt_a] = butter(3, f_c/(f_s/2));  % (order, cutoff freq)
    u_t = 0:(1/f_s):t_end;
    u = u_a * (rand(length(u_t), 1) - 0.5);
    u = filter(filt_b, filt_a, u);  % defines the cart velocity!
    
    du = diff(u)/(1/f_s);   % get cart acceleration by differentiation!
    du(end+1) = du(end);
    du(end*5/6:end) = 0;
%     figure();subplot(121);plot(u,'b');subplot(122);plot(du,'r')
    
    u_f = @(t) interp1(u_t, du, t);

% Constant discontinuous
%     u = (randi(6,[180,1])-u_a/2);
%     u_t = linspace(0, t_end, length(u));
%     u_f = @(t) interp1(u_t, u, t, 'previous');

% Sin
%     u_t = 0:(1/f_s):t_end;
%     u = sin(u_t*4);
%     u_f = @(t) interp1(u_t, u, t);

% Pulses
%     nk = t_end*f_s; % number of samples
%     u_t = linspace(0, t_end, nk);
%     u = zeros([nk,1]);
%     u(0.02*nk:0.05*nk) = 30000;
%     u(0.055*nk:0.085*nk) = -30000;
%     u_f = @(t) interp1(u_t, u, t);
tic
odefun = @(t, X)triplePendCart(t, X, u_f, params);
toc

%% Solve
disp("Solving...")
x0 = [0, 0, 0, 0, 0, 0, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
sol = ode15s(odefun, tspan, x0);

% x = (logspace(0, 0.3, 1200) - 1) * t_end;
% x = linspace(0, t_end, 1000);
x = 0:0.025:t_end;
y = deval(sol, x);
results = array2table([x', y', u_f(x)']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'phi3', 'Ds', 'Dphi1','Dphi2', 'Dphi3', 'u'};

%% Plot
% disp("Stackedplot...")
% figure()
% stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
% grid on

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
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
disp("Calculating energies...")

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
disp("Animating...")

h = figure('Position', [10 10 1200 900]);

a1 = subplot(1,2,1);
hold on
grid on
xlim([-1 1])
ylim([-1 1])
axis equal

a2 = subplot(3,2,2);
xlim([0 t_end])
ylim([min(KE)-10, max(KE)+10])
ylabel("Kinetic")
l2 = animatedline(a2);

a3 = subplot(3,2,4);
xlim([0 t_end])
ylim([min(PE)-10, max(PE)+10])
ylabel("Potential")
l3 = animatedline(a3);

a4 = subplot(3,2,6);
xlim([0 t_end])
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
    plot(a1, [pc(1), pc(1)+u_f(x(k))/20], [pc(2), pc(2)], 'r', 'LineWidth', 2)
    a1.XLim = [-1+q(k,1), 1+q(k,1)];
    a1.YLim = [-1, 1];


    addpoints(l2, x(k), KE(k))
    addpoints(l3, x(k), PE(k))
    addpoints(l4, x(k), TE(k))
    
    drawnow
    frames(k) = getframe(gcf);
    
end

writerObj = VideoWriter('triplePendulumCart_tmp');
writerObj.FrameRate = 20;

open(writerObj);
for k = 2:length(frames)
   frame = frames(k);
   writeVideo(writerObj, frame);
end
close(writerObj);
    
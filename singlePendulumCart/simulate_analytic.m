%% Initialize
clear all;
close all;
clc;

savefile = 'singlePend.csv';

l_1 = 0.3;
a_1 = 0.18;
m_1 = 2;
I_1 = 0.6*m_1*a_1^2;
b_1 = 0.12;
m_c = 0.5;
g = 9.81;

params = [m_1,...             % pendulum weights
            I_1,...           % penndulum moments of inertia around CoMs
            a_1,...          % pendulum distances from joints to CoMs
            l_1,...           % pendulum lengths
            b_1,...      % joint damping coefficients
            m_c,...
            g];
        
%% Create external input signal
disp("Creating input signal...")
t_end = 120;     % simulation time
f_s = 100;     % sampling
f_c = 3;        % cutoff
u_a = 3;      % input power

% Random walk
    [filt_b, filt_a] = butter(3, f_c/(f_s/2));  % (order, cutoff freq)
    u_t = 0:(1/f_s):t_end;
    u = u_a * (rand(length(u_t), 1) - 0.5);
    u = filter(filt_b, filt_a, u);  % defines the cart position!
    limit = 0.3;
    u(u>limit) = limit;
    u(u<(-limit)) = -limit;  % ensures that the cart position doesn't go much further than 0.5
    u = filter(filt_b, 0.5*filt_a, u);  % smoothens the trajectory (necessary due to the cutoffs
    u = diff(u)/(1/f_s);   % get cart velocity by differentiation!
    du = diff(u)/(1/f_s);   % get cart acceleration == input signal
    
    du(end+1: end+2) = du(end); % extend the input signal to fix the signal length
    
    % define the input signal as a time-dependent function
    u_f = @(t) interp1(u_t, du, t);

%% Solve
disp("Solving...")

% Create system of ODE functions for the solver
odefun = @(t, X)singlePendCartFull(t, X, u_f, params);

x0 = [0, pi, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc

% x = (logspace(0, 0.3, 1200) - 1) * t_end;
% x = linspace(0, t_end, 1000);
x = 0:0.002:t_end;
y = deval(sol, x);
results = array2table([x', y', u_f(x)']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'Ds', 'Dphi1', 'u'};
writetable(results, savefile)

%% Plot
% disp("Stackedplot...")
% figure()
% stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
% grid on

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
q = [results.s, results.phi1];
dq = [results.Ds, results.Dphi1];

P_c = @(q) [q(1); 0];        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))];         % Pendulum 1

dP_c = @(q, dq) [dq(1); 0];

dP_1 = @(q,dq) [dq(1) + a_1*cos(q(2))*dq(2);
                a_1*sin(q(2))*dq(2)];

%% Calculate kinetic, potential and total energies (for animation plots)
disp("Calculating energies...")

T = @(q,dq) (0.5 * m_1 * dP_1(q,dq)' * dP_1(q,dq) + ...
    0.5 * I_1 * dq(2)^2 + ...
    0.5 * m_c * dP_c(q,dq)' * dP_c(q,dq));  % Kinetic energy

V = @(q) (g * m_1 * P_1(q)'*[0;1]);

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
grid on
xlim([-1 1])
ylim([-1 1])
pbaspect([1 1 1])
hold on

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

frames = [getframe(gcf)];

for k = 1:5:length(q)
    cla(a1)
    
    pc = P_c(q(k, :));
    p1 = P_1(q(k, :));
    
    title(a1, x(k))
    plot(a1, [pc(1), p1(1)], [pc(2), p1(2)], 'kO-', 'LineWidth', 3)
    plot(a1, [pc(1), pc(1)+u_f(x(k))/20], [pc(2), pc(2)], 'r', 'LineWidth', 2)

    addpoints(l2, x(k), KE(k))
    addpoints(l3, x(k), PE(k))
    addpoints(l4, x(k), TE(k))
    
    drawnow
%     frames(end+1) = getframe(gcf);
    
end
% 
% writerObj = VideoWriter('singlePendulumCart');
% writerObj.FrameRate = 20;
% 
% open(writerObj);
% for k = 2:length(frames)
%    frame = frames(k);
%    writeVideo(writerObj, frame);
% end
% close(writerObj);
    
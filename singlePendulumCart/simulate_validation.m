%% Initialize
clear all;
close all;
clc;

% savefile = 'singlePendID.csv';

l_1 = 0.3;
a_1 = 0.18;
m_1 = 1;
I_1 = 0.6*m_1*a_1^2;
b_1 = 0.15;
b_c = 15;
m_c = 0.8;
g = 9.81;

params = [m_1,...             % pendulum weights
            I_1,...           % penndulum moments of inertia around CoMs
            a_1,...          % pendulum distances from joints to CoMs
            l_1,...           % pendulum lengths
            b_1,...         % joint damping coefficients
            b_c,...
            m_c,...
            g];
        
%% Create external input signal
disp("Creating input signal...")
t_end = 20;     % simulation time
f_s = 100;     % sampling
f_c = 4;        % cutoff
u_a = 0.1;      % input power

% Random walk
    [filt_b, filt_a] = butter(3, f_c/(f_s/2));  % (order, cutoff freq)
    u_t = 0:(1/f_s):t_end;
    u = u_a * (rand(length(u_t), 1) - 0.5);
    u = filter(filt_b, filt_a, u);  % defines the cart position!
    u(500:800) = 0;
    limit = 0.3;
    u(u>limit) = limit;
    u(u<(-limit)) = -limit;  % ensures that the cart position doesn't go much further than 0.5
    u = filter(filt_b, 0.5*filt_a, u);  % smoothens the trajectory (necessary due to the cutoffs
    u = diff(u)/(1/f_s);   % get cart velocity by differentiation!
    du = diff(u)/(1/f_s);   % get cart acceleration == input signal
    
    du(end+1: end+2) = du(end); % extend the input signal to fix the signal length
    
    % define the input signal as a time-dependent function
    u_f = @(t) interp1(u_t, du, t);

%% Solve identified model
disp("Solving...")

% Create system of ODE functions for the solver
odefun = @(t, X)identified_model2(t, X, u_f);

x0 = [0, pi/4, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc

x = 0:0.002:t_end;
y = deval(sol, x);
results = array2table([x', y', u_f(x)']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'Ds', 'Dphi1', 'u'};

%% Solve analytic model
disp("Solving...")

% Create system of ODE functions for the solver
odefun = @(t, X)singlePendCartFull(t, X, u_f, params);

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc

y = deval(sol, x);
results_true = array2table([x', y', u_f(x)']);
results_true.Properties.VariableNames = {'t', 's', 'phi1', 'Ds', 'Dphi1', 'u'};
%% Plot
% disp("Stackedplot...")
% figure()
% stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
% grid on

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
q_id = [results.s, results.phi1];
q_true = [results_true.s, results_true.phi1];

P_c = @(q) [q(1); 0];        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))];         % Pendulum 1

%% Animation
disp("Animating...")

h = figure('Position', [0 0 1920 1080]);

t = tiledlayout(2,4, 'TileSpacing', 'tight', 'Padding', 'tight');

a1 = nexttile([2 2]);
grid on
xlim([-1 1])
ylim([-1 1])
pbaspect([1 1 1])
hold on

state_vars = results.Properties.VariableNames(2:end-1);
tile_locs = [3, 7, 4, 8];
for i = 1:length(state_vars)
    disp(i)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    ylabel(var);
    lx(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue', 'LineStyle', ':');
    lx_true(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'red');
    xlim(tspan);
%     ylim([ min(results.(var)) max(results.(var))]);
    grid on
end


frames = [getframe(h)];

for k = 1:8:length(q_id)
    cla(a1)
    
    yline(a1, 0, 'k-', 'LineWidth', 1.5)
    
    pc_id = P_c(q_id(k, :));
    p1_id = P_1(q_id(k, :));
    
    pc_true = P_c(q_true(k, :));
    p1_true = P_1(q_true(k, :));
    
    title(a1, x(k))
    plot(a1, [pc_id(1), p1_id(1)], [pc_id(2), p1_id(2)], 'bx:', 'LineWidth', 3)
    plot(a1, [pc_true(1), p1_true(1)], [pc_true(2), p1_true(2)], 'rO-', 'LineWidth', 3)
    plot(a1, [0, u_f(x(k))/100], [0.5, 0.5], 'g', 'LineWidth', 2)
    
    for i = 1:length(state_vars)
       var = state_vars{i};
       addpoints(lx(i), x(k), results.(var)(k));
       addpoints(lx_true(i), x(k), results_true.(var)(k))
       ax(i).XLim = [x(k)-3, x(k)+1];
    end
    
    drawnow
    frames(end+1) = getframe(h);
    
end
    
writerObj = VideoWriter('singlePendulumCart');
writerObj.FrameRate = 20;

open(writerObj);
for k = 2:length(frames)
   frame = frames(k);
   writeVideo(writerObj, frame);
end
close(writerObj);
%% Initialize
clear all;
close all;
clc;

% savefile = 'singlePendID.csv';

l_1 = 0.4; l_2 = 0.4;
a_1 = 0.2; a_2 = 0.2;
m_1 = 1; m_2 = 1; m_c = 1;
I_1 = 0.8*m_1*a_1^2; I_2 = 0.8*m_2*a_2^2;
b_1 = 0.02; b_2 = 0.01; b_c = 5;
g = 9.81;

params = [m_c, m_1, m_2,...             % pendulum weights
            I_1, I_2,...           % penndulum moments of inertia around CoMs
            a_1, a_2,...          % pendulum distances from joints to CoMs
            l_1, l_2,...           % pendulum lengths
            b_1, b_2, b_c,...     % joint damping coefficients
            g];
        
%% Create external input signal
disp("Creating input signal...")
t_end = 150;     % simulation time
f_s = 1000;     % sampling
f_c = 1;        % cutoff
u_a = 1;      % input power


% Random walk
    [filt_b, filt_a] = butter(3, f_c/(f_s/2));  % (order, cutoff freq)
    u_t = 0:(1/f_s):t_end;
    squarewave = ((square(u_t, 95)+1)/2)';
    u = u_a * (rand(length(u_t), 1) - 0.5);
    u = filter(filt_b, filt_a, u);  % defines the cart position!
    u = u .* squarewave;
    
    squarewave = ones(size(u));
    squarewave(1:end*1/4) = 0.0;
    u = u .* squarewave;

    limit = 3000;
    u(u>limit) = limit;
    u(u<(-limit)) = -limit;  % ensures that the cart position doesn't go much further than 0.5
    u = filter(filt_b, 0.125*filt_a, u);  % smoothens the trajectory (necessary due to the cutoffs
    u = diff(u)/(1/f_s);   % get cart velocity by differentiation
    du = diff(u)/(1/f_s);   % get cart acceleration == input signal
    
    du(end+1: end+2) = du(end); % extend the input signal to fix the signal length
    
%     define the input signal as a time-dependent function
    u_f = @(t) interp1(u_t, du, t);
%     u_f = @(t) interp1(u_t, u, t);

%% Solve identified model
disp("Solving...")

% Create system of ODE functions for the solver
odefun = @(t, X)doublePendCartIdentified(t, X, u_f);

x0 = [0, pi/4, 0, 0, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc

x = 0:0.02:t_end;
y = deval(sol, x);
results_id = array2table([x', y', u_f(x)']);
results_id.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'Ds', 'Dphi1','Dphi2', 'u'};

%% Solve analytic model
disp("Solving...")
% Define the system of ODEs
odefun = @(t, X)doublePendCartAnalytical(t, X, u_f, params);

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc

y = deval(sol, x);
results_true = array2table([x', y', u_f(x)']);
results_true.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'Ds', 'Dphi1','Dphi2', 'u'};

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
q_id = [results_id.s, results_id.phi1, results_id.phi2];
q_true = [results_true.s, results_true.phi1, results_true.phi2];

P_c = @(q) [q(1); 0];        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))];         % Pendulum 1

P_2 = @(q) [l_2*sin(q(3)) + l_1*sin(q(2)) + q(1);
    -l_2*cos(q(3)) - l_1*cos(q(2))];         % Pendulum 2

%% Animation
disp("Animating...")

h = figure('Position', [0 0 1920 1080]);

t = tiledlayout(2,5, 'TileSpacing', 'tight', 'Padding', 'tight');

a1 = nexttile([2 2]);
grid on
lims = 0.8;
xlim([-lims lims])
ylim([-lims lims])
pbaspect([1 1 1])
hold on

state_vars = results_true.Properties.VariableNames(2:end-1);
tile_locs = [3, 8, 4, 9, 5, 10];
ylabels = {'$x_1\,[m]$','$x_2\,[rad]$','$x_3\,[rad]$',...
    '$x_4\,[m\,s^{-1}]$', '$x_5\,[rad\,s^{-1}]$', '$x_6\,[rad\,s^{-1}]$'};
for i = 1:length(state_vars)
    disp(i)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    ax(i).YLabel.Interpreter = 'latex';    
    ax(i).YLabel.String = ylabels{i};
    ax(i).YLabel.FontSize = 20;
    lx_identified(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue', 'LineStyle', ':');
    lx_analytical(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'red');
    xlim(tspan);
%     ylim([ min(results.(var)) max(results.(var))]);
    grid on
end


frames = [getframe(h)];

for k = 1:length(q_id)
    cla(a1)
    
    yline(a1, 0, 'k-', 'LineWidth', 1.5)
    
    pc_id = P_c(q_id(k, :));
    p1_id = P_1(q_id(k, :));
    p2_id = P_2(q_id(k, :));
    
    pc_true = P_c(q_true(k, :));
    p1_true = P_1(q_true(k, :));
    p2_true = P_2(q_true(k, :));
    
    title(a1, x(k))
    % Plot identified model frame
    plot(a1, [pc_id(1), p1_id(1)], [pc_id(2), p1_id(2)], 'bo:', 'LineWidth', 4)
    plot(a1, [p1_id(1), p2_id(1)], [p1_id(2), p2_id(2)], 'bo:', 'LineWidth', 4)
    
    % Plot analytical model frame
    plot(a1, [pc_true(1), p1_true(1)], [pc_true(2), p1_true(2)], 'ro-', 'LineWidth', 4)
    plot(a1, [p1_true(1), p2_true(1)], [p1_true(2), p2_true(2)], 'ro-', 'LineWidth', 4)
    
    plot(a1, [0, u_f(x(k))/100], [0.5, 0.5], 'k', 'LineWidth', 5)
    
    for i = 1:length(state_vars)
       var = state_vars{i};
       addpoints(lx_identified(i), x(k), results_id.(var)(k));
       addpoints(lx_analytical(i), x(k), results_true.(var)(k))
       ax(i).XLim = [x(k)-3, x(k)+1];
    end
    
    drawnow
    frames(end+1) = getframe(h);
    
end
    
writerObj = VideoWriter('doublePendulumCart');
writerObj.FrameRate = 20;

open(writerObj);
for k = 2:length(frames)
   frame = frames(k);
   writeVideo(writerObj, frame);
end
close(writerObj);

%% Animation
% disp("Animating...")
% 
% h = figure('Position', [0 0 1920 1080]);
% 
% t = tiledlayout(5,1, 'TileSpacing', 'tight', 'Padding', 'tight');
% 
% 
% state_vars = results.Properties.VariableNames(2:end-1);
% tile_locs = [1,2,3,4];
% for i = 1:length(state_vars)
%     var = state_vars{i};
%     ax(i) = nexttile(tile_locs(i));
%     ylabel(var);
%     lx(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue', 'LineStyle', ':');
%     lx_true(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'red');
% %     ylim([ min(results.(var)) max(results.(var))]);
%     grid on
% end
% 
%     ax(5) = nexttile(5);
%     lx(5) = animatedline(ax(5), 'LineWidth', 2, 'Color', 'black', 'LineStyle', '-');
%     grid on
% 
% 
% k=3000;
% ylabels = {'$x_1\,[m]$','$x_2\,[rad]$','$x_3\,[m\,s^{-1}]$','$x_4\,[rad\,s^{-1}]$'};
% 
% 
% for i = 1:length(state_vars)
%     var = state_vars{i};
%     addpoints(lx(i), x(1:k), results.(var)(1:k));
%     addpoints(lx_true(i), x(1:k), results_true.(var)(1:k))
%     
%     ax(i).YLabel.Interpreter = 'latex';    
%     ax(i).YLabel.String = ylabels{i};
%     ax(i).YLabel.FontSize = 13;
%     legend(ax(i), {'Identified model', 'Real model'}, 'location', 'southwest')
%     
% end
% 
% ax(5).XLabel.Interpreter = 'latex';
% ax(5).XLabel.String = '$Time\,t\,[s]$';
% ax(5).XLabel.FontSize=13;
% 
%     addpoints(lx(5), x(1:k), results.(var)(1:k));
%     
%     ax(5).YLabel.Interpreter = 'latex';    
%     ax(5).XLabel.Interpreter = 'latex';
%     ax(5).YLabel.String = '$u\,[N]$';
%     ax(5).YLabel.FontSize = 13;
%     
%     



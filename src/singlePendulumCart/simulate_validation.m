%% Initialize
disp("Initializing...")
clear all;
close all;
clc;
addpath('analytical_models/', 'models/');

% savefile = 'singlePendID.csv';

l_1 = 0.3;
a_1 = 0.18;
m_1 = 1;
I_1 = 0.7*m_1*a_1^2;
b_1 = 0.01;
b_c = 10;
% m_c = 0.8;
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
f_s = 1000;     % sampling frequency
dt = 1/f_s;     % sampling period
N = 2^13;     % number of samples
t_end = (N-1) * dt; % simulation time

% White noise
t_res = t_end / N;    % time resolution
f_res = f_s / N;      % freq resolution
wnum = -(N/2):(N/2-1);  % wave numbers
freq = wnum * f_res;  % frequencies

time = 0:dt:t_end;
    %% Random white noise prescribing the cart position

    t = tiledlayout(2, 1, 'TileSpacing', 'tight', 'Padding', 'tight');
    
    
    ax1 = nexttile();
    u = white_noise(N, 1, 1);  % change seed to 0 for training set
    f_cutoff = 8;
    u = spectral_filter(u, ceil(f_cutoff/f_res)); % filter out high frequencies
    plot(time, u, '--b', 'linewidth', 1.5)
    hold on
    grid on
    yline(0, 'k-', 'linewidth', 3)
    % multiply by a square wave signal so the cart returns to 0 periodically
    squarewave = ((square(time, 80)+1)/2)';
    u = u .* squarewave;
    u = spectral_filter(u, ceil(f_cutoff/f_res)); % filter out high frequencies
    plot(time, u, '-b', 'linewidth', 2)

    
    ax2 = nexttile();
    % set the maximal distance from origin as 0.5
    max_val = 0.5;
    u = u ./ max(abs(u)) .* max_val;
    plot(time, u, 'r--', 'linewidth', 1.5)
    hold on
    yline(0, 'k-', 'linewidth', 3)
    % damp the movement at the beginning of the experiment 
    dampfun = ones(length(u), 1);
    dampfun(1:end/3) = linspace(0.15, 1, length(u)/3);
    u = u .* dampfun;
    plot(time, u, 'r-', 'linewidth', 2)
    
    axes(ax1);
    ylabel('$\mathrm{Random\ process\ signal}\ [-]$', 'interpreter', 'latex',...
    'fontsize', 20)
    legend({'Random process signal (RPS)', '',...
        'RPS after multiplication by squarewave & re-filtering'})
    
    axes(ax2);
    ylabel('$\mathrm{Reference\ cart\ trajectory}\ [m]$', 'interpreter', 'latex',...
    'fontsize', 20)
    grid on
    legend({'Cart trajectory computed from RPS by rescaling', '',...
        'Cart trajectory with slow start'})

    du = diff(u)/dt;   % get cart velocity by differentiation
    ddu = diff(du)/dt;   % get cart acceleration == input signal
    
    ddu(end+1: end+2) = ddu(end); % extend the input signal to fix the signal length
    ddu = 16 * ddu;
    
%define the input signal as a time-dependent function
    u_f = @(t) interp1(time, ddu, t);
    
%% Simulate the trajectory of both models
disp("Solving...")

% Create system of ODE functions for the solver
% odefun_real = @(t, x)singlePendCartFull(t, x, u_f, params);
odefun_real = @(t, x)identified_model_real_(t, x, u_f);
odefun_model = @(t, x)identified_model_real_bs_(t, x, u_f);
% odefun_model = @(t, x)identified_model_noisy_bootstrapped_(t, x, u_f);

x0 = [0, pi, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
tic
sol_real = ode45(odefun_real, tspan, x0);
sol_model = ode45(odefun_model, tspan, x0);
toc

results_real = get_results(sol_real, u_f, time);
results_model = get_results(sol_model, u_f, time);

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
q_model = [results_model.s, results_model.phi1];
q_real = [results_real.s, results_real.phi1];

P_c = @(q) [q(1); 0]';        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))]';         % Pendulum 1

%% Animation
disp("Animating...")

h = figure('Position', [0 0 1920 1080]);

t = tiledlayout(2,4, 'TileSpacing', 'tight', 'Padding', 'tight');

a1 = nexttile([2 2]);
grid on
xlim([-0.6 0.6])
ylim([-0.6 0.6])
pbaspect([1 1 1])
hold on

traj_real = animatedline(a1, 'MaximumNumPoints', 500, 'LineWidth', 2, 'Color', '#17BECF', 'LineStyle', '-');
pend_real = animatedline(a1, 'MaximumNumPoints', 2, 'LineWidth', 4,...
    'Color', '#1F77B4', 'LineStyle', '-', 'Marker', 'O', 'MarkerSize', 10,...
    'MarkerFaceColor', 'blue');

traj_model = animatedline(a1, 'MaximumNumPoints', 500, 'LineWidth', 2, 'Color', '#FF7F0E', 'LineStyle', '-');
pend_model = animatedline(a1, 'MaximumNumPoints', 2, 'LineWidth', 4,...
    'Color', '#D62728', 'LineStyle', '-', 'Marker', 'O', 'MarkerSize', 10,...
    'MarkerFaceColor', 'red');


force = animatedline(a1, 'MaximumNumPoints', 2, 'LineWidth', 4,...
    'Color', '#2CA02C', 'LineStyle', '-', 'Marker', 'd', 'MarkerSize', 6,...
    'MarkerFaceColor', '#2CA02C');

yline(a1, 0, 'k-', 'LineWidth', 1.5)

legend({'', 'SINDy model', '', 'Bootstrapped model', 'Input', ''})

state_vars = results_real.Properties.VariableNames(2:end-1);
tile_locs = [3, 7, 4, 8];
ylabels = {'$x_1\,[m]$','$x_2\,[rad]$','$x_3\,[m\,s^{-1}]$','$x_4\,[rad\,s^{-1}]$'};
for i = 1:length(state_vars)
    disp(i)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    yline(ax(i), 0, 'black', 'linewidth', 2)
    ax(i).YLabel.Interpreter = 'latex';    
    ax(i).YLabel.String = ylabels{i};
    ax(i).YLabel.FontSize = 20;
    lx_model(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', '#D62728', 'LineStyle', '--');
    lx_real(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', '#1F77B4');
    xlim(tspan);
    grid on
end

s = 10;  % animation time step

update_traj = @(traj, data) (...
    addpoints(traj,...
    data(:,1), data(:,2)));

update_plot = @(plot, data, k) (...
    addpoints(plot, time(k-s:k), data(k-s:k)));

update_line = @(line, point1, point2)(...
        addpoints(line,...
        [point1(1), point2(1)],...
        [point1(2), point2(2)]));

frames = [getframe(h)];

    
for k = s+1:s:length(q_real)
    
    pc_true = P_c(q_real(k, :));
    p1_true = P_1(q_real(k, :));
    
    pc_model = P_c(q_model(k, :));
    p1_model = P_1(q_model(k, :));
    
    traj_data = table(q_real(k-s:k, :));
    xy_traj_real = table2array(rowfun(P_1, traj_data));
    
    traj_data = table(q_model(k-s:k, :));
    xy_traj_model = table2array(rowfun(P_1, traj_data));
    
    title(a1, time(k))
    update_line(pend_real, pc_true, p1_true);
    update_traj(traj_real, xy_traj_real);
    
    update_line(pend_model, pc_model, p1_model);
    update_traj(traj_model, xy_traj_model);
    
    update_line(force, [u_f(time(k))/10000, 0.5], [0, 0.5]);
    
    for i = 1:length(state_vars)
       var = state_vars{i};
       update_plot(lx_real(i), results_real.(var), k)
       update_plot(lx_model(i), results_model.(var), k)
       ax(i).XLim = [time(k)-2, time(k)+0.2];
    end
    
    drawnow
    frames(end+1) = getframe(h);
    
end
    
writerObj = VideoWriter('singlePendulumCart_realmodel2');
writerObj.FrameRate = 20;

open(writerObj);
for k = 2:length(frames)
   frame = frames(k);
   writeVideo(writerObj, frame);
end
close(writerObj);

%%
h = figure('Position', [0 0 1920*0.7 1080*0.7]);
t = tiledlayout(2,2, 'TileSpacing', 'tight', 'Padding', 'tight');
state_vars = results_real.Properties.VariableNames(2:end-1);
tile_locs = [1, 3, 2, 4];
ylabels = {'$x_1\,[m]$','$x_2\,[rad]$','$x_3\,[m\,s^{-1}]$','$x_4\,[rad\,s^{-1}]$'};
for i = 1:length(state_vars)
    disp(i)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    ax(i).YLabel.Interpreter = 'latex';    
    ax(i).YLabel.String = ylabels{i};
    ax(i).YLabel.FontSize = 24;
    yline(ax(i), 0, 'black', 'linewidth', 2) 
    lx_real(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', '#1F77B4');
    lx_model(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', '#D62728', 'LineStyle', ':');
    xlim(tspan);
    grid on
end

    for i = 1:length(state_vars)
        var = state_vars{i};
        addpoints(lx_real(i), time, results_real.(var))
%         addpoints(lx_model(i), time, results_model.(var))
    end

    legend(ax(3), {'', 'SINDy model', 'Bootstrapped model'}, 'fontsize', 20)
%%

function results = get_results(sol, u_f, x)
    y = deval(sol, x);
    results = array2table([x', y', u_f(x)']);
    results.Properties.VariableNames = {'t',...
        's', 'phi1', 'Ds', 'Dphi',...
        'u'};
end

function noise = white_noise(N, cols, seed)
    % N must be even!
    
    rng(seed);

    phase = ( rand(N/2-1, cols)-0.5 ) + ...
        1j*( rand(N/2-1, cols)-0.5 );  % random complex vector
    phase = phase ./ abs(phase); % normalize to 1

    ones_row = ones(1,cols) * (1+0*1j);
    noise_hat = [ones_row; flip(conj(phase), 1); ones_row; phase];
    noise_hat = fftshift(noise_hat);
    noise = ifft(noise_hat, 'symmetric');
end

function out = spectral_filter(in, cutoff_idx)
    in_hat = fft(in);
    in_hat = fftshift(in_hat);
    
    mid = ceil(length(in_hat)/2);
    filt = zeros(length(in_hat), 1);
    filt(mid-cutoff_idx:mid+cutoff_idx) = 1;
    
    out_hat = in_hat .* filt;
    out_hat = ifftshift(out_hat);
    out = ifft(out_hat, 'symmetric');
end

%% Initialize
disp("Initializing...")
clear all;
close all;
clc;
addpath('analytical_models/');

datapath = fullfile('..', '..', 'data', 'singlePend', 'simulated');
savefile = "singlePend_val.csv";

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
N = 2^14;     % number of samples
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
    f_cutoff = 3;
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
    
%%
    du = diff(u)/dt;   % get cart velocity by differentiation
    ddu = diff(du)/dt;   % get cart acceleration == input signal
    
    ddu(end+1: end+2) = ddu(end); % extend the input signal to fix the signal length
    ddu = 2 * ddu;
    
%define the input signal as a time-dependent function
    u_f = @(t) interp1(time, ddu, t);

%% Solve
disp("Solving...")

% Create system of ODE functions for the solver
odefun = @(t, X)singlePendCartFull(t, X, u_f, params);

x0 = [0, pi*30/32, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc

% x = (logspace(0, 0.3, 1200) - 1) * t_end;
% x = linspace(0, t_end, 1000);
x = 0:1/f_s:t_end;
y = deval(sol, x);
dy = odefun(x, y);
results_true = array2table([x', y', u_f(x)', dy(3:4, :)']);
results_true.Properties.VariableNames = {'t',...
    's', 'phi1', 'Ds', 'Dphi',...
    'u',...
    'DDs', 'DDphi'};

writetable(results_true, fullfile(datapath, savefile))
% results_true = readtable('singlePend.csv');

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
q_true = [results_true.s, results_true.phi1];

P_c = @(q) [q(1); 0]';        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))]';         % Pendulum 1

%% Animation
disp("Animating...")

h = figure('Position', [0 0 1920 1080]);

t = tiledlayout(2,4, 'TileSpacing', 'tight', 'Padding', 'tight',...
    'GridSize', [2, 4]);

a1 = nexttile([2 2]);
grid on
xlim([-0.6 0.6])
ylim([-0.6 0.6])
axis manual
pbaspect([1 1 1])
hold on
traj = animatedline(a1, 'MaximumNumPoints', 2000, 'LineWidth', 2, 'Color', 'cyan', 'LineStyle', '-');
pend = animatedline(a1, 'MaximumNumPoints', 2, 'LineWidth', 4,...
    'Color', 'blue', 'LineStyle', '-', 'Marker', 'O', 'MarkerSize', 10,...
    'MarkerFaceColor', 'blue');
force = animatedline(a1, 'MaximumNumPoints', 2, 'LineWidth', 3,...
    'Color', 'black', 'LineStyle', '-', 'Marker', 'd', 'MarkerSize', 5,...
    'MarkerFaceColor', 'black');

state_vars = results_true.Properties.VariableNames(2:end-3);
tile_locs = [3, 7, 4, 8];
ylabels = {'$x_1\,[m]$','$x_2\,[rad]$','$x_3\,[m\,s^{-1}]$','$x_4\,[rad\,s^{-1}]$'};
for i = 1:length(state_vars)
    disp(i)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    ax(i).YLabel.Interpreter = 'latex';    
    ax(i).YLabel.String = ylabels{i};
    ax(i).YLabel.FontSize = 20;
    lx(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue', 'LineStyle', ':');
    lx_true(i) = animatedline(ax(i), 'LineWidth', 2.5, 'Color', 'black');
    xlim(tspan);
%     ylim([ min(results.(var)) max(results.(var))]);
    grid on
end

s = 100;  % animation time step

update_traj = @(traj, data) (...
    addpoints(traj,...
    data(:,1), data(:,2)));

update_plot = @(plot, data, k) (...
    addpoints(plot, x(k-s:k), data(k-s:k)));

update_line = @(line, point1, point2)(...
        addpoints(line,...
        [point1(1), point2(1)],...
        [point1(2), point2(2)]));

frames = [getframe(h)];

yline(a1, 0, 'k-', 'LineWidth', 1.5)
    
for k = s+1:s:length(q_true)
    
    xlim(a1, [-0.6 0.6])
    ylim(a1, [-0.6 0.6])
    pc_true = P_c(q_true(k, :));
    p1_true = P_1(q_true(k, :));
    
    traj_data = table(q_true(k-s:k, :));
    xy_traj = table2array(rowfun(P_1, traj_data));
    
    title(a1, x(k))
    update_line(pend, pc_true, p1_true);
    update_line(force, [u_f(x(k))/200, 0.5], [0, 0.5]);
    update_traj(traj, xy_traj);
    
    for i = 1:length(state_vars)
       var = state_vars{i};
%        addpoints(lx_true(i), x(k), results_true.(var)(k))
       update_plot(lx_true(i), results_true.(var), k)
       ax(i).XLim = [x(k)-3, x(k)+0.1];
    end
    
    
    drawnow
    frames(end+1) = getframe(h);
    
end
%     
% writerObj = VideoWriter('singlePendulumCart2');
% writerObj.FrameRate = 20;
% 
% open(writerObj);
% for k = 2:length(frames)
%    frame = frames(k);
%    writeVideo(writerObj, frame);
% end
% close(writerObj);

%%
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
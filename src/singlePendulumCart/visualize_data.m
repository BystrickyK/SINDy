%% Initialize
disp("Initializing...")
clear all;
close all;
clc;
addpath('analytical_models/', 'models/');

% savefile = 'singlePendID.csv';


datapath = string(['.', filesep, '..', filesep, '..', filesep, 'data',...
    filesep, 'singlePend',...
    filesep, 'real',...
    filesep, 'processed_measurements.csv']);

l_1 = 0.3;

results_real = readtable(datapath);
results_real = results_real(:, :);

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
q_real = [results_real.x_1, results_real.x_2];

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

force = animatedline(a1, 'MaximumNumPoints', 2, 'LineWidth', 4,...
    'Color', '#2CA02C', 'LineStyle', '-', 'Marker', 'd', 'MarkerSize', 6,...
    'MarkerFaceColor', '#2CA02C');

yline(a1, 0, 'k-', 'LineWidth', 1.5)

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
    grid on
end

s = 50;  % animation time step
time = results_real.t;
input = results_real.u;

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
    
    traj_data = table(q_real(k-s:k, :));
    xy_traj_real = table2array(rowfun(P_1, traj_data));

    title(a1, time(k))
    update_line(pend_real, pc_true, p1_true);
    update_traj(traj_real, xy_traj_real);
    
    update_line(force, [input(k)/10000, 0.5], [0, 0.5]);
    
    for i = 1:length(state_vars)
       var = state_vars{i};
%        addpoints(lx_true(i), x(k), results_true.(var)(k))
       update_plot(lx_real(i), results_real.(var), k)
       ax(i).XLim = [time(k)-2, time(k)+0.2];
    end
    
    drawnow
    frames(end+1) = getframe(h);
    
end
    
writerObj = VideoWriter('singlePendulumCart_measurements');
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
        addpoints(lx_model(i), time, results_model.(var))
    end

    legend(ax(3), {'', 'Reference model', 'SINDy model'}, 'fontsize', 20)
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

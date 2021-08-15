%% Initialize
disp("Initializing...")
addpath('analytical_models', 'identified_models')
clear all;
close all;
clc;

        
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

%% Generate the input signal by band-filtering white noise
noise = white_noise(N, 3, 0);

f_cutoff = 4;
noise_bandlimited = spectral_filter(noise, ceil(f_cutoff/f_res));

max_val = [5, 5, 5];
noise_bandlimited = noise_bandlimited ./ max(noise_bandlimited) .* max_val;

% plot(time, noise_bandlimited)
% define the input signal as a time-dependent function
control_law = @(x) [0, 0, 0];
random_process = @(t) interp1(time, noise_bandlimited, t, 'previous');
u_fun = @(t, x) (control_law(x) + random_process(t));
%% Solve
disp("Solving...")
% Define the system of ODEs
odefun_real = @(t, x)identified_model_sgn_clean_(t, x, u_fun);
odefun_model = @(t, x)identified_model_sgn_noisy_(t, x, u_fun);

x0 = [-15, -15, 15]'; % initial conditions
tspan = [0, t_end]; % time span

% Solve the system of ODEs
tic
sol_real = ode45(odefun_real, tspan, x0);
sol_model = ode45(odefun_model, tspan, x0);
toc

%%

x = (0:dt:t_end)';
tic
results_real = get_results(sol_real, odefun_real, u_fun, x);
results_model = get_results(sol_model, odefun_model, u_fun, x);
toc

%% Animation
disp("Animating...")

scale = 0.9;
h = figure('Position', [10 10 1920*scale 1080*scale]);

t = tiledlayout(3,4, 'TileSpacing', 'tight', 'Padding', 'compact');

a1 = nexttile([3, 2]);
grid on
traj_real = animatedline('MaximumNumPoints', 8000, 'LineWidth', 2.5, 'Color', '#1F77B4');
point_real = animatedline('MarkerSize', 6, 'Color', 'blue',...
        'Marker', 'o', 'MaximumNumPoints', 1, 'MarkerFaceColor', '#1F77B4');
    
traj_model = animatedline('MaximumNumPoints', 8000, 'LineWidth', 2.5, 'Color', '#D62728',...
    'linestyle', '--');
point_model = animatedline('MarkerSize', 6, 'Color', 'red',...
        'Marker', 'o', 'MaximumNumPoints', 1, 'MarkerFaceColor', '#D62728');
    
view(3)
pbaspect([1 1 1])
xlim([min(results_model.x_1)-3, 3+max(results_model.x_1)])
ylim([min(results_model.x_2)-3, 3+max(results_model.x_2)])
zlim([min(results_model.x_3)-3, 3+max(results_model.x_3)])
ax = gca;
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 25;
    ax.XLabel.String = '$x_1$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 25;
    ax.YLabel.String = '$x_2$';
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 25;
    ax.ZLabel.String = '$x_3$';
    legend({'Reference model', '', 'Identified model', ''}, 'location', 'northwest',...
        'fontsize', 20)
hold on

state_vars = {'$x_1$', '$x_2$', '$x_3$'};
tile_locs = [3, 7, 11];
for i = 1:length(state_vars)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    yline(ax(i), 0, 'color', 'black', 'linewidth', 2)
    ax(i).YLabel.Interpreter = 'latex';
    ax(i).YLabel.FontSize = 20;
    ax(i).YLabel.String = var;
    lx_real(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', '#1F77B4');
    lx_model(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', '#D62728', 'linestyle', '--');
    xlim(tspan);
    grid on
end

der_vars = {'$\dot{x}_1$', '$\dot{x}_2$', '$\dot{x}_3$'};
tile_locs = [4, 8, 12];
for i = 1:length(state_vars)
    var = der_vars{i};
    adx(i) = nexttile(tile_locs(i));
    yline(adx(i), 0, 'color', 'black', 'linewidth', 2)
    adx(i).YLabel.Interpreter = 'latex';
    adx(i).YLabel.FontSize = 20;
    adx(i).YLabel.String = var;
    ldx_real(i) = animatedline(adx(i), 'LineWidth', 2, 'Color', '#1F77B4');
    ldx_model(i) = animatedline(adx(i), 'LineWidth', 2, 'Color', '#D62728', 'linestyle', '--');
    xlim(tspan);
    grid on
end

% input_vars = {'$u_1$', '$u_2$', '$u_3$'};
% 
% tile_locs = [5, 10, 15];
% for i = 1:length(state_vars)
%     var = input_vars{i};
%     au(i) = nexttile(tile_locs(i));
%     au(i).YLabel.Interpreter = 'latex';
%     au(i).YLabel.FontSize = 20;
%     au(i).YLabel.String = var;
%     lu(i) = animatedline(au(i), 'LineWidth', 2, 'Color', 'black');
%     xlim(tspan);
%     grid on
% end


% Load data
state_real = table2array(results_real(1:end, 2:4));
dot_state_real = table2array(results_real(1:end, 5:7));

state_model = table2array(results_model(1:end, 2:4));
dot_state_model = table2array(results_model(1:end, 5:7));


% inputs = table2array(results_real(1:end, 8:10));

frames = [getframe(h)];

s = 20;
update_traj = @(traj, state, k) (...
    addpoints(traj,...
    state(k-s:k,1), state(k-s:k,2), state(k-s:k,3)));

update_plot = @(plot, data, k, i) (...
    addpoints(plot, x(k-s:k), data(k-s:k, i)));

for k = (s+1):s:length(x)

    title(a1, x(k))
    
    update_traj(traj_real, state_real, k)
    update_traj(traj_model, state_model, k)
    
    update_traj(point_real, state_real, k)
    update_traj(point_model, state_model, k)

    view(a1, k/5/s, 30);
%     view(a1, 45, 30);
    
    for i = 1:length(state_vars)
       update_plot(lx_real(i), state_real, k, i)
       update_plot(lx_model(i), state_model, k, i)
       ax(i).XLim = [x(k)-3, x(k)+0.5];
       update_plot(ldx_real(i), dot_state_real, k, i)
       update_plot(ldx_model(i), dot_state_model, k, i)
       adx(i).XLim = [x(k)-3, x(k)+0.5];
%        update_plot(lu(i), inputs, k, i)
%        au(i).XLim = [x(k)-5, x(k)+2];
    end
    
    drawnow
    frames(end+1) = getframe(h);
    
end

writerObj = VideoWriter('lorenz_validation');
writerObj.FrameRate = 20;

open(writerObj);
for k = 2:length(frames)
   frame = frames(k);
   writeVideo(writerObj, frame);
end
close(writerObj);
%%

scale = 0.9;
h = figure('Position', [10 10 1920*scale 1080*scale]);

t = tiledlayout(3,2, 'TileSpacing', 'compact', 'Padding', 'tight');

state_vars = {'$x_1$', '$x_2$', '$x_3$'};
tile_locs = [1, 3, 5];
for i = 1:length(state_vars)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    ax(i).YLabel.Interpreter = 'latex';
    ax(i).YLabel.FontSize = 20;
    ax(i).YLabel.String = var;
    lx_real(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue');
    lx_model(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'red', 'linestyle', '--');
    xlim(tspan);
    grid on
    addpoints(lx_real(i), x, state_real(:,i));
    addpoints(lx_model(i), x, state_model(:,i));
end
xlabel('$Time\ t [s]$', 'FontSize', 18, interpreter='latex')
linkaxes(ax, 'x')
title(ax(1), 'State variables', 'interpreter', 'latex',...
    'fontsize', 20)


der_vars = {'$\dot{x}_1$', '$\dot{x}_2$', '$\dot{x}_3$'};
tile_locs = [2, 4, 6];
for i = 1:length(state_vars)
    var = der_vars{i};
    adx(i) = nexttile(tile_locs(i));
    adx(i).YLabel.Interpreter = 'latex';
    adx(i).YLabel.FontSize = 22;
    adx(i).YLabel.String = var;
    ldx_real(i) = animatedline(adx(i), 'LineWidth', 2, 'Color', 'blue');
    ldx_model(i) = animatedline(adx(i), 'LineWidth', 2, 'Color', 'red', 'linestyle', '--');
    xlim(tspan);
    grid on
    addpoints(ldx_real(i), x, dot_state_real(:,i));
    addpoints(ldx_model(i), x, dot_state_model(:,i));
end
xlabel('$Time\ t [s]$', 'FontSize', 20, interpreter='latex')
linkaxes(adx, 'x')
title(adx(1), 'State derivative variables', 'interpreter', 'latex',...
    'fontsize', 20)

legend(adx(1), {'Reference model', 'Identified model'}, 'fontsize', 16,...
    'location', 'northeastoutside')

%%
scale = 0.9;
h = figure('Position', [10 10 1080*scale 1080*scale]);

t = tiledlayout(3,3, 'TileSpacing', 'tight', 'Padding', 'compact');

ax = nexttile([3,3]);
grid on
traj_real = animatedline('LineWidth', 1.5, 'Color', 'blue');
point_real = animatedline('MarkerSize', 6, 'Color', 'blue',...
        'Marker', 'o', 'MaximumNumPoints', 1, 'MarkerFaceColor', 'blue');
    
traj_model = animatedline('LineWidth', 1.5, 'Color', 'red',...
    'linestyle', '--');
point_model = animatedline('MarkerSize', 6, 'Color', 'red',...
        'Marker', 'o', 'MaximumNumPoints', 1, 'MarkerFaceColor', 'red');
    
view(3)
pbaspect([1 1 1])
xlim([min(results_model.x_1)-3, 3+max(results_model.x_1)])
ylim([min(results_model.x_2)-3, 3+max(results_model.x_2)])
zlim([min(results_model.x_3)-3, 3+max(results_model.x_3)])
ax = gca;
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 25;
    ax.XLabel.String = '$x_1$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 25;
    ax.YLabel.String = '$x_2$';
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 25;
    ax.ZLabel.String = '$x_3$';
    legend({'Reference model', '', 'Identified model', ''}, 'location', 'northwest',...
        'fontsize', 20)
hold on

k = 4;
addpoints(traj_real,...
    state_real(1:end/k,1),...
    state_real(1:end/k,2),...
    state_real(1:end/k,3));
addpoints(traj_model,...
    state_model(1:end/k,1),...
    state_model(1:end/k,2),...
    state_model(1:end/k,3));
addpoints(point_model,...
    state_model(end/k,1),...
    state_model(end/k,2),...
    state_model(end/k,3));
addpoints(point_real,...
    state_real(end/k,1),...
    state_real(end/k,2),...
    state_real(end/k,3));

%% Animation
% disp("Animating...")
% 
% scale = 0.9;
% h = figure('Position', [10 10 1080*scale 1080*scale]);
% 
% t = tiledlayout(6, 2, 'TileSpacing', 'tight', 'Padding', 'tight');
% 
% a1 = nexttile([3, 1]);
% grid on
% traj = animatedline('MaximumNumPoints', 10000, 'LineWidth', 1.5, 'Color', 'blue');
% point = animatedline('MarkerSize', 5, 'Color', 'black',...
%         'Marker', 'o', 'MaximumNumPoints', 1, 'MarkerFaceColor', 'black');
% view(3)
% pbaspect([1 1 1])
% xlim([min(results.x_1) max(results.x_1)])
% ylim([min(results.x_2) max(results.x_2)])
% zlim([min(results.x_3) max(results.x_3)])
% ax = gca;
%     ax.XLabel.Interpreter = 'latex';
%     ax.XLabel.FontSize = 25;
%     ax.XLabel.String = '$x_1$';
%     ax.YLabel.Interpreter = 'latex';
%     ax.YLabel.FontSize = 25;
%     ax.YLabel.String = '$x_2$';
%     ax.ZLabel.Interpreter = 'latex';
%     ax.ZLabel.FontSize = 25;
%     ax.ZLabel.String = '$x_3$';
% hold on
% 
% state_vars = {'$x_1$', '$x_2$', '$x_3$'};
% state = table2array(results(1:end, 2:4));
% tile_locs = [2, 4, 6];
% for i = 1:length(state_vars)
%     var = state_vars{i};
%     ax(i) = nexttile(tile_locs(i));
%     ax(i).YLabel.Interpreter = 'latex';
%     ax(i).YLabel.FontSize = 20;
%     ax(i).YLabel.String = var;
%     lx(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue');
%     xlim(tspan);
%     grid on
% end
% 
% der_vars = {'$\dot{x}_1$', '$\dot{x}_2$', '$\dot{x}_3$'};
% dot_state = table2array(results(1:end, 5:7));
% tile_locs = [7, 9, 11];
% for i = 1:length(state_vars)
%     var = der_vars{i};
%     adx(i) = nexttile(tile_locs(i));
%     adx(i).YLabel.Interpreter = 'latex';
%     adx(i).YLabel.FontSize = 20;
%     adx(i).YLabel.String = var;
%     ldx(i) = animatedline(adx(i), 'LineWidth', 2, 'Color', 'red');
%     xlim(tspan);
%     grid on
% end
% 
% input_vars = {'$u_1$', '$u_2$', '$u_3$'};
% inputs = table2array(results(1:end, 8:10));
% tile_locs = [8, 10, 12];
% for i = 1:length(state_vars)
%     var = input_vars{i};
%     au(i) = nexttile(tile_locs(i));
%     au(i).YLabel.Interpreter = 'latex';
%     au(i).YLabel.FontSize = 20;
%     au(i).YLabel.String = var;
%     lu(i) = animatedline(au(i), 'LineWidth', 2, 'Color', 'black');
%     xlim(tspan);
%     grid on
% end
% 
% 
% frames = [getframe(h)];
% 
% s = 300;
% for k = (s+1):s:length(x)
% 
%     title(a1, x(k))
%     
%     addpoints(traj, state(k-s:k,1), state(k-s:k,2), state(k-s:k,3));
%     addpoints(point, state(k,1), state(k,2), state(k,3));
% %     view(a1, k/5/s, 30);
%     view(a1, 45, 30);
%     
%     for i = 1:length(state_vars)
%        addpoints(lx(i), x(k-s:k), state(k-s:k, i));
%        ax(i).XLim = [x(k)-9, x(k)];
%        addpoints(ldx(i), x(k-s:k), dot_state(k-s:k, i));
%        adx(i).XLim = [x(k)-9, x(k)];
%        addpoints(lu(i), x(k-s:k), inputs(k-s:k, i));
%        au(i).XLim = [x(k)-9, x(k)];
%     end
%     
%     drawnow
%     frames(end+1) = getframe(h);
%     
% end

%%
function noise = white_noise(N, cols, seed)
    % N must be even!
    
    rng(seed);

    phase = ( rand(N/2-1, cols)-0.5 ) + ...
        1j*( rand(N/2-1, cols)-0.5 );  % random complex vector
    phase = phase ./ abs(phase); % normalize to 1

    ones_row = ones(1,3) * (1+0*1j);
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

function results = get_results(sol, odefun, u_fun, x)
    y = deval(sol, x)';
    tmp = [x, y];
    tmp = num2cell(tmp, 2);

    accels = cellfun(@(in) odefun(in(1), in(2:4)), tmp, 'UniformOutput', false);
    accels = cell2mat(accels);
    accels = reshape(accels, 3, [])';

    inputs = cellfun(@(row) u_fun(row(1), row(2:4)), tmp, 'UniformOutput', false);
    inputs = cell2mat(inputs);
    inputs = reshape(inputs, [], 3);

    results = array2table([x, y, accels, inputs]);
    results.Properties.VariableNames = {'t', 'x_1', 'x_2', 'x_3', 'dx_1', 'dx_2','dx_3', 'u_1', 'u_2', 'u_3'};
end
    
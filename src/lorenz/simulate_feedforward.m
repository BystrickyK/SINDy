%% Initialize
disp("Initializing...")
addpath('analytical_models')
clear all;
close all;
clc;

savefile = 'lorenz_sim_sgn.csv';
savepath = ['.' filesep '..' filesep '..' filesep 'data' filesep 'lorenz' filesep savefile];

params.sigma = 10;
params.beta = 8/3;
params.rho = 28;
        
        
%% Create external input signal
disp("Creating input signal...")
f_s = 1000;     % sampling frequency
dt = 1/f_s;     % sampling period
N = 2^16;     % number of samples
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
% control_law = @(x) [-0.25*(x(1)-20), -0.25*(x(2)-20), -0.25*(x(3)-20)];
random_process = @(t) interp1(time, noise_bandlimited, t, 'previous');
u_fun = @(t, x) (control_law(x) + random_process(t));
%%
% a = noise_bandlimited(:,1);
% a_hat = fftshift(fft(a));
% figure()
% subplot(121)
% plot(time, a, 'linewidth', 1.5, 'color', 'red')
% xlim([0 10])
% xlabel('$Time\ t [s]$', 'FontSize', 18, interpreter='latex')
% ylabel('$Value\ u\ [1]$', 'FontSize', 18, interpreter='latex')
% grid on
% subplot(122)
% plot(freq, abs(a_hat).^2, 'linewidth', 1.5)
% xlim([-10 10])
% xlabel('$Frequency\ \omega\ [\frac{rad}{s}]$', 'FontSize', 18, interpreter='latex')
% ylabel('$Power\ P_u\ [1^2]$', 'FontSize', 18, interpreter='latex')
% grid on

%% Solve
disp("Solving...")
% Define the system of ODEs
odefun = @(t, x)lorenz(t, x, u_fun, params);

x0 = [-15, -15, 15]'; % initial conditions
tspan = [0, t_end]; % time span

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc



x = (0:dt:t_end)';
y = deval(sol, x)';
tmp = [x, y];
tmp = num2cell(tmp, 2);
accels = cellfun(@(in) odefun(in(1), in(2:4)), tmp, 'UniformOutput', false);
accels = cell2mat(accels);
accels = reshape(accels, 3, [])';
results = array2table([x, y, accels, u_fun(x, y)]);
results.Properties.VariableNames = {'t', 'x_1', 'x_2', 'x_3', 'dx_1', 'dx_2','dx_3', 'u_1', 'u_2', 'u_3'};
writetable(results, savepath)


%% Animation
disp("Animating...")

scale = 0.9;
h = figure('Position', [10 10 1920*scale 1080*scale]);

t = tiledlayout(3,5, 'TileSpacing', 'tight', 'Padding', 'tight');

a1 = nexttile([3, 2]);
grid on
traj = animatedline('MaximumNumPoints', 10000, 'LineWidth', 1.5, 'Color', 'blue');
point = animatedline('MarkerSize', 5, 'Color', 'black',...
        'Marker', 'o', 'MaximumNumPoints', 1, 'MarkerFaceColor', 'black');
view(3)
pbaspect([1 1 1])
xlim([min(results.x_1) max(results.x_1)])
ylim([min(results.x_2) max(results.x_2)])
zlim([min(results.x_3) max(results.x_3)])
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
hold on

state_vars = {'$x_1$', '$x_2$', '$x_3$'};
state = table2array(results(1:end, 2:4));
tile_locs = [3, 8, 13];
for i = 1:length(state_vars)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    ax(i).YLabel.Interpreter = 'latex';
    ax(i).YLabel.FontSize = 20;
    ax(i).YLabel.String = var;
    lx(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue');
    xlim(tspan);
    grid on
end

der_vars = {'$\dot{x}_1$', '$\dot{x}_2$', '$\dot{x}_3$'};
dot_state = table2array(results(1:end, 5:7));
tile_locs = [4, 9, 14];
for i = 1:length(state_vars)
    var = der_vars{i};
    adx(i) = nexttile(tile_locs(i));
    adx(i).YLabel.Interpreter = 'latex';
    adx(i).YLabel.FontSize = 20;
    adx(i).YLabel.String = var;
    ldx(i) = animatedline(adx(i), 'LineWidth', 2, 'Color', 'red');
    xlim(tspan);
    grid on
end

input_vars = {'$u_1$', '$u_2$', '$u_3$'};
inputs = table2array(results(1:end, 8:10));
tile_locs = [5, 10, 15];
for i = 1:length(state_vars)
    var = input_vars{i};
    au(i) = nexttile(tile_locs(i));
    au(i).YLabel.Interpreter = 'latex';
    au(i).YLabel.FontSize = 20;
    au(i).YLabel.String = var;
    lu(i) = animatedline(au(i), 'LineWidth', 2, 'Color', 'black');
    xlim(tspan);
    grid on
end


frames = [getframe(h)];

s = 100;
for k = (s+1):s:length(x)

    title(a1, x(k))
    
    addpoints(traj, state(k-s:k,1), state(k-s:k,2), state(k-s:k,3));
    addpoints(point, state(k,1), state(k,2), state(k,3));
%     view(a1, k/5/s, 30);
    view(a1, 45, 30);
    
    for i = 1:length(state_vars)
       addpoints(lx(i), x(k-s:k), state(k-s:k, i));
       ax(i).XLim = [x(k)-9, x(k)+2];
       addpoints(ldx(i), x(k-s:k), dot_state(k-s:k, i));
       adx(i).XLim = [x(k)-9, x(k)+2];
       addpoints(lu(i), x(k-s:k), inputs(k-s:k, i));
       au(i).XLim = [x(k)-9, x(k)+2];
    end
    
    drawnow
    frames(end+1) = getframe(h);
    
end

writerObj = VideoWriter('lorenz_feedforward_horizontal');
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
    
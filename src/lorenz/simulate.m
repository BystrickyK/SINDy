%% Initialize
disp("Initializing...")
clear all;
close all;
clc;

savefile = 'lorenz_simdata.csv';

params.sigma = 10;
params.beta = 8/3;
params.rho = 28;
        
        
%% Create external input signal
disp("Creating input signal...")
t_end = 60;     % simulation time
f_s = 1000;     % sampling
f_c = 0.5;        % cutoff
u_a = 1000;      % input power
n = t_end * f_s;

% Random walk
    [filt_b, filt_a] = butter(3, f_c/(f_s/2));  % (order, cutoff freq)
    u_t = 0:(1/f_s):t_end;
    squarewave = ((square(u_t, 70)+1)/2)';
    u = u_a * (rand(length(u_t), 3) - 0.5);
    u = filter(filt_b, filt_a, u); 
    
%     define the input signal as a time-dependent function
    u_f = @(t) interp1(u_t, u, t);  

%% Solve


disp("Solving...")
% Define the system of ODEs
odefun = @(t, x)lorenz(t, x, u_f, params);

x0 = [1, 2, 3]'; % initial conditions
tspan = [0, t_end]; % time span

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc


% x = (logspace(0, 0.3, 1200) - 1) * t_end;
% x = linspace(0, t_end, 1000);
x = (0:1/f_s:t_end)';
y = deval(sol, x)';
tmp = [x, y];
tmp = num2cell(tmp, 2);
accels = cellfun(@(in) odefun(in(1), in(2:4)), tmp, 'UniformOutput', false);
accels = cell2mat(accels);
accels = reshape(accels, 3, [])';
results = array2table([x, y, accels, u_f(x)]);
results.Properties.VariableNames = {'t', 'x_1', 'x_2', 'x_3', 'dx_1', 'dx_2','dx_3', 'u_1', 'u_2', 'u_3'};
writetable(results, 'lorenz_sim.csv')


%% Plot
% disp("Stackedplot...")
% figure()
% stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
% grid on

%% Animation
disp("Animating...")

scale = 0.9;
h = figure('Position', [10 10 1920*scale 1080*scale]);

t = tiledlayout(3,5, 'TileSpacing', 'tight', 'Padding', 'tight');

a1 = nexttile([3, 2]);
grid on
traj = animatedline('MaximumNumPoints', 15000, 'LineWidth', 2, 'Color', 'blue');
view(3)
pbaspect([1 1 1])
xlim([-30 30])
ylim([-30 30])
zlim([0 50])
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

s = 200;
for k = (s+1):s:length(x)

    title(a1, x(k))
    
    addpoints(traj, state(k-s:k,1), state(k-s:k,2), state(k-s:k,3));
    view(a1, k/5/s, 45);
    
    for i = 1:length(state_vars)
       addpoints(lx(i), x(k-s:k), state(k-s:k, i));
       ax(i).XLim = [x(k)-12, x(k)+2];
       addpoints(ldx(i), x(k-s:k), dot_state(k-s:k, i));
       adx(i).XLim = [x(k)-12, x(k)+2];
       addpoints(lu(i), x(k-s:k), inputs(k-s:k, i));
       au(i).XLim = [x(k)-12, x(k)+2];
    end
    
    drawnow
%     frames(end+1) = getframe(h);
    
end

% writerObj = VideoWriter('triplePendulumCart');
% writerObj.FrameRate = 40;
% 
% open(writerObj);
% for k = 2:length(frames)
%    frame = frames(k);
%    writeVideo(writerObj, frame);
% end
% close(writerObj);

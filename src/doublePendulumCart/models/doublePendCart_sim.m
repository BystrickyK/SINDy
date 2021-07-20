%% Initialize
disp("Initializing...")
clear all;
close all;
clc;

savefile = 'doublePend.csv';

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
t_end = 160;     % simulation time
f_s = 1000;     % sampling
f_c = 1;        % cutoff
u_a = 2;      % input power
n = t_end * f_s;

% Random walk
    [filt_b, filt_a] = butter(3, f_c/(f_s/2));  % (order, cutoff freq)
    u_t = 0:(1/f_s):t_end;
    squarewave = ((square(u_t, 70)+1)/2)';
    u = u_a * (rand(length(u_t), 1) - 0.5);
    u = filter(filt_b, filt_a, u);  % defines the cart position!
    u = u .* squarewave;
    
    squarewave2 = ones(size(u));
    squarewave2(1:end/6) = 0.25;
    u = u .* squarewave2;

    limit = 0.4;
    u(u>limit) = limit;
    u(u<(-limit)) = -limit;  % ensures that the cart position doesn't go much further than 0.5
    u = filter(filt_b, 0.25*filt_a, u);  % smoothens the trajectory (necessary due to the cutoffs
    u = diff(u)/(1/f_s);   % get cart velocity by differentiation
    du = diff(u)/(1/f_s);   % get cart acceleration == input signal
    
    du(end+1: end+2) = du(end); % extend the input signal to fix the signal length
    
    f = figure();
    plot(du./9.81);
    ylabel('Acceleration (G)')
    xlabel('Sample index')
    
    f2 = figure();
    plot(cumsum(cumsum(du./f_s)./f_s));
    ylabel('Double integrated acceleration')
    xlabel('Index')
    
%     define the input signal as a time-dependent function
    u_f = @(t) interp1(u_t, du, t);
%     u_f = @(t) interp1(u_t, u, t);
    inputdata = array2table([u_t', du]);
    
    savepath = 'input_signals/input_acc_mid';
    writetable(inputdata, strcat(savepath, '.csv'));
    saveas(f, savepath, 'fig')
    saveas(f, savepath, 'png')
    saveas(f2, strcat(savepath, '_dInt'), 'png')

%% Solve
disp("Solving...")
% Define the system of ODEs
odefun = @(t, X)doublePendCartFullAnalytical(t, X, u_f, params);

x0 = [0, pi/4, 0, 0, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
tic
sol = ode45(odefun, tspan, x0);
toc


% x = (logspace(0, 0.3, 1200) - 1) * t_end;
% x = linspace(0, t_end, 1000);
x = 0:1/f_s:t_end;
y = deval(sol, x);
accels = odefun(x,y);
accels = accels(4:6,:)';
results = array2table([x', y', accels, u_f(x)']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'Ds', 'Dphi1','Dphi2', 'DDs', 'DDphi1', 'DDphi2', 'u'};
writetable(results, 'doublePendSimData.csv')


%% Plot
disp("Stackedplot...")
figure()
stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
grid on

%% Define important coordinates and their derivatives
disp("Defining important coordinates...")
q = [results.s, results.phi1, results.phi2];
dq = [results.Ds, results.Dphi1, results.Dphi2];

P_c = @(q) [q(1); 0];        % Cart

P_1 = @(q) [l_1*sin(q(2)) + q(1);
    -l_1*cos(q(2))];         % Pendulum 1

P_2 = @(q) [l_2*sin(q(3)) + l_1*sin(q(2)) + q(1);
    -l_2*cos(q(3)) - l_1*cos(q(2))];         % Pendulum 2


dP_c = @(q, dq) [dq(1); 0];

dP_1 = @(q,dq) [dq(1) + a_1*cos(q(2))*dq(2);
                a_1*sin(q(2))*dq(2)];

dP_2 = @(q,dq) [dq(1) + a_2*cos(q(3))*dq(3) + l_1*cos(q(2))*dq(2);
                a_2*sin(q(3))*dq(3) + l_1*sin(q(2))*dq(2)];
            


%% Animation
disp("Animating...")

scale = 0.9;
h = figure('Position', [10 10 1920*scale 1080*scale]);

t = tiledlayout(2,5, 'TileSpacing', 'tight', 'Padding', 'tight');

a1 = nexttile([2, 2]);
grid on
xlim([-1 1])
ylim([-1 1])
pbaspect([1 1 1])
hold on

state_vars = results.Properties.VariableNames(2:end-4);
tile_locs = [3, 8, 4, 9, 5, 10];
for i = 1:length(state_vars)
    disp(i)
    var = state_vars{i};
    ax(i) = nexttile(tile_locs(i));
    ylabel(var);
    lx(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue');
    xlim(tspan);
    grid on
end


frames = [getframe(h)];

for k = round(length(q)/5*2) : 12 : length(q)
    cla(a1)
    
    pc = P_c(q(k, :));
    p1 = P_1(q(k, :));
    p2 = P_2(q(k, :));
    
    title(a1, x(k))
    plot(a1, [pc(1), p1(1)], [pc(2), p1(2)], 'kO-', 'LineWidth', 3)
    plot(a1, [p1(1), p2(1)], [p1(2), p2(2)], 'rO-', 'LineWidth', 3)
    plot(a1, [pc(1), pc(1)+u_f(x(k))/20], [pc(2), pc(2)], 'r', 'LineWidth', 2)  
    
    for i = 1:length(state_vars)
       var = state_vars{i};
       addpoints(lx(i), x(k), results.(var)(k)/g);
       ax(i).XLim = [x(k)-12, x(k)+2];
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
%     
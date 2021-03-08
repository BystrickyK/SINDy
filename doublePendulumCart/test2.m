%% Initialize
disp("Initializing...")
clear all;
close all;
clc;

l_1 = 0.33; l_2 = 0.33;
a_1 = 0.2; a_2 = 0.2;
m_1 = 4; m_2 = 3; m_c = 1;
I_1 = 0.9*m_1*a_1^2; I_2 = 0.9*m_2*a_2^2;
b_1 = 0.1; b_2 = 0.08;
g = 9.81;

params = [m_1, m_2,...             % pendulum weights
            I_1, I_2,...           % penndulum moments of inertia around CoMs
            a_1, a_2,...          % pendulum distances from joints to CoMs
            l_1, l_2,...           % pendulum lengths
            b_1, b_2,...     % joint damping coefficients
            g];
        
        
%% Create external input signal
disp("Creating input signal...")
t_end = 100;     % simulation time
f_s = 100;     % sampling
f_c = 1;        % cutoff (2)
u_a = 7;      % input power (4)

% Random walk
    [filt_b, filt_a] = butter(3, f_c/(f_s/2));  % (order, cutoff freq)
    u_t = 0:(1/f_s):t_end;
    u = u_a * (rand(length(u_t), 1) - 0.5);
    u = filter(filt_b, filt_a, u);  % defines the cart position!
    u(u>0.5) = 0.5;
    u(u<(-0.5)) = -0.5;  % ensures that the cart position doesn't go much further than 0.3
    
%     u(end*20/t_end:35/t_end*end) = 0;
%     u(end*35/t_end:37/t_end*end) = -0.5; % cart moves to -0.5
%     u(end*37/t_end:40/t_end*end) = 0.5; % cart moves to 0.5
%     u(end*40/t_end:41/t_end*end) = -0.5; % cart moves to -0.5
%     
%     u(41/90*end:49/90*end) = 0; % cart moves to 0
% 
%     u(90/t_end*end:end) = 0;
    
%     figure();subplot(121);plot(u,'k');hold on
    u = filter(filt_b, filt_a, u);  % smoothens the trajectory (necessary due to the cutoffs)
%     plot(u,'b');
    
    u = diff(u)/(1/f_s);   % get cart velocity by differentiation!
    u = diff(u)/(1/f_s);   % get cart acceleration == input signal
    
    u(end+1:end+2) = u(end); % extend the input signal to fix the signal length
%     subplot(122);plot(u./9.81,'r');
%     subplot(121);plot(cumsum(cumsum(u))/length(u),'r-.');
    % Create the input signal as a time-dependent function
    u_f = @(t) interp1(u_t, u, t);

% Constant discontinuous
%     u = (randi(6,[180,1])-u_a/2);
%     u_t = linspace(0, t_end, length(u));
%     u_f = @(t) interp1(u_t, u, t, 'previous');

% Sin
%     u_t = 0:(1/f_s):t_end;
%     u = sin(u_t*4);
%     u_f = @(t) interp1(u_t, u, t);

% Pulses
%     nk = t_end*f_s; % number of samples
%     u_t = linspace(0, t_end, nk);
%     u = zeros([nk,1]);
%     u(0.02*nk:0.05*nk) = 30000;
%     u(0.055*nk:0.085*nk) = -30000;
%     u_f = @(t) interp1(u_t, u, t);


%% Solve
disp("Solving...")
% Define the system of ODEs
odefun = @(t, X)doublePendCart(t, X, u_f, params);

x0 = [0, 0, 0, 0, 0, 0]; % initial conditions
tspan = [0 t_end]; % time span

% Solve the system of ODEs
sol = ode45(odefun, tspan, x0);

% x = (logspace(0, 0.3, 1200) - 1) * t_end;
% x = linspace(0, t_end, 1000);
x = 0:0.002:t_end;
y = deval(sol, x);
results = array2table([x', y', u_f(x)']);
results.Properties.VariableNames = {'t', 's', 'phi1', 'phi2', 'Ds', 'Dphi1','Dphi2', 'u'};
writetable(results, 'doublePendSimData.csv')
%% Plot
% disp("Stackedplot...")
% figure()
% stackedplot(results, '-|k', 'XVariable', 't', 'LineWidth', 1.25)
% grid on

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

h = figure('Position', [10 10 1500 1500]);

a1 = subplot(2,2,1);
grid on
xlim([-2 2])
ylim([-2 2])
pbaspect manual
hold on

a2 = subplot(2,2,3);
yline(0, '-.', 'LineWidth', 2)
ylabel('Cart acceleration [G]');
grid on
xlim(tspan)
ylim([ min(results.u)/g max(results.u)/g])
l_u = animatedline(a2, 'LineWidth', 2, 'Color', 'red');

state_vars = results.Properties.VariableNames(2:end-1);
for i = 1:length(state_vars)
    var = state_vars{i};
    ax(i) = subplot(length(state_vars), 2, i*2);
    ylabel(var);
    lx(i) = animatedline(ax(i), 'LineWidth', 2, 'Color', 'blue');
    xlim(tspan);
%     ylim([ min(results.(var)) max(results.(var))]);
    grid on
end


frames = [getframe(h)];

for k = 1:20:length(q)
    cla(a1)
    
    pc = P_c(q(k, :));
    p1 = P_1(q(k, :));
    p2 = P_2(q(k, :));
    
    title(a1, x(k))
    plot(a1, [pc(1), p1(1)], [pc(2), p1(2)], 'kO-', 'LineWidth', 3)
    plot(a1, [p1(1), p2(1)], [p1(2), p2(2)], 'rO-', 'LineWidth', 3)
    plot(a1, [pc(1), pc(1)+u_f(x(k))/20], [pc(2), pc(2)], 'r', 'LineWidth', 2)  
    
    addpoints(l_u, x(k), results.u(k)/g);
    a2.XLim = [x(k)-12, x(k)+2];
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
%% Initialization
clearvars, clc
syms m_c m_1 m_2 m_3   % masses
syms I_1 I_2 I_3 % moments of inertia
syms a_1 a_2 a_3 % pendulum joint to CoM distance
syms l_1 l_2 l_3 % pendulum lengths
syms b_1 b_2 b_3 % joint damping coefficients

assume([m_c, m_1, m_2, m_3], 'positive')
assumeAlso([m_c, m_1, m_2, m_3], 'real')

assume([I_1, I_2, I_3], 'positive')
assumeAlso([I_1, I_2, I_3], 'real')

assume([a_1, a_2, a_3], 'positive')
assumeAlso([a_1, a_2, a_3], 'real')

assume([l_1, l_2, l_3], 'positive')
assumeAlso([l_1, l_2, l_3], 'real')

assume([b_1, b_2, b_3], 'positive')
assumeAlso([b_1, b_2, b_3], 'real')


syms t     % Time (not a coordinate)
assume(t, 'real')
assumeAlso(t, 'positive')

%% Generalized coordinates
syms s(t)           % Cart position
ds = diff(s, t);    % Cart velocity

syms phi_1(t) phi_2(t) phi_3(t)     % Pendulum angles from y axis (downward position == 0)
dPhi_1 = diff(phi_1, t);             % Pendulum angle velocities
dPhi_2 = diff(phi_2, t);
dPhi_3 = diff(phi_3, t);

q = [s; phi_1; phi_2; phi_3];  % Vector of generalized coordinates
q_i = q(t);  % This step is necessary in order to index singular elements
assume(q_i, 'real')

dq = diff(q, t);  % Vector of generalized velocities
dq_i = dq(t);  % This step is necessary in order to index singular elements
assume(dq_i, 'real')

%% CoM coordinates
P_c = [s; 0];        % Cart

P_1 = [a_1*sin(phi_1) + s;
    -a_1*cos(phi_1)];         % Pendulum 1

P_2 = [a_2*sin(phi_2) + l_1*sin(phi_1) + s;
    -a_2*cos(phi_2) - l_1*cos(phi_1)];         % Pendulum 2

P_3 = [a_3*sin(phi_3) + l_2*sin(phi_2) + l_1*sin(phi_1) + s;
    -a_3*cos(phi_3) - l_2*cos(phi_2) - l_1*cos(phi_1)];         % Pendulum 3

P_c = P_c(t);
P_1 = P_1(t);
P_2 = P_2(t);
P_3 = P_3(t);

assumeAlso(P_c, 'real')
assumeAlso(P_1, 'real')
assumeAlso(P_2, 'real')
assumeAlso(P_3, 'real')

%% CoM velocities
dP_c = diff(P_c, t);     % Cart
dP_1 = diff(P_1, t);     % Pendulum 1
dP_2 = diff(P_2, t);     % Pendulum 2
dP_3 = diff(P_3, t);     % Pendulum 3

assumeAlso(dP_c, 'real')
assumeAlso(dP_1, 'real')
assumeAlso(dP_2, 'real')
assumeAlso(dP_3, 'real')


%% Generalized force vector
syms F real
Tau_i = [F; 0; 0; 0];

%% Kinetic energy T
T = 0.5 * m_1 * dP_1' * dP_1 + ...
    0.5 * m_2 * dP_2' * dP_2 + ...
    0.5 * m_3 * dP_3' * dP_3 + ...
    0.5 * I_1 * dPhi_1^2 + ...
    0.5 * I_2 * dPhi_2^2 + ...
    0.5 * I_3 * dPhi_3^2 + ...
    0.5 * m_c * dP_c' * dP_c;

%% Potential energy V
g = 9.0665;      % Gravitational acceleration

V = g * m_1 * P_1(2) + ...
    g * m_2 * P_2(2) + ...
    g * m_3 * P_3(2);

%% Lagrangian
L = simplify(T - V);

%% Rayleigh dissipation function R
R = 0.5 * b_1 * dPhi_1^2 + ...
    0.5 * b_2 * (dPhi_2 - dPhi_1)^2 + ...
    0.5 * b_3 * (dPhi_3 - dPhi_2)^2;

%% Equations of motion
eqs = sym(zeros(4,1));
for i = 1:length(q_i)
    eqs(i) = diff( diff(L, dq_i(i)), t) - diff(L, q_i(i)) + diff(R, dq_i(i)) == Tau_i(i);
end

% Simplification -> Cart acceleration given purely by system input
syms u real
eqs(1) = diff(dq_i(1), t) == u;

%% Convert EoMs to a system of 1st order state derivative equations
[VF, state_vars, R] = reduceDifferentialOrder(eqs, q_i);
[M, F] = massMatrixForm(VF, state_vars);
assumeAlso(state_vars, 'real')
f = M\F;

%% Generate state derivative function
comment = ['Input form (t, X, u, parameters)',...
    'State vars: ',...
    join(string(state_vars), '; ')];

odeFunction(f, state_vars, u,...
    [m_1, m_2, m_3, I_1, I_2, I_3, a_1, a_2, a_3, ...
    l_1, l_2, l_3, b_1, b_2, b_3], ...
    'File', 'triplePendCart2', ...
    'Comments', comment);

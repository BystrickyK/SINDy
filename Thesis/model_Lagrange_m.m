%Parameters
clearvars, clc
syms m_c m_1 m_2 m_3 positive real  % masses
syms I_1 I_2 I_3 positive real      % moments of inertia
syms a_1 a_2 a_3 positive real     % pendulum joint to CoM distance
syms l_1 l_2 l_3 positive real     % pendulum lengths
syms b_1 b_2 b_3 positive real     % joint damping coefficients

syms t positive real     % Time (not a coordinate)
%% Generalized coordinates
syms s(t)           % Cart position
ds = diff(s, t);    % Cart velocity

syms phi_1(t) phi_2(t) phi_3(t)      % Pendulum angles from y axis (downward position == 0)
dPhi_1 = diff(phi_1, t);             % Pendulum angle velocities
dPhi_2 = diff(phi_2, t);
dPhi_3 = diff(phi_3, t);

q = [s; phi_1; phi_2; phi_3];  % Vector of generalized coordinates
q_i = q(t);  % This step is necessary in order to index singular elements

dq = diff(q, t);  % Vector of generalized velocities
dq_i = dq(t);  % This step is necessary in order to index singular elements
%% CoM coordinates
P_c = [s; 0];        % Cart

P_1 = [a_1*sin(phi_1) + s;
    -a_1*cos(phi_1)];         % Pendulum 1

P_2 = [a_2*sin(phi_2) + l_1*sin(phi_1) + s;
    -a_2*cos(phi_2) - l_1*cos(phi_1)];         % Pendulum 2

P_3 = [a_3*sin(phi_3) + l_2*sin(phi_2) + l_1*sin(phi_1) + s;
    -a_3*cos(phi_3) - l_2*cos(phi_2) - l_1*cos(phi_1)];         % Pendulum 3

%% CoM velocities
dP_c = diff(P_c, t);     % Cart
dP_1 = diff(P_1, t);     % Pendulum 1
dP_2 = diff(P_2, t);     % Pendulum 2
dP_3 = diff(P_3, t);     % Pendulum 3

%% Generalized force vector
syms F(t)
Tau = [F; 0; 0; 0];
Tau_i = Tau(t);

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

tmpP_1 = P_1(t); % temporary variables have to be initialized with 't',
tmpP_2 = P_2(t); % vector indexing is impossible otherwise (because
tmpP_3 = P_3(t); % round brackets are used for both indexing and function calling)

V = g * m_1 * tmpP_1(2) + ...
    g * m_2 * tmpP_2(2) + ...
    g * m_3 * tmpP_3(2);

clearvars tmpP_1 tmpP_2 tmpP_3

%% Lagrangian
L = simplify(T - V);

%% Rayleigh dissipation function R
R = 0.5 * b_1 * dPhi_1^2 + ...
    0.5 * b_2 * (dPhi_2 - dPhi_1)^2 + ...
    0.5 * b_3 * (dPhi_3 - dPhi_2)^2;

eqs = sym(zeros(4,1));
for i = 1:length(q_i)
    eqs(i) = diff( diff(L, dq_i(i)), t) - diff(L, q_i(i)) + diff(R, dq_i(i)) == Tau_i(i);
end

% Simplification -> Cart acceleration given purely by system input
syms u real
eqs(1) = diff(dq_i(1), t) == u;
%% Solution
% syms ddq [4 1] real
% vars = ddq
% eqns = subs(eqs, diff(q, t, 2), vars)
% 
% sols = solve(eqns, vars, 'Real', true)

% VF == system of 1st order ODEs
[VF, subs] = odeToVectorField(eqs);
matlabFunction(VF, 'File', 'triplePendCart');
writematrix(latex(VF), 'waytoolong.txt');
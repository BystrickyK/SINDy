function out = identified_model_sqn_clean_(t, x, u_fun)
  %IDENTIFIED_MODEL_SQN_CLEAN  Autogenerated by sympy
  %   Code generated with sympy 1.7.1
  %
  %   See http://www.sympy.org/ for more information.
  %
  %   This file is part of 'project'
  x_1 = x(1);
  x_2 = x(2);
  x_3 = x(3);
  
  u = u_fun(t, x);
  u_1 = u(1);
  u_2 = u(2);
  u_3 = u(3);
  % unsupported: sgn
  out1 = -10.0*x_1 + 10.0*x_2 + 39.99999*sgn(u_1);
  out2 = 10.0*u_2.*sin(u_1) - x_1.*x_3 + 28.0*x_1 - x_2;
  out3 = u_3.*x_1 + x_1.*x_2 - 2.66667*x_3;

  
  out = [out1, out2, out3]';
end
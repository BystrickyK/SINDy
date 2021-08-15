function out = identified_model_real_(t, x, u_fun)
  %IDENTIFIED_MODEL_NOISY  Autogenerated by sympy
  %   Code generated with sympy 1.7.1
  %
  %   See http://www.sympy.org/ for more information.
  %
  %   This file is part of 'project'

  u = u_fun(t);
  
  x_1 = x(1);
  x_2 = x(2);
  x_3 = x(3);
  x_4 = x(4);
  
  out1 = (0.01142*u.*cos(x_2).^2 - 0.0226*u + 2.29663*x_3 - 0.02736*x_4.^2.*sin(x_2))./(0.5374*cos(x_2).^2 - 1.0);
  out2 = (-0.02791*u.*cos(x_2).^2 + 0.09094*u.*cos(x_2) - 8.91299*x_3.*cos(x_2) + 0.04447*x_4.^2.*sin(2.0*x_2) + 42.87518*sin(x_2) - 7.65352*sin(2.0*x_2))./(0.29978*cos(x_2) - 1.0);

  out = [x_3, x_4, out1, out2]';
  
end
function expr = doublePendCartFullAnalytical(t,in2,u,in4)
%DOUBLEPENDCARTFULLANALYTICAL
%    EXPR = DOUBLEPENDCARTFULLANALYTICAL(T,IN2,PARAM14,IN4)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    06-May-2021 00:42:07

%Input form (t, X, u, parameters)
%State vars: 
%s(t); phi_1(t); phi_2(t); Dst2(t); Dphi_1t2(t); Dphi_2t2(t)
Dphi_1t2 = in2(5,:);
Dphi_2t2 = in2(6,:);
Dst2 = in2(4,:);
param14 = u(t);
param15 = in4(:,1);
param16 = in4(:,2);
param17 = in4(:,3);
param18 = in4(:,4);
param19 = in4(:,5);
param20 = in4(:,6);
param21 = in4(:,7);
param22 = in4(:,8);
param24 = in4(:,10);
param25 = in4(:,11);
param26 = in4(:,12);
param27 = in4(:,13);
phi_1 = in2(2,:);
phi_2 = in2(3,:);
t2 = cos(phi_1);
t3 = cos(phi_2);
t4 = sin(phi_1);
t5 = sin(phi_2);
t6 = Dphi_1t2.^2;
t7 = Dphi_2t2.^2;
t8 = param16.^2;
t9 = param17.^2;
t10 = param20.^2;
t11 = param20.^3;
t12 = param21.^2;
t13 = param21.^3;
t14 = param22.^2;
t15 = param22.^3;
t24 = param15.*param18.*param19;
t25 = param16.*param18.*param19;
t26 = param17.*param18.*param19;
t16 = t2.^2;
t17 = t2.^3;
t18 = t3.^2;
t19 = t3.^3;
t20 = t4.^2;
t21 = t4.^3;
t22 = t5.^2;
t23 = t5.^3;
t27 = Dphi_1t2.*param16.*param17.*param20.*param21.*param25.*t2.*t3;
t28 = Dphi_1t2.*param15.*param17.*param21.*param22.*param25.*t2.*t3;
t29 = Dphi_1t2.*param16.*param17.*param21.*param22.*param25.*t2.*t3;
t30 = Dphi_2t2.*param16.*param17.*param20.*param21.*param25.*t2.*t3;
t31 = Dphi_2t2.*param15.*param17.*param21.*param22.*param25.*t2.*t3;
t32 = Dphi_2t2.*param16.*param17.*param21.*param22.*param25.*t2.*t3;
t33 = Dphi_1t2.*param15.*param17.*param21.*param22.*param25.*t4.*t5;
t34 = Dphi_1t2.*param16.*param17.*param21.*param22.*param25.*t4.*t5;
t35 = Dphi_2t2.*param15.*param17.*param21.*param22.*param25.*t4.*t5;
t36 = Dphi_2t2.*param16.*param17.*param21.*param22.*param25.*t4.*t5;
t50 = Dphi_1t2.*param21.*param22.*param25.*t4.*t5.*t9;
t51 = Dphi_2t2.*param21.*param22.*param25.*t4.*t5.*t9;
t66 = param16.*param20.*param22.*t2.*t3.*t4.*t5.*t9.*t12.*2.0;
t70 = param15.*t2.*t3.*t4.*t5.*t9.*t12.*t14.*2.0;
t71 = param16.*t2.*t3.*t4.*t5.*t9.*t12.*t14.*2.0;
t37 = param15.*param16.*param19.*t10.*t16;
t38 = param15.*param17.*param18.*t12.*t18;
t39 = param16.*param17.*param19.*t10.*t16;
t40 = param15.*param17.*param19.*t14.*t16;
t41 = param16.*param17.*param18.*t12.*t18;
t42 = param16.*param17.*param19.*t14.*t16;
t43 = param15.*param16.*param19.*t10.*t20;
t44 = param15.*param17.*param18.*t12.*t22;
t45 = param16.*param17.*param19.*t10.*t20;
t46 = param15.*param17.*param19.*t14.*t20;
t47 = param16.*param17.*param18.*t12.*t22;
t48 = param16.*param17.*param19.*t14.*t20;
t49 = param16.*param17.*param19.*param20.*param22.*t16.*2.0;
t52 = param19.*t8.*t10.*t20;
t53 = param18.*t9.*t12.*t22;
t54 = param19.*t9.*t14.*t20;
t56 = param15.*param16.*param17.*t10.*t12.*t16.*t18;
t57 = param15.*param16.*param17.*t10.*t12.*t16.*t22;
t58 = param15.*param16.*param17.*t10.*t12.*t18.*t20;
t59 = param15.*param16.*param17.*t10.*t12.*t20.*t22;
t60 = param16.*t9.*t10.*t12.*t16.*t22;
t61 = param17.*t8.*t10.*t12.*t18.*t20;
t62 = param15.*t9.*t12.*t14.*t16.*t22;
t63 = param15.*t9.*t12.*t14.*t18.*t20;
t64 = param16.*t9.*t12.*t14.*t16.*t22;
t65 = param16.*t9.*t12.*t14.*t18.*t20;
t67 = param16.*t9.*t10.*t12.*t20.*t22;
t68 = param17.*t8.*t10.*t12.*t20.*t22;
t69 = param16.*param20.*param22.*t9.*t12.*t16.*t22.*2.0;
t73 = -t70;
t74 = -t71;
t55 = -t49;
t72 = -t69;
t75 = t24+t25+t26+t37+t38+t39+t40+t41+t42+t43+t44+t45+t46+t47+t48+t52+t53+t54+t55+t56+t57+t58+t59+t60+t61+t62+t63+t64+t65+t66+t67+t68+t72+t73+t74;
t76 = 1.0./t75;
et1 = param14.*param18.*param19+param20.*t2.*t34-param20.*t2.*t36+param20.*t4.*t6.*t25+param22.*t4.*t6.*t26+param21.*t5.*t7.*t26+param20.*t4.*t6.*t41+param22.*t4.*t6.*t39+param20.*t4.*t6.*t42+param21.*t5.*t7.*t39+param20.*t4.*t6.*t47+param21.*t5.*t7.*t45+param22.*t4.*t6.*t53+param21.*t5.*t7.*t54+param22.*t4.*t6.*t60-Dst2.*param18.*param19.*param26+param14.*param16.*param19.*t10.*t16+param14.*param16.*param19.*t10.*t20+param14.*param17.*param18.*t12.*t18+param14.*param17.*param19.*t14.*t16+param14.*param17.*param18.*t12.*t22+param14.*param17.*param19.*t14.*t20+param19.*t6.*t8.*t11.*t21+param18.*t7.*t9.*t13.*t23+param19.*t6.*t9.*t15.*t21+param19.*param27.*t2.*t4.*t8.*t10+param18.*param27.*t3.*t5.*t9.*t12+param19.*param27.*t2.*t4.*t9.*t14+param19.*t4.*t6.*t8.*t11.*t16+param19.*t4.*t6.*t9.*t15.*t16+param18.*t5.*t7.*t9.*t13.*t18+param14.*t9.*t12.*t14.*t16.*t22+param14.*t9.*t12.*t14.*t18.*t20+Dphi_1t2.*param16.*param19.*param20.*param24.*t2;
et2 = Dphi_1t2.*param16.*param19.*param20.*param25.*t2-Dphi_1t2.*param17.*param18.*param21.*param25.*t3+Dphi_1t2.*param17.*param19.*param22.*param24.*t2+Dphi_1t2.*param17.*param19.*param22.*param25.*t2-Dphi_2t2.*param16.*param19.*param20.*param25.*t2+Dphi_2t2.*param17.*param18.*param21.*param25.*t3-Dphi_2t2.*param17.*param19.*param22.*param25.*t2-Dst2.*param16.*param19.*param26.*t10.*t16-Dst2.*param16.*param19.*param26.*t10.*t20-Dst2.*param17.*param18.*param26.*t12.*t18-Dst2.*param17.*param19.*param26.*t14.*t16-Dst2.*param17.*param18.*param26.*t12.*t22-Dst2.*param17.*param19.*param26.*t14.*t20+Dphi_1t2.*param22.*param24.*t2.*t9.*t12.*t22-Dphi_1t2.*param21.*param25.*t3.*t9.*t14.*t20+Dphi_1t2.*param22.*param25.*t2.*t9.*t12.*t22+Dphi_2t2.*param21.*param25.*t3.*t9.*t14.*t20-Dphi_2t2.*param22.*param25.*t2.*t9.*t12.*t22-Dst2.*param26.*t9.*t12.*t14.*t16.*t22-Dst2.*param26.*t9.*t12.*t14.*t18.*t20+param16.*param17.*param19.*param22.*t6.*t10.*t21+param16.*param17.*param19.*param20.*t6.*t14.*t21+param14.*param16.*param17.*t10.*t12.*t16.*t18;
et3 = param14.*param16.*param17.*t10.*t12.*t16.*t22+param14.*param16.*param17.*t10.*t12.*t18.*t20+param14.*param16.*param17.*t10.*t12.*t20.*t22+param17.*t6.*t8.*t11.*t12.*t18.*t21+param16.*t7.*t9.*t10.*t13.*t16.*t23+param17.*t6.*t8.*t11.*t12.*t21.*t22+param16.*t7.*t9.*t10.*t13.*t20.*t23-Dphi_1t2.*param16.*param17.*param21.*param25.*t3.*t10.*t16+Dphi_1t2.*param16.*param17.*param20.*param24.*t2.*t12.*t18+Dphi_1t2.*param16.*param17.*param20.*param25.*t2.*t12.*t18-Dphi_1t2.*param16.*param17.*param21.*param25.*t3.*t10.*t20+Dphi_1t2.*param16.*param17.*param20.*param24.*t2.*t12.*t22+Dphi_1t2.*param16.*param17.*param20.*param25.*t2.*t12.*t22+Dphi_2t2.*param16.*param17.*param21.*param25.*t3.*t10.*t16-Dphi_2t2.*param16.*param17.*param20.*param25.*t2.*t12.*t18+Dphi_2t2.*param16.*param17.*param21.*param25.*t3.*t10.*t20-Dphi_2t2.*param16.*param17.*param20.*param25.*t2.*t12.*t22-Dst2.*param16.*param17.*param26.*t10.*t12.*t16.*t18-Dst2.*param16.*param17.*param26.*t10.*t12.*t16.*t22-Dst2.*param16.*param17.*param26.*t10.*t12.*t18.*t20-Dst2.*param16.*param17.*param26.*t10.*t12.*t20.*t22;
et4 = -Dphi_1t2.*param22.*param24.*t3.*t4.*t5.*t9.*t12+Dphi_1t2.*param21.*param25.*t2.*t4.*t5.*t9.*t14-Dphi_1t2.*param22.*param25.*t3.*t4.*t5.*t9.*t12-Dphi_2t2.*param21.*param25.*t2.*t4.*t5.*t9.*t14+Dphi_2t2.*param22.*param25.*t3.*t4.*t5.*t9.*t12+param16.*param17.*param19.*param20.*param22.*param27.*t2.*t4.*2.0-param16.*param20.*param22.*t7.*t9.*t13.*t16.*t23+param18.*param22.*t2.*t3.*t5.*t6.*t9.*t12+param19.*param21.*t2.*t3.*t4.*t7.*t9.*t14+param16.*param27.*t3.*t5.*t9.*t10.*t12.*t16+param17.*param27.*t2.*t4.*t8.*t10.*t12.*t18+param16.*param27.*t3.*t5.*t9.*t10.*t12.*t20+param17.*param27.*t2.*t4.*t8.*t10.*t12.*t22+param16.*param20.*t6.*t9.*t12.*t14.*t18.*t21+param16.*param22.*t6.*t9.*t10.*t12.*t21.*t22-param14.*t2.*t3.*t4.*t5.*t9.*t12.*t14.*2.0+param17.*t4.*t6.*t8.*t11.*t12.*t16.*t18+param16.*t5.*t7.*t9.*t10.*t13.*t16.*t18+param17.*t4.*t6.*t8.*t11.*t12.*t16.*t22+param16.*t5.*t7.*t9.*t10.*t13.*t18.*t20;
et5 = Dphi_1t2.*param16.*param17.*param20.*param21.*param22.*param25.*t3.*t16-Dphi_2t2.*param16.*param17.*param20.*param21.*param22.*param25.*t3.*t16+Dst2.*param26.*t2.*t3.*t4.*t5.*t9.*t12.*t14.*2.0-param16.*param17.*param19.*param20.*param21.*param22.*t5.*t7.*t16+param16.*param20.*param22.*param27.*t2.*t4.*t9.*t12.*t18-param16.*param20.*param22.*param27.*t3.*t5.*t9.*t12.*t16+param16.*param20.*param22.*param27.*t2.*t4.*t9.*t12.*t22-param16.*param20.*param22.*param27.*t3.*t5.*t9.*t12.*t20+param16.*param20.*param22.*t2.*t4.*t7.*t9.*t13.*t19-param16.*param20.*param22.*t5.*t7.*t9.*t13.*t16.*t18+param16.*param22.*t3.*t5.*t6.*t9.*t10.*t12.*t17-param16.*param20.*t3.*t5.*t6.*t9.*t12.*t14.*t17+param16.*param20.*t4.*t6.*t9.*t12.*t14.*t16.*t18+param16.*param17.*param19.*param20.*param21.*param22.*t2.*t3.*t4.*t7+param16.*param20.*param22.*t2.*t3.*t4.*t7.*t9.*t13.*t22+param16.*param22.*t2.*t3.*t5.*t6.*t9.*t10.*t12.*t20-param16.*param20.*t2.*t3.*t5.*t6.*t9.*t12.*t14.*t20;
et6 = -t27+t28+t29+t30-t31-t32+t33+t34-t35-t36+t50-t51+Dphi_1t2.*param15.*param19.*param24+Dphi_1t2.*param15.*param19.*param25+Dphi_1t2.*param16.*param19.*param24+Dphi_1t2.*param16.*param19.*param25+Dphi_1t2.*param17.*param19.*param24+Dphi_1t2.*param17.*param19.*param25-Dphi_2t2.*param15.*param19.*param25-Dphi_2t2.*param16.*param19.*param25-Dphi_2t2.*param17.*param19.*param25+Dphi_1t2.*param24.*t9.*t12.*t22+Dphi_1t2.*param25.*t9.*t12.*t22-Dphi_2t2.*param25.*t9.*t12.*t22+param14.*param16.*param19.*param20.*t2+param14.*param17.*param19.*param22.*t2+param19.*param20.*param27.*t4.*t8+param19.*param22.*param27.*t4.*t9+param15.*param16.*param19.*param20.*param27.*t4+param16.*param17.*param19.*param20.*param27.*t4+param15.*param17.*param19.*param22.*param27.*t4+param16.*param17.*param19.*param22.*param27.*t4+param14.*param22.*t2.*t9.*t12.*t22+param19.*t2.*t4.*t6.*t8.*t10+param19.*t2.*t4.*t6.*t9.*t14-Dst2.*param16.*param19.*param20.*param26.*t2-Dst2.*param17.*param19.*param22.*param26.*t2+Dphi_1t2.*param15.*param17.*param24.*t12.*t18;
et7 = Dphi_1t2.*param15.*param17.*param25.*t12.*t18+Dphi_1t2.*param16.*param17.*param24.*t12.*t18+Dphi_1t2.*param16.*param17.*param25.*t12.*t18+Dphi_1t2.*param15.*param17.*param24.*t12.*t22+Dphi_1t2.*param15.*param17.*param25.*t12.*t22+Dphi_1t2.*param16.*param17.*param24.*t12.*t22+Dphi_1t2.*param16.*param17.*param25.*t12.*t22-Dphi_2t2.*param15.*param17.*param25.*t12.*t18-Dphi_2t2.*param16.*param17.*param25.*t12.*t18-Dphi_2t2.*param15.*param17.*param25.*t12.*t22-Dphi_2t2.*param16.*param17.*param25.*t12.*t22-Dst2.*param22.*param26.*t2.*t9.*t12.*t22+param14.*param16.*param17.*param20.*t2.*t12.*t18+param14.*param16.*param17.*param20.*t2.*t12.*t22+param19.*param21.*param22.*t3.*t4.*t7.*t9+param17.*param20.*param27.*t4.*t8.*t12.*t18+param15.*param22.*param27.*t4.*t9.*t12.*t18+param16.*param22.*param27.*t4.*t9.*t12.*t18+param16.*param20.*param27.*t4.*t9.*t12.*t22+param17.*param20.*param27.*t4.*t8.*t12.*t22-param14.*param22.*t3.*t4.*t5.*t9.*t12+param15.*param22.*t4.*t7.*t9.*t13.*t19+param16.*param20.*t2.*t7.*t9.*t13.*t23+param16.*param22.*t4.*t7.*t9.*t13.*t19;
et8 = -param15.*param22.*t2.*t7.*t9.*t13.*t23-param16.*param22.*t2.*t7.*t9.*t13.*t23-Dst2.*param16.*param17.*param20.*param26.*t2.*t12.*t18-Dst2.*param16.*param17.*param20.*param26.*t2.*t12.*t22+Dst2.*param22.*param26.*t3.*t4.*t5.*t9.*t12+param16.*param17.*param19.*param20.*param22.*t2.*t4.*t6.*2.0+param16.*param17.*param19.*param20.*param21.*t2.*t5.*t7-param15.*param17.*param19.*param21.*param22.*t2.*t5.*t7+param15.*param17.*param19.*param21.*param22.*t3.*t4.*t7-param16.*param17.*param19.*param21.*param22.*t2.*t5.*t7+param16.*param17.*param19.*param21.*param22.*t3.*t4.*t7+param15.*param16.*param17.*param20.*param27.*t4.*t12.*t18+param15.*param16.*param17.*param20.*param27.*t4.*t12.*t22+param16.*param20.*param27.*t2.*t3.*t5.*t9.*t12-param15.*param22.*param27.*t2.*t3.*t5.*t9.*t12-param16.*param22.*param27.*t2.*t3.*t5.*t9.*t12+param16.*param20.*t2.*t5.*t7.*t9.*t13.*t18-param15.*param22.*t2.*t5.*t7.*t9.*t13.*t18-param16.*param22.*t2.*t5.*t7.*t9.*t13.*t18+param15.*param22.*t3.*t4.*t7.*t9.*t13.*t22;
et9 = param16.*param22.*t3.*t4.*t7.*t9.*t13.*t22+param17.*t2.*t4.*t6.*t8.*t10.*t12.*t18+param15.*t2.*t4.*t6.*t9.*t12.*t14.*t18-param15.*t3.*t5.*t6.*t9.*t12.*t14.*t16+param16.*t2.*t4.*t6.*t9.*t12.*t14.*t18-param16.*t3.*t5.*t6.*t9.*t12.*t14.*t16+param17.*t2.*t4.*t6.*t8.*t10.*t12.*t22-param15.*t2.*t4.*t6.*t9.*t12.*t14.*t22+param15.*t3.*t5.*t6.*t9.*t12.*t14.*t20-param16.*t2.*t4.*t6.*t9.*t12.*t14.*t22+param16.*t3.*t5.*t6.*t9.*t12.*t14.*t20+param16.*param20.*param22.*t3.*t5.*t6.*t9.*t12.*t16+param16.*param20.*param22.*t2.*t4.*t6.*t9.*t12.*t22.*2.0-param16.*param20.*param22.*t3.*t5.*t6.*t9.*t12.*t20;
et10 = t27-t28-t29-t30+t31+t32-t33-t34+t35+t36-t50+t51-Dphi_1t2.*param15.*param18.*param25-Dphi_1t2.*param16.*param18.*param25-Dphi_1t2.*param17.*param18.*param25+Dphi_2t2.*param15.*param18.*param25+Dphi_2t2.*param16.*param18.*param25+Dphi_2t2.*param17.*param18.*param25-Dphi_1t2.*param25.*t8.*t10.*t20-Dphi_1t2.*param25.*t9.*t14.*t20+Dphi_2t2.*param25.*t8.*t10.*t20+Dphi_2t2.*param25.*t9.*t14.*t20+param14.*param17.*param18.*param21.*t3+param18.*param21.*param27.*t5.*t9+param15.*param17.*param18.*param21.*param27.*t5+param16.*param17.*param18.*param21.*param27.*t5+param14.*param21.*t3.*t9.*t14.*t20+param18.*t3.*t5.*t7.*t9.*t12-Dst2.*param17.*param18.*param21.*param26.*t3-Dphi_1t2.*param15.*param16.*param25.*t10.*t16-Dphi_1t2.*param16.*param17.*param25.*t10.*t16-Dphi_1t2.*param15.*param16.*param25.*t10.*t20-Dphi_1t2.*param15.*param17.*param25.*t14.*t16-Dphi_1t2.*param16.*param17.*param25.*t10.*t20-Dphi_1t2.*param16.*param17.*param25.*t14.*t16;
et11 = -Dphi_1t2.*param15.*param17.*param25.*t14.*t20-Dphi_1t2.*param16.*param17.*param25.*t14.*t20+Dphi_2t2.*param15.*param16.*param25.*t10.*t16+Dphi_2t2.*param16.*param17.*param25.*t10.*t16+Dphi_2t2.*param15.*param16.*param25.*t10.*t20+Dphi_2t2.*param15.*param17.*param25.*t14.*t16+Dphi_2t2.*param16.*param17.*param25.*t10.*t20+Dphi_2t2.*param16.*param17.*param25.*t14.*t16+Dphi_2t2.*param15.*param17.*param25.*t14.*t20+Dphi_2t2.*param16.*param17.*param25.*t14.*t20+Dphi_1t2.*param16.*param17.*param20.*param22.*param25.*t16.*2.0-Dphi_2t2.*param16.*param17.*param20.*param22.*param25.*t16.*2.0-Dphi_1t2.*param21.*param22.*param24.*t4.*t5.*t9-Dst2.*param21.*param26.*t3.*t9.*t14.*t20+param14.*param16.*param17.*param21.*t3.*t10.*t16+param14.*param16.*param17.*param21.*t3.*t10.*t20+param18.*param21.*param22.*t2.*t5.*t6.*t9+param16.*param21.*param27.*t5.*t9.*t10.*t16+param15.*param21.*param27.*t5.*t9.*t14.*t16+param16.*param21.*param27.*t5.*t9.*t10.*t20+param16.*param21.*param27.*t5.*t9.*t14.*t16+param17.*param21.*param27.*t5.*t8.*t10.*t20-param14.*param21.*t2.*t4.*t5.*t9.*t14+param17.*param21.*t3.*t6.*t8.*t11.*t21;
et12 = param15.*param21.*t5.*t6.*t9.*t15.*t17+param16.*param21.*t5.*t6.*t9.*t15.*t17-param15.*param21.*t3.*t6.*t9.*t15.*t21-param16.*param21.*t3.*t6.*t9.*t15.*t21+Dphi_1t2.*param16.*param17.*param20.*param21.*param24.*t2.*t3-Dphi_1t2.*param15.*param17.*param21.*param22.*param24.*t2.*t3-Dphi_1t2.*param16.*param17.*param21.*param22.*param24.*t2.*t3-Dphi_1t2.*param15.*param17.*param21.*param22.*param24.*t4.*t5-Dphi_1t2.*param16.*param17.*param21.*param22.*param24.*t4.*t5-Dst2.*param16.*param17.*param21.*param26.*t3.*t10.*t16-Dst2.*param16.*param17.*param21.*param26.*t3.*t10.*t20+Dst2.*param21.*param26.*t2.*t4.*t5.*t9.*t14-param14.*param16.*param17.*param20.*param21.*param22.*t3.*t16+param16.*param17.*param18.*param20.*param21.*t3.*t4.*t6+param15.*param17.*param18.*param21.*param22.*t2.*t5.*t6-param15.*param17.*param18.*param21.*param22.*t3.*t4.*t6+param16.*param17.*param18.*param21.*param22.*t2.*t5.*t6-param16.*param17.*param18.*param21.*param22.*t3.*t4.*t6+param15.*param16.*param17.*param21.*param27.*t5.*t10.*t16+param15.*param16.*param17.*param21.*param27.*t5.*t10.*t20;
et13 = param16.*param20.*param21.*param22.*param27.*t5.*t9.*t16.*-2.0-param16.*param20.*param21.*param22.*param27.*t5.*t9.*t20-param17.*param20.*param21.*param22.*param27.*t5.*t8.*t20+param17.*param21.*param27.*t2.*t3.*t4.*t8.*t10-param15.*param21.*param27.*t2.*t3.*t4.*t9.*t14-param16.*param21.*param27.*t2.*t3.*t4.*t9.*t14+param16.*param21.*param22.*t5.*t6.*t9.*t10.*t17-param16.*param20.*param21.*t5.*t6.*t9.*t14.*t17.*2.0-param17.*param21.*param22.*t3.*t6.*t8.*t10.*t21+param16.*param20.*param21.*t3.*t6.*t9.*t14.*t21+param17.*param21.*t3.*t4.*t6.*t8.*t11.*t16-param15.*param21.*t3.*t4.*t6.*t9.*t15.*t16-param16.*param21.*t3.*t4.*t6.*t9.*t15.*t16+param15.*param21.*t2.*t5.*t6.*t9.*t15.*t20+param16.*param21.*t2.*t5.*t6.*t9.*t15.*t20+param16.*t3.*t5.*t7.*t9.*t10.*t12.*t16-param15.*t2.*t4.*t7.*t9.*t12.*t14.*t18+param15.*t3.*t5.*t7.*t9.*t12.*t14.*t16-param16.*t2.*t4.*t7.*t9.*t12.*t14.*t18;
et14 = param16.*t3.*t5.*t7.*t9.*t10.*t12.*t20+param16.*t3.*t5.*t7.*t9.*t12.*t14.*t16+param15.*t2.*t4.*t7.*t9.*t12.*t14.*t22-param15.*t3.*t5.*t7.*t9.*t12.*t14.*t20+param16.*t2.*t4.*t7.*t9.*t12.*t14.*t22-param16.*t3.*t5.*t7.*t9.*t12.*t14.*t20+Dst2.*param16.*param17.*param20.*param21.*param22.*param26.*t3.*t16-param15.*param16.*param17.*param20.*param21.*param22.*param27.*t5.*t20-param14.*param16.*param17.*param20.*param21.*param22.*t2.*t4.*t5+param16.*param20.*param21.*param22.*param27.*t2.*t3.*t4.*t9-param17.*param20.*param21.*param22.*param27.*t2.*t3.*t4.*t8+param15.*param16.*param17.*param21.*param22.*t5.*t6.*t10.*t17-param15.*param16.*param17.*param21.*param22.*t3.*t6.*t10.*t21-param17.*param21.*param22.*t3.*t4.*t6.*t8.*t10.*t16+param16.*param20.*param21.*t3.*t4.*t6.*t9.*t14.*t16+param16.*param20.*param22.*t2.*t4.*t7.*t9.*t12.*t18-param16.*param20.*param22.*t3.*t5.*t7.*t9.*t12.*t16.*2.0+param16.*param21.*param22.*t2.*t5.*t6.*t9.*t10.*t20;
et15 = param16.*param20.*param21.*t2.*t5.*t6.*t9.*t14.*t20.*-2.0-param16.*param20.*param22.*t2.*t4.*t7.*t9.*t12.*t22+Dst2.*param16.*param17.*param20.*param21.*param22.*param26.*t2.*t4.*t5-param15.*param16.*param17.*param20.*param21.*param22.*param27.*t2.*t3.*t4-param15.*param16.*param17.*param21.*param22.*t3.*t4.*t6.*t10.*t16+param15.*param16.*param17.*param21.*param22.*t2.*t5.*t6.*t10.*t20;
expr = [Dst2;Dphi_1t2;Dphi_2t2;t76.*(et1+et2+et3+et4+et5);-t76.*(et6+et7+et8+et9);-t76.*(et10+et11+et12+et13+et14+et15)];

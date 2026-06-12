%%% Parametros de Falla %%%
% Copyleft: Cesar Jimenez 2010
clc
disp ('   PARAMETROS DE LA GEOMETRIA DE RUPTURA')
disp ('Ecuaciones de escalamiento de Papazachos 2004')
disp ('         Copyleft: Cesar Jimenez')
disp ('   ')
Mw = input ('Magnitud (Mw>6.5)  = ');
if Mw >= 6.5
  L = 10^(0.55*Mw-2.19);   % (km) Papazachos 2004
  fprintf ('%s %4.1f %s\n' ,'Longitud L = ',L,'km');
  W = 10^(0.31*Mw-0.63);   % (km)
  fprintf ('%s %4.1f %s\n' ,'Ancho    W = ',W,'km');
  D = (10^(0.64*Mw-2.78))/100;   % (m)
  
  M0 = 10^(1.5*Mw+9.1);
  %Mw = (log10(M0)-9.1)/1.5
  u = 4.0e10; % (N/m2) coeficiente de rigidez promedio
  D_new = M0/(u*(L*1000)*(W*1000));
  fprintf ('%s %4.2f %s\n' ,'Slip     D = ',D_new,'m');
  fprintf ('%s' ,'Momento M0 = '); disp(M0);
    
  % Valores maximo y minimo de M para std = 0.18
  disp ('Magnitud maxima y minima para std = 0.18')
  M_max = (log10(L)+0.18+2.19)/0.55
  M_min = (log10(L)-0.18+2.19)/0.55
    
  %M0 = u*(L*1000)*(W*1000)*D
  %fprintf ('%s %4.1f %s\n' ,'Momento Mo = ',M0,'N*m');
  %%% M0 = u*L*W*D   %%%  L en m,  W en m, D en m
  S = 10^(0.86*Mw-2.82);   % (km2)

  a = 1.11*0.5642*W;  %a = sqrt(S*e/pi);
  b = 0.90*0.5642*L;  %b = a/e;
  %fprintf ('%s %4.1f %s\n' ,'Semieje A      = ',a,'km');
  %fprintf ('%s %4.1f %s\n' ,'Semieje B      = ',b,'km');
elseif Mw <= 6.5
    disp ('La magnitud debe ser > 6.5')
end

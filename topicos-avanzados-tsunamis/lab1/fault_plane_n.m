% Falla multiple para m x n segmentos
% Calcular las coordenadas de las subfuentes, teniendo 
% en cuenta las ecuaciones de esfericidad de la Tierra
% Aplicable para zonas de latitud alta (lat > 20)
% Copyright: Cesar Jimenez
% Update: 22 Ago 2016
  
  clear all; close all; clc
  xep = -74.654; yep=-16.00; zep =18; % coordenadas epicentro
  xo = -74.50; yo = -16.50; % extremo inferior (o ext sup para Japon)
  L = 160000; % Largo o longitud del pano de falla; (en metros)
  W = 70000; % Ancho del plano de falla; (metros)
  H = 5000;  % Profundidad de la parte superior de la falla
  slip=3.0; % dislocacion de la falla (m)
  Az= 310;     % strike, rumbo, azimuth
  echado = 18; % (dip), buzamiento
  rake = 63;   % rake o slip angle
% *************************************************************
   W1 = W*cos(echado*pi/180); % *****************************************
   beta = atan(W1/L)*180/pi; % ******************************************
   alfa = Az - 270;% ****************************************************
   h = sqrt(L*L+W1*W1);% ************************************************
   a = 0.5*h*sin((alfa+beta)*pi/180)/1000;% *****************************
   b = 0.5*h*cos((alfa+beta)*pi/180)/1000;% *****************************
   if xo < 0 xo = xo + 360; end
   if xep < 0 xep = xep + 360; end
   % Epicentroide o centro de gravedad (xe, ye)
   xe = xo - km2deg(b); %xo = xe+km2deg(b);% ***************************
   ye = yo + km2deg(a); %yo = ye-km2deg(a);% ***************************
disp ('Cargando archivo, espere ...')
load xya;
load grid_a.grd;
I0=find(abs(xa-xo) == min(abs(xa-xo)) ); I0=I0(1);
J0=find(abs(ya-yo) == min(abs(ya-yo)) ); J0=J0(1);

figure; hold on;
contour(xa,ya,grid_a',[4000 4000],'k');
contour(xa,ya,-grid_a'), grid, colorbar, axis equal
contour(xa,ya,grid_a',[0 0],'black');
text (282.384, -11.122,'Huacho');
text (282.852, -12.052,'Callao');
text (283.600, -13.090,'Canete');
axis equal; grid on; zoom on;

% Leer archivo de replicas
disp ('Elegir: (1) Si graficar replicas')
disp ('        (0) No graficar replicas')
s = input ('Eleccion: ');

if s == 1
  fname = 'replicas.txt';
  fid = fopen(fname, 'r');
  feof(fid) = 0;
  lat = [];   lon = [];
  while feof(fid) == 0
    linea2 = fgetl(fid);
    if linea2 == -1
       break
    end
    lat = [lat, str2num(linea2(26:33))];
    lon = [lon, str2num(linea2(34:42))];
  end
  if lon < 0 lon = lon + 360; end
  hold on
  plot (lon,lat,'o'), grid on
  clear lon lat
end
% Fin de leer Replicas

dip=echado*pi/180; 
a1=-(Az-90)*pi/180; a2=-(Az)*pi/180;
r1=L; r2=W*cos(dip);
r1=r1/(60*1853); r2=r2/(60*1853);
sx(1)=0;          sy(1)=0;
sx(2)=r1*cos(a1); sy(2)=r1*sin(a1);
sx(4)=r2*cos(a2); sy(4)=r2*sin(a2);
sx(3)=sx(4)+sx(2);sy(3)=sy(4)+sy(2);
sx(5)=sx(1)      ;sy(5)=sy(1);

sx=sx + xa(I0); sy=sy + ya(J0);
plot(sx,sy,'k','linewidth',2);
px=xa(I0); py=ya(J0); % origen del plano de falla de acuerdo al modelo de tsunamis
plot(px,py,'ro',xo,yo,'bo'), grid on
plot (xep,yep,'*'), zoom on

% Calcula la posicion de cada sub-falla
m = input ('Numero de sub-fallas en el eje "W": m = ');
n = input ('Numero de sub-fallas en el eje "L": n = ');

disp ('Elegir: (1) Tierra plana (lat < 18) ')
disp ('        (2) Tierra esferica (lat > 18) ')
s = input ('Eleccion: ');

if s == 1
   W1 = km2deg(W1/1000);
   Ld = km2deg(L/1000);
   hh = W/m*sin(echado*pi/180);
   P1 = [xo,yo,H];
   ux = W1/m*[cos(Az*pi/180),-sin(Az*pi/180),0]; % vector unit
   uy = Ld/n* [sin(Az*pi/180),cos(Az*pi/180),0]; % vector unit
   uz = [0,0,hh]; %  vector vertical no unit
   P = [];
   for j = 1:n
      for i = 1:m
        PP = P1+(i-1)*ux+(j-1)*uy+(i-1)*uz;
        P = [P; PP];
      end
   end
end

if s == 2
  L0 = L/n;
  W0 = W/m;

  ST=Az; DI=echado;
  lon1 = xo; lat1 = yo;
  phai=lat1; 
  rad=pi/180.0;

  phai=phai*pi/180.0;
  a=6377137.0;
  f=1.0/298.25722;
  e2=2.0*f-f*f;

  % l1 long
  l1=(pi/180.0)*((a*0.001*cos(phai))/sqrt(1.0-e2*(sin(phai)^2)));
  % l4 lat
  l4=(pi/648000.0)*(a*(1.0-e2)/((1.0-e2*(sin(phai)^2))^1.5));
 
  l1=l1*1000.0;
  l4=l4*3600.0;

  cstr = cos(ST*rad);
  sstr = sin(ST*rad);
  cdip = cos(DI*rad);
  sdip = sin(DI*rad);

  lon2 = lon1 + L*sstr/l1;
  lat2 = lat1 + L*cstr/l4;

  lon3 = lon2 + W*cdip*cstr/l1;
  lat3 = lat2 - W*cdip*sstr/l4;
   
  lon4 = lon3 - L*sstr/l1;
  lat4 = lat3 - L*cstr/l4;

  for j = 1:m
    lon(1,j) = (j-1)/m*lon4+(m-j+1)/m*lon1;
    lat(1,j) = (j-1)/m*lat4+(m-j+1)/m*lat1;
    for i = 1:n-1
      lon(i+1,j) = lon(i,j) + L0*sstr/l1;
      lat(i+1,j) = lat(i,j) + L0*cstr/l4;
    end
  end

  % Convierte de matriz a vector
  lon_v = reshape( lon',numel(lon),1);
  lat_v = reshape( lat',numel(lat),1);

  B = [lon_v lat_v];
  plot(lon_v,lat_v,'o'), grid on, axis equal

  lon_r = [lon1 lon2 lon3 lon4 lon1];
  lat_r = [lat1 lat2 lat3 lat4 lat1];
  hold on, plot(lon_r,lat_r)

  % Calculo de profundidad
  hh = W/m*sin(echado*pi/180);
  P = [];
  for j = 1:n
    for i = 1:m
        PP = H+(i-1)*hh;
        P = [P; PP];
    end
  end
  B = [B P]; P = B;
end

hold on
plot(P(:,1),P(:,2),'bo')

disp ('Se crearan los archivos pfallaXX.inp')
L_new = L/n; W_new = W/m;
fprintf ('%s %4.0f %s %6.0f\n','L_new =',L_new,'W_new =',W_new);

for k = 1:m*n    
   k = num2str(k);
      if length(k) < 2, k=['0',k]; end
   fname = ['pfalla',k,'.inp'];
   k = str2num(k);
   fid = fopen (fname, 'w');
   I0=find(abs(xa-P(k,1)) == min(abs(xa-P(k,1))) ); I0=I0(1);
   J0=find(abs(ya-P(k,2)) == min(abs(ya-P(k,2))) ); J0=J0(1);
   fprintf ('%s %2.0f %s %4.0f %4.0f %7.0f\n','Subfalla',k,':',I0,J0,P(k,3));
   fprintf (fid,'%3.0f \n',I0);
   fprintf (fid,'%3.0f \n',J0);
   fprintf (fid,'%2.1f \n',slip); %(k));
   fprintf (fid,'%6.1f \n',L_new);
   fprintf (fid,'%6.1f \n',W_new);
   fprintf (fid,'%3.1f \n',Az);
   fprintf (fid,'%3.1f \n',echado);
   fprintf (fid,'%3.1f \n',rake);
   fprintf (fid,'%3.1f \n',P(k,3));
   fclose all;
end

% Calcular los coordenadas de grilla de las estaciones
A = load('tidal.txt');
if A(:,1) < 0 A(:,1) = A(:,1) + 360; end
fname = 'tidal.dat';
fid = fopen (fname, 'w');
[p q] = size(A);
disp ('')
disp ('Ubicacion de los mareografos')
for k = 1:p    
   I0=find(abs(xa-A(k,1)) == min(abs(xa-A(k,1))) ); I0=I0(1);
   J0=find(abs(ya-A(k,2)) == min(abs(ya-A(k,2))) ); J0=J0(1);
   fprintf ('%4.0f %4.0f %4.0f \n',k,I0,J0);
   fprintf (fid,'%4.0f %4.0f %4.0f \n',k,I0,J0);
   plot (A(k,1),A(k,2),'ro'), grid on, axis equal
end
fclose (fid);
title ('Blue: subfaults,  Red: tidal stations')
zoom on

% Calculo de la grilla de deformacion: IDS IDE JDS JDE
disp ('Creando parametros: IDS IDE JDS JDE') 
IDS = xe - km2deg(0.70*L/1000);
I0=find(abs(xa-IDS) == min(abs(xa-IDS))); IDS=I0(1);
fprintf ('%s %4.0f \n','IDS = ',IDS);
IDE = xe + km2deg(0.70*L/1000);
I0=find(abs(xa-IDE) == min(abs(xa-IDE))); IDE=I0(1);
fprintf ('%s %4.0f \n','IDE = ',IDE);
JDS = ye - km2deg(0.82*L/1000);
J0=find(abs(ya-JDS) == min(abs(ya-JDS))); JDS=J0(1);
fprintf ('%s %4.0f \n','JDS = ',JDS);
JDE = ye + km2deg(0.78*L/1000);
J0=find(abs(ya-JDE) == min(abs(ya-JDE))); JDE=J0(1);
fprintf ('%s %4.0f \n','JDE = ',JDE);

IDS=1; IDE=length(xa); JDS=1; JDE=length(ya);
save xyo.mat xo yo I0 J0 L W Az echado IDS IDE JDS JDE

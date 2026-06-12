% Changes the format of the Gebco or Etopo output to raster format
% Copyleft: Cesar Jimenez 2011
% Update: 22 Ago 2011
% 
clear, clc, close all
disp ('Changes the format of Gebco output file')
disp ('Previously you must to put off the header of this file')
dir *.asc, dir *.txt
fname = input ('File name = ','s');
lat = input ('Latitud inferior   = ');
lon = input ('Longitud izquierda = ');
resol = input ('Resolucion de la grilla (seg) = ');

if lon < 0
    lon = lon + 360;
end
disp ('The file is being readed, wait ...')
A = load (fname);
[JA IA] = size(A)

if IA == 3
    disp ('El formato del archivo debe ser raster o matricial')
    break
end
A = -A';

% Especular reflexion
for j = 1:JA
     k = JA-j+1;
     B(:,j) = A(:,k);
     if mod(j,10)==0
         fprintf ('%6.0f %s %5.0f\n',j,' of',JA);
     end
end
clear A; % ahorrar memoria RAM

% Condicon de frontera para evitar inestabilidad
%disp ('Condicion de frontera para evitar inestabilidad')
% [B] = boundary_a(B); 

% Escribir a un archivo
fid = fopen ('grid_a.grd','w');
disp ('Ahora escribiendo a archivo ...');
for k = 1:IA
     fprintf (fid,'%6.0f',B(k,:));
     fprintf (fid,'\r\n');
end
disp ('The data was saved as grid_a.grd')
disp ('The bathymetry is positive (+)')
fclose all;

% Creando valores de coordenadas
 xa = 0:IA-1; xa = resol*xa/3600; xa = xa + lon; %267; %extremo izquierdo
 ya = 0:JA-1; ya = resol*ya/3600; ya = ya + lat;%(-43); %extremo inferior
 
% Saving geographical coordinates for all grids
 save xya xa ya;
 disp ('The file xya.mat was created');
 figure, hold on
 if JA*IA > 6000000
     contour(xa-360,ya,-B'), grid, colorbar, axis equal
     contour(xa-360,ya,-B',[0 0],'black')
 else
     hold on
     pcolor(xa-360,ya,-B'), shading flat, axis equal, grid
     contour(xa-360,ya,-B',[0 0],'black')
 end
xlim ([min(xa-360) max(xa-360)])
ylim ([min(ya) max(ya)])
grid on
title ('Gebco topography and bathimetry')

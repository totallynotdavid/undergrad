% Extraer batimetria a partir de Etopo5
% Copyleft: Cesar Jimenez 19 Set 2011
% Debe estar en el directorio Etopo5
% Updated: 24 Jul 2015

clc, help extraer_etopo5, clear
disp ('Limites del rectangulo geografico: ')
lat_min = input('Lat_min = ');
lat_max = input('Lat_max = ');
lon_min = input('Lon_min = ');
lon_max = input('Lon_max = ');
if lon_min < 0 lon_min = lon_min + 360; end
if lon_max < 0 lon_max = lon_max + 360; end    

[A maplegend] = etopo(1,[lat_min lat_max], [lon_min lon_max]);
B = -A'; A = B;
[A] = boundary_a(A); % evitar inestabilidad de frontera
save grid_a5.grd A -ascii
disp ('Se grabo el archivo grid_a5.grd')

lon = maplegend(3);
lat = maplegend(2);
resol = 3600/maplegend(1);

[IA JA] = size(A);
xa = 0:IA-1; xa = resol*xa/3600; xa = xa + lon;
ya = JA-1:-1:0; ya = resol*ya/3600; ya = -ya + lat;
save xya.mat xa ya -mat
disp ('Se grabo el archivo xya.mat')

pcolor(xa,ya,-A'); shading flat; colorbar; axis equal, grid on
xlim([min(xa) max(xa)]), ylim([min(ya) max(ya)])
title ('Batimetria Etopo5')
xlabel ('Longitud')
ylabel ('Latitud')


% Convierte un formato matricial (raster) a formato xyz 
% El formato matricial debe proporcionar la matriz A y
% la leyenda "maplegend"
% Update: 13 Ago 2015
clear, close all, clc
dir *.mat
archivo = input ('Nombre de archivo de datos (*.mat):  = ','s');
load (archivo);  
load perfil.txt
lonp = perfil(:,2);
latp = perfil(:,1);

selec = input('Elegir opcion: (1)Todo el rectangulo (2)Solo la topografia: ','s');
if selec == '1'
  [lat,lon,z] = findm (A, maplegend);
  [lat0,lon0,z0] = findm (A==0, maplegend);
  z0 = z0-1;
  xyz = [lon+360 lat -z; lon0+360 lat0 -z0];
  disp ('Topografia (-) invertida para el modelo TIME')
  save salida.txt xyz -ascii
  disp ('Se copio un archivo "salida.txt"')
end

if selec =='2'
%%%%% otro metodo %%%%%
cota = input('Cota >= ');
corr_z = 1;
delta = 1/maplegend(1);
lat0 = maplegend(2);
lon0 = maplegend(3);
[m n] = size(A);
lat = []; lon = []; z = [];
for i = 1:m
    for j = 1:n
        if (A(i,j) >= cota & A(i,j) < 9999) %30)
            lat = [lat; lat0+(i-1)*delta];
            lon = [lon; lon0+(j-1)*delta];
            z   = [z  ; A(i,j)-corr_z];
        end
    end
    fprintf('%5.0f \n',i);
end
%%% correccion de latitud
[lat_s,lon_w] = setltln (A, maplegend,1,1);
lat_n = maplegend(2);
correccion = (lat_n+lat_s);
lat = -lat + correccion + abs(lat_n-lat_s);

N = length(lon);
fid = fopen('salida.txt','w');
for k = 1:N
  if z(k) == 5
    fprintf(fid,'%10.6f %10.6f %7.2f',lon(k),lat(k),z(k));
    fprintf(fid,'\r\n');
  end
end
fclose (fid);

xyz = [lon+360 lat -z];  % topografia negativa para el Modelo Time
save topofino.txt xyz -ascii
plot(lonp,latp,lon, lat,'.'), grid on, zoom on, axis equal
axis ([min(lon)-0.1 max(lon)+0.1 min(lat)-0.1 max(lat)+0.1])
disp ('Se creo el archivo "topofino.txt"')
end

% Plot 3D
figure
plot3 (lon,lat,z,'.'), grid on
title ('Grafico en 3D')


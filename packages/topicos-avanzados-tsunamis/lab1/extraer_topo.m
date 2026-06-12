%Extraer una matriz mas pequeña del topo90 %%%
%Copyleft = Cesar Jimenez cesarsud86@hotmail.com
clear%, clc
dir *.mat
archivo = input ('Nombre de archivo de datos:  = ','s');
load (archivo);  
disp ('Introducir las coordenadas del rectangulo: ')
lat_s = input ('Latitud inf = ');
lat_n = input ('Latitud sup = ');
if (lat_n < lat_s)
    disp ('Los datos estan permutados');
    return;
end
lon_w = input ('Longitud izq = ');
lon_e = input ('Longitud der = ');
if (lon_e < lon_w) 
    disp ('Los datos estan permutados');
    return;
end

[r1 c1] = setpostn (A, maplegend, lat_n, lon_w);
[r2 c2] = setpostn (A, maplegend, lat_s, lon_e);
[m n] = size(A);
if m-r1 == -1   % solo para Aster
    r1 = m-1;   % solo para Aster
end             % solo para Aster
[lati,long] = setltln(A,maplegend,r1,c1);  % Importante
A = A(m-r1:m-r2, c1:c2);
% maplegend = [maplegend(1) lat_n lon_w];
maplegend = [maplegend(1) lati long];
[p q] = size(A);

%lat_n lon_w] = setltln(A, maplegend,p,1); %correccion 
%maplegend = [maplegend(1) lat_n lon_w];
xllcenter = maplegend(3);
yllcenter = maplegend(2);
cellsize = 1/maplegend(1);
i = 1:p; 
vlat(i) = yllcenter-(i-1)*cellsize;
j = 1:q; 
vlon(j) = xllcenter+(j-1)*cellsize;
[lon, lat] = meshgrid(vlon,vlat);
%
% [p q] = size(A);
for i = 1:p
    for j = 1:q
        if A(i,j) == -9999
            A(i,j) = -1;
        end
    end
end
meshc(vlon,vlat,A);
file = input ('Archivo de datos *.mat a crear:  = ','s');
eval(['save ' file,' A maplegend']);
file

%load perfil.txt;  %costa.txt; 
%costa = perfil;
%latc = costa(:,1);
%lonc = costa(:,2);

%[lats,longs] = findm(A<=runup & A>0, maplegend);
%              findm(A>=runup ,maplegend);
%correccion = (lat_n+lat_s);
%lats_corregido = -lats+correccion;
%plot(longs, lats_corregido,'.',lonc,latc,'.') %, lon_w, lat_n,'o')
%grid, zoom, axis equal
%axis ([lon_w lon_e lat_s lat_n])
%title ('Mapa de Inundacion')
%xlabel ('Longitud')
%ylabel ('Latitud')

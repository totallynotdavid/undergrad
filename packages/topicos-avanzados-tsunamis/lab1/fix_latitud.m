archivo = input('Nombre de archivo de datos (*.txt): ', 's');

A = load(archivo);

lon = A(:, 1);
lat = A(:, 2);
elev = A(:, 3);

lat_corrected = -lat;
lon_corrected = lon;

A_corrected = [lon_corrected, lat_corrected, elev];

corrected_file_name = [archivo(1:end-4), '_c.txt'];
save(corrected_file_name, 'A_corrected', '-ascii');

figure;
scatter(lon_corrected, lat_corrected, 15, elev, 'filled');
colorbar;
grid on;
zoom on;
xlabel('Longitud');
ylabel('Latitud');
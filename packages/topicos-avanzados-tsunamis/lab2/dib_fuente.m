% Dibujar fuente sismica del resultado de la inversion
% Copyleft: Cesar Jimenez 01 Jun 2011
% Updated: 28 Dic 2013
clear, close all, clc
disp ('Cargando archivo, espere... ')
load xya.mat
load xyo.mat
dir *.grd
fname = input ('Archivo de deformacion: ','s');
A = load (fname);

[m n] = size(A);
if m > IDE-IDS+1
    A = A(IDS:IDE,JDS:JDE);
end
maximo = ceil(max(max(A)));
%maximo = 4;

load grid_a.grd;
B = grid_a(IDS:IDE,JDS:JDE); clear grid_a
xb = xa(IDS:IDE);
yb = ya(JDS:JDE);
if xb > 180
    xb = xb-360;
end    

figure, hold on
pcolor(xb,yb,A'); shading flat; colorbar; 
caxis([-0.5*maximo maximo]); axis equal, grid on;
%contour(xb,yb,-B');
contour(xb,yb,B',[0 0],'k');
contour(xb,yb,B',[5000 5000],'b');
axis ([xb(1) xb(end) yb(1) yb(end)])
text (84.708,28.147,'* Epicentro'); % epicentro 
%text (142.37, 38.30,'*'); % epicentro Japon 2011
text (-77.10, -12.05,'Callao');
text (-76.21, -13.71,'Pisco');
text (-75.16, -15.36,'Marcona');
text (-74.25, -15.85,'Chala');
text (-78.61, -09.07,'Chimbote');
text (-79.02, -08.11,'Trujillo');
text (-79.84, -06.77,'Chiclayo');
text (-72.71, -16.62,'Camana');
text (-71.34, -29.96,'Coquimbo');
text (-70.64, -33.47,'Santiago');
text (-72.41, -35.33,'Constitucion');
text (-73.106, -36.695,'Concepcion');
%text (139.73, 35.65,'Tokio');
%text (140.87, 38.26,'Sendai');
text (85.33, 27.70,'* Katmandu');
text (86.925,27.988,'* Everest');
title ('Deformacion vertical (m)')
pause(1)

% Filtro Laplaciano
cod = input ('Filtro Laplaciano: y n? ','s');
if cod == 'y'
H = [-4 1 0 1 0 0 0 0 0
     1 -4 1 0 1 0 0 0 0
     0 1 -4 0 0 1 0 0 0
     1 0 0 -4 1 0 1 0 0
     0 1 0 1 -4 1 0 1 0
     0 0 1 0 1 -4 0 0 1
     0 0 0 1 0 0 -4 1 0
     0 0 0 0 1 0 1 -4 1
     0 0 0 0 0 1 0 1 -4];
%H = -1/25*[1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1;];
F = -imfilter(A,H);
F = F*max(max(A))/max(max(F));
%F = smoothn(A,'robust',1);
figure, hold on
pcolor(xb,yb,F'); shading flat; colorbar; 
caxis([-0.5*maximo maximo]); axis equal, grid on;
contour(xb,yb,B',[0 0],'k');
contour(xb,yb,B',[5000 5000],'b');
axis ([xb(1) xb(end) yb(1) yb(end)])
title ('Filtro Laplaciano')
save deform_f.grd F -ascii
disp ('Se grabo la fuente filtrada deform_f.grd')

figure, hold on
mesh (xb,yb,F'), caxis([-0.5*maximo maximo]);
contour(xb,yb,B',[0 0],'k');
contour(xb,yb,B',[5000 5000],'b');
text (-77.10, -12.05,'Callao');
title ('Deformacion vertical filtrada 3D')
end

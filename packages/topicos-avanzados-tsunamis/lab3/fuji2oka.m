% Cambio de formato del archivo de fuente sismica
% de Caltech o de Fujii
% La salida de este programa es la entrada de def_oka.f
% Copyleft: Cesar Jimenez Mar 2011
% Updated: 18 Ene 2016

clear, clc
dir *.txt, dir pfalla*.*
load xya % xya.mat: archivo con los vectores longitud xa y latitud ya
fname = input ('Archivo de entrada: ','s');
A = load (fname);
[Np n] = size(A);
disp ('Elegir entre 3 formatos de entrada: ')
form = input('Fuji (1), Caltech (2), Pulido (3), Kan (4), Fault (5): ');

if form == 1   % lee formato Fuji
  L = A(:,1);
  W = A(:,2);
  H = A(:,3);
  strike = A(:,4);
  dip = A(:,5);
  rake = A(:,6);
  slip = A(:,7);
  yo = A(:,8); % lat
  xo = A(:,9); % lon
end

if form == 2   % Lee formato Caltech
  L0 = input ('Dx along strike (km) = ');  
  W0 = input ('Dy along dip (km) = ');    
  L = L0*ones(Np,1)*1000; 
  W = W0*ones(Np,1)*1000; 
  xo = A(:,2); % lon
  yo = A(:,1); % lat
  H = A(:,3)*1000; % en m
  slip = A(:,4)/100; % metros
  rake = A(:,5);
  strike = A(:,6);
  dip = A(:,7);
end

if form == 3   % Lee formato Pulido
  L = 20*ones(Np,1)*1e3; % m
  W = 20*ones(Np,1)*1e3; % m
  xo = A(:,1); % lon
  yo = A(:,2); % lat
  slip = A(:,3)/100; % metros
  rake = A(:,4);
  strike = A(:,5);
  dip = A(:,6);
  H = A(:,7)*1e3; % m
end

if form == 4   % lee formato Kanamori
  L = 1000*A(:,1);
  W = 1000*A(:,2);
  H = 1000*A(:,3);
  strike = A(:,4);
  dip = A(:,5);
  rake = A(:,6);
  slip = A(:,7);
  yo = A(:,8); % lat
  xo = A(:,9); % lon
end

if form == 5   % lee formato fault_plane
  xo = A(:,1);
  yo = A(:,2);
  slip = A(:,3);
  L = A(:,4);
  W = A(:,5);
  strike = A(:,6);
  dip = A(:,7);
  rake = A(:,8);
  H = A(:,9);
end

% Cambiar longitud a formato 0-360
if xo < 0
      xo = xo + 360;
end
  
% Cambiar formato de coord. geograficas a coord. de grilla
I = [];   J = [];
for i = 1:Np
    I0=find(abs(xa-xo(i)) == min(abs(xa-xo(i))) ); %I=I0(i)
    J0=find(abs(ya-yo(i)) == min(abs(ya-yo(i))) ); %J=J0(i)
    I = [I; I0(1)];
    J = [J; J0(1)];
end

% Escribir la salida al archivo pfalla.inp
fid = fopen ('pfalla_inv.inp','w');
for k = 1:Np
    fprintf(fid,'%5.0f %5.0f %6.2f %8.1f %8.1f %6.1f %6.1f %6.1f %8.1f'...
    ,I(k),J(k),slip(k),L(k),W(k),strike(k),dip(k),rake(k),H(k));
    fprintf(fid,'\r\n');
end

%fprintf(fid,'%5.0f ',I');      fprintf(fid,'\r\n');
%fprintf(fid,'%5.0f ',J');      fprintf(fid,'\r\n');
%fprintf(fid,'%5.2f ',slip');   fprintf(fid,'\r\n');
%%fprintf(fid,'%5.0f ',L*1000'); fprintf(fid,'\r\n');
%fprintf(fid,'%5.0f ',L'); fprintf(fid,'\r\n');
%%fprintf(fid,'%5.0f ',W*1000'); fprintf(fid,'\r\n');
%fprintf(fid,'%5.0f ',W'); fprintf(fid,'\r\n');
%fprintf(fid,'%5.1f ',strike'); fprintf(fid,'\r\n');
%fprintf(fid,'%5.1f ',dip');    fprintf(fid,'\r\n');
%fprintf(fid,'%5.1f ',rake');   fprintf(fid,'\r\n');
%%fprintf(fid,'%5.0f ',H*1000'); fprintf(fid,'\r\n');
%fprintf(fid,'%5.0f ',H'); fprintf(fid,'\r\n');

fclose all;
type pfalla_inv.inp
disp ('Se creo el archivo pfalla_inv.inp')
disp ('Cambiar el valor de NP en el programa def_oka.f')
disp (Np)

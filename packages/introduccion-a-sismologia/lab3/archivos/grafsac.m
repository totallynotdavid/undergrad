function varargout = grafsac(varargin)
% grafsac Application M-file for grafsac.fig
%    FIG = grafsac launch grafsac GUI.
%    grafsac('callback_name', ...) invoke the named callback.
% Last Modified by GUIDE v2.5 15-Nov-2019 15:39:02
% Copyleft Cesar Jimenez 2011 cjimenezt@unmsm.edu.pe
% Updated: 12 May 2018

if nargin == 0  % LAUNCH GUI
    clear, clc
	fig = openfig(mfilename,'reuse');
	% Generate a structure of handles to pass to callbacks, and store it. 
	handles = guihandles(fig);
    ai = 1;              % Comparte informacion    
    handles.ai = ai;     % con los callbacks
    contador = 1;
    handles.contador = contador;
    system ('dir *.SAC /b/o > lista');
    polaridad_P = [' EP '];
    handles.polP = polaridad_P;
    polaridad_S = [' ES'];
    handles.polS = polaridad_S;
    fmin = 1.1; 
    handles.fmin = fmin;
    eje = 1;
    handles.eje = eje;
    guidata(fig, handles);
	if nargout > 0
		varargout{1} = fig;
	end
elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK
	try
		if (nargout)
			[varargout{1:nargout}] = feval(varargin{:}); % FEVAL switchyard
		else
			feval(varargin{:}); % FEVAL switchyard
		end
	catch
		disp(lasterr);
	end
end
%| ABOUT CALLBACKS:
%| GUIDE automatically appends subfunction prototypes to this file, and 
%| sets objects' callback properties to call them through the FEVAL 
%| switchyard above. This comment describes that mechanism.
%|
%| Each callback subfunction declaration has the following form:
%| <SUBFUNCTION_NAME>(H, EVENTDATA, HANDLES, VARARGIN)
%|
%| The subfunction name is composed using the object's Tag and the 
%| callback type separated by '_', e.g. 'slider2_Callback',
%| 'figure1_CloseRequestFcn', 'axis1_ButtondownFcn'.
%|
%| H is the callback object's handle (obtained using GCBO).
%|
%| EVENTDATA is empty, but reserved for future use.
%|
%| HANDLES is a structure containing handles of components in GUI using
%| tags as fieldnames, e.g. handles.figure1, handles.slider2. This
%| structure is created at GUI startup using GUIHANDLES and stored in
%| the figure's application data using GUIDATA. A copy of the structure
%| is passed to each callback.  You can store additional information in
%| this structure at GUI startup, and you can change the structure
%| during callbacks.  Call guidata(h, handles) after changing your
%| copy to replace the stored original so that subsequent callbacks see
%| the updates. Type "help guihandles" and "help guidata" for more
%| information.
%|
%| VARARGIN contains any extra arguments you have passed to the
%| callback. Specify the extra arguments by editing the callback
%| property in the inspector. By default, GUIDE sets the property to:
%| <MFILENAME>('<SUBFUNCTION_NAME>', gcbo, [], guidata(gcbo))
%| Add any extra arguments after the last argument, before the final
%| closing parenthesis.

% --------------------------------------------------------------------
function varargout = Sismo_Callback(h, eventdata, handles, varargin)
fname = get(handles.edit2,'string');
n = 1;
if fname(1:5) == ['lista']   % Leer lista de eventos sismicos
    fid = fopen(fname,'r');
    if fid == -1  %%% | fid >= 3)
      linea = ' No existe la lista';
      t = 0; x = 0; plot (t,x)
      text (-0.18,0.1,linea)
      return 
    end
    Fl = fread(fid);
    list = char(Fl');
    
    fid = fopen(fname, 'r');
    i = 0;       A = [];
    feof(fid) = 0;
    while feof(fid) == 0
      i = i+1;
      linea = fgetl(fid);
      N = length(linea);
      
      if N == 45  linea = [linea,''];   end
      if N == 44  linea = [linea,' '];   end
      if N == 43  linea = [linea,'  '];   end
      if N == 42  linea = [linea,'   '];   end
      if N == 41  linea = [linea,'    '];   end
      if N == 40  linea = [linea,'     '];   end
      if N == 39  linea = [linea,'      '];   end
      if N == 38  linea = [linea,'       '];   end
      if N == 37  linea = [linea,'        '];   end
      if N == 36  linea = [linea,'         '];   end
      if N == 35  linea = [linea,'          '];   end
      if N == 34  linea = [linea,'           '];   end
      if N == 33  linea = [linea,'            '];   end
      if N == 32  linea = [linea,'             '];   end
      if N == 31  linea = [linea,'              '];   end
      if N == 30  linea = [linea,'               '];   end
      if N == 29  linea = [linea,'                '];   end
      if N == 28  linea = [linea,'                 '];   end
      if N == 27  linea = [linea,'                  '];   end
      if N == 26  linea = [linea,'                   '];   end
      if N == 25  linea = [linea,'                    '];   end
      if N == 24  linea = [linea,'                     '];   end
      if N == 23  linea = [linea,'                      '];   end
      if N == 22  linea = [linea,'                       '];   end
      if N == 21  linea = [linea,'                        '];   end
      if N == 20  linea = [linea,'                         '];   end
      if N == 19  linea = [linea,'                          '];   end
      if N == 18  linea = [linea,'                           '];   end
      if N == 17  linea = [linea,'                            '];   end
      if N == 16  linea = [linea,'                             '];   end
      if N == 15  linea = [linea,'                              '];   end
      if N == 14  linea = [linea,'                               '];   end
      if N == 13  linea = [linea,'                                '];   end
      if N == 12  linea = [linea,'                                 '];   end
      if N == 11  linea = [linea,'                                  '];   end
      if N == 10  linea = [linea,'                                   '];   end
      if N == 09  linea = [linea,'                                    '];   end
      if N == 08  linea = [linea,'                                     '];   end
      if N == 07  linea = [linea,'                                      '];   end
      if N == 06  linea = [linea,'                                       '];   end
      if N == 05  linea = [linea,'                                        '];   end
   
      if linea == -1
        break
      end
      A = cat(1,A,linea);
    end
    [m n] = size(A);
        
    ai = handles.ai;
    contador = handles.contador ;
    fname = A(contador,:);
    % Correccion para Linux
    nulo = find(fname==' ');
    maximo = min(nulo);
    fname = A(contador,1:maximo-1);
    % Fin correccion Linux
    handles.ai = ai;
    handles.m = m;
    set (handles.edit2, 'visible','off')
    set (handles.text3, 'visible','off')
    set (handles.Sismo, 'visible','off')
    set (handles.elmismo,'visible','on')
    set (handles.next,   'visible','on')
    set (handles.tres,   'visible','on')
    if contador > 1
       set (handles.anterior,'visible','on')
    end
else
    list = [' '];
end
fid = fopen(fname, 'r');
set (handles.elmismo,'visible','on')
set (handles.tres,   'visible','on')
if fid == -1
    linea = 'No existe el archivo';
    t = 0;      x = 0;
    plot (t,x)
    text (-0.18,0.1,linea)
    return  
end
ss = dir(fname);
filesize = ss.bytes;
if filesize > 20000000  % 20 Mb
    linea = 'Archivo demasiado grande';
    t = 0; x = 0; plot (t,x)
    text (-0.18,0.1,linea)
    return
end
%                 Leer formato SAC         %%%%
B = rsac(fname);
%t = B(:,1);
y = B(:,2);
header = B(:,3);
N = header(80);
lat = header(32);
lon = header(33);
elev= header(34);
dist= header(51); % km
Az  = header(52); % azimut
%estacion, the station name, is a string stored in the 110th header variable
%kstnm = KSTNM (First 3 letters of the station name)
fseek(fid,4*110,-1);
estacion = char(fread(fid,4)');
fseek(fid,158*4,-1);

T = header(1); Fs = 1/T;
ss = header(75)+header(76)/1000;
k = 1:N;
t = T.*(k-1)+ss; clear k;
y = y - mean(y);

E = 0.5*sum(y.*y); % energia de la sennal

anno = header(71);
suma = header(72);    %%% dia juliano
hora = num2str(header(73));
if length(hora) == 1 
    hora = ['0',hora]; end
minu= num2str(header(74));
if length(minu) == 1 
    minu = ['0',minu]; end
seg = num2str(ss);
if length(seg) == 1 
    seg = ['0',seg]; end
if length(seg) == 2
    seg = [seg,'.00']; end

%%%%%
ad = 0; dd = 0; mes = '00'; % por defecto
if (mod(anno,4)==0 & suma>59) ad = 1; end;
  if ((suma>=01) & (suma<=31))   
      mes = '01';  dd=suma;  end;
  if ((suma>=32) & (suma<=59+ad))   
      mes = '02';  dd=suma-31; end;
  if ((suma>=60+ad) & (suma<=90+ad))   
      mes = '03'; dd=suma-59-ad; end;
  if ((suma>=91+ad) & (suma<=120+ad))  
      mes = '04'; dd=suma-90-ad; end;
  if ((suma>=121+ad) & (suma<=151+ad)) 
      mes = '05'; dd=suma-120-ad; end;
  if ((suma>=152+ad) & (suma<=181+ad)) 
      mes = '06'; dd=suma-151-ad; end;
  if ((suma>=182+ad) & (suma<=212+ad)) 
      mes = '07'; dd=suma-181-ad; end;
  if ((suma>=213+ad) & (suma<=243+ad)) 
      mes = '08'; dd=suma-212-ad; end;
  if ((suma>=244+ad) & (suma<=273+ad)) 
      mes = '09'; dd=suma-243-ad; end;
  if ((suma>=274+ad) & (suma<=304+ad)) 
      mes = '10'; dd=suma-273-ad; end;
  if ((suma>=305+ad) & (suma<=334+ad)) 
      mes = '11'; dd=suma-304-ad; end;
  if ((suma>=335+ad) & (suma<=365+ad)) 
      mes = '12'; dd=suma-334-ad; end;
dia = num2str(dd);
if length(dia) == 1 
    dia = ['0',dia]; end

%%%%%
tiempo = [num2str(anno),' ',mes,' ',dia,' ',hora,':',minu,':',seg];

eje = handles.eje;
if eje ~= 1
    delete(gca), delete(gca), delete(gca)
    axes(handles.ejes)
    eje = 1;
end
maximo = abs(max(y));
if maximo == 0
    maximo = 0.1;
end
plot (t,y), grid on, zoom on %xon
axis ([ss t(N) -2*maximo 2*maximo]);
title([fname,'       ',tiempo,'       Fs = ',num2str(Fs),'       N = ',num2str(N)])
ym = y;
filtr = 0;
handles.fname = fname;
handles.sennal = y; % Comparte informacion con otro callback
handles.ym = ym;
handles.tdata = t;
handles.fecha = tiempo;
handles.T = T;
handles.ss = ss;
handles.estacion = estacion;
handles.lat = lat;
handles.lon = lon;
handles.elev = elev;
handles.Az = Az;
handles.list = list;
handles.filtr = filtr;
fmin = 1.1;   % umbral del filtro
handles.fmin = fmin;
handles.eje = eje;
handles.N = N;
handles.n = n;
contador_trf = 0;
handles.contador_trf = contador_trf;
guidata(h, handles)
set (handles.trf,     'visible','on')
set (handles.filtro,  'visible','on')
set (handles.filtro2, 'visible','on')
set (handles.fcorte1, 'visible','on')
set (handles.fcorte2, 'visible','on')
set (handles.calculo, 'visible','on')
set (handles.grabar_p,'visible','on')
fclose(fid);

% --------------------------------------------------------------------
function varargout = anterior_Callback(h, eventdata, handles, varargin)
clear t y ym;
list = handles.list;
ai = handles.ai;
contador = handles.contador ;
n = handles.n;
ai = ai-(n+2);
contador = contador - 1;
if ai < length(list)
   %handles.ai = ai;
   handles.contador = contador;
   guidata(h, handles);
   Sismo_Callback(h, eventdata, handles, varargin)
end
%if ai < 14
if contador < 2
    set (handles.anterior, 'visible','off')
end

% --------------------------------------------------------------------
function varargout = elmismo_Callback(h, eventdata, handles, varargin)
fname = handles.fname;
tiempo = handles.fecha;
T = handles.T; Fs = 1/T;
eje = handles.eje;
ss = handles.ss;
contador_trf = handles.contador_trf;

if contador_trf == 1 
   cla
   delete(gca), delete(gca)
end
axes(handles.ejes)
t = handles.tdata;
N = length(t);
ym = handles.ym;
plot (t,ym), grid, zoom on
axis ([ss t(N) -1.2*abs(max(ym)) 1.2*abs(max(ym))]);
title([fname,'       ',tiempo(1:16),'       Fs = ',num2str(Fs),'       N = ',num2str(N)]);
y = ym;
filtr = 0;
fmin = 0.10;
eje = 1;
handles.eje = eje;
handles.fmin = fmin;
handles.filtr = filtr;
handles.sennal = y;
contador_trf = 0;
handles.contador_trf = contador_trf;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = next_Callback(h, eventdata, handles, varargin)
clear t y ym;
fmin = 0.10;
handles.fmin = fmin;
list = handles.list;
ai = handles.ai;
contador = handles.contador ;
m = handles.m;
eje = handles.eje;
contador_trf = handles.contador_trf;

if contador_trf == 1 
   cla
   delete(gca), delete(gca)
end

n = handles.n;
ai = ai+(n+2);
contador = contador + 1;
%if ai < length(list)
if contador <= m
   %handles.ai = ai;
   handles.contador = contador;
   guidata(h, handles);
   cla
   title (' ')
   Sismo_Callback(h, eventdata, handles, varargin)
else
   t = 0;      x = 0;
   plot (t,x), grid on
   title ('** FIN DE LISTA ** FIN DE LISTA ** FIN DE LISTA **')
   salir_Callback(h, eventdata, handles, varargin)   
end

% --------------------------------------------------------------------
function varargout = trf_Callback(h, eventdata, handles, varargin)
t = handles.tdata;
y = handles.sennal;
T = handles.T; Fs = 1/T;
contador_trf = handles.contador_trf;

if contador_trf == 0 
set (handles.ejes,  'visible','off')
cla
end

if contador_trf == 1 
   cla
   delete(gca), delete(gca)
end

N = length(y);
if mod(N,2) == 1
    N = N-1;
end
Y = fft(y,N);
Pyy = Y.*conj(Y) / N;    % valor absoluto
f = (1/T)*(0:N/2)/ N;
axes('position',[.07  .55  .67  .28]);
plot (f,Pyy(1:N/2+1)), grid on, zoom xon
xlim ([0 max(f)/2])
title('Espectro de frecuencias de la senal')
xlabel('Frecuencia (Hz)')
axes('position',[.07  .15  .67  .28]);
loglog(f,Pyy(1:N/2+1)), grid on, zoom xon
%xlim ([log10(min(f)) log10(max(f))])
title('Espectro de frecuencias logaritmico')
xlabel('Frecuencia (Hz)')
contador_trf = 1;
handles.contador_trf = contador_trf;
guidata(h, handles)

% --------------------------------------------------------------------
% funcion automatico
function varargout = ondaP_Callback(h, eventdata, handles, varargin)
t = handles.tdata;
y = handles.sennal;
polaridad_P = handles.polP;
polaridad_S = handles.polS;
filtr = handles.filtr;
T = handles.T; Fs = round(1/T);
estacion = handles.estacion;
tiempo = handles.fecha;
anno = tiempo(1:4);
mes  = tiempo(6:7);
dia  = tiempo(9:10);
hora = tiempo(12:13);
minu  = tiempo(15:16);
%seg  = tiempo(18:22);
seg  = tiempo(18:end);
ss   = str2num(seg);
N    = length(y);  
[A,I]= max(abs(y));

y1 = hilbert(y);
y2  = sqrt(y.*y+y1.*conj(y1));  %Envolvente positiva
y2 = y2/max(y2);

y3 = abs(y);  % Valor absoluto
y3 = y3/max(y3);   

% Calcular la fase P
comp = mean(y3(1:50*Fs)); %(1:60)
if comp < 0.0001
    comp = 10*comp;  
end
if comp > 0.050  % 0.020
    comp = 0.5*comp;  % para niveles muy ruidosos
end

if filtr == 1
  for m = 10:I
    if y3(m) > 7*comp %   %0.135 
        m;
        break
    end  
  end
  for k = 1:m
   if y3(m-k)-y3(m-k-1)<=0&y3(m-k)<0.035
    Ip = m-k;
    tp = (Ip+1)*T+ss;  % Ip*T+ss
    break
   end  
  end
end

if filtr == 0
  for m = 10:I
    if y3(m) > 5.5*comp    %0.135 
        m;
        break
    end  
  end
  for k = 1:m
   if y3(m-k)-y3(m-k-1)<=0 & y3(m-k)<0.015
    Ip = m-k; 
    tp = (Ip)*T+ss;
    break
   end  
  end
end
xp = [tp,tp];
handles.tiempop = tp;

% Calcular SNR
Ip = floor((tp-ss)/T);
Np = floor(10/T);
despues = max(y(Ip:Ip+Np));
antes = mean(abs(y(Ip-Np:Ip)));
SNR = num2str(round(despues/antes));
% Fin SNR

% Polaridad de la fase P
for j = 1:3;
    dy(j) = 100*(y(Ip+j)-y(Ip+j-1))/A;  % derivada normalizada a 100%
end
pendiente = mean(dy);
if pendiente > 2.0 
    disp('IPC')
end
if pendiente < -2.0 
    disp('IPD')
end
if abs(pendiente) < 2.0
%    disp('EP')
end

% Calcular la fase S
if I-Ip < 5*Fs %100
  Ix = Ip+6*Fs; %120;
else
  Ix = round((Ip+I)/2);
end

for n = Ix:N  
    if y2(n) > 0.45 
        n;
        break
    end  
end
for k = Ix:n+Ix
  if (y2(n+Ix-k)-y2(n+Ix-k-1)<0)&y2(n+Ix-k)<0.08
    Is = n+Ix-k;
    ts = (Is)*T+ss;
    break
  end  
end
xs = [ts,ts];
handles.tiempos = ts;

% Calcular el periodo usando TRF
Ns = round(5/T);    %128;    % Ns = numero de puntos TRF
k = 1:Ns;
ys= y(Is+k);
Y = fft(ys,Ns);
Pyy = Y.*conj(Y) / Ns;
f = (1/T)*(0:Ns/2)/ Ns;
delta_f = f(2)-f(1);
[amp_espectro,J] = max (abs(Pyy));
fs = delta_f*(J-1);
per = 1/(fs+eps);
if per > 10
   per = 9.99;
end
handles.per = per;
guidata(h, handles)

% Calcular la amplitud
if A < 1 A = (max(abs(y(Is:Is+round(6/T))))); end
if A>= 1 A = round(max(abs(y(Is:Is+round(6/T))))); end
handles.A = A;

% Calcular la duracion
prom_f = sum(abs(y(N-20:N)))/21;
if filtr == 1
    prom_f = 2*prom_f;  % 3
end
for k = N:-1:I
  if y(k) > 4*prom_f     % 4
    k;
    td = k*T+ss;
    break
  else
    td = N*T+ss;  
  end  
end
dur = round(td-tp);
handles.tiempod = td;
guidata(h, handles)

% Escribir resultados en la leyenda
min1 = str2num(minu);
if tp > 60.0
    min1 = min1 + floor(tp./60);
    tp = mod(tp,60);
end
t1 = num2str(tp);
if tp < 10
    t1 = ['0',t1];
end
if length(t1)==4
    t1 = [t1,'0']; 
end
if length(t1) == 2
    t1 = [t1,'.00']; 
end
if length(t1) > 5
    t1 = t1(1:5);
end
min2 = str2num(minu);
if ts > 60.0
    min2 = min2 + floor(ts./60);
    ts = mod(ts,60);
end
t2 = num2str(ts);
if ts < 10
    t2 = ['0',t2];
end
if length(t2)==4, 
    t2 = [t2,'0'];
end
if length(t2) == 2, 
    t2 = [t2,'.00'];
end
if length(t2) > 5
    t2 = t2(1:5);
end
Amp = num2str(A);
j = length(Amp);
if j == 3
    Amp = ['     ',Amp];
end
if j == 4
    Amp = ['    ',Amp];
end
if j == 5
    Amp = ['   ',Amp];
end
if j == 6
    Amp = ['  ',Amp];
end
if j == 7
    Amp = [' ',Amp];
end

P = num2str(per);
if length(P) == 1
    P = [P,'.00'];
end
if length(P) == 3
    P = [P,'0'];
end

D = num2str(dur);
if length(D) == 2
    D = [' ',D];
end

if min1 > 59 
    min1 = min1 - 60;
    hora = hora + 1;
end
if min2 > 59 
    min2 = min2 - 60;
%   hora = hora + 1;
end

if min1 < 10
    min1 = ['0',num2str(min1)];
else min1 = num2str(min1);
end
if min2 < 10
    min2 = ['0',num2str(min2)];
else min2 = num2str(min2);
end

linea=[estacion,'  ',dia,mes,anno,' EP ',hora,min1,t1,' ES',hora,min2,t2,' ',P(1:4),' ',Amp,'  ',D];
xd = [td,td];
yd = [-.65*max(y), .65*max(y)];
plot (t,y, xp,yd,'red', xs,yd,'red',xd,yd,'red'), grid on
axis ([ss t(N) -1.2*abs(max(y)) 1.2*abs(max(y))]);
title(linea) %['Hora GMT: ',tiempo(1:16)])
xlabel(['SNR = ',SNR]);
text (xp(1),.7*max(y),'P','color','red')
text (xs(1),.7*max(y),'S','color','red')
zoom on
set (handles.grabar, 'visible','on')

% --------------------------------------------------------------------
function varargout = filtro_Callback(h, eventdata, handles, varargin)
t = handles.tdata;
y = handles.sennal;
fname = handles.fname;
T = handles.T;
ss = handles.ss;
N = handles.N;
tiempo = handles.fecha;
[A,I]= max(abs(y));
Fs = 1/T; % Frecuencia de muestreo     
fcorte = get(handles.fcorte1,'string');
fcorte = str2num(fcorte);

% [b, a] = butter (5,fcorte/(Fs/2));   % Filtro pasabajo
% El 2do argumento en butter indica la frecuencia de corte
% normalizado a la mitad de la frecuencia de muestreo.
%fmin = handles.fmin;
%fcorte = fmin;     
%if fmin < 10 
   [b, a] = butter (5,fcorte/(Fs/2),'high'); %pasa-alta
   yf = filtfilt (b,a,y);
   [Af,I] = max(abs(yf));
   y = yf;,%  y = (A/Af)*yf;  % Normalizar la senal filtrada
 %  yf = 1*yf/Af;    % Normalizado a Uno
   P = log10(0.5*sum(y.*y)/N);
   prom = mean(abs(yf));
   prom_inicio = mean(abs(yf(1:20)));
   sta_lta = prom_inicio/prom;
   plot (t,y,'red'), grid on, zoom on
   axis ([ss t(N) -1.2*abs(max(y)) 1.2*abs(max(y))]);
   title([fname,'       ',tiempo(1:16),'       Fs = ',num2str(Fs),'       N = ',num2str(N)]);
   xlabel(['log(Potencia) = ',num2str(P),'       Amplitud = ',num2str(floor(Af)),'     ',num2str(sta_lta)]);
%end
filtr = 1;
%fmin = fmin + 1;
handles.sennal = y;
handles.filtr = filtr;
%handles.fmin = fmin;
guidata(h, handles)

% ------------------------------------------------------------
function filtro2_Callback(h, eventdata, handles)
% hObject    handle to filtro2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
t = handles.tdata;
y = handles.sennal;
fname = handles.fname;
T = handles.T;
ss = handles.ss;
N = handles.N;
tiempo = handles.fecha;
[A,I]= max(abs(y));
Fs = 1/T; % Frecuencia de muestreo  
fcorte = get(handles.fcorte2,'string');
fcorte = str2num(fcorte);
% fcorte = 6.0;
[b, a] = butter (5,fcorte/(Fs/2));   % Filtro pasabajo
% El 2do argumento en butter indica la frecuencia de corte
% normalizado a la mitad de la frecuencia de muestreo.
yf = filtfilt (b,a,y);
   [Af,I] = max(abs(yf));
   y = (A/Af)*yf;      % Normalizar la se�al filtrada
   plot (t,y,'red'), grid on, zoom on
   axis ([ss t(N) -1.2*abs(max(y)) 1.2*abs(max(y))]);
   title([fname,'       ',tiempo(1:16),'       Fs = ',num2str(Fs),'       N = ',num2str(N)])
handles.sennal = y;
% handles.filtr = filtr;
% handles.fmin = fmin;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = limpiar_Callback(h, eventdata, handles, varargin)
cla
clc
%t = 0;      x = 0;
%plot (t,x), grid on
title (' ')
set (handles.Sismo, 'visible','on')
set (handles.text3, 'visible','on')
set (handles.edit2, 'visible','on')
clear

%--------------------------------------------------------------------
function varargout = edit2_Callback(h, eventdata, handles, varargin)
fname = get(handles.edit2,'string');

% --------------------------------------------------------------------
function varargout = edit3_Callback(h, eventdata, handles, varargin)
fprintf('%s%7.2f\n', '     Duracion = ',ss);

% --------------------------------------------------------------------
function varargout = leer_P_Callback(h, eventdata, handles, varargin)
t = handles.tdata;
y = handles.sennal;
T = handles.T;
N = length(y);
tiempo = handles.fecha;

ss  = str2num(tiempo(18:end));
[tp,y1] = ginput(1);
xp = [tp,tp];
tp = round(100*tp)/100;
yp = [-.65*max(y), .65*max(y)];
% Calcular SNR
Ip = floor((tp-ss)/T);
Np = floor(10/T);
despues = max(y(Ip:Ip+Np));
antes = mean(abs(y(Ip-Np:Ip)));
SNR = num2str(round(despues/antes));
% Fin SNR
plot (t,y, xp,yp,'red'), grid on
axis ([ss t(N) -1.2*abs(max(y)) 1.2*abs(max(y))]);
title(['Hora GMT: ',tiempo(1:16)])
xlabel(['SNR = ',SNR]);
text (tp,.7*max(y),'P','color','red')
zoom on
handles.tiempop = tp;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = leer_S_Callback(h, eventdata, handles, varargin)
t = handles.tdata;
y = handles.sennal;
N = length(y);
T = handles.T;
tp = handles.tiempop;
tiempo = handles.fecha;
ss  = str2num(tiempo(18:end));
[ts,y2] = ginput(1);
ts = round(100*ts)/100;
xp = [tp,tp];
xs = [ts,ts];
ys = [-.65*max(y), .65*max(y)];
plot (t,y, xp,ys,'red', xs,ys,'red'), grid on
axis ([ss t(N) -1.2*abs(max(y)) 1.2*abs(max(y))]);
title(['Hora GMT: ',tiempo(1:16)])
text (tp,.7*max(y),'P','color','red')
text (ts,.7*max(y),'S','color','red')
zoom on
% Calcular el periodo
Is = floor((ts-ss)/T);
Ns = round(5/T);  %128;    % Ns = numero de puntos TRF

k = 1:Ns;
y1= y(Is+k);
Y = fft(y1,Ns);
Pyy = Y.*conj(Y) / Ns;
f = (1/T)*(0:Ns/2)/ Ns;
delta_f = f(2)-f(1);
[amp_espectro,J] = max (abs(Pyy));
fs = delta_f*(J-1);
per = 1/(fs+eps);
if per > 10
   per = 9.99;
end
% Calcular la amplitud
A = (max(abs(y(Is:Is+round(6/T)))));
if A < 1 A = (max(abs(y(Is:Is+round(6/T))))); end
if A>= 1 A = round(max(abs(y(Is:Is+round(6/T))))); end
handles.A = A;
handles.tiempos = ts;
handles.per = per;
guidata(h, handles)

Ip = round((tp-ss)/T);
% Polaridad de la fase P
for j = 1:3;
    dy(j) = 100*(y(Ip+j)-y(Ip+j-1))/A;  % derivada normalizada a 100%
end
pendiente = mean(dy);
if pendiente > 2.0 
    disp('IPC')
end
if pendiente < -2.0 
    disp('IPD')
end
if abs(pendiente) < 2.0
    disp('EP')
end

% --------------------------------------------------------------------
function varargout = leer_duracion_Callback(h, eventdata, handles, varargin)
t = handles.tdata;
y = handles.sennal;
N = length(y);
tp = handles.tiempop;
ts = handles.tiempos;
tiempo = handles.fecha;
ss = handles.ss;
[td,y3] = ginput(1);
td = round(100*td)/100;
xp = [tp,tp];
xs = [ts,ts];
xd = [td,td];
yd = [-.65*max(y), .65*max(y)];
plot (t,y, xp,yd,'red', xs,yd,'red',xd,yd,'red'), grid on
axis ([ss t(N) -1.2*abs(max(y)) 1.2*abs(max(y))]);
title(['Hora GMT: ',tiempo(1:16)])
xlabel(['Duracion = ',num2str(td-xp(1))]);
text (tp,.7*max(y),'P','color','red')
text (ts,.7*max(y),'S','color','red')
zoom on
handles.tiempod = td;
guidata(h, handles)
set (handles.grabar, 'visible','on')

% --------------------------------------------------------------------
function varargout = ep_Callback(h, eventdata, handles, varargin)
set (handles.ipd,'value',0)
set (handles.ipc,'value',0)
opcion = get(handles.ep,'value');
if opcion == 1
    polaridad_P = [' EP '];
end
handles.polP = polaridad_P;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = ipc_Callback(h, eventdata, handles, varargin)
set (handles.ep, 'value',0)
set (handles.ipd,'value',0)
opcion = get(handles.ipc,'value');
if opcion == 1
    polaridad_P = [' IPC'];
end
handles.polP = polaridad_P;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = ipd_Callback(h, eventdata, handles, varargin)
set (handles.ep, 'value',0)
set (handles.ipc,'value',0)
opcion = get(handles.ipd,'value');
if opcion == 1
    polaridad_P = [' IPD'];
end
handles.polP = polaridad_P;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = es_Callback(h, eventdata, handles, varargin)
set (handles.is,'value',0)
opcion1 = get(handles.es,'value');
if opcion1 == 1
    polaridad_S = [' ES'];
end
handles.polS = polaridad_S;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = is_Callback(h, eventdata, handles, varargin)
set (handles.es,'value',0)
opcion1 = get(handles.is,'value');
if opcion1 == 1
    polaridad_S = [' IS'];
end
handles.polS = polaridad_S;
guidata(h, handles)

% --------------------------------------------------------------------
function varargout = grabar_Callback(h, eventdata, handles, varargin)
estacion = handles.estacion;
tp = handles.tiempop;
ts = handles.tiempos;
td = handles.tiempod;
y  = handles.sennal;
polaridad_P = handles.polP;
polaridad_S = handles.polS;
per = handles.per;
A = handles.A;
tiempo = handles.fecha;
anno = tiempo(1:4);
mes  = tiempo(6:7);
dia  = tiempo(9:10);
hora = tiempo(12:13);
minu  = tiempo(15:16);
seg  = tiempo(18:end);
ss   = str2num(seg);
dur = round(td-tp);

% Escribir resultados a un archivo de texto
fname2 = 'salida.txt';
fid = fopen(fname2, 'a');
min1 = str2num(minu);
if tp > 60.0
    min1 = min1 + floor(tp./60);
    tp = mod(tp,60);
end
t1 = num2str(tp);
if tp < 10
    t1 = ['0',t1];
end
if length(t1)==4
    t1 = [t1,'0']; 
end
if length(t1) == 2
    t1 = [t1,'.00']; 
end
if length(t1) > 5
    t1 = t1(1:5);
end
min2 = str2num(minu);
if ts > 60.0
    min2 = min2 + floor(ts./60);
    ts = mod(ts,60);
end
t2 = num2str(ts);
if ts < 10
    t2 = ['0',t2];
end
if length(t2)==4, 
    t2 = [t2,'0'];
end
if length(t2) == 2, 
    t2 = [t2,'.00'];
end
if length(t2) > 5
    t2 = t2(1:5);
end
Amp = num2str(A);
j = length(Amp);
if j == 3
    Amp = ['     ',Amp];
end
if j == 4
    Amp = ['    ',Amp];
end
if j == 5
    Amp = ['   ',Amp];
end
if j == 6
    Amp = ['  ',Amp];
end
if j == 7
    Amp = [' ',Amp];
end
P = num2str(per);
if length(P) == 1
    P = [P,'.00'];
end
if length(P) == 3
    P = [P,'0'];
end

D = num2str(dur);
if length(D) == 2
    D = [' ',D];
end

if min1 > 59 
    min1 = min1 - 60;
    hora = (str2num(hora)+1);
end
if min2 > 59 
    min2 = min2 - 60;
%   hora = hora + 1;
end

if min1 < 10
    min1 = ['0',num2str(min1)];
else min1 = num2str(min1);
end
if min2 < 10
    min2 = ['0',num2str(min2)];
else min2 = num2str(min2);
end
if hora < 10
    hora = ['0',num2str(hora)];
else hora = num2str(hora);
end

linea=[estacion,'  ',dia,mes,anno,polaridad_P,hora,min1,t1,polaridad_S,hora,min2,t2,' ',P(1:4),' ',Amp,'  ',D];
title (linea)
fprintf('%s\n',linea);
fprintf(fid, '%s%',linea);
fprintf(fid, '\r\n');
% fprintf(fid, '%s%\n',linea);
% fprintf(fid, '\n');
fclose(fid);
set (handles.grabar, 'visible','off')

% -------------------------------------------------------------------
function grabar_p_Callback(h, eventdata, handles)
% hObject    handle to grabar_p (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
estacion = handles.estacion;
lat = handles.lat; 
lon = handles.lon; 
elev= handles.elev; 
Az  = handles.Az;
polaridad_P = handles.polP; length(polaridad_P);

if polaridad_P == ' IPC' pol = 1;  end
if polaridad_P == ' IPD' pol = -1; end
if polaridad_P == ' EP ' pol = 0; end
    
% Escribir resultados a un archivo de texto
fname1 = 'estacion.txt';
fname2 = 'polaridad.txt';
fid1 = fopen(fname1, 'a');
fid2 = fopen(fname2, 'a');
%if length(estacion) == 3
%    estacion = [estacion,' '];
%end
linea=[estacion,'  ',num2str(lat),'  ',num2str(lon),'  ',polaridad_P];
title (linea)
fprintf('%4s %9.3f %10.3f %6.0f\n',estacion,lat,lon,pol);

fprintf(fid1,'%4s %8.3f %9.3f %7.1f',estacion,lat,lon,elev);
fprintf(fid1, '\r\n');

fprintf(fid2, '%4s %4.0f',estacion,pol);
fprintf(fid2, '\r\n');
fclose(fid1);
fclose(fid2);
set (handles.grabar_p, 'visible','off')
handles.polaridad_P = polaridad_P;
guidata(h, handles)

% --------------------------------------------------------------------
function tres_Callback(h, eventdata, handles, varargin)
return
y = handles.sennal;
t = handles.tdata;
set (handles.text5, 'visible','off')
set (handles.ejes,  'visible','off')
cla
fname = handles.fname;
fid = fopen(fname,'r');
i = 0;

T = 1/str2num(Fs);
seg  = tiempo(16:21);
ss   = str2num(seg);
fname2 = 'sal2';
fid2 = fopen(fname2, 'wt');
fname3 = 'sal3';
fid3 = fopen(fname3, 'wt');

fclose(fid); %fclose(fid1); 
fclose(fid2); fclose(fid3);
load sal2;
load sal3;
%N = length(sal2);
y1 = y;
y2 = sal2 - mean(sal2);
y3 = sal3 - mean(sal3);
eje(1) = axes('position',[.07  .65  .67  .22]);
plot(t,y1), grid, zoom xon
ylabel ('Vertical')
eje(2) = axes('position',[.07  .37  .67  .22]);
plot(t,y2), grid, zoom xon
ylabel ('Norte')
eje(3) = axes('position',[.07  .089 .67  .22]);
plot(t,y3), grid, zoom xon
ylabel ('Este')
handles.eje = eje;
guidata(h, handles)

% -------------------------------------------------------------
function Borrar_Callback(h, eventdata, handles, varargin)
% hObject    handle to Borrar (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fname = handles.fname;
fclose all;
delete (fname)
next_Callback(h, eventdata, handles, varargin)

% --------------------------------------------------------------------
function varargout = salir_Callback(h, eventdata, handles, varargin)
opcion=questdlg(' Desea Salir del Software ?','Visualizador de Sismos',...
                    'Si','No','cancel ','cancel ') ;            
switch opcion,
case 'No';          
    elmismo_Callback(h, eventdata, handles, varargin)   
case 'Si';
    close all
    fclose all;
    clear all
case 'cancel';      
end                

% --- Executes during object creation, after setting all properties.
function fcorte2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fcorte2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function varargout = fcorte2_Callback(h, eventdata, handles, varargin)
  fcorte = get(handles.fcorte2,'string');
  fcorte = str2num(fcorte);



function fcorte1_Callback(hObject, eventdata, handles)
% hObject    handle to fcorte1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of fcorte1 as text
%        str2double(get(hObject,'String')) returns contents of fcorte1 as a double


% --- Executes during object creation, after setting all properties.
function fcorte1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fcorte1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

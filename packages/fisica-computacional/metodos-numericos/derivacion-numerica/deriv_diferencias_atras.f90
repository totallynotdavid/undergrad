program deriv_diferencias_atras
implicit none
integer n
real,dimension(100,100)::F
print*,'Numero de puntos a leer:'
read*,n
!Lectura de datos
call lectura(F,n,8)
print*,'Se leyo correctamente los datos'
print*,''
!Tabla de diferencias
call diferencias(F,n,9)
print*,'Se genero correctamente la tabla de diferencias hacia adelante en el archivo tabla_diferencias_atras.dat'
print*,''

call derivadas(F,n,10)
print*,'Se genero correctamente las derivadas permitidas en el archivo derivadas.dat'
print*,''
end program

subroutine lectura(F,n,indicador)
implicit none
integer i,indicador,n
real,dimension(100,100)::F
open(indicador,file='puntos.dat',status='old')
do i=1,n,1
read(indicador,*)F(i,1:2)
end do
end subroutine

subroutine diferencias(F,n,indicador)
implicit none
integer i,j,indicador,n
real,dimension(100,100)::F
open(indicador,file='tabla_diferencias_atras.dat',status='unknown')
do j=3,n+1,1
do i=2,n,1
F(i,j)=F(i,j-1)-F(i-1,j-1)
end do
end do
j=2
do i=1,n,1
write(*,50)F(i,1:j)
write(indicador,50)F(i,1:j)
j=j+1
end do
50 format(100f10.4)
end subroutine


subroutine derivadas(F,n,indicador)
integer i,j,t,indicador,n,k,m
real,dimension(100,100)::F,D,A
open(indicador,file='derivadas.dat',status='unknown')
print*,'Orden de truncamiento de la derivada:'
read*,t
do while (t>=n)
print*,'El orden de truncamiento debe ser menor a la cantidad de datos/puntos!'
print*,''
print*,'Orden de truncamiento de la derivada'
read*,t
end do
A=0
D=0

j=2
do i=1,n,1
A(i,1:j)=F(i,1:j)
j=j+1
end do

do i=1,n,1
D(i,1:2) = F(i,1:2)
end do
m=n
do j=1,n-1,1
do i=1,t,1
D(m,3) = D(m,3) + (A(m,i+2))/(i*(A(m,1)-A(m-1,1)))
end do
m=m-1
end do
k = n-t
m = n
do i=1,k,1
write(indicador,50)D(m,1:3)
m = m-1
end do
50 format(100f12.4)
end subroutine
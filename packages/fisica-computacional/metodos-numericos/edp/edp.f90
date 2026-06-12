program edp_parabolica
implicit none
integer(kind=4) n,i
real x0, xf, t0, gamma
real, parameter::dx=1, dt=0.02, alpha=10
real a, b, c
real, dimension(101,11)::P		!P(tiempo,posición) == P(#filas,#columnas)
!character(64),dimension(101)::nombre
open(8,file='edp.dat',status='unknown')
open(9,file='edp2.dat',status='unknown')
open(10,file='edp3.dat',status='unknown')
call datos(x0, a, xf, b, t0, c)
!Resolviendo la parte temporal mediante Euler simple usando las condiciones iniciales

P(1,1:11)=c			!Colocando las condiciones iniciales
P(1:101,1)=a		!Colocando las condiciones de frontera
P(1:101,11)=b		!Colocando las condiciones de frontera
gamma = alpha*dt/dx**2
!Temperatura
do n=1,100,1
do i=2,10,1
P(n+1,i)=P(n,i)+gamma*(P(n,i-1)-2*P(n,i)+P(n,i+1))
end do
end do

!Guardando la matriz P(tiempo,posicion)
do n=1,101,1
write(8,40)P(n,1:11)
end do

!Guardando los valores de (t,x,P(t,x))
do n=1,101,1
do i=1,11,1
write(9,50)(n-1)*dt,i-1,P(n,i)
end do
end do

!Guardando los valores de (x,t,P(t,x))
do i=1,11,1
do n=1,101,1
write(10,60)i-1,(n-1)*dt,P(n,i)
end do
end do

print*,'Finalizado'

40 format(11F10.4)
50 format(F6.3,I5,F10.4)
60 format(I5,F6.3,F10.4)
end program


subroutine datos(x0, a, xf, b, t0, c)
implicit none
real x0, xf, t0
real a, b, c
!Condiciones de frontera
print*,'Ingresar x0 y f(x0,t):'
read*,x0,a
print*,'Ingresar xf y f(xf,t):'
read*,xf,b
!Condiciones iniciales
print*,'Ingresar t0 y f(x,t0):'
read*,t0,c
end subroutine
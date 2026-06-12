!*************************************
!Método de Euler para EDO de 2do orden
!Alvaro Siesquén
!*************************************
program euler2
implicit none
real z0,v0,z,v,t,a
real Ek,Ep
real,parameter::h=0.05
print*,'Altura inicial z0:'
read*,z0
print*,'Velocidad inicial v0:'
read*,v0
open(8,file='caida_vertical.dat',status='unknown')
t=0
write(8,fmt=*)'#	t	z	v	Ek	E'
write(8,50)t,z0,v0,Ek(v),Ek(v0)+Ep(z0)
do while (z0>0)
v = v0+h*a(v0)
z = z0 + h*v
t = t+h
z0 = z
v0 = v
if (z>0)then
write(8,50)t,z,v,Ek(v),Ek(v)+Ep(z)
end if
end do
50 format(5f12.4)
print*,'Programa finalizado con exito'
print*,'Los datos se guardaron en caida_vertical.dat'
end program

real function a(v)
real, parameter::g=9.81,m=2.0,k=2.8
a = -g - k*v/m
end function

real function Ek(v)
real v
real, parameter::m=2.0
Ek = (m*v**2)/2
end function

real function Ep(z)
real z
real, parameter::m=2.0,g=9.81
Ep=m*g*z
end function
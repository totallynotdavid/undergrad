program osciladores
implicit none
integer n
real, parameter::h=.01, delta=0.25
real y0, y1, z1, y2, z2, t, m1, m2, k1,k2
real g1, g2
real p1, p2, p3, p4
real l1, l2, l3, l4
real r1, r2, r3, r4
real s1, s2, s3, s4

print*,'---Masas oscilando por 3 resortes---'
print*,'Digite m1 y m2:'
read*,m1,m2

print*,'Tiempo inicial'
read*,t
print*,'x1 inicial'
read*,y1
print*,'dx1/dt inicial'
read*,z1
print*,'x2 inicial'
read*,y2
print*,'dx2/dt inicial'
read*,z2

open(8, file='tres_resortes_datos.dat', status='unknown')
open(9, file='retorno.dat', status='unknown')
write(8, fmt=*)'t	x1		dx1/dt		x2		dx2/dt'
!z = y' = f
!z' = y'' = g
k1 = 250
k2 = 250
n=1
do while(k1>0)
	p1 = h*z1
	l1 = h*g1(y1,y2,m1,k1)
	p2 = h*(z1 + l1/2.)
	l2 = h*g1(y1 + p1/2.,y2,m1,k1)
	p3 = h*(z1 + l2/2.)
	l3 = h*g1(y1 + p2/2.,y2,m1,k1)
	p4 = h*(z1 + l3)
	l4 = h*g1(y1 + p3,y2,m1,k1)

	r1 = h*z2
	s1 = h*g2(y1,y2,m2,k2)
	r2 = h*(z2 + s1/2.)
	s2 = h*g2(y1,y2 + r2/2.,m2,k2)
	r3 = h*(z2 + s2/2.)
	s3 = h*g2(y1,y2+r2/2.,m2,k2)
	r4 = h*(z2+s3)
	s4 = h*g2(y1,y2+r3,m2,k2)

	write(8,40)t,y1,z1,y2,z2
	y1 = y1 + (p1 + 2*p2 + 2*p3 + p4)/6.
	z1 = z1 + (l1 + 2*l2 + 2*l3 + l4)/6.
	y2 = y2 + (r1 + 2*r2 + 2*r3 + r4)/6.
	z2 = z2 + (s1 + 2*s2 + 2*s3 + s4)/6.
	t = t + h
	k1 = k1 - delta
	k2 = k2 - delta
	if (n>1) then
	write(9,*)y1,y0
	end if
	n = n+1
	y0 = y1
end do
40 format(5f10.4)

print*,'Tiempo final: ',t-h
print*,'Programa finalizado'
end program

real function g1(y1,y2,m1,k1)
implicit none
real y1,y2,m1,k1
real, parameter::k=250.
	g1 = (- k1*y1 + k*(y2-y1))/m1
end function

real function g2(y1,y2,m2,k2)
implicit none
real y1,y2,m2,k2
real, parameter::k=250.
	g2 = (- k2*y2 - k*(y2-y1))/m2
end function

program pendulo_doble
implicit none
integer i
real, parameter::h=.01, g=9.81, m1=1.0, m2=2.0, c1=1.0, c2=1.0
real t, y1, z1, y2, z2
real g1, g2
real k1, k2, k3, k4
real l1, l2, l3, l4
real p1, p2, p3, p4
real q1, q2, q3, q4
real K, V, E
T = 0
V = 0
E = 0
print*,'Tiempo inicial'
read*,t
print*,'theta_1 inicial'
read*,y1
print*,'velocidad angular 1 inicial'
read*,z1

print*,'theta_2 inicial'
read*,y2
print*,'velocidad angular 2 inicial'
read*,z2

open(1, file='pendulo_doble_datos.dat', status='unknown')

write(1, fmt=*)'tiempo		theta1	w1	theta2		w2	T	V	E'
!z = y' = f
!z' = y'' = g

do i=1,100
	k1 = h*z1
	l1 = h*g1(y1,z1,y2,z2,t)
	k2 = h*(z1 + l1/2.)
	l2 = h*g1(y1 + k1/2.,z1 + l1/2.,y2,z2, t + h/2.)
	k3 = h*(z1 + l2/2.)
	l3 = h*g1(y1 + k2/2.,z1 + l2/2.,y2,z2, t + h/2.)
	k4 = h*(z1 + l3)
	l4 = h*g1(y1 + k3,z1 + l3,y2, z2, t + h)

	p1 = h*z2
	q1 = h*g2(y1,z1,y2,z2,t)
	p2 = h*(z2 + q1/2.)
	q2 = h*g2(y1,z1,y2 + p1/2.,z2 + q1/2.,t + h/2.)
	p3 = h*(z2 + q2/2.)
	q3 = h*g2(y1,z1,y2 + p2/2.,z2 + q2/2.,t + h/2.)
	p4 = h*(z2 + q3)
	q4 = h*g2(y1,z1,y2 + p3,z2 + q3,t + h)
	
	K = m1*c1**2*z1**2/2 + m2*(c1**2*z1**2+c2**2*z2**2+2*c1*c2*z1*z2*cos(y1-y2))/2
	V = -(m1+m2)*g*c1*cos(y1) - m2*g*c2*cos(y2)
	E = T + V

	write(1,50)t,y1,z1,y2,z2, K, V, E
	y1 = y1 + (k1 + 2*k2 + 2*k3 + k4)/6.
	z1 = z1 + (l1 + 2*l2 + 2*l3 + l4)/6.
	y2 = y2 + (p1 + 2*p2 + 2*p3 + p4)/6.
	z2 = z2 + (q1 + 2*q2 + 2*q3 + q4)/6.
	t = t + h
end do
50 format(8f10.4)
print*,'Finalizado'
end program

real function g1(y1,z1,y2,z2,t)
implicit none
real,parameter::g=9.81, m1=1.0, m2=2.0,l1=1.0, l2=1.0
real y1,z1,y2,z2,t
	g1 = (-g*(2*m1+m2)*sin(y1)-m2*g*sin(y1-2*y2)-2*sin(y1-y2)*m2*(z2**2*l2+z1**2*l1*cos(y1-y2)))/( l1*(2*m1+m2-m2*cos(2*y1-2*y2)))
end function

real function g2(y1,z1,y2,z2,t)
implicit none
real,parameter::g=9.81, m1=1., m2=2.,l1=1., l2=1.
real y1,z1,y2,z2,t
	g2 = ( 2*sin(y1-y2)*(z1**2*l1*(m1+m2) + g*(m1+m2)*cos(y1) + z2**2*l2*m2*cos(y1-y2)) )/( l2*(2*m1 + m2 - m2*cos(2*y1-2*y2)) )
end function

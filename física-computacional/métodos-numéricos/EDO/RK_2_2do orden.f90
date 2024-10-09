program RK_modelo
implicit none
integer i
real, parameter::h=.01
real y, z, t
real f, g
real k1, k2, k3, k4
real l1, l2, l3, l4

print*,'Tiempo inicial'
read*,t
print*,'Y inicial'
read*,y
print*,'dY/dt inicial'
read*,z

open(1, file='rungekutta_datos.dat', status='unknown')

write(1, fmt=*)'x       y'
!z = y' = f
!z' = y'' = g

do i=1,100
	k1 = h*f(y,z,t)
	l1 = h*g(y,z,t)
	k2 = h*f(y + k1/2.,z + l1/2.,t + h/2.)
	l2 = h*g(y + k1/2.,z + l1/2.,t + h/2.)
	k3 = h*f(y + k2/2.,z + l2/2.,t + h/2.)
	l3 = h*g(y + k2/2.,z + l2/2.,t + h/2.)
	k4 = h*f(y + k3,z + l3,t + h)
	l4 = h*g(y + k3,z + l3,t + h)

	write(1,*)t,y
	y = y + (k1 + 2*k2 + 2*k3 + k4)/6.
	z = z + (l1 + 2*l2 + 2*l3 + l4)/6.
	t = t + h
end do
end program

real function f(y,z,t)
implicit none
real y, z, t
	f = z + 0*(t + y)
end function

real function g(y,z,t)
implicit none
real y,z,t
	g = - 200*y -0*z + 0*t
end function
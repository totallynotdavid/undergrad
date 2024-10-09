!Derivada por desarrollo de Taylor para tan(1)
program deriv_taylor
implicit none
real h,fd,bd,cd,c
real,dimension(3)::salto
integer i
salto=(/0.1,0.05,0.02/)
do i=1,3
h = salto(i)
!Taylor dos puntos
bd = (tan(1.)-tan(1-h))/h
fd = (tan(1+h)-tan(1.))/h
cd = (tan(1+h)-tan(1-h))/(2*h)
print*,h
print*,bd,100*(c(1.)-bd)/c(1.)
print*,fd,100*(c(1.)-fd)/c(1.)
print*,cd,100*(c(1.)-cd)/c(1.)
!taylor tres puntos
bd = (3*tan(1.) -4*tan(1-h) + tan(1-2*h) )/(2*h)
fd = (-tan(1+2*h) + 4*tan(1+h) -3*tan(1.) )/(2*h)
print*,bd,100*(c(1.)-bd)/c(1.)
print*,fd,100*(c(1.)-fd)/c(1.)
print*,''
end do
end program

real function c(x)
real x
	c = 1/cos(x)**2
end function

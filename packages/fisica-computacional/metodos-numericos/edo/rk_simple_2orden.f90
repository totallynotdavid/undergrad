program rk_simple_2orden
implicit none
integer i,n
real, parameter::h=0.1
real x,y,xf,k1,k2,f

print*,'x inicial:'
read*,x
print*,'y inicial:'
read*,y
print  *,'x final'
read*,xf
n = (xf-x)/h
do i=1,n,1
k1 = h*f(y,x)
k2 = h*f(y+k1,x+h)
y = y + (k1+k2)/2
x = x+h
print *, x,y
end do
end program

real function f(y,x)
real x
f = 2*x+0*y
end function

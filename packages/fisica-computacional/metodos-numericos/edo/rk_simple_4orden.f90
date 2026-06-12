program rk_simple_4orden
implicit none
integer i,n
real, parameter::h=0.1
real x,y,xf,k1,k2,k3,k4,f

print*,'x inicial:'
read*,x
print*,'y inicial:'
read*,y
print  *,'x final'
read*,xf
n = (xf-x)/h
do i=1,n,1
k1 = h*f(y,x)
k2 = h*f(y+k1/2,x+h/2)
k3 = h*f(y+k2/2,x+h/2)
k4 = h*f(y+k3,x+h)
y = y + (k1+2*k2+2*k3+k4)/6
x = x+h
print *,x,y
end do
50 format (2f8.3)
end program

real function f(y,x)
real x
f = 2*x+0*y
end function		  

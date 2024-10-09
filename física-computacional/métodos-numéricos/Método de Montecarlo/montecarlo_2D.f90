program montecarlo2d
implicit none
integer i, seed, N
real a,b,c,d,s,f
print*,'------ 2-D MonteCarlo Method ------'
print*,'Write the seed: '
read*,seed
print*,'Write the number of points: '
read*,N

print*,'Write the range of x: '
read*,a,b

print*,'Write the range of y: '
read*,c,d

call srand(seed)
s = 0
do i=1,N
s = s + f(a + (b-a)*ran(),c + (d-c)*ran())
end do
print*,(d-c)*(b-a)*s/N
end program

real function f(x,y)
real x,y
f = 9*x**2*y**2
end function
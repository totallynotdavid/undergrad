program integral_simple
implicit none
integer i, seed, n
real f,integral,a,b,s,sigma
real,parameter::pi=3.14159265

print*,'---- Montecarlo Method ----'
print*,'Write the seed: '
read*,seed
print*,'Write the number of points: '
read*,n
print*,'Write the range of x: '
read*,a,b
call srand(seed)
integral = 0
s = 0
do i=1,n,1
integral = integral + f(a + (b-a)*ran())
s = s+ a + (b-a)*ran()
end do

integral = (b-a)*integral/n
s = s/n

print*,'Integral: ',
print*,'Error: ',pi-integral
end program

real function f(x)
real x
f = sqrt(4.-x**2)
end function
program biseccion
implicit none
real a,b,c,epsilon,f
print*,'--- Bisección ---'
print*,'Digite los intervalos'
read*,a,c
do while (f(a)*f(c)>0 .and. a .ne. c)
print*,'Digite intervalos válidos'
read*,a,c
end do
print*,'Digite la tolerancia permitida:'
read*,epsilon
a = min(a,c)
c = max(a,c)
call metodo(a,c,epsilon)
print*,'Programa finalizado'
end program

real function f(x)
real x
f = x**3+1
end function

subroutine metodo(a,c,epsilon)
real a,c,b,epsilon,f
real x
do while (abs(c-a)>epsilon)
b = (a+c)/2
if (f(b)>=0) then
c = b
end if
if (f(b)<0)then
a = b
end if
end do
x = (a+c)/2
print 50,x
50 format (f5.2)
end subroutine
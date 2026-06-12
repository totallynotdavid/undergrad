program newton_raphson
implicit none
integer i
real x0,x1,f,d,epsilon
print*,'Ingrese x0:'
read*,x0
print*,'Digite el error permitido:'
read*,epsilon
x1 = x0 - f(x0)/d(x0)
i=1
do while (abs(x1-x0)>epsilon)
x0=x1
x1 = x1 - f(x1)/d(x1)
i=i+1
end do
print*,'x = ',x1
print*,'Numero de iteraciones: ' ,i
end program

!Escribir la funcion y la derivada
real function f(x)
real x
f = x**3+5*x**2+2
end function

real function d(x)
real x
d = 3*x**2+10*x
end function

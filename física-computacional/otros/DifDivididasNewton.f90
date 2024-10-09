program DiferenciasDividas2
  implicit none
  real :: x(0:6), f(0:6,0:6), suma, xint=0.28, p
  integer :: i,j
  x=(/0.1, 0.2, 0.4, 0.7, 1.0, 1.2, 1.3/)
  f=0.0
  f(:,0)=(/.99750, .99002, .96040, .88122, .76520, .67113, .62009/)
  do j = 1, 6
    do i = 0,6-j
      f(i,j)=(f(i+1,j-1)-f(i,j-1))/(x(i+j)-x(i))
    end do
  end do
  suma = 0.0
  do i = 1, 6
    p=1.0
    do j = 1, i
      p = p * (xint-x(j-1))
    end do
    p= f(0,i) * p
    suma = suma + p
  end do
  suma = suma +f(0,0)
  print*, "el valor interpolado es: ", suma
end program
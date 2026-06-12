program Ilagrange
implicit none
 real :: x(0:3),y(0:3),xint=2.8, suma=0.0,f
 integer :: i,j,n=3
 x=(/1.,2.,3.,4./)
 y = (/0.671,0.620,0.567,0.52/)
 print *, suma
 do i =0,n
    f=1
    do j =0,n
      if (i /=j) then
      f =f*(xint -x(j))/(x(i)-x(j))
      end if
    end do
    suma = suma +f*y(i)
 end do
  print*, "El valor interpolado es: ", suma
end program

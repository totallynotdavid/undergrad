program errores3
implicit none
real :: theta(12),d(12)
integer :: i
 theta(1)=0.1
 do i =1,12
    d(i) = cos(1.0) - 0.5*theta(i)*sin(1.0)
    print *,i,theta(i),d(i)
    theta(i+1)= theta(i)/10
 end do


end program

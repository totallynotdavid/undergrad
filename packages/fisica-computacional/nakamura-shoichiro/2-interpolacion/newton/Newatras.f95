program Newatras
    implicit none
    real :: Q(0:7),F(0:7,0:7),p,Qi = 1000.0,Ni=0.0,h=200.
    integer:: i,j
    Q(:) =(/500.,700.,900.,1100.,1300.,1500.,1700.,1900./)
    F=0.0; p = (Qi -Q(7))/h
    F(:,0)=(/365.,361.6,370.74,379.68,384.46,395.5,395.95,397./)
    !Tablas de diferencias hacia atras
    do j=1,7
        do i=j,7
            F(i,j)= F(i,j-1)-F(i-1,j-1)
        end do
    end do
    do i=0,7
     Qi = Qi + pcombinatorio(p,i)*F(7,i)
    end do
    print*, "La potencia interpolada es: ", Qi
    contains
    function factorial(k)
    implicit none
    integer:: i,factorial,k
    factorial =1
    do i =1,k
        factorial =factorial*k
    end do
    end function factorial
    function pcombinatorio(p,k)
     implicit none
     real :: p,pcombinatorio
     integer :: i,k
     if (k == 0) then
      pcombinatorio =1
     else
      pcombinatorio =1
      do i = 0,k-1
        pcombinatorio = pcombinatorio*(p+i)
      end do
       pcombinatorio = pcombinatorio/factorial(k)
      end if
    end function
end program

program errores1
    implicit none
    double precision :: suma =1.0
    integer :: i
    do i =1,10000
     suma =suma +0.00001
    end do
    print *, suma
end program

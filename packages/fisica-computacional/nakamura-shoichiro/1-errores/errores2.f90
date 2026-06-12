program errores2
    implicit none
    double precision :: suma =1.0,total
    integer :: i ,j
    do i =1,100
        total = 0.0
        do j=1,100
        total = total +0.00001
        end do
        suma =suma +total
     end do
    print *, suma
end program

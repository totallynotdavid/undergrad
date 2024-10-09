program DiferenciasDividas
    implicit none
    real :: x(0:6), f(0:6,0:6)
    integer :: i,j
    x=(/0.1,0.2,0.4,0.7,1.0,1.2,1.3/)
    f = 0.0
    f(:,0)=(/.99750,.99002,.96040,.88122,.76520,.67113,.62009/)
    do j=1,6
        do i = 0,6-j
            f(i,j)=((f(i+1,j-1))-f(i,j-1))/(x(i+j)-x(i))
        end do
    end do
    do i=0,6
        print*,(f(i,j),j=0,6)
    end do
end program

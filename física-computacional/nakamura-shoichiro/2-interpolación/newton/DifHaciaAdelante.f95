program DifHaciaAdelante
    implicit none
    real :: y(1:6),m(1:6,1:6)
    integer:: i,j
    m=0.0
    m(:,1)=(/1.143,1.0,0.828,0.667,0.533,0.428/)
    do j=2,6
        do i=1,7-j
        m(i,j)= m(i+1,j-1)-m(i,j-1)
        end do
    end do
    do i=1,6
        print*,(m(i,j),j=1,6)
    end do

end program

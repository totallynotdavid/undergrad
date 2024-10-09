program findingpi
    implicit none
    real :: pi, rnd(2), x, y
    integer:: M=0, N=10000,i
    do i = 1,N
        call random_number(rnd)
        x=rnd(1)
        y=rnd(2)
        if (x**2+y**2<=1.) then
            M=M+1
        end if
    end do
    pi = 4.*real(M)/N 
    print *, "el valor aproximado de pi es ", pi
end program
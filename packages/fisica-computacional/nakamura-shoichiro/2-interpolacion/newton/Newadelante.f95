program Newadelante
    implicit none
    Real :: T(0:6),Ri, Ti=162.,F(0:6,0:6),s,h=10.
    integer :: i,j
    F=0.0
    F(:,0)=(/35.5,37.8,43.6,45.7,47.3,50.1,51.2/)
    T=(/150.,160.,170.,180.,190.,200.,210./)
    s=(Ti-T(0))/h
    do j=1,6
        do i=1,6-j
        F(i,j)=F(i+1,j-1)-F(i,j-1)
        end do
    end do
    Ri=F(0,0)
    do i = 1,6
     Ri = Ri + fcombinat(s,i)*F(0,i)
    end do
    print*, "el valor interpolado es: ", Ri
    contains
    function factorial(n)
    implicit none
    integer:: factorial,n,i
    factorial=1
     do i=1,n
        factorial =factorial*i
     end do
     end function
   function fcombinat(s,i)
    implicit none
    real ::s,fcombinat
    integer ::i,k
    fcombinat=1.0
    do k =0,i-1
    fcombinat =fcombinat*(s-k)
    end do
    fcombinat = fcombinat/factorial(i)
    end function
end program

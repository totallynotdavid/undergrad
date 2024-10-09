program evaluarintegrales
  implicit none
  real :: Fn, b=1., a=0., x, rnd, pi=4*atan(1.0)
  real :: dif(5), logdif(5), logn(5)
  integer :: n(5), i, j
  n(1) = 1000
  do i=2,5
    n(i)=n(i-1)*10
    print*, n(i)
  end do
  do j = 1,5
    Fn=0.
    do i = 1, n(j)
      call random_number(rnd)
      x     = rnd
      Fn    = Fn + f(x)
    end do
    Fn      = Fn/n(j)
    Fn      = (b-a)*Fn
    dif(j)  = abs(Fn-pi)
    logdif(j) = log(dif(j))
    logn(j)   = log(real(n(j)))
  end do

  open(file="integral.txt", unit=7)

  do i = 1,5
    write(7,*) logn(i), logdif(i)
  end do

  contains 
  function f(x)
    implicit none
    real :: f,x
    f = 4 * sqrt(1-x**2)
  end function 
end program
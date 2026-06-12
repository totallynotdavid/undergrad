subroutine compute_integral_1d(n, a, b, integral)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: a, b
    real, intent(out) :: integral
    integer :: i
    real :: x, acc

    acc = 0.0
    do i = 1, n
        call random_number(x)
        x = a + (b - a) * x
        acc = acc + sqrt(4.0 - x * x)
    end do

    integral = (b - a) * acc / real(n)
end subroutine

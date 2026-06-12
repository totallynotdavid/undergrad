subroutine compute_integral_2d(n, a, b, c, d, integral)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: a, b, c, d
    real, intent(out) :: integral
    integer :: i
    real :: x, y, acc

    acc = 0.0
    do i = 1, n
        call random_number(x)
        call random_number(y)
        x = a + (b - a) * x
        y = c + (d - c) * y
        acc = acc + 9.0 * x * x * y * y
    end do

    integral = (b - a) * (d - c) * acc / real(n)
end subroutine

subroutine compute_pi(n, pi_estimate)
    implicit none
    integer, intent(in) :: n
    real, intent(out) :: pi_estimate
    integer :: i, k
    real :: x, y

    k = 0
    do i = 1, n
        call random_number(x)
        call random_number(y)
        x = 2.0 * x - 1.0
        y = 2.0 * y - 1.0
        if (x * x + y * y <= 1.0) then
            k = k + 1
        end if
    end do

    pi_estimate = 4.0 * real(k) / real(n)
end subroutine

subroutine compute_pi(n, result)
    integer, intent(in) :: n
    real, intent(out) :: result
    integer :: i, k
    real :: x, y

    k = 0
    do i = 1, n
        call random_number(x)
        call random_number(y)
        if ((x - 0.5)**2 + (y - 0.5)**2 <= 0.25) then
            k = k + 1
        end if
    end do
    result = 4.0 * real(k) / real(n)
end subroutine
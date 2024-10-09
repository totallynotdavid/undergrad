Program epsilon
implicit none
real :: eps
eps = 1
do while (eps + 1 > 1)
eps = eps/2
end do
print*, "El valor de epsilon del ordenador es", eps
End Program
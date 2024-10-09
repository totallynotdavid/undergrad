program pi_montecarlo
implicit none
integer i, k, seed, N
real x,y
print*,'---- Pi mediante montecarlo ----'
print*,'Write the seed: '
read*,seed
print*,'Number of points: '
read*,N
call srand(seed)
k = 0
do i=1,N
x = ran()
y = ran()
if ( (x-0.5)**2. + (y-0.5)**2. <= 0.25 ) then
k = k+1
end if
end do
print*,'Valor aproximado de Pi: ' ,4*real(k)/real(N)
end program
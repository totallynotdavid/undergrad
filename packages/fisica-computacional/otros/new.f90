Program new
  implicit none
  integer :: n, suma
  real :: hello
  n = 1
  print*, hello
  if (suma > hello) then
    print*, 'suma es mayor'
  end if
  if (hello > suma) then
    print*, 'hello es mayor'
  end if
  if (hello == suma) then
    print*, 'iguales'
  end if
  suma = 5
  hello = suma + 1
  print*, hello
  
end program
Program bin2dec
  implicit none
  integer :: n, c, i, r, suma
  print*, "numero binario a convertir a decimal"
  read(*,*) n
  c=n
  suma = 0
  i = 0
  do while (c /= 0)
    r=mod(c,10)
    c=c/10
    suma = suma + (r*2**i)
    i=i+1
    print*, "binario en decimal equivale a :", suma
  end do
End Program
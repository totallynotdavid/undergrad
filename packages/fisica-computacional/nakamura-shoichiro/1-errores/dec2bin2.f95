program dec2bin
    implicit none
    integer:: n,c,r,i,suma
    print*, "ingrese el numero entero en base 10"
    read *,n
    c = n
    i=0
    do while (c /= 0)
     r = mod(c,2)
     suma = suma + (r*10**i)
     c = c/2
     i =i+1
    end do
    print *, n, "es equivalente a: ", suma
end program

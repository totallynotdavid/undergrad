program dec2bin
    implicit none
    integer:: n,c,r,vec1(10),vec2(10),i,j
    print*, "ingrese el numero entero en base 10"
    read *,n
    c = n; vec1=0; vec2=vec1
    i= 0; j=1
    do while (c /= 0)
     i =i+1
     r = mod(c,2)
     vec1(i)=r
     c = c/2
    end do
    do i =10,1,-1
      vec2(i) = vec1(j)
      j =j+1
    end do

    print *, vec2
end program

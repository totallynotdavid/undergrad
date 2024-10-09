program curvaDePlanck
implicit none

! Temperatura
integer :: T

! Matriz de temperaturas a recorrer en bucle
integer, parameter, dimension(4) :: dplanck = (/ 2000, 6000, 10000, 14000 /)

! Rango de longitudes de onda para la curva
real, parameter :: lambda_min = 1       ! Longitud de onda mínima (nm)
real, parameter :: lambda_max = 4000.0  ! Longitud de onda máxima (nm)
real, parameter :: lambda_step = 1    ! Tamaño del paso de longitud de onda (nm)

! Constantes de la fórmula de la curva de Planck
real, parameter :: h = 6.62607004e-34   ! Constante de Planck (J*s)
real, parameter :: c = 2.99792458e+8    ! Velocidad de la luz (m/s)
real, parameter :: k = 1.38064852e-23   ! Constante de Boltzmann (J/K)

! Declarando variables
integer :: i
real :: lambda                  ! Longitud de onda (nm)
real :: B_lambda                ! Valor de la curva de Planck para la longitud de onda dada (W/m^2*sr*nm)
character(len=50) :: filename   ! Nombre del archivo CSV

! Bucle para cada temperatura
do i = 1, 4

    ! Fijando la temperatura actual
    T = dplanck(i)

    ! Abrir y nombrar los archivos CSV de salida
    write(filename, '(A,I0,A)') 'curvaDePlanck_', T, '.csv'
    open(unit=10, file=filename, status='replace', action='write')

    ! Escribe la línea de encabezado en el archivo
    write(10, '(A)') 'x,y'

    ! Valor inicial de la longitud de onda
    lambda = lambda_min

    ! Bucle para cada longitud de onda en el rango especificado
    do while (lambda <= lambda_max)
        ! Valor de la curva de Planck para la longitud de onda dada
        B_lambda = (2.0 * h * c ** 2 / (lambda * 1e-9) ** 5) / (exp(h * c / (lambda * 1e-9 * k * T)) - 1.0)

        ! Guardar la longitud de onda y el valor correspondiente de la curva de Planck en el archivo CSV
        write(10, '(ES25.10, ",", ES25.10)') lambda, B_lambda

        ! Aumentar la longitud de onda para el siguiente paso
        lambda = lambda + lambda_step
    end do

    ! Cerrar el archivo CSV
    close(10)

end do

end program curvaDePlanck

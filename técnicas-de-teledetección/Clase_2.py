import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import osgeo.gdal
import osgeo.gdal_array
filepath = r"/workspaces/teledeteccion/LC08_L1TP_007068_20240427_20240427_02_RT_refl.tif" #Ruta de la imagen
#l. Usando Libreria GDAL
raster_b4 = osgeo.gdal.Open(filepath) #Abrimos el archivo y almacenamos en una raviable raster
print(type(raster_b4)) #chequeamos en tipo de la variable raster
print(raster_b4.GetProjection()) #Proyeccion
print("Columnas (X):",raster_b4.RasterXSize) #Dimensiones
print("Filas (Y):",raster_b4.RasterYSize)
print("Bandas:",raster_b4.RasterCount) #Numbero de bandas
raster_b4.GetMetadata()
#Metadata del raster
band_4 = raster_b4.GetRasterBand(1) #Convierte el dataset raster a Banda en una variable separada
print(type(band_4)) #Chequea en tipo de variable
osgeo.gdal.GetDataTypeName(band_4.DataType) #Tipo de dato de los valores
if band_4.GetMinimum() is None or band_4.GetMaximum() is None: #Calculo de estadisticas necesarias
    band_4.ComputeStatistics(0)
    print("Calculo de Estadisticas:")
band_4.GetMetadata() #Metada de la banda
#Imprime solo la metadata seleccionada:
print ("[VALOR NO DATA] = ", band_4.GetNoDataValue()) # none
print ("[ MIN ] = ", band_4.GetMinimum())
print ("[ MAX ] = ", band_4.GetMaximum())
array_4 = band_4.ReadAsArray() #Convierte la Banda a un Array bidimensional
array_4x= osgeo.gdal_array.LoadFile(filepath) #Lectura Directa de un array usanda gdal_array
print(array_4x.min())
nodata = band_4.GetNoDataValue() #Encuentra los valores NoData en la Banda
array_4x = np.ma.masked_equal(array_4x, nodata) #Enamascara los valores NoData de la imagen
type(array_4x)
print(array_4x.min(),array_4x.max())
img_4 = plt.imshow(array_4x[0, :, :])
plt.show()
raster_b4 = None
band_4 = None

# save the image
plt.imsave('img_4.png', array_4x[0, :, :])
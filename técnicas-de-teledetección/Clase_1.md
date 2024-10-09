# P-01: INSTALACION Y CONFIGURACION (06/04/2024)

La presente practica tiene por objeto instalar y configurar los programas, librerías y IDE necesarios para el procesamiento digital de imágenes de teledetección.
1. Instalar ANACONDA: https://www.anaconda.com/download
2. Crear entorno virtual usando conda: 
    ```
    conda create fcf-sr -c conda-forge python=3
    conda activate fcf-sr
    conda install -c conda-forge nose numpy matplotlib scipy python-dateutil gdal pyproj scikit-image pyhdf pyresample netcdf4 h5py pandas xarray rasterio earthpy earthengine-api
    conda install -c conda-forge py6s
    conda install -c conda-forge ipython Jupyter
    ```
3. Usando entono virtual fcf-sr:
    - Abrir Anaconda Prompt y dirigirse a su directorio de trabajo
    - Activar entorno virtual digitando en el prompt: `actívate fcf-sr`
    - Ejecutar Jupyter Notebook, escribiendo en el prompt: `jupyter-notebook`
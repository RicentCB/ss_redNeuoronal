## Para usar
Se necesita tener instalado el manejador de paquetes "pip3", y correr los siguientes comandos, para instalar la librererias.

```bash
    pip3 install matplotlib
    pip3 install tensorflow
```

## Nuevos modelos
Se necesita tener una carpeta, con subcarpetas con los grupos los cuales se clasificara la imagen.

### Configuración
- Las siguientes variables deben ser modificadas
```python
    # Nombre de archivo a guardar la red
    saveFileName = "*.h5"   # Nombre de modelo entrenado
    # Path con imagenes
    data_dir = ""           # Ruta de carpeta con grupos

    # Parametros para el "cargador"
    img_height = 0          # Altura de la imane
    img_width = 0           # Ancho de la imane
```
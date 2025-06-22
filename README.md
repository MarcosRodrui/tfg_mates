# TFG Matemáticas: Experimentos KANs

Este repositorio contiene todo los experimentos sobre KANs utilizados en la elaboración de mi TFG de matemáticas. El orden natural para examinar los ejemplos es el siguiente:
- `00_ejemplo_mult`. El ejemplo más sencillo de todos. Muestra como una KAN [2,2,1] puede aprender la multiplicación entre dos variables.
- `01_simplificacion`. Este ejemplo es esencial ya que muestra el proceso de simplificación de una KAN. Este proceso es utilizado en el resto de cuardernillos, ya que es una de las herramientas de esta arquitectura más útiles.
- `02_interpretabilidad`. Contiene varios cuadernillos en los que se muestra como se puede redescubrir la ley de gravitación universal utilizando una KAN.
- `03_edps`. Contiene una librería que permite comparar una KAN y un MLP en la tarea de resolver una EDP utilizando el método PINN. Además, se incluyen algunos cuadernillos que utilizan la librería a modo de ejemplo.

## Requisitos
Se requiere tener instalado Python junto con [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) o [Anaconda](https://www.anaconda.com/) para gestionar las librerías. Las librerías necesarias se encuentran en el fichero `enviroment.yml`.

## Instalación
Clona el repositorio.
```bash
git clone https://github.com/MarcosRodrui/tfg_mates
cd tfg_mates
```

Crea el entorno Conda.
```bash
conda env create -f environment.yml
```

Activa el entorno.
```bash
conda activate tfg-env
```

## Documentación
Todo el código utiliza la librería [pykan](https://github.com/KindXiaoming/pykan). Por esta razón, se recomienda revisar la [documentación oficial](https://kindxiaoming.github.io/pykan/) ante cualquier duda.
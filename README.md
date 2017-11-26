# red-neuronal-keras-mnist
Keras es una libreria para python que facilita el uso de diferentes backends para machine learning.


para evitar conflictos, se puede usar un ambiente virtual exclusivo para proyectos con keras

```pip install virtualenv```

creamos el ambiente
```mkvirtualenv keras```

si mkvirtualenv no funciona, esto puede ayudar
``` source `which virtualenvwrapper.sh` ```

para trabajar en el ambiente virtual
```workon keras```     


#Instalar keras

instalar prerequisitos:

```$ pip install numpy scipy
$ pip install scikit-learn
$ pip install pillow
$ pip install h5py```

Se pueden usar varios backends como Tensor Flow o Theano

Instalar Theano (por facilidad de instalación)
```pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git```

Instalar keras
```pip install keras```

#Modificar el archivo de configuración keras.json

Keras usa por defecto Tensor Flow, para cambiar la configuración, primero, hay que llamar a keras haciendo un import, para que se cree el archivo keras.json y poder modificarlo.

```python
>>> import keras
>>> quit()```

ahora hay que editar el archivo, que debería estar en ~/.keras/keras.json  

y debería quedar algo así:
```{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow" }
```

from keras.datasets import mnist
import matplotlib.pyplot as plt
# carga (y descarga si es necesario) la base de datos MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Muestra 4 imagenes en escala de grises
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# muestra el plot
plt.show()
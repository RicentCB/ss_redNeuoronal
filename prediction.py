import numpy as np

import tensorflow as tf
from tensorflow import keras

# Path con imagenes
data_dir = './images/'

# Parametros para el "cargador"
batch_size = 32
img_height = 180
img_width = 180

# Obtener clases
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
class_names = train_ds.class_names

# RECREAR MODELO
model = keras.models.load_model('trainedModel.h5')

# PREDICCION DE NUEVOS DATOS
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "\n\nLa imgen pertenece a \"{}\" con un {:.2f} porcentaje de exactitud."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
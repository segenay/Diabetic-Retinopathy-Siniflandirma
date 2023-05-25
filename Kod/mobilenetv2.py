import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dense,MaxPool2D,Conv2D,Flatten,Dropout
from keras.utils import img_to_array
import cv2
import numpy as np
import pandas as pd

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Görüntüleri düzenleme işlemleri yapılır (eğitim için)
train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   )
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 subset='training',
                                                 class_mode = 'categorical',
                                                 )

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg' #modelin çıkışı sabit boyutlu bir vektör olur ve eğitim sürecinde daha az parametre kullanılır.
)

inputs = pretrained_model.input


x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output) #İlk katman 128 gizli nörona sahiptir ve ReLU aktivasyon fonksiyonu kullanılarak çıkış hesaplanır.
x = tf.keras.layers.Dense(128, activation='relu')(x) #İkinci katman, bir önceki katmanın çıktısını girdi olarak alır ve aynı şekilde 128 gizli nörondan oluşur.
outputs = tf.keras.layers.Dense(5, activation='softmax')(x) # Son katman ise 5 sınıf için softmax aktivasyonu kullanarak tam bağlı bir çıktı katmanıdır.

model = tf.keras.Model(inputs=inputs, outputs=outputs)
#yeni model oluşturulur, ön işlem görmüş girdileri alır ve yeni eklenen yoğun katmanlar tarafından sınıflandırılmış çıktılar verir.

''' Bir modeli compile özelliğini kullanarak derleyebiliriz.
Derleme için bir optimize edici ve bir kayıp fonksiyonu belirlememiz gerekir.
'''
optimizer = tf.keras.optimizers.legacy.Adam(lr = 0.0001) #lr(learn rate): Öğrenme katsayısı overfitting (aşırı uyum) i engellemek için kullanılan bir parametredir.
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy',tf.keras.metrics.Recall()]
)
'''
Optimizer:w değerlerinin iyileştirilmesi için kullanılan optimizasyon algoritmaları
loss: Her eğitimden sonra elde edilen değerler ile gerçek değerler arasındaki hata farkının hesaplanmasıdır.
metrics: Eğitim aşamasında her epoch sonrasında sonuçları değerlendirmek için bir sınama işlemi yapmaktadır.
Kullanılan “accuracy”, modelin başarısını inceleyebilmek için kullanılan yaygın bir metriktir.
'''

history = model.fit(
    training_set,
    validation_data=test_set,
    batch_size = 32,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Eğitim ve doğrulama başarısı')
plt.show()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Eğitim ve doğrulama hatası')
plt.show()


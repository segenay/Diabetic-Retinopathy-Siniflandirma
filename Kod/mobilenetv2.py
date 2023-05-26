import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dense,MaxPool2D,Conv2D,Flatten,Dropout
from keras.utils import img_to_array
from keras.callbacks import EarlyStopping
import cv2
import numpy as np
import pandas as pd

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   )
test_datagen = ImageDataGenerator(rescale = 1./255)
# ImageDataGenerator => veriden en iyi sekilde yararlanmak icin, rastgele donusumle veriyi artcriyoruz
# rescale => islemi kolaylastirmak icin 0-255 arasindaki degerleri 0-1 arasindaki degerler haline getirir.

# verilerin islenmesi icin uygun hale getirilmesi
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

early_stop = EarlyStopping(monitor="val_loss", patience = 3,restore_best_weights=True)
# izlenen metrik iyile�meyi durdurunca egitimi durdurulmali
# monitor => izlenemesi gerek miktar
# patience => egitimin durdurulacagi iyile�tirme olmayan epoch sayisi

history = model.fit_generator(training_set,callbacks=[early_stop], validation_data=test_set,epochs=15)
# fit_generator =>> gercek zamanli veri aktarimini saglar, bellege sigdirilmasi gereken cok b�y�k veri k�mesi oldugunda 
# training_set => alinacak egitim verisi
# test_set => dogrulama verileri, test verileri verildi
# epoch => modelin egitilip duruma gore agirliklarin g�ncellendigi her adim

# grafik cizdirme
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Egitim ve dogrulama basarisi')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.plot(history.history['loss'])
plt.plot( history.history['val_loss'])
plt.title('Egitim ve dogrulama hatasi')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()




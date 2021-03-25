# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 19:46:34 2018

@author: ATA
"""
#Kütüphane
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Verisetini yükleme. Bazı satır ve sütunları doğruluk artsın diye
#sildim
veriSeti=pd.read_csv("kalp_veriseti.csv",na_values=['.'])
veriSeti=veriSeti.sample(frac=1)

#kalp_hastaligi_varligi sütununu verinin diğerinden ayırdım.
values_series=veriSeti['kalp_hastaligi_varligi']
x_data=veriSeti.pop('kalp_hastaligi_varligi')

#Giriş(x) ve Çıkış(y) verilerini eğitim ve test olarak ayırdım.
train_x=veriSeti[0:100]
train_y=x_data[0:100]
train_x = train_x.values
train_y = train_y.values
test_x = veriSeti[100:]
test_y = x_data[100:]
test_x_ = test_x.values
test_y_ = test_y.values


#YSA oluşturdum.
model=Sequential()

#1. Gizli Katman
model.add(Dense(8,input_dim=13,activation="relu"))

#2. Gizli Katman
model.add(Dense(16,activation="relu"))

#3.Gizli Katman
model.add(Dense(32,activation="relu"))

#4.Gizli Katman
model.add(Dense(64,activation="relu"))


#Çıktı Katmanı(ya 0 ya 1 olabilir)
model.add(Dense(1,activation="sigmoid"))

#Özet
model.summary()

#Modeli çalıştırma
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Sinir Ağını Eğittim(Model Fitting)
history = model.fit(train_x,
                    train_y,
                    epochs=200,
                    batch_size=80, 
                    validation_data=(test_x, test_y)
                    )

#Batch Size eğimi tanımlar ve ne sıklıkla ağırlık güncelleneceğini belirtir.
#Epochs yüksek Batch Size düşük olması daha fazla avantaj sağlar.


#Sonuç
results = model.evaluate(test_x, test_y)
print(results)

# Doğruluk ve Kayıp İçin Grafikler
history_dict = history.history
history_dict.keys()
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Eğitim Kaybı')
plt.plot(epochs, val_loss, 'b', label='Geçerlilik Kaybı')
plt.title('Eğitim ve Doğruluk Kaybı')
plt.xlabel('Tur Sayısı')
plt.ylabel('Kayıp')
plt.legend()

plt.show()
plt.clf()   
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Eğitim Doğruluğu')
plt.plot(epochs, val_acc, 'b', label='Geçerlilik Doğruluğu')
plt.title('Eğitim ve Geçerlilik Doğruluğu')
plt.xlabel('Tur Sayısı')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

#Yapay Sinir Ağı Testi
sonuclar=model.evaluate(test_x,test_y)
print("Doğruluk: %.2f%%" %(sonuclar[1]*100))
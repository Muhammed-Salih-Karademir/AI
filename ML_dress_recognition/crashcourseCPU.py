import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import os

#GPU ile CPU arasında geçiş yapmak için eğer GPU'ya geçiş yapmak istersen alltaki kodu yorum satırı yap.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#mnist fashion datasetimizi indiriyoruz.
data = keras.datasets.fashion_mnist

#dataseti, eğitim ve etiketleri, test ve etiketleri olmak üzeri yüklüyoruz.
(train_images,train_labels), (test_images, test_labels) = data.load_data()

#etiketler 0 ile 9 rakamı arasında olduğu için
#rakamların hangi isme karşılık geldiğini bildiriyoruz
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#datasetteki resimleriminiz piksel değerleri 0 ile 255 arasında. Bu sebeble
#işimizi daha da kolaylaştırmak için her bir resimi 255e bölüyoruz.
#Böylelikle bütün resimlerimiz 0 ile 255 e karşılık gelen 0 ile 1 arasına map'lenmiş oluyor.

train_images = train_images / 255.0
test_images  = test_images  / 255.0
start_time = time.time()
#modelimi yaratırken yine kolaylık ve giriş katmanın hidden katmanlara bağlanması içifmain
#28x28lik veri setimizi düzleştirerek 1x758'lik hale getiriyoruz.
model = keras.Sequential([
    #giriş katmanımızı düzleştiriyoruz.
    keras.layers.Flatten(input_shape=(28,28)),
    #ilk hidden katmanımızı 128 norendan oluşturuoruz ve değerlerin aşırı büyümemesi içi
    #relu fonksiyonnu ekleyerek 0 ile 1 arasına getiriyoruz.
        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(256, activation="relu"),
    #çıkış katmanımız ise toplamda 10 nörondan oluşuyor.
        #TODO softmax araştırılacak****
    keras.layers.Dense(10, activation="softmax")

    ])
#bazı parametrelerle modelemizi init ediyoruz.
#TODO optimizer ve loss fonksiyonları araştırılacak****
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#modelimize eğitim resimlerini ve etiketleri veriyoruz ve 5 kez ağı döndürmesini istiyoruz.
model.fit(train_images,train_labels,epochs=5)

#eğittiğimiz modelin çıktılarını class_names içindeki değerler gibi vermek için bu kodu yazıyoruz.
prediction = model.predict(test_images)
#np.argmax fonksiyonu bir dizi içerisindeki en büyük değerin indexini döndürür.
#for döngüsü ile ilk 5 resmimizi ekrana taminleriyle beraber bastırıyoruz.
print("--- %s seconds ---" % (time.time() - start_time))
#for i in range(5):
#    plt.grid(False)
#    plt.imshow(test_images[i],cmap=plt.cm.binary)
#    plt.xlabel("Actual: " + class_names[test_labels[i]])
#    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
#    plt.show()

#eğittiğimiz modele eğitimin test edilmesi için test resimlerini ve etiketlerini veriyoruz.
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("tested Acc:",test_acc)

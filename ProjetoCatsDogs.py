
# # Projeto Cats and Dogs.ipynb 
# # Que estava no google Colab 
# # talvez não esteja mais lá https://colab.research.google.com/drive/110ehTpsjB63QKmAItWId3RxV_zwWwRfS#scrollTo=NEr5oL5Xc6V5 




# # Baixa as imagens
# !wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

# # dezipa em data
# !unzip kagglecatsanddogs_5340.zip -d /content/data

    
# # -----------------------------------------------------------------------------


# #imports
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from PIL import Image
# import numpy as np
# import shutil

# import os
# import shutil
# import random
# #from sklearn.model_selection import train_test_split

# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# # caminho para as pastas

# # caminho_imagens_processadas = "/content/data/imagens_processadas"
# # caminho_imagens_processadas_cat = "/content/data/imagens_processadas/Cat"
# # caminho_imagens_processadas_dog = "/content/data/imagens_processadas/Dog"
# caminho_train = "/content/data/train"
# caminho_test = "/content/data/test"

# # verefica se o caminho_imagens_processadas
# # if not os.path.exists(caminho_imagens_processadas):
# #     os.makedirs(caminho_imagens_processadas)

# def is_image(filename):
#     try:
#         Image.open(filename).verify()
#         return True
#     except:
#         return False
    
    
    
# # -----------------------------------------------------------------------------


# # import os
# # import shutil
# # import random
# # from sklearn.model_selection import train_test_split


# caminho_processadas_cat = "/content/data/PetImages/Cat"
# caminho_processadas_dog = "/content/data/PetImages/Dog"
# caminho_train = "/content/data/train"
# caminho_test = "/content/data/test"

# # Cria as pastas de trino e de teste dos cats e dos Dogs
# os.makedirs(os.path.join(caminho_train, 'Cat'), exist_ok= True)
# os.makedirs(os.path.join(caminho_train, 'Dog'), exist_ok= True)
# os.makedirs(os.path.join(caminho_test, 'Cat'), exist_ok= True)
# os.makedirs(os.path.join(caminho_test, 'Dog'), exist_ok= True)


# # Verifica se a imagem é uma imagem, retorna o tipo da imagem  cat ou dog
# def listar_imagens_tipo():
#     imagens_tipo = []

#     imagens_cat = os.listdir(caminho_processadas_cat)
#     for imagem in imagens_cat:
#         imagens_path = os.path.join(caminho_processadas_cat, imagem)
#         if os.path.isfile(imagens_path) and is_image(imagens_path):
#             imagens_tipo.append((imagens_path, 'Cat'))

#     imagens_dog = os.listdir(caminho_processadas_dog)
#     for imagem in imagens_dog:
#         imagens_path = os.path.join(caminho_processadas_dog, imagem)
#         if os.path.isfile(imagens_path) and is_image(imagens_path):
#             imagens_tipo.append((imagens_path, 'Dog'))

#     return imagens_tipo


    
# # -----------------------------------------------------------------------------


# # divide as imagens em treino e teste
# def dividir_imagens(imagens_tipo,slit_radio=0.8):
#     imagens_path, labels = zip(*imagens_tipo)

#     X_train, X_test, y_train, y_test = train_test_split(imagens_path, labels, test_size=1-slit_radio, random_state=42,shuffle=True)

#     for imagens_path, tipo in zip(X_train, y_train):
#       destino = os.path.join(caminho_train, tipo, os.path.basename(imagens_path))
#       if os.path.exists(imagens_path):
#         shutil.move(imagens_path, destino)
#       else:
#         print(f"A imagem {imagens_path} não existe.")

#     for imagens_path, tipo in zip(X_test, y_test):
#       destino = os.path.join(caminho_test, tipo, os.path.basename(imagens_path))
#       if os.path.exists(imagens_path):
#         shutil.move(imagens_path, destino)
#       else:
#         print(f"A imagem {imagens_path} não existe.")

    
# # -----------------------------------------------------------------------------

# imagens_tipo = listar_imagens_tipo()
# dividir_imagens(imagens_tipo)

    
# # -----------------------------------------------------------------------------

# # import tensorflow as tf
# # from tensorflow.keras import layers, models
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator


# train_dir = '/content/data/train'
# test_dir = '/content/data/test'

# barch_size = 32
# img_height = 128
# img_width = 128

# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width),
#     batch_size=barch_size,
#     class_mode='binary'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_height, img_width),
#     batch_size=barch_size,
#     class_mode='binary'
# )
    
# # -----------------------------------------------------------------------------

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# # Construir o modelo de CNN // o que é CNN    é a rede neurtal?
# model = Sequential()

# #Camada de entrada de convolução e ReLU
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3))) #1
# model.add(MaxPooling2D((2, 2)))

# #Camada de entrada de convolução e pooling
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))

# #Vamada Flatten para converter 2D em Vetor
# model.add(layers.Flatten())

# #Camada densa com ReLU
# model.add(layers.Dense(64, activation='relu'))

# #Camada de saída com ativação sigmoid
# model.add(layers.Dense(1, activation='sigmoid'))

# #Compilação do modelo
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# #Treinamneto do modelo
# history = model.fit(train_generator, steps_per_epoch = train_generator.samples // barch_size,
#                     epochs=10, #10
#                     validation_data=test_generator,
#                     validation_steps=test_generator.samples // barch_size)

# model.save('/content/pets/modelo_CatsAndDogs.h5')


    
# # -----------------------------------------------------------------------------

# # Avaliar o modelo
# # test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
# # print(f"Test Accuracy: {test_accuracy:.2f}")

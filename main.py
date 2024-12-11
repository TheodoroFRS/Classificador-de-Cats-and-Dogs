import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# pip install streamlit pandas tensorflow numpy

# Congiguração da página
st.set_page_config(page_title="Classificador de Cats e Dogs", page_icon="😺🐶")

#Título
st.title("Classificador de Cats😺 e Dogs🐶")
st.write("Faça o upload da imagem para encaxar em uma classificação entre Cats😺 e Dogs🐶")

#Carregar o modelo treinado
model_path = "modelo_CatsAndDogs.h5"
model = load_model(model_path)


#Função para processar a imagem
def process_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize((128,128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Componente para upload da imagem
uploaded_image = st.file_uploader("Envie uma imagem de um gato ou cachorro", type=["jpg","jpeg","png"])

# Inferência e exibição de resultado
if uploaded_image is not None:
    # Mostrar a imagem carregada
    st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)

    # Processar a imagem
    img_array = process_image(uploaded_image)

    # Realizar a predição
    prediction = model.predict(img_array)
    class_name = "Dog🐶" if prediction >= 0.5 else "Cat😺"
    confidence = prediction[0][0] if prediction >= 0.5 else 1 - prediction [0][0]

    # Exibir o resultado
    st.write(f"## Resultado: **{class_name}**")
    st.write(f"Confiança: **{confidence * 100:.2f}**")

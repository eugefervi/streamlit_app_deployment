

#C:\Users\eugef\Documents\geno\empresa\glowup\MVPs\subtono\streamlit_app\env\Scripts

#C:\Users\eugef\Documents\geno\empresa\glowup\MVPs\subtono\streamlit_app\env\project\scripts

#streamlit run app3.py



import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image
import time
import urllib.request

logo_silbon = 'https://www.silbonshop.com/on/demandware.static/-/Library-Sites-SilbonSharedLibrary/es/dw8ce524c4/logo.svg'
base_url = "https://www.silbonshop.com"

database = pd.read_csv("silbon_products.csv")
del database['Unnamed: 0']
database.columns = ['Product URL', 'Product Name', 'Selling Price', 'Discount',
       'Image URLs', 'cathegory', 'base', 'genero', 'subcathegory', 'producto']
len(database['producto'].unique())
# seleccionamos los productos de interés porque no tiene sentido
# decirle a la gnte que seleccione monederos también. Eso está bien
# tenerlo para la combinación de productos y completar el look
interes = database[(database['subcathegory']=='Ropa')|
                   (database['producto']=='Corbatas y pajaritas')|
                   (database['producto']=='Gafas de sol y cordones')|
                   (database['producto']=='Gafas de sol')|
                   (database['producto']=='Sombreros y gorras')]


opciones = list(interes['producto'].unique())

# Cargamos el modelo de IA preentrenado para identificar el género
# Utilizamos el modelo MobileNetV2 con transfer learning
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def main():
    
    st.columns(3)[1].markdown(f"[![Foo]({logo_silbon})]({base_url})")
    #st.columns(3)[1].image(logo_silbon, width=200)

    st.title("Descubre lo que te favorece :wink:")

    # Variable de estado para controlar el flujo de la aplicación
    uploaded_image = pantalla1()
    if 'state' not in st.session_state:
        st.session_state.state = 'pantalla1'
    if st.session_state.state == 'pantalla2':
        pantalla2()

    if st.session_state.state == 'pantalla3':
        pantalla3(uploaded_image, opciones)

def pantalla1():
    st.subheader("Toma una foto")
    st.write("Asegúrate de que haya buenas condiciones de luz :sun_with_face: y de no estar maquillad@ :no_entry_sign:")

    #image = st.file_uploader("Sube una imagen o toma una foto", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key='file_uploader')
    image = st.camera_input("Take a picture")
    
    if image is not None:
        #display_main_image(image, caption="Imagen subida", link = False)
        if st.button("Analizar"):
            st.session_state.state = 'pantalla2'
        return image

def pantalla2():
    with st.spinner("Tu imagen está siendo analizada..."):
        time.sleep(1)  # Simular el análisis de la imagen
    #st.success("¡Análisis completado!")
    st.session_state.state = 'pantalla3'

def pantalla3(uploaded_image, opciones):
    st.success("¡Análisis completado!")

    st.subheader("Elige los productos que buscas")
    st.write("Te quedarán ideales ;)")
    
    # Opciones de selección
    seleccion = st.multiselect("Selecciona los productos que buscas:", opciones)

    if seleccion:
        st.subheader("Resultados")
        #st.write("Productos seleccionados:")
        #for producto in seleccion:
        #    st.write(f"- {producto}")
        st.write("Productos encontrados en la base de datos:")
        filtered_database = database[database['producto'].isin(seleccion)]
        if uploaded_image is not None:
            gender = identify_gender(uploaded_image)
            filtered_database = filtered_database[filtered_database['genero'] == gender]

            # Dividir los elementos en columnas de 3 elementos
            cols = st.columns(3)
            for idx, (_, row) in enumerate(filtered_database.iterrows()):
                # vamos a revisar el tamaño de las imagenes para que tengan todas el mismo y no se decuadre
                urllib.request.urlretrieve(row['Image URLs'], "gfg.png")
                img = Image.open("gfg.png")
                if img.size == (256, 341):
                    with cols[idx % 3]:
                        st.markdown(f"[![Foo]({row['Image URLs']})]({row['Product URL']})")
                        #st.markdown("[![Foo](http://www.google.com.au/images/nav_logo7.png)](http://google.com.au/)")
    
                        #display_main_image(row['imagen'], caption=row['producto'], link = 'https://www.zara.com/es/')
                        #display_external_image(row['imagen'], link = row['Image URLs'])

def identify_gender(image):
    # Preprocesamiento de la imagen para la clasificación de género
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # Identificación del género
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    gender = 'Mujer' if decoded_predictions[0][0][1] in ['lipstick', 'wig'] else 'Hombre'

    return gender

def display_main_image(image, caption, link = False):
    # Descargar y redimensionar la imagen para mostrarla en la aplicación
    img = Image.open(image).convert('RGB')
    img.thumbnail((300, 300))  # Redimensionar la imagen
    if link == False:
        st.image(img, caption=caption, use_column_width=False)
    else:
        st.image(img, caption=caption, use_column_width=False)
        st.markdown(f"[![Link]({img})]({link})", unsafe_allow_html=True)

def display_external_image(img, link = False):
    st.markdown(f"[![Foo]({img})]({link})")


if __name__ == "__main__":
    main()

    
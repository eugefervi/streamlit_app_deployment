
# on premise
#C:\Users\eugef\Documents\geno\empresa\glowup\MVPs\subtono\streamlit_app\env\Scripts

#C:\Users\eugef\Documents\geno\empresa\glowup\MVPs\subtono\streamlit_app\env\project\scripts\streamlit_app_deployment

#streamlit run app3.py

# git bash
# C:/Users/eugef/Documents/geno/empresa/glowup/MVPs/subtono/streamlit_app/env/project/scripts/streamlit_app_deployment



import streamlit as st
import pandas as pd
from PIL import Image
import time
import urllib.request
import gender_detection as gd


logo_silbon = 'https://www.silbonshop.com/on/demandware.static/-/Library-Sites-SilbonSharedLibrary/es/dw8ce524c4/logo.svg'
base_url = "https://www.silbonshop.com"
products_path = "silbon_products.csv"

@st.cache_data
def read_clean_dataframe(products_path):
    database = pd.read_csv(products_path)
    del database['Unnamed: 0']
    database.columns = ['Product URL', 'Product Name', 'Selling Price', 'Discount',
           'Image URLs', 'cathegory', 'base', 'genero', 'subcathegory', 'producto']
    return database
    
@st.cache_data
def filter_dataframe(database):
    
    filtered_df = database[(database['subcathegory']=='Ropa')|
                       (database['producto']=='Corbatas y pajaritas')|
                       (database['producto']=='Gafas de sol y cordones')|
                       (database['producto']=='Gafas de sol')|
                       (database['producto']=='Sombreros y gorras')]
    return filtered_df



def main():
    
    st.columns(3)[1].markdown(f"[![Foo]({logo_silbon})]({base_url})")
    #st.columns(3)[1].image(logo_silbon, width=200)

    st.title("Descubre lo que te favorece :wink:")

    st.subheader("Sácate una foto")
    st.write("Asegúrate de que haya buenas condiciones de luz :sun_with_face: y de no estar maquillad@ :no_entry_sign: para obtener un resultado más preciso")

    #image = st.file_uploader("Sube una imagen o toma una foto", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key='file_uploader')
    image = st.camera_input("")
    if image is not None:
        pantalla1(image)
        if 'state' not in st.session_state:
            st.session_state.state = 'pantalla1'
        if st.session_state.state == 'pantalla2':
            pantalla2()
    
        if st.session_state.state == 'pantalla3':
            pantalla3(image, products_path)
    
    else:
        st.session_state.state = 'state'


def pantalla1(image):

    if image is not None:
        #display_main_image(image, caption="Imagen subida", link = False)
        if st.button("Analizar"):
            st.session_state.state = 'pantalla2'

def pantalla2():
    with st.spinner("Tu imagen está siendo analizada..."):
        time.sleep(1)  # Simular el análisis de la imagen
    #st.success("¡Análisis completado!")
    st.session_state.state = 'pantalla3'

def pantalla3(uploaded_image, products_path):
    st.success("¡Análisis completado!")

    st.subheader("Selecciona los productos que deseas")
    database = read_clean_dataframe(products_path)
    
    #Hacemos que se muestren únicamente los productos del sexo de la persona que se hace el servicio
    gender = gd.identify_gender(uploaded_image)
    
    interes = filter_dataframe(database)
    
    opciones = list(interes['producto'][interes['genero']==gender].unique())
    # Opciones de selección
    seleccion = st.multiselect("", opciones)

    if seleccion:
        st.subheader("Resultados")

        st.write("A continuación se mostrarán los productos que más te favorocen en función del tono de tu piel")
        st.write("Pinchando en los productos podrás comprarlos directamente en la página web")
        st.write("Te quedarán ideales :dancer:")

        filtered_database = database[database['producto'].isin(seleccion)]
        if uploaded_image is not None:
            
            filtered_database = filtered_database[filtered_database['genero'] == gender]

            # Dividir los elementos en columnas de 3 elementos
            cols = st.columns(3)
            for idx, (_, row) in enumerate(filtered_database.iterrows()):
                # vamos a revisar el tamaño de las imagenes para que tengan todas el mismo y no se decuadre
                try:
                    urllib.request.urlretrieve(row['Image URLs'], "gfg.png")
                    img = Image.open("gfg.png")

                    if img.size == (256, 341):
                        with cols[idx % 3]:
                            st.markdown(f"[![Foo]({row['Image URLs']})]({row['Product URL']})")
                except:
                    pass
                    


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

    


#C:\Users\eugef\Documents\geno\empresa\glowup\MVPs\subtono\streamlit_app\env\Scripts

#C:\Users\eugef\Documents\geno\empresa\glowup\MVPs\subtono\streamlit_app\env\project\scripts\streamlit_app_deployment

#streamlit run app3.py

# C:/Users/eugef/Documents/geno/empresa/glowup/MVPs/subtono/streamlit_app/env/project/scripts/streamlit_app_deployment



import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image
import time
import urllib.request
import cv2
import numpy as np


logo_silbon = 'https://www.silbonshop.com/on/demandware.static/-/Library-Sites-SilbonSharedLibrary/es/dw8ce524c4/logo.svg'
base_url = "https://www.silbonshop.com"

database = pd.read_csv("silbon_products.csv")
del database['Unnamed: 0']
database.columns = ['Product URL', 'Product Name', 'Selling Price', 'Discount',
       'Image URLs', 'cathegory', 'base', 'genero', 'subcathegory', 'producto']
len(database['producto'].unique())
# seleccionamos los productos de interés porque no tiene sentido
# decirle a la gnte que seleccione, monederos también. Eso está bien
# tenerlo para la combinación de productos y completar el look

def filter_dataframe(database):
    filtered_df = database[(database['subcathegory']=='Ropa')|
                       (database['producto']=='Corbatas y pajaritas')|
                       (database['producto']=='Gafas de sol y cordones')|
                       (database['producto']=='Gafas de sol')|
                       (database['producto']=='Sombreros y gorras')]
    return filtered_df


#opciones = list(interes['producto'].unique())

# Cargamos el modelo de IA preentrenado para identificar el género
# Utilizamos el modelo MobileNetV2 con transfer learning
model = tf.keras.applications.MobileNetV2(weights='imagenet')


# The gender model architecture
# https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
GENDER_MODEL = 'weights/deploy_gender.prototxt'
# The gender model pre-trained weights
# https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
GENDER_PROTO = 'weights/gender_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the gender classes
GENDER_LIST = ['Hombre', 'Mujer']
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "weights/deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"


# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

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
        pantalla3(uploaded_image, database)

def pantalla1():
    st.subheader("Sácate una foto")
    st.write("Asegúrate de que haya buenas condiciones de luz :sun_with_face: y de no estar maquillad@ :no_entry_sign: para obtener un resultado más preciso")

    #image = st.file_uploader("Sube una imagen o toma una foto", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key='file_uploader')
    image = st.camera_input("")
    
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

def pantalla3(uploaded_image, database):
    st.success("¡Análisis completado!")

    st.subheader("Selecciona los productos que deseas")
    
    #Hacemos que se muestren únicamente los productos del sexo de la persona que se hace el servicio
    gender = identify_gender(uploaded_image)
    
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



def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def display_img(title, img):
    """Displays an image on screen and maintains the output until the user presses a key"""
    # Display Image on screen
    cv2.imshow(title, img)
    # Mantain output until user presses a key
    cv2.waitKey(0)
    # Destroy windows when user presses a key
    cv2.destroyAllWindows()
    
def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)


def identify_gender(image):
    """Predict the gender of the faces showing in the image"""
    # To read image file buffer with OpenCV:
    bytes_data = image.getvalue()    
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # img = cv2.resize(img, (frame_width, frame_height))
    # Take a copy of the initial image and resize it
    #frame = img.copy()
    if cv2_img.shape[1] > 400:
        cv2_img = image_resize(cv2_img, width=400)
    # predict the faces
    faces = get_faces(cv2_img)
    # Loop over the faces detected
    # for idx, face in enumerate(faces):
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = cv2_img[start_y: end_y, start_x: end_x]
        # image --> Input image to preprocess before passing it through our dnn for classification.
        # scale factor = After performing mean substraction we can optionally scale the image by some factor. (if 1 -> no scaling)
        # size = The spatial size that the CNN expects. Options are = (224*224, 227*227 or 299*299)
        # mean = mean substraction values to be substracted from every channel of the image.
        # swapRB=OpenCV assumes images in BGR whereas the mean is supplied in RGB. To resolve this we set swapRB to True.
        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        #gender_confidence_score = gender_preds[0][i]
        return gender



if __name__ == "__main__":
    main()

    
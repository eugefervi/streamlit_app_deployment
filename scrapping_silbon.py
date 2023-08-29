# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:08:33 2023

@author: eugef
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = "https://www.silbonshop.com"


# Intentamos coger todas las urls de productos de la página web para luego
# meterlas en un bucle y sacar todos los productos
url = "https://www.silbonshop.com/es_es/home"
res = requests.get(url)
sou = BeautifulSoup(res.content, "html.parser")
links = sou.find_all("a")
len(links)
links = [link['href'] for link in links]

# Eliminamos valores duplicados
links = list(set(links))
links = {'links':links}
links = pd.DataFrame(links)

# Los procesamos para que estén bien y nos quedamos solo con los buenos
arrange = base_url+'/es'
links['links'] = links['links'].str.replace('^/es', arrange, regex=True)
links = links[links['links'].str.contains(base_url)]
links = links['links'].drop_duplicates().to_frame()
links = links.reset_index(drop=True).to_frame()

# Ahora pasamos a coger todos los productos de los links de silbon
products = pd.DataFrame()
for url in links['links']:
    try:
    
        # Realizar la solicitud GET a la página web
        response = requests.get(url)
        
        # Parsear el contenido HTML con BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Obtener la categoría del producto
        final_cathegory = soup.find("div", class_="category-breadcrumb")
        final_cathegory = final_cathegory.find("div", class_="category-crumb")
        final_cathegory = final_cathegory.find("span", class_="category-name").text.strip()
        
        path_cathegory = soup.find_all("a", class_="category-name")
        final_path = ''
        for path in path_cathegory:
            temp = path.text.strip()
            final_path = temp + '/' + final_path
            #print(final_path)
        
        cathegories = final_path + final_cathegory
        
        # Encontrar todos los elementos que contienen información de productos
        product_items = soup.find_all("div", class_="product")
        
        # Lista para almacenar la información de los productos
        image_urls = []
        product_urls = []
        product_names = []
        selling_prices = []
        dctos = []
        
        # Iterar a través de los elementos de los productos
        for product_item in product_items:
            # Obtener la URL del producto
            product_url = product_item.find("div", class_="image-container")
            product_url = product_url.find("a")['href']
            product_url = base_url+product_url
            product_urls.append(product_url)
            #print(product_url)
            
        
            # Obtener el nombre del producto
            product_name = product_item.find("div", class_="tile-body")
            product_name = product_name.find("a", class_="link").text.strip()
            product_names.append(product_name)
            #print(product_name)
            
         
            # Obtener el precio original y rebajado (si está disponible)
            price = product_item.find("div", class_="price")
            selling_price = float(price.find("span", class_="value")['content'])
            try:
                dcto = float(price.find("span", class_="dto").text.strip()[1:-1])
            except:
                dcto = 0
            selling_prices.append(selling_price)
            dctos.append(dcto)
            #print(selling_price)
            #print(dcto)
        
            
            
            # Obtener las imágenes del producto
            try:
                images = product_item.find("img", class_='tile-image')['data-src']
                #print(images)
                #print('#####################################siguiente')
            except:        
                res = requests.get(product_url)
                sou = BeautifulSoup(res.content, "html.parser")
                images = sou.find("div", class_='primary-images')
                images = images.find("img")['data-src']
                #print(images)
                #print('#####################################siguiente')  
            image_urls.append(images)
            
                
            # Almacenar la información en un diccionario
            data = {
                "Product URL": product_urls,
                "Product Name": product_names,
                "Selling Price": selling_prices,
                "Discount": dctos,
                "Image URLs": image_urls,
                "cathegory" : cathegories
            }
            
            
        print(url)
        temp = pd.DataFrame(data)
        products = pd.concat([products, temp], ignore_index=True)
        print(len(products))
    except:
        print(url)
        pass

# Procesamos y clasificamos el dataframe para que no haya repetidos y
# podamos sacar las categorías

products = products[~products['cathegory'].str.contains('todo')]
products = products[~products['cathegory'].str.contains('rebajas', case = False)]
products = products[~products['cathegory'].str.contains('Colecciones', case = False)]
products = products[~products['cathegory'].str.contains('Novedades', case = False)]
products = products[~products['cathegory'].str.contains('Novedades', case = False)]
products = products[products['cathegory']!='Home/Mujer']
products = products[products['cathegory']!='Home/Hombre']
products = products[products['cathegory']!='Home/Niño']
products = products[products['cathegory']!='Home/Co-creación']

products[["1","2","3","4"]] = products["cathegory"].str.split("/", expand = True)
products.dropna(inplace=True)

products.columns = ['Product URL', 'Product Name', 'Selling Price', 'Discount',
       'Image URLs', 'cathegory', '1', 'genero' , '3', 'producto']

products.to_csv('scripts/silbon_products.csv')

PROYECTO INDIVIDUAL 01 
MACHINE LEARNING OPERATIONS (MLOps)

Abraham Jorge Fernando

CONTEXTO

Se solicita colocarse en el rol de un Data Scientist, desarrollando un conjunto de datos en crudo
obtenidos a partir de tres(03) DataSets en formato Json, los cuales son datos sobre la plataforma Steam. A continuacion se 
explica la metodologia utilizada en cada archivo, con el objetivo de implementar los conocimientos adquiridos y asi obtener
un Minimun Viable Product(MVP) para poder administrar consultas de forma localmente y remotamente, a partir de la implementacion
de API deployada en un servicio en la nube, y la aplicacion de un modelo de Maching Learning.


DATOS

Para este proyecto se proporcionaron tres archivos JSON:

australian_user_reviews.json es un dataset que contiene los comentarios que los usuarios realizaron sobre los juegos que 
consumen, además de datos adicionales como si recomiendan o no ese juego, emoticones de gracioso y estadísticas de si el 
comentario fue útil o no para otros usuarios. También presenta el id del usuario que comenta con su url del perfil y el 
id del juego que comenta.
australian_users_items.json es un dataset que contiene información sobre los juegos que juegan todos los usuarios, así 
como el tiempo acumulado que cada usuario jugó a un determinado juego.
output_steam_games.json es un dataset que contiene datos relacionados con los juegos en sí, como los título, el 
desarrollador, los precios, características técnicas, etiquetas, entre otros datos.

En el documento Diccionario de datos se encuetran los detalles de cada una de las variables de los conjuntos de datos.



EXTRACCION, TRANSFORMACION Y CARGA DE DATOS (ETL), ANALISIS EXPLORATORIO DE LOS DATOS(EDA)

Cabe destacar que en este proceso, se tomo la decision de realizar el analisis exploratorio en conjunto con la transformacion
y limpieza de datos.

ETL_steam_games.ipynb

Se importan librerias a utilizar en el procesamiento de los datos, abriendo el archivo en formato de un DataFrame, realizando
una vista previa de la cabecera, eliminando datos nulos que no se consideraron importantes, como asi tambien la eliminacion
de columnas que no se utilizaran para el resultado final.
Se observa una columna comprimida en formato json, la cual es descomprimida para su analisis, aplicando "dummies" lo cual
consiste en que por cada genero, se coloca un marcador numerico del 0 o 1 para cada juego, siendo mas facil asi para
luego su tratamiento y procesamiento. 
Se verifican columnas y formato del nuevo archivo, y se procede a guardarlo en un .csv.


ETL_users_reviews.ipynb

Se comienza por la importacion de librerias a utilizar, luego de la apertura del archivo, se procede a la exploracion 
del mismo observando que podria contener datos duplicados. En un analisis mas profundo sobre esta situacion, se 
llega a la conclusion de no borra estas entradas duplicadas, ya que son datos repetidos por los mismos usuarios, pero 
que hacen referencia a diferentes opiniones sobre diferentes juegos.
Se observa la columna "reviews", la cual contiene datos anidados en formato json, convirtiendola en un dataframe
para su posterior analisis. Se procede a descomprimir dicha columna, adjuntando a cada usuario con su propio comentario
desanidado para no perder diccho dato de vital importancia.
Se eliminan columnas innecesarias para el Analisis de Sentimientos que se pretende aplicar, trabajando sobre un codigo
el cual a partir de un modelo de ML, analiza el comentario y coloca un dato numerico, siendo este 0 para comentarios
Negativos, 1 para comentarios Neutrales, y 2 comentarios Positivos; como tambien asi 1 para la faltante de comentarios
o reviews.
Se procede a trabajar sobre la columna Fecha, dandole el formato correspondiente para su correcto uso posteriormente. Se 
verifica el dataframe final y se guarda el archivo en formato .csv.



ETL_users_items.ipynb

Luego de la importacion de librerias a utilizar y su apertura correspondiente del archivo, se verifica el contenido
del dataframe creado, eliminando las columnas que se consideran irrelevantes para su posterior trabajo. 
Se verifican estructura del dataframe y la informacion  correspondiente, y se toma la decision de eliminar datos
de entradas duplicadas tomando como referencia las columnas "item_name" y "user_id", ya que las mismas nos proporcionaron
que hay datos repetidos, pero corresponden a diferentes items. Se guarda el dataset en formato .csv.


mergeDataSets.ipynb

En este codigo, se pretende juntar los datasets anteriormente trabajados en un unico datasets.
Luego de la apertura de los archivos en diferentes Dataframes, se analiza y se prepara uno de los archivos, 
renombrando columnas que son importantes para su union o merged, como asi tambien la correcta estructuracion 
de los tipos de datos. Se procede a la union, verificando visualmente su resultado. Se continua con 
la eliminacion de columnas irrelevantes, como asi tambien una reestructuracion en el ordenamiento
de las columnas para su correcta lectura. Posteriormente, se toma la decision de la eliminacion de datos
de las filas, cuyas horas jugadas superen las 100 en la columna "playtime_forever"; esto se realiza
debido a que en la union de los diferentes dataframes, el archivo ha quedado muy grande para su 
posterior procesamiento, entonces, con esta accion nos aseguramos de reducir el tamaño del dataset.
Se continua con la revision del tercer dataframe para su union con el conjunto anterior, modificando
columnas que se consideran importantes para la union. Por cosiguiente, se reestructura los nombres y 
posiciones de las columnas, dejando asi el dataset en un formato limpio y legible para su uso.
IMPORTANTE, para su posterior uso, se decide la reduccion de dicho datasets final, ya que para
su carga en diferentes plataformas como ser GitHub y Render, no nos va a permitir el tamaño de dicho
archivo y procesamiento; es por ello que se toma la decision de reducir lo mas eficazmente posible
la cantidad de filas del conjunto, quedandonos con un Dataframe de 2000 filas y 30 columnas.
Se procede por ultimo al guardado final en formato .csv.


EDA_df_reducido_final.ipynb

Luego de la importacion de las librerias a utilizar para dicho analisis, se trabaja sobre el archivo anteriormente
guardado pero NO REDUCIDO, esto se debe a que en la reduccion corresponde unicamente para el posterior
trabajo y consultas de las plataformas.
En un analisis exploratorio inicial, procedemos a verificar la estructura del dataframe, observando su tamaño
y formatos de columnas. Se observan vaores nulos y se implemeta el cambio de esos valores correspondientes a la columna de
"developer" por el string "N/D" (No Data).
Se trabaja sobre un dataframe nuevo con la columa "playtime_forever", para la busqueda de posibles "Outliers", se verifica
los minimos y maximos de dicho dataframe ya que los datos en ella correspnden a horas jugadas. Se procede a realizar
operaciones para la division en quertiles de los datos, creando una plantilla o mascara para su posterior comapracion.
Como resultado de dicho analisis, se observa que no se encuentran datos Outliers que puedan influir en nuestro posterior
uso. Igualmente, se procede a graficar los datos obtenidos para su correcta verifcacion visual mediante  dos (02) graficos.
Posteriormente, se continua con el analisis de la columna "posted_year", la cual es relevante para el tratamiento de los
datos. Se convierte en un dataframe para su trabajo, verificando valores atipicos, y graficando relacionando las
cantidades de publicaciones ppr año.
Finalmente, se trabaja sobre la columna de "dummies" creados, analizando su correcta estructura en los nombres de las
columnas, y graficando por la cantidad de jugadores para cada categoria de juego.


main.py

En el siguiente codigo, se trabaja sobre las consignas del trabajo solicitado, con la creacion de los EndPoints solicitados
que van a ser consumidos mediante el deploy localmente con FastApi y remotamente con Render.
def message: muestra un mensaje sobre el trabajo tratado, con el nombre del creador de dicho proyecto.
def PlayTimeGenre: devuelve el año, con mas horas jugadas para un jugador en la categoria de juego que se le es pasado 
por parametro.
def UserForGenre: retorna el usuario que mas horas jugo para un genero especifico pasado como parametro.
UsersRecommend: a partir del año ingresado como entrada, y del Sentiment_Score busca los desarrolladores
mas recomendados por los usuarios.
def UserWorstDeveloper: recibe como parametro de entrada el año deseado, realizando una busqueda por años y por el 
sentiment_score, y devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado.
def sentiment_analysis: recibe como parametro de entrada la empresa desarrolladora, y devuelve un diccionario con el 
nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se 
encuentren categorizados con un análisis de sentimiento como valor.

A continuacion, se procede a crear el modelo de Machine Lerning, y a entrenarlo con el dataset ya cargado, para su 
posterior consumo mediante consultas relacionadas.
def recomendar_juego: recibe como parametro de entrada, la ID de un juego, y mediante su entrenamiento ya realizado
nos devuelve una lista con juegos relacionados a la categoria.



Ademas de los codigos antes mencionados, que se encuentran alojados en una carpeta y clasificados, en el conjunto
de archivos se encuentran para su observacion los datasets comprimidos(para ahorrar espacio de uso en las plataformas),
archivo Propuestas.txt que contienen las consignas reales para dicha planificacion del proyecto, archivo
requerements.txt el cual contiene las librerias a utilizar para el correcto deployment en plataformas, y archivo
df_reducido_final.csv que es con el cual se realizan todas las consultas y el entrenamiento del Modelo de Machine Learning.






 
















































































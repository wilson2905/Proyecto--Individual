from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(debug=True)
df_merged=pd.read_csv("df_reducido_final.csv", encoding="utf-8")


@app.get('/')
def message():
    return 'PROYECTO INTEGRADOR ABRAHAM JORGE FERNANDO'

##Creacion de los endpoints

@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str) -> dict:
    """ 
    Obtiene el año con más horas jugadas para un género específico. 

    Genre: Genero a obtener el número de años de lanzamiento con más horas, ejemplos de parametros
    "Action", "Casual", "Education"

    Returns:
        Año con más horas jugadas para un género específico teniendo en cuenta los generos disponibles.
    """
    genre = genre.capitalize()
    if genre not in df_merged.columns:
        return {"Error": f"Género {genre} no encontrado en el dataset."}
    else:
        genre_df = df_merged[df_merged[genre] == 1]
        year_playtime_df = genre_df.groupby('posted_year')['playtime_forever'].sum().reset_index()
        max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'posted_year']
        return {"Género": genre, f"Año de lanzamiento con más horas jugadas para Género {genre} :": int (max_playtime_year)}




@app.get("/UserForGenre/{genero}")
def UserForGenre(genero):
    """
        Recibe un genero de juego como entrada, ejemplos de parametros
        "Action", "Casual", "Education"
        Retorna el usuario que acumula mas horas jugadas para el genero dado como parametro, y una 
        lista de la acumulacion de horas jugadas por año
        
    """
    
    
    # Se filtra el DataFrame por el género dado
    genero=genero.capitalize()
    genre_df = df_merged[df_merged[genero] == 1]  # SE selecciona las filas donde el género tiene valor 1
    
    if genre_df.empty:
        return "No se encontraron datos para el género proporcionado"
    
    # SE busca el usuario con más horas jugadas para el género
    usuario_mas_horas = genre_df.groupby('user_id')['playtime_forever'].sum().idxmax()
    
    # Calcula la acumulación de horas jugadas por año para ese género
    horas_por_anio = genre_df.groupby('posted_year')['playtime_forever'].sum().reset_index()
    horas_por_anio = horas_por_anio.rename(columns={'posted_year': 'Año', 'playtime_forever': 'Horas'})
    horas_por_anio = horas_por_anio.to_dict(orient='records')
    
    # SE crea el diccionario de retorno
    retorno = {
        "Usuario con más horas jugadas para " + genero: usuario_mas_horas,
        "Horas jugadas": horas_por_anio
    }
    
    return retorno





@app.get("/UsersRecommend/{anio}")
def UsersRecommend(anio:int):
    """A partir del año ingresado como entrada, y del Sentiment_Score busca los desarrolladores
    mas recomendados por los usuarios
    ejemplo de parametro de entrada 2010, 2011, 2012, 2013..

    Devuelve el top 3 de juegos MÁS recomendados por usuarios 
    para el año dado
    """
    
    
    # Filtracion de las revisiones para el año dado y que sean recomendadas
    revisiones_anio = df_merged[(df_merged['posted_year'] ==(anio)) & (df_merged['recommend'] == True)]

    # Filtracion de las revisiones con comentarios positivos o neutrales (sentiment_score 1 o 2)
    revisiones_positivas_neutrales = revisiones_anio[df_merged['sentiment_score'].isin([1, 2])]

    # se cuenta las recomendaciones por juego
    juegos_recomendados = revisiones_positivas_neutrales['item_name'].value_counts().head(3)

    # Creacion de la lista de los juegos más recomendados
    top_3_juegos_recomendados = []
    for puesto, (juego, _) in enumerate(juegos_recomendados.items(), 1):
        top_3_juegos_recomendados.append({"Puesto " + str(puesto): juego})

    return top_3_juegos_recomendados




@app.get("/UsersWorstDeveloper/{anio}")
def UsersWorstDeveloper(anio:int):
    """
    Recibe como parametro de entrada el año deseado, realizando una busqueda por años y por el 
    sentiment_score
    ejemplo de parametro de entrada 2010, 2011, 2012, 2013..
    
     Devuelve el top 3 de desarrolladoras con juegos MENOS 
    recomendados por usuarios para el año dado.
    """
    
    # Filtracion de las revisiones para el año dado y que no sean recomendadas
    revisiones_anio = df_merged[(df_merged['posted_year'] == anio) & (df_merged['recommend'] == False)]

    # Filtracion de las revisiones con comentarios negativos (sentiment_score 0)
    revisiones_negativas = revisiones_anio[df_merged['sentiment_score'] == 0]

    # se cuenta las no recomendaciones por desarrolladora
    desarrolladoras_no_recomendadas = revisiones_negativas['developer'].value_counts().head(3)

    # se Crea la lista de las desarrolladoras menos recomendadas
    top_3_developers_no_recomendadas = []
    for puesto, (developer, _) in enumerate(desarrolladoras_no_recomendadas.items(), 1):
        top_3_developers_no_recomendadas.append({"Puesto " + str(puesto): developer})

    return top_3_developers_no_recomendadas




@app.get("/sentiment_analysis/{empresa_desarrolladora}")
def sentiment_analysis(empresa_desarrolladora):
    
    """ recibe como parametro de entrada la empresa desarrolladora,
    se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con
    la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con 
    un análisis de sentimiento como valor.
    
    ejemplos de nombres de entrada Valve, dev4play, BlueLine Games
    """
    
    empresa_desarrolladora=empresa_desarrolladora.capitalize()
    # se filtra las revisiones por la empresa desarrolladora especificada
    revisiones_empresa = df_merged[df_merged['developer'] == empresa_desarrolladora]

    # conteo de los registros por análisis de sentimiento
    sentimiento_por_empresa = revisiones_empresa['sentiment_score'].value_counts()

    #creacion del diccionario con los resultados del análisis de sentimiento
    resultado_analisis_sentimiento = {empresa_desarrolladora: [
        f"Negative = {sentimiento_por_empresa.get(0, 0)}",
        f"Neutral = {sentimiento_por_empresa.get(1, 0)}",
        f"Positive = {sentimiento_por_empresa.get(2, 0)}"
    ]}

    return resultado_analisis_sentimiento



#ENTRENAMIENTO MODELO MACHINE LEARNING Y ENDPOINT ITEM-ITEM

# se seleccionan las columnas relevantes para calcular la similitud
columnas_relevantes = ['item_name', 'Action', 'Casual', 'Indie', 'Simulation', 'Strategy', 'Free to Play', 'RPG', 'Sports', 'Adventure', 'Racing', 'Early Access', 'Massively Multiplayer', 'Animation &amp; Modeling', 'Video Production', 'Utilities', 'Web Publishing', 'Education', 'Software Training', 'Design &amp; Illustration', 'Audio Production', 'Photo Editing', 'Accounting']
datos_relevantes = df_merged[columnas_relevantes]
caracteristicas_juegos = datos_relevantes.drop(['item_name'], axis=1)
# se normalizan los datos si fuese  necesario
scaler = MinMaxScaler()
datos_normalizados = scaler.fit_transform(caracteristicas_juegos)
# se calcula la similitud del coseno entre juegos basado en las características
similarity_matrix = cosine_similarity(datos_normalizados)


#endpint recomendacion item-item
@app.get("/recomendar_juego/{id_producto}")
def recomendar_juego(id_producto: int):
    """recibe como parametro de entrada, la ID de un juego, y mediante su entrenamiento ya realizado
    nos devuelve una lista con juegos relacionados a la categoria.
    ejemplo de parametros de ingreso: 21100, 250180"""
    # se encuentra el índice del juego seleccionado
    try:
        juego_seleccionado_idx = df_merged[df_merged['item_id'] == id_producto].index[0]
    except IndexError:
        return {"mensaje": "El juego ingresado no se encontró en la base de datos."}

    # se obtiene las puntuaciones de similitud para el juego seleccionado
    similarity_scores = similarity_matrix[juego_seleccionado_idx]

    juegos_recomendados = similarity_scores.argsort()[-2:-7:-1]  # 5 juegos más similares menos el seleccionado

    juegos_recomendados_list = []
    for idx in juegos_recomendados:
        juego_recomendado = df_merged.iloc[idx]['item_name']
        similitud = similarity_scores[idx]
        juegos_recomendados_list.append({"nombre": juego_recomendado, "similitud": similitud})

    return {"juegos_recomendados": juegos_recomendados_list}


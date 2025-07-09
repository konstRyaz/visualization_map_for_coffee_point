import streamlit as st
import folium
import numpy as np
import pandas as pd
import random
from shapely import wkt # Теперь импортируем shapely для работы с геометриями

# --- Заголовок приложения ---
st.set_page_config(layout="wide")
st.title("Карта выгодности для открытия кофеен: Визуализация POI")
st.write("Визуализация потенциально выгодных мест для открытия кофеен вокруг реальных точек интереса (POI) из OpenStreetMap.")

# --- Функция для имитации скора модели ---
@st.cache_data
def get_model_score(lat, lon, poi_type=None):
    """
    Имитирует получение скора от модели для конкретной точки.
    Скор от 0 (невыгодно) до 1 (очень выгодно).
    """
    base_score = random.uniform(0.1, 0.9)

    # Можно добавить "бонусы" за определенные типы POI
    if poi_type in ['subway_station', 'bus_stop', 'university', 'mall', 'restaurant', 'cafe', 'bar']:
        base_score += random.uniform(0.1, 0.2)

    return min(1.0, max(0.0, base_score))

# --- Функция для загрузки и обработки данных POI ---
@st.cache_data
def load_and_process_pois(filepath):
    df = pd.read_csv(filepath)

    # --- НОВАЯ ЛОГИКА ДЛЯ ОБРАБОТКИ РАЗЛИЧНЫХ ТИПОВ ГЕОМЕТРИЙ (POINT, POLYGON, MULTIPOLYGON и т.д.) ---
    def get_coords_from_wkt(wkt_string):
        try:
            geom = wkt.loads(wkt_string)
            
            if geom.geom_type == 'Point':
                # Для точек возвращаем их прямые координаты
                # WKT-формат: POINT (долгота широта), поэтому x - долгота, y - широта
                return geom.y, geom.x 
            elif geom.geom_type in ['Polygon', 'LineString', 'MultiPoint', 'MultiPolygon', 'MultiLineString']:
                # Для полигонов, линий и мульти-геометрий берем центроид
                # Centroid - это объект Point, поэтому получаем его координаты
                centroid = geom.centroid
                return centroid.y, centroid.x
            else:
                # Для других неподдерживаемых типов геометрий
                return None, None
        except Exception as e:
            # Обработка ошибок при парсинге WKT (например, некорректные строки)
            # st.warning(f"Ошибка парсинга WKT строки '{wkt_string[:50]}...': {e}") # Отладочная строка, можно убрать
            return None, None

    # Применяем новую функцию к столбцу 'geometry'
    df[['latitude', 'longitude']] = df['geometry'].apply(lambda x: pd.Series(get_coords_from_wkt(x)))
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    # Объединяем столбцы типов POI в один 'poi_type'
    # Этот список type_columns должен соответствовать тегам OSM, которые присутствуют в вашем файле.
    # Если у вас есть другие столбцы, содержащие тип POI, добавьте их сюда.
    # Порядок важен: bfill берет первое не-NaN значение справа налево, поэтому более специфичные теги должны быть левее.
    type_columns = ['amenity', 'shop', 'leisure', 'building', 'tourism', 'office', 'healthcare', 'sport', 'natural', 'craft', 'historic', 'landuse', 'public_transport'] 
    
    # Создаем столбец 'poi_type', беря первое не-NaN значение из списка type_columns
    # Используем df.get(col) чтобы избежать ошибки, если столбца нет в DataFrame
    # Создаем временный DataFrame только из существующих type_columns
    existing_type_columns = [col for col in type_columns if col in df.columns]
    
    if existing_type_columns:
        df['poi_type'] = df[existing_type_columns].bfill(axis=1).iloc[:, 0]
    else:
        df['poi_type'] = 'unknown' # Если нет ни одного из этих столбцов, ставим 'unknown'

    df['poi_type'] = df['poi_type'].fillna('other') # Заменяем оставшиеся NaN в poi_type на 'other'

    # Очистка строк с отсутствующими координатами (после обработки shapely таких должно быть меньше)
    df = df.dropna(subset=['latitude', 'longitude'])

    # Применяем модель для каждой точки
    df['score'] = df.apply(lambda row: get_model_score(row['latitude'], row['longitude'], row['poi_type']), axis=1)

    return df

# --- Вспомогательная функция для определения цвета по скору ---
def get_gradient_color(score, min_score, max_score):
    if max_score == min_score:
        normalized_score = 0.5
    else:
        normalized_score = (score - min_score) / (max_score - min_score)

    if normalized_score < 0.5:
        r = 1
        g = normalized_score * 2
        b = 0
    else:
        r = 1 - (normalized_score - 0.5) * 2
        g = 1
        b = 0
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# --- Интерфейс Streamlit: Выбор города и загрузка данных ---
city_selection = st.sidebar.radio("Выберите город для визуализации:", ("Москва", "Санкт-Петербург"), index=0)
radius_meters = st.sidebar.slider("Радиус области вокруг POI (метры):", 50, 500, 150, step=50)
min_score_filter = st.sidebar.slider("Показывать POI с минимальным скором:", 0.0, 1.0, 0.1, 0.05)


if city_selection == "Москва":
    filepath = "moscow_pois.csv" # Возвращаемся к оригинальному файлу OSMnx
    map_center = [55.7558, 37.6173]
    zoom_start = 10
    map_title = "Москва: Выгодность кофеен вокруг POI (из OSMnx, все геометрии)"
else:
    # Здесь вам нужно будет либо создать spb_pois.csv с аналогичной структурой
    # и обработать его так же, либо временно отключить этот выбор
    filepath = "spb_pois.csv" # Предполагаем, что этот файл тоже из OSMnx
    map_center = [59.9343, 30.3351]
    zoom_start = 11
    map_title = "Санкт-Петербург: Выгодность кофеен вокруг POI (из OSMnx)"

filtered_pois_df = pd.DataFrame()
pois_df = pd.DataFrame()

st.info(f"Загрузка и обработка данных для {city_selection} из файла '{filepath}'...")
try:
    pois_df = load_and_process_pois(filepath)
    st.success(f"Загружено {len(pois_df)} точек интереса для {city_selection}.")

    if not pois_df.empty:
        filtered_pois_df = pois_df[pois_df['score'] >= min_score_filter].copy()
    else:
        st.warning("Загруженный датасет пуст или не содержит данных.")

    st.write(f"Отображается {len(filtered_pois_df)} точек из {len(pois_df)} (с учетом фильтрации по скору).")

except FileNotFoundError:
    st.error(f"Файл '{filepath}' не найден. Убедитесь, что он находится в той же директории, что и ваш скрипт Streamlit.")
except Exception as e:
    st.error(f"Произошла ошибка при загрузке или обработке данных: {e}")
    st.error("Пожалуйста, проверьте формат файла и названия столбцов.")


m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")

st.subheader(map_title)
st.markdown("---")

if not filtered_pois_df.empty:
    min_overall_score = filtered_pois_df['score'].min()
    max_overall_score = filtered_pois_df['score'].max()

    if min_overall_score == max_overall_score:
        min_overall_score = 0.0
        max_overall_score = 1.0

    for index, row in filtered_pois_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        score = row['score']
        # Для OSMnx файлов название POI обычно в столбце 'name'
        name = row['name'] if pd.notna(row['name']) else "Неизвестный POI"
        poi_type = row['poi_type']

        color = get_gradient_color(score, min_overall_score, max_overall_score)

        folium.Circle(
            location=[lat, lon],
            radius=radius_meters,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            tooltip=f"Прогноз выгодности: {score:.2f}"
        ).add_to(m)

    st.markdown(f"На карте отображены круги радиусом **{radius_meters} метров** вокруг каждой точки интереса.")
    st.markdown("Цвет круга показывает скор выгодности:")
    st.markdown("- **Красный**: Менее выгодно")
    st.markdown("- **Желтый**: Средняя выгодность")
    st.markdown("- **Зеленый**: Очень выгодно")

else:
    st.warning("Нет точек POI для отображения после фильтрации или произошла ошибка загрузки данных.")

folium_figure = folium.Figure(width=900, height=520)
folium_figure.add_child(m)
st.components.v1.html(folium_figure._repr_html_(), height=540, scrolling=False)

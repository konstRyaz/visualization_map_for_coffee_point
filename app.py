import streamlit as st
import folium
import numpy as np
import pandas as pd
import random
from shapely import wkt
from folium.plugins import HeatMap

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

    if poi_type in ['subway_station', 'bus_stop', 'university', 'mall', 'restaurant', 'cafe', 'bar']:
        base_score += random.uniform(0.1, 0.2)

    return min(1.0, max(0.0, base_score))

# --- Функция для загрузки и обработки данных POI ---
@st.cache_data
def load_and_process_pois(filepath):
    df = pd.read_csv(filepath)

    def get_coords_from_wkt(wkt_string):
        try:
            geom = wkt.loads(wkt_string)
            
            if geom.geom_type == 'Point':
                return geom.y, geom.x 
            elif geom.geom_type in ['Polygon', 'LineString', 'MultiPoint', 'MultiPolygon', 'MultiLineString']:
                centroid = geom.centroid
                return centroid.y, centroid.x
            else:
                return None, None
        except Exception as e:
            return None, None

    df[['latitude', 'longitude']] = df['geometry'].apply(lambda x: pd.Series(get_coords_from_wkt(x)))

    type_columns = ['amenity', 'shop', 'leisure', 'building', 'tourism', 'office', 'healthcare', 'sport', 'natural', 'craft', 'historic', 'landuse', 'public_transport'] 
    
    existing_type_columns = [col for col in type_columns if col in df.columns]
    
    if existing_type_columns:
        df['poi_type'] = df[existing_type_columns].bfill(axis=1).iloc[:, 0]
    else:
        df['poi_type'] = 'unknown'

    df['poi_type'] = df['poi_type'].fillna('other')

    df = df.dropna(subset=['latitude', 'longitude'])

    df['score'] = df.apply(lambda row: get_model_score(row['latitude'], row['longitude'], row['poi_type']), axis=1)

    return df

# --- Вспомогательная функция для определения цвета по скору (для кругов) ---
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


# --- Боковая панель (левая) ---
st.sidebar.header("Настройки приложения")

# 1. Выбор режима настроек (Обычный / Расширенный)
settings_mode = st.sidebar.radio("Режим настроек:", ("Обычный", "Расширенный"), index=0)

# 2. Общие настройки (всегда видны)
city_selection = st.sidebar.radio("Выберите город для визуализации:", ("Москва", "Санкт-Петербург"), index=0)
show_heatmap = st.sidebar.checkbox("Показать тепловую карту", False)

# Инициализация переменных для избежания ошибок, если они не будут заданы в "Обычном" режиме
# Эти значения будут использоваться, если не переопределены в "Расширенном" режиме
radius_meters = 150
fill_opacity_value = 0.4
heatmap_radius = 20
heatmap_blur = 15
heatmap_min_opacity = 0.3
heatmap_max_opacity = 0.8
min_score_filter = 0.1

# 3. Расширенные настройки (показываются только если выбран режим "Расширенный")
if settings_mode == "Расширенный":
    st.sidebar.markdown("---") # Разделитель
    st.sidebar.subheader("Расширенные параметры:")

    # Фильтр по минимальному скору - теперь в расширенных
    min_score_filter = st.sidebar.slider("Показывать POI с минимальным скором:", 0.0, 1.0, 0.1, 0.05)

    if not show_heatmap: # Настройки для кругов
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Настройки для кругов:**")
        radius_meters = st.sidebar.slider("Радиус области вокруг POI (метры):", 50, 500, 150, step=50)
        fill_opacity_value = st.sidebar.slider("Прозрачность кругов (0.0 - полностью прозрачный, 1.0 - полностью непрозрачный):", 0.0, 1.0, 0.4, 0.05)
    else: # Настройки для тепловой карты
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Настройки для тепловой карты:**")
        heatmap_radius = st.sidebar.slider("Радиус влияния точек на тепловой карте:", 5, 50, 20, 5)
        heatmap_blur = st.sidebar.slider("Размытие тепловой карты:", 5, 50, 15, 5)
        heatmap_min_opacity = st.sidebar.slider("Минимальная прозрачность тепловой карты:", 0.0, 1.0, 0.3, 0.1)
        heatmap_max_opacity = st.sidebar.slider("Максимальная прозрачность тепловой карты:", 0.0, 1.0, 0.8, 0.1)

# 4. Руководство пользователя (всегда видимо, но сворачиваемо)
st.sidebar.markdown("---")
with st.sidebar.expander("📖 Руководство пользователя"):
    st.markdown("""
    **Приветствие:**
    Это интерактивное веб-приложение предназначено для визуализации потенциально выгодных мест для открытия кофеен, основываясь на данных о точках интереса (POI) из OpenStreetMap и симулированном прогнозе выгодности. Приложение позволяет исследовать территорию и настраивать отображение данных в соответствии с вашими потребностями.

    **Основные разделы приложения:**
    1.  **Основная панель:** Отображает интерактивную карту с выбранным типом визуализации (круги или тепловая карта).
    2.  **Боковая панель (левая):** Содержит все настройки и фильтры для управления отображением данных.

    ---

    **Настройки на боковой панели (левая часть экрана):**

    **1. Общие настройки (всегда видны):**

    * **Режим настроек:**
        * **Обычный:** Отображает только основные настройки для быстрой работы.
        * **Расширенный:** Открывает доступ ко всем дополнительным параметрам визуализации.
    * **Выберите город для визуализации:**
        * Позволяет переключаться между данными для **Москвы** и **Санкт-Петербурга**.
    * **Показать тепловую карту (чекбокс):**
        * Этот переключатель позволяет выбрать основной режим визуализации:
            * **Включен (галочка стоит):** Приложение покажет данные в виде **тепловой карты**. Этот режим лучше подходит для оценки общей "температуры" выгодности по большим областям.
            * **Выключен (галочки нет):** Приложение покажет данные в виде **отдельных кругов** вокруг каждой точки интереса.
        * **Внимание:** Настройки ниже меняются в зависимости от выбранного режима визуализации и выбранного "Режима настроек"!

    **2. Расширенные параметры (появляются только если выбран режим "Расширенный"):**

    * **Показывать POI с минимальным скором:**
        * **Тип:** Слайдер от 0.0 до 1.0.
        * **Назначение:** Позволяет отфильтровать точки интереса, отображая только те, чей прогнозируемый скор выгодности равен или выше заданного значения. Если вы хотите видеть только самые "выгодные" места, увеличьте это значение. Это особенно полезно для уменьшения "шума" на карте и фокусировки на наиболее перспективных областях, а также помогает уменьшить перегрев тепловой карты в областях с высокой, но не обязательно "качественной" плотностью.

    * **Настройки для режима "Круги" (если чекбокс "Показать тепловую карту" ВЫКЛЮЧЕН):**
        * **Радиус области вокруг POI (метры):**
            * **Тип:** Слайдер (от 50 до 500 метров).
            * **Назначение:** Определяет размер круга, который будет нарисован вокруг каждой точки интереса. Это позволяет визуализировать зону влияния каждого POI.
        * **Прозрачность кругов (0.0 - полностью прозрачный, 1.0 - полностью непрозрачный):**
            * **Тип:** Слайдер (от 0.0 до 1.0).
            * **Назначение:** Контролирует степень прозрачности кругов.
                * **0.0:** Круги полностью прозрачны (невидимы).
                * **1.0:** Круги полностью непрозрачны.
                * **Рекомендация:** Установите значение от 0.3 до 0.6, чтобы круги просвечивали и позволяли видеть карту под ними, особенно в местах их наложения.
            * **Цвет кругов:** Красный цвет означает низкий скор выгодности, желтый — средний, зеленый — высокий. При наведении курсора на круг отображается его точный прогноз выгодности.

    * **Настройки для режима "Тепловая карта" (если чекбокс "Показать тепловую карту" ВКЛЮЧЕН):**
        * **Радиус влияния точек на тепловой карте:**
            * **Тип:** Слайдер (от 5 до 50).
            * **Назначение:** Определяет, насколько широко каждая точка интереса "распространяет" свой "жар" (скор выгодности) по карте. Больший радиус делает тепловую карту более гладкой и обобщенной, меньший — более детализированной и локализованной.
        * **Размытие тепловой карты:**
            * **Тип:** Слайдер (от 5 до 50).
            * **Назначение:** Контролирует степень "размытости" тепловой карты. Чем больше значение, тем более плавными и менее резкими будут переходы между горячими и холодными зонами.
        * **Минимальная прозрачность тепловой карты:**
            * **Тип:** Слайдер (от 0.0 до 1.0).
            * **Назначение:** Определяет минимальный уровень прозрачности для самых "холодных" (менее выгодных) областей тепловой карты.
        * **Максимальная прозрачность тепловой карты:**
            * **Тип:** Слайдер (от 0.0 до 1.0).
            * **Назначение:** Определяет максимальный уровень прозрачности для самых "горячих" (наиболее выгодных) областей тепловой карты.
            * **Цвет тепловой карты:** Соответствует вашей логике: **Красные** области означают низкий скор выгодности, **желтые** — средний, **зеленые** — высокий. Чем ярче и насыщеннее зеленый цвет, тем выше прогнозируемая выгода в этой области.
    """)

# --- Основная логика приложения: загрузка данных и отрисовка карты ---
if city_selection == "Москва":
    filepath = "moscow_pois.csv"
    map_center = [55.7558, 37.6173]
    zoom_start = 10
    map_title = "Москва: Выгодность кофеен вокруг POI (из OSMnx, все геометрии)"
else:
    filepath = "spb_pois.csv"
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

    if show_heatmap:
        heat_data = filtered_pois_df[['latitude', 'longitude', 'score']].values.tolist()
        
        custom_gradient = {0.0: 'red', 0.5: 'yellow', 1.0: 'green'}

        HeatMap(heat_data,
                radius=heatmap_radius,
                blur=heatmap_blur,
                min_opacity=heatmap_min_opacity,
                max_opacity=heatmap_max_opacity,
                gradient=custom_gradient
                ).add_to(m)

        st.markdown(f"Отображена **тепловая карта** на основе скора выгодности POI.")
        st.markdown("Цвет тепловой карты показывает скор выгодности:")
        st.markdown("- **Красный**: Менее выгодно")
        st.markdown("- **Желтый**: Средняя выгодность")
        st.markdown("- **Зеленый**: Очень выгодно")
        st.markdown("Более яркие и насыщенные области указывают на более высокую концентрацию выгодных POI.")
    else:
        for index, row in filtered_pois_df.iterrows():
            lat, lon = row['latitude'], row['longitude']
            score = row['score']
            name = row['name'] if pd.notna(row['name']) else "Неизвестный POI"
            poi_type = row['poi_type']

            color = get_gradient_color(score, min_overall_score, max_overall_score)

            folium.Circle(
                location=[lat, lon],
                radius=radius_meters,
                color=color,
                weight=0,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity_value,
                tooltip=f"Прогноз выгодности: {score:.2f}"
            ).add_to(m)

        st.markdown(f"На карте отображены **круги** радиусом **{radius_meters} метров** вокруг каждой точки интереса.")
        st.markdown("Цвет круга показывает скор выгодности:")
        st.markdown("- **Красный**: Менее выгодно")
        st.markdown("- **Желтый**: Средняя выгодность")
        st.markdown("- **Зеленый**: Очень выгодно")
        st.markdown(f"Текущая прозрачность кругов (непрозрачность): **{fill_opacity_value:.2f}**")

else:
    st.warning("Нет точек POI для отображения после фильтрации или произошла ошибка загрузки данных.")

folium_figure = folium.Figure(width=900, height=520)
folium_figure.add_child(m)
st.components.v1.html(folium_figure._repr_html_(), height=540, scrolling=False)

import streamlit as st
import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap

# --- Заголовок приложения ---
st.set_page_config(layout="wide")
st.title("Карта выгодности для открытия кофеен: Визуализация POI")
st.write("Визуализация потенциально выгодных мест для открытия кофеен на основе данных POI с предсказаниями выгодности.")

# --- Функция для загрузки и обработки данных POI ---
@st.cache_data
def load_and_process_pois(filepath):
    df = pd.read_csv(filepath)

    # Проверяем наличие всех необходимых колонок: lat, lon, score
    required_columns = ['lat', 'lon', 'score']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Ошибка: Входной CSV файл '{filepath}' должен содержать столбец '{col}'.")
            st.stop()

    df = df.dropna(subset=['lat', 'lon', 'score'])
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})

    return df

# --- Вспомогательная функция для определения цвета по скору (для кругов) ---
def get_gradient_color(score, min_score, max_score):
    if max_score == min_score:
        normalized_score = 0.5
    else:
        normalized_score = (score - min_score) / (max_score - min_score)

    normalized_score = max(0.0, min(1.0, normalized_score))

    if normalized_score < 0.5:
        r = 1
        g = normalized_score * 2
        b = 0
    else:
        r = 1 - (normalized_score - 0.5) * 2
        g = 1
        b = 0
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# --- Инициализация session_state для всех параметров ---
if 'radius_meters' not in st.session_state: st.session_state.radius_meters = 150
if 'fill_opacity_value' not in st.session_state: st.session_state.fill_opacity_value = 0.4
if 'heatmap_radius' not in st.session_state: st.session_state.heatmap_radius = 20
if 'heatmap_blur' not in st.session_state: st.session_state.heatmap_blur = 15
if 'heatmap_min_opacity' not in st.session_state: st.session_state.heatmap_min_opacity = 0.3
if 'heatmap_max_opacity' not in st.session_state: st.session_state.heatmap_max_opacity = 0.8
if 'min_score_filter' not in st.session_state: st.session_state.min_score_filter = 0.1
if 'city_selection' not in st.session_state: st.session_state.city_selection = "Москва"
if 'show_heatmap' not in st.session_state: st.session_state.show_heatmap = False
if 'fixed_gradient' not in st.session_state: st.session_state.fixed_gradient = False

# --- Боковая панель (левая) ---
st.sidebar.header("Настройки приложения")

# 1. Выбор режима настроек (Обычный / Расширенный)
settings_mode = st.sidebar.radio("Режим настроек:", ("Обычный", "Расширенный"), index=0)

# 2. Общие настройки (всегда видны)
st.session_state.city_selection = st.sidebar.radio("Выберите город для визуализации:", ("Москва", "Санкт-Петербург"), key='city_radio')
st.session_state.show_heatmap = st.sidebar.checkbox("Показать тепловую карту", key='heatmap_checkbox')

# 3. Расширенные настройки (показываются только если выбран режим "Расширенный")
if settings_mode == "Расширенный":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Расширенные параметры:")

    st.session_state.min_score_filter = st.sidebar.slider("Показывать POI с минимальным скором:", 0.0, 10000.0, float(st.session_state.min_score_filter), 100.0, key='min_score_slider')
    st.session_state.fixed_gradient = st.sidebar.checkbox("Фиксировать цветовой градиент (0 - 10000)", st.session_state.fixed_gradient, key='fixed_gradient_checkbox')


    if not st.session_state.show_heatmap:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Настройки для кругов:**")
        st.session_state.radius_meters = st.sidebar.slider("Радиус области вокруг POI (метры):", 50, 500, st.session_state.radius_meters, step=50, key='radius_slider')
        st.session_state.fill_opacity_value = st.sidebar.slider("Прозрачность кругов (0.0 - полностью прозрачный, 1.0 - полностью непрозрачный):", 0.0, 1.0, st.session_state.fill_opacity_value, 0.05, key='opacity_slider')
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Настройки для тепловой карты:**")
        st.session_state.heatmap_radius = st.sidebar.slider("Радиус влияния точек на тепловой карте:", 5, 50, st.session_state.heatmap_radius, 5, key='hm_radius_slider')
        st.session_state.heatmap_blur = st.sidebar.slider("Размытие тепловой карты:", 5, 50, st.session_state.heatmap_blur, 5, key='hm_blur_slider')
        st.session_state.heatmap_min_opacity = st.sidebar.slider("Минимальная прозрачность тепловой карты:", 0.0, 1.0, st.session_state.heatmap_min_opacity, 0.1, key='hm_min_op_slider')
        st.session_state.heatmap_max_opacity = st.sidebar.slider("Максимальная прозрачность тепловой карты:", 0.0, 1.0, st.session_state.heatmap_max_opacity, 0.1, key='hm_max_op_slider')

# 4. Руководство пользователя (всегда видимо, но сворачиваемо)
st.sidebar.markdown("---")
with st.sidebar.expander("📖 Руководство пользователя"):
    st.markdown("""
    **Приветствие:**
    Это интерактивное веб-приложение предназначено для визуализации потенциально выгодных мест для открытия кофеен, основываясь на данных о точках интереса (POI) с **предварительно рассчитанным прогнозом выгодности (скор)**. Приложение позволяет исследовать территорию и настраивать отображение данных в соответствии с вашими потребностями.

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
        * **Тип:** Слайдер от 0 до 10000.
        * **Назначение:** Позволяет отфильтровать точки интереса, отображая только те, чей прогнозируемый скор выгодности равен или выше заданного значения. Это полезно для фокусировки на наиболее перспективных областях.
    * **Фиксировать цветовой градиент (0 - 10000):**
        * **Тип:** Чекбокс.
        * **Назначение:** При включении (отмечен) цвет кругов будет рассчитываться исходя из полного диапазона возможных скоров от 0 (красный) до 10000 (зеленый), независимо от применяемого фильтра "Показывать POI с минимальным скором". Это обеспечивает постоянство цветовой шкалы и позволяет сравнивать скоры между собой напрямую, без перекалибровки градиента под отфильтрованный набор данных. Если отключен, градиент цвета кругов будет адаптироваться к минимальному и максимальному скорам *отображаемых* POI.

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

# --- Кэшированная функция для создания и отрисовки карты ---
@st.cache_data(show_spinner=False)
def get_folium_map(current_city_selection, current_show_heatmap, current_min_score_filter,
                   current_radius_meters, current_fill_opacity_value,
                   current_heatmap_radius, current_heatmap_blur, current_heatmap_min_opacity, current_heatmap_max_opacity,
                   current_fixed_gradient,
                   pois_df):

    if current_city_selection == "Москва":
        map_center = [55.7558, 37.6173]
    else:
        map_center = [59.9343, 30.3351]
    zoom_start = 10

    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")

    filtered_pois_df = pois_df[pois_df['score'] >= current_min_score_filter].copy()

    if not filtered_pois_df.empty:
        min_displayed_score = filtered_pois_df['score'].min()
        max_displayed_score = filtered_pois_df['score'].max()
        
        color_min_score = 0.0
        color_max_score = 10000.0 
        if not current_fixed_gradient:
            color_min_score = min_displayed_score
            color_max_score = max_displayed_score
            if color_min_score == color_max_score:
                color_min_score = 0.0
                color_max_score = 10000.0

        if current_show_heatmap:
            heat_data = filtered_pois_df[['latitude', 'longitude', 'score']].values.tolist()
            if max_displayed_score > 0:
                scaled_heat_data = [[lat, lon, score / max_displayed_score] for lat, lon, score in heat_data]
            else:
                scaled_heat_data = heat_data

            custom_gradient = {0.0: 'red', 0.5: 'yellow', 1.0: 'green'}
            HeatMap(scaled_heat_data,
                    radius=current_heatmap_radius,
                    blur=current_heatmap_blur,
                    min_opacity=current_heatmap_min_opacity,
                    max_opacity=current_heatmap_max_opacity,
                    gradient=custom_gradient
                    ).add_to(m)
        else:
            for index, row in filtered_pois_df.iterrows():
                lat, lon = row['latitude'], row['longitude']
                score = row['score']
                
                color = get_gradient_color(score, color_min_score, color_max_score)
                folium.Circle(
                    location=[lat, lon],
                    radius=current_radius_meters,
                    color=color,
                    weight=0,
                    fill=True,
                    fill_color=color,
                    fill_opacity=current_fill_opacity_value,
                    tooltip=f"Прогноз выгодности: {score:.0f}"
                ).add_to(m)
    return m

# --- Основная логика приложения: загрузка данных и отрисовка карты ---
current_city = st.session_state.city_selection
if current_city == "Москва":
    filepath = "moscow_pois.csv"
else:
    filepath = "spb_pois.csv"

try:
    current_pois_df = load_and_process_pois(filepath)
    st.info(f"Загрузка и обработка данных для {current_city} из файла '{filepath}'...")
    st.success(f"Загружено {len(current_pois_df)} точек интереса для {current_city}.")
except FileNotFoundError:
    st.error(f"Файл '{filepath}' не найден. Убедитесь, что он находится в той же директории, что и ваш скрипт Streamlit.")
    st.stop()
except Exception as e:
    st.error(f"Произошла ошибка при загрузке или обработке данных: {e}")
    st.error("Пожалуйста, убедитесь, что файл содержит столбцы 'lat', 'lon' и 'score'.")
    st.stop()


st.write(f"Отображается {len(current_pois_df[current_pois_df['score'] >= st.session_state.min_score_filter])} точек из {len(current_pois_df)} (с учетом фильтрации по скору).")


# Вызываем кэшированную функцию для получения объекта карты
m = get_folium_map(st.session_state.city_selection,
                   st.session_state.show_heatmap,
                   st.session_state.min_score_filter,
                   st.session_state.radius_meters,
                   st.session_state.fill_opacity_value,
                   st.session_state.heatmap_radius,
                   st.session_state.heatmap_blur,
                   st.session_state.heatmap_min_opacity,
                   st.session_state.heatmap_max_opacity,
                   st.session_state.fixed_gradient,
                   current_pois_df)

# Отображение карты и пояснений
st.subheader(f"{st.session_state.city_selection}: Выгодность кофеен вокруг POI")
st.markdown("---")

# Пояснения к визуализации
if not current_pois_df[current_pois_df['score'] >= st.session_state.min_score_filter].empty:
    if st.session_state.show_heatmap:
        st.markdown(f"Отображена **тепловая карта** на основе скора выгодности POI.")
        st.markdown("Цвет тепловой карты показывает скор выгодности:")
        st.markdown("- **Красный**: Менее выгодно")
        st.markdown("- **Желтый**: Средняя выгодность")
        st.markdown("- **Зеленый**: Очень выгодно")
        st.markdown("Более яркие и насыщенные области указывают на более высокую концентрацию выгодных POI.")
    else:
        st.markdown(f"На карте отображены **круги** радиусом **{st.session_state.radius_meters} метров** вокруг каждой точки интереса.")
        st.markdown("Цвет круга показывает скор выгодности:")
        st.markdown("- **Красный**: Менее выгодно")
        st.markdown("- **Желтый**: Средняя выгодность")
        st.markdown("- **Зеленый**: Очень выгодно")
        st.markdown(f"Текущая прозрачность кругов (непрозрачность): **{st.session_state.fill_opacity_value:.2f}**")
        if st.session_state.fixed_gradient:
            st.markdown("*Цветовой градиент кругов зафиксирован от 0 (красный) до 10000 (зеленый).*")
        else:
            st.markdown("*Цветовой градиент кругов адаптируется к диапазону скоров отображаемых POI.*")
else:
    st.warning("Нет точек POI для отображения после фильтрации или произошла ошибка загрузки данных.")


folium_figure = folium.Figure(width=900, height=520)
folium_figure.add_child(m)
st.components.v1.html(folium_figure._repr_html_(), height=540, scrolling=False)

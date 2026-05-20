import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Инференс моделей", page_icon="assets/icon.png")
st.title("Предсказание моделей ML")
st.markdown("---")

@st.cache_resource
def load_models():
    models = {}
    
    try:
        # полиномиальная регрессия
        models['Полиномиальная регрессия'] = joblib.load('models/ml1_polynomial.pkl')
        st.success("ML1: Полиномиальная регрессия загружена")
    except Exception as e:
        st.error(f"Ошибка загрузки ML1: {e}")
    
    try:
        # Gradient Boosting
        models['Gradient Boosting'] = joblib.load('models/ml2_gbr.pkl')
        st.success("ML2: Gradient Boosting загружена")
    except Exception as e:
        st.error(f"Ошибка загрузки ML2: {e}")
    
    try:
        # CatBoost
        models['CatBoost'] = joblib.load('models/ml3_catboost.pkl')
        st.success("ML3: CatBoost загружена")
    except Exception as e:
        st.error(f"Ошибка загрузки ML3: {e}")
    
    try:
        # Bagging
        models['Bagging'] = joblib.load('models/ml4_bagging.pkl')
        st.success("ML4: Bagging загружена")
    except Exception as e:
        st.error(f"Ошибка загрузки ML4: {e}")
    
    try:
        # Stacking
        models['Stacking'] = joblib.load('models/ml5_stacking.pkl')
        st.success("ML5: Stacking загружена")
    except Exception as e:
        st.error(f"Ошибка загрузки ML5: {e}")
    
    try:
        # Keras
        models['Keras'] = load_model('models/ml6_keras.h5', compile=False)
        st.success("ML6: Keras загружена")
    except Exception as e:
        st.error(f"Ошибка загрузки ML6: {e}")
    
    return models

st.subheader("Загрузка моделей")
models = load_models()
st.markdown("---")

@st.cache_data
def load_bounds():
    df = pd.read_csv('data/house_sales_preprocessed.csv')
    bounds = {}
    for col in df.columns:
        bounds[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    return bounds

bounds = load_bounds()

if len(models) > 0:
    selected_model = st.selectbox(
        "Выберите модель для предсказания",
        list(models.keys()),
        help="Выберите одну из 6 обученных моделей"
    )
    st.markdown("---")
    
    st.subheader("Введите характеристики дома")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input(
            "Количество спален (bedrooms)", 
            min_value=int(bounds['bedrooms']['min']), 
            max_value=int(bounds['bedrooms']['max']), 
            value=3,
            help=f"От {bounds['bedrooms']['min']:.0f} до {bounds['bedrooms']['max']:.0f}"
        )
        
        bathrooms = st.number_input(
            "Количество ванных комнат (bathrooms)", 
            min_value=float(bounds['bathrooms']['min']), 
            max_value=float(bounds['bathrooms']['max']), 
            value=2.0, step=0.25,
            help=f"От {bounds['bathrooms']['min']:.1f} до {bounds['bathrooms']['max']:.1f}"
        )
        
        sqft_living = st.number_input(
            "Жилая площадь (sqft_living) - кв. футы", 
            min_value=float(bounds['sqft_living']['min']), 
            max_value=float(bounds['sqft_living']['max']), 
            value=2000.0, step=1.0,
            help=f"От {bounds['sqft_living']['min']:.0f} до {bounds['sqft_living']['max']:.0f} кв. футов"
        )
        
        sqft_lot = st.number_input(
            "Площадь участка (sqft_lot) - кв. футы", 
            min_value=float(bounds['sqft_lot']['min']), 
            max_value=float(bounds['sqft_lot']['max']), 
            value=5000.0, step=1.0
        )
        
        floors = st.number_input(
            "Количество этажей (floors)", 
            min_value=float(bounds['floors']['min']), 
            max_value=float(bounds['floors']['max']), 
            value=1.5, step=0.5
        )
    
    with col2:
        waterfront = st.selectbox(
            "Вид на воду (waterfront)", [0, 1], 
            format_func=lambda x: "Да" if x == 1 else "Нет"
        )
        
        view = st.selectbox(
            "Качество вида из дома (view)", [0, 1, 2, 3, 4],
            format_func=lambda x: ["Плохой", "Средний", "Хороший", "Очень хороший", "Отличный"][x]
        )
        
        condition = st.selectbox(
            "Состояние дома (condition)", [1, 2, 3, 4, 5],
            format_func=lambda x: ["Плохое", "Среднее", "Хорошее", "Очень хорошее", "Отличное"][x-1]
        )
        
        grade = st.number_input(
            "Общая оценка качества (grade)", 
            min_value=1, max_value=13, value=7,
            help="1-13, где 13 - наилучшее качество"
        )
        
        yr_built = st.number_input(
            "Год постройки (yr_built)", 
            min_value=int(bounds['yr_built']['min']), 
            max_value=int(bounds['yr_built']['max']), 
            value=1980,
            help=f"От {int(bounds['yr_built']['min'])} до {int(bounds['yr_built']['max'])}"
        )
    
    st.markdown("---")
    
    with st.expander("Дополнительные параметры"):
        col3, col4 = st.columns(2)
        with col3:
            sqft_above = st.number_input(
                "Площадь над землей (sqft_above)", 
                min_value=float(bounds['sqft_above']['min']), 
                max_value=float(bounds['sqft_above']['max']), 
                value=1500.0, step=1.0
            )
            
            sqft_basement = st.number_input(
                "Площадь подвала (sqft_basement)", 
                min_value=float(bounds['sqft_basement']['min']), 
                max_value=float(bounds['sqft_basement']['max']), 
                value=0.0, step=1.0
            )
            
            yr_renovated = st.number_input(
                "Год последнего ремонта (yr_renovated)", 
                min_value=int(bounds['yr_renovated']['min']), 
                max_value=int(bounds['yr_renovated']['max']), 
                value=0, step=1
            )
            
            st.markdown("---")
            st.markdown("**Дата продажи**")
            
            sale_year = st.number_input(
                "Год продажи (sale_year)", 
                min_value=int(bounds['sale_year']['min']), 
                max_value=int(bounds['sale_year']['max']), 
                value=2014
            )
            
            sale_month = st.number_input(
                "Месяц продажи (sale_month)", 
                min_value=int(bounds['sale_month']['min']), 
                max_value=int(bounds['sale_month']['max']), 
                value=6
            )
            
            sale_day = st.number_input(
                "День продажи (sale_day)", 
                min_value=int(bounds['sale_day']['min']), 
                max_value=int(bounds['sale_day']['max']), 
                value=15
            )
            
            sale_dayofweek = st.number_input(
                "День недели (sale_dayofweek, 0=пн, 6=вс)", 
                min_value=int(bounds['sale_dayofweek']['min']), 
                max_value=int(bounds['sale_dayofweek']['max']), 
                value=3
            )

        with col4:
            lat = st.number_input(
                "Широта (lat)", 
                min_value=float(bounds['lat']['min']), 
                max_value=float(bounds['lat']['max']), 
                value=47.5, 
                format="%.4f"
            )
            
            lng = st.number_input(
                "Долгота (long)", 
                min_value=float(bounds['long']['min']), 
                max_value=float(bounds['long']['max']), 
                value=-122.2, 
                format="%.4f"
            )
            
            sqft_living15 = st.number_input(
                "Ср. площадь 15 соседей (sqft_living15)", 
                min_value=float(bounds['sqft_living15']['min']), 
                max_value=float(bounds['sqft_living15']['max']), 
                value=1800.0, step=1.0
            )
            
            sqft_lot15 = st.number_input(
                "Ср. участок 15 соседей (sqft_lot15)", 
                min_value=float(bounds['sqft_lot15']['min']), 
                max_value=float(bounds['sqft_lot15']['max']), 
                value=7500.0, step=1.0
            )
    
    if st.button("Рассчитать стоимость", type="primary"):
        if selected_model == 'Keras':
            # только нужные 6 признаков
            input_data = pd.DataFrame([{
                'sqft_living': sqft_living,
                'grade': grade,
                'sqft_above': sqft_above,
                'lat': lat,
                'long': lng,
                'sqft_living15': sqft_living15
            }])

            # scaler = joblib.load('models/scaler_keras.pkl')
            # input_data = scaler.transform(input_data)
        
        # для всех остальных моделей (22 признака)
        else:
            input_data = pd.DataFrame([{
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft_living': sqft_living,
                'sqft_lot': sqft_lot,
                'floors': floors,
                'waterfront': waterfront,
                'view': view,
                'condition': condition,
                'grade': grade,
                'sqft_above': sqft_above,
                'sqft_basement': sqft_basement,
                'yr_built': yr_built,
                'yr_renovated': yr_renovated,
                'lat': lat,
                'long': lng,
                'sqft_living15': sqft_living15,
                'sqft_lot15': sqft_lot15,
                'sale_year': sale_year,
                'sale_month': sale_month,
                'sale_day': sale_day,
                'sale_dayofweek': sale_dayofweek,
                'years_existed': sale_year - yr_built + 1,
            }])
        
        try:
            prediction = models[selected_model].predict(input_data)

            # число из результата
            if hasattr(prediction, 'flatten'):
                prediction = float(prediction.flatten()[0])
            elif hasattr(prediction, '__getitem__'):
                prediction = float(prediction[0])
            else:
                prediction = float(prediction)
            
            st.markdown("---")
            st.subheader("Результат предсказания")
            
            col_result1, col_result2 = st.columns([1, 2])
            with col_result1:
                st.metric("Стоимость дома", f"${prediction:,.2f}")
            with col_result2:
                st.write(f"**Использованная модель:** {selected_model}")
                st.caption(f"Цена указана в долларах США (USD)")
            
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.info("Возможно, модели требуют другого набора признаков. Проверьте предобработку данных.")
else:
    st.error("Не удалось загрузить ни одну модель")

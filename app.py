import streamlit as st

st.set_page_config(
    page_title="РГР: ML Дашборд",
    page_icon="assets/icon.png",
    layout="wide"
)

st.title("Разработка Web-приложения для инференса моделей ML и анализа данных")

st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    try:
        st.image("assets/developer.jpg", width=200)
    except:
        st.warning("Разработчик не добавил фотографию")

with col2:
    st.header("Разработчик")
    st.subheader("ФИО: Пирогова Екатерина Алексеевна")
    st.subheader("Группа: ФИТ-241")
    st.subheader("Тема РГР: Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

st.markdown("---")

st.markdown("""
### Цель работы
Создать веб-приложение на Streamlit для демонстрации работы моделей машинного обучения 
и визуализации данных о продажах недвижимости.

### Задачи
1.  Разработать многостраничное приложение с информацией о разработчике, датасете и визуализациях.
2.  Реализовать страницу инференса для получения предсказаний от 6 обученных моделей ML.
3.  Развернуть приложение и подготовить отчет о проделанной работе.
""")

st.markdown("---")

st.subheader("Ссылки на проект")

col_link1, col_link2 = st.columns(2)

with col_link1:
    st.markdown("""
    ### GitHub репозиторий
    [Перейти на GitHub](https://github.com/KaPie2/rgr_ml)
    """)

with col_link2:
    st.markdown("""
    ### Streamlit Cloud
    [Открыть веб-приложение](https://khq63y967nhv6ww9qpwy7i.streamlit.app/)
    """)
import streamlit as st

st.set_page_config(page_title="О разработчике", page_icon="assets/icon.png")

st.title("О разработчике")

col1, col2 = st.columns([1, 3])

with col1:
    try:
        st.image("assets/developer.jpg", width=200)
    except:
        st.warning("Разработчик не добавил фотографию")

with col2:
    st.header("Пирогова Екатерина Алексеевна")
    st.subheader("Группа: ФИТ-241")
    st.subheader("Тема: Разработка Web-приложения для инференса моделей ML")

st.markdown("---")

st.subheader("Контакты:")
st.write("Email: pirogovakata7@gmail.com")
st.write("Телефон: +7 960 988 32 16")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Визуализации", page_icon="assets/icon.png")
st.title("Визуализация зависимостей в датасете")

df = pd.read_csv("data/house_sales_preprocessed.csv")
st.success(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")

st.markdown("---")

st.subheader("1. Распределение цены недвижимости")
plt.figure(figsize=(10, 5))
sns.histplot(df['price_k'], kde=True, color='steelblue')
plt.xlabel('Цена (тыс. $)')
plt.ylabel('Частота')
plt.title('Распределение цены недвижимости')
plt.grid(True, alpha=0.3)
st.pyplot(plt)

with st.expander("Описание графика"):
    st.write("""
    - **Правостороннее распределение** (длинный хвост справа)
    - Основная масса домов в диапазоне **0-1000 тыс. $**
    - Есть дома-выбросы стоимостью более **5000 тыс. $** (элитная недвижимость)
    """)

st.subheader("2. Зависимость цены от жилой площади")
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='sqft_living', y='price_k', alpha=0.3, s=15, color='green')
plt.xlabel('Жилая площадь (кв. футы)')
plt.ylabel('Цена (тыс. $)')
plt.title('Зависимость цены от жилой площади')
plt.grid(True, alpha=0.3)
st.pyplot(plt)

with st.expander("Описание графика"):
    st.write("""
    - **Положительная корреляция** между площадью и ценой
    - При увеличении площади растёт и цена
    - Видны несколько выбросов (очень дорогие дома с большой площадью)
    """)

st.subheader("3. Влияние качества дома на цену")

plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='grade', y='price_k', hue='grade', palette='coolwarm', legend=False)
plt.xlabel('Оценка качества (grade) - от 1 (плохо) до 13 (отлично)')
plt.ylabel('Цена (тыс. $)')
plt.title('Распределение цен в зависимости от качества дома')
plt.grid(True, alpha=0.3, axis='y')
st.pyplot(plt)

with st.expander("Описание графика"):
    st.write("""
    - Чем выше grade, тем выше **медианная цена** и шире разброс
    - При grade 10-12 появляются **дорогие выбросы** (элитное жилье)
    - Grade 1-5 встречается редко, данных мало
    """)

st.subheader("4. Корреляция признаков")
key_cols = ['price_k', 'sqft_living', 'grade', 'sqft_above', 'bathrooms', 'bedrooms', 'sqft_living15', 'view', 'condition']
corr_matrix = df[key_cols].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Корреляция признаков (Spearman)')
st.pyplot(plt)

with st.expander("Описание графика"):
    st.write("""
    ### Что сильно влияет на цену:
    - **grade** (качество) - 0.66
    - **sqft_living** (жилая площадь) - 0.64
    - **sqft_above** (площадь над землей) - 0.54
    
    ### Что слабо влияет:
    - **condition** (состояние) - 0.02
    - **bedrooms** (спальни) - 0.36
    
    ---
    ### Вывод:
    **Качество и размер дома** - главные факторы стоимости
    """)

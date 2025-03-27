import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Загружаем модель
loaded_model = joblib.load('best_actor_award_predictor.pkl')

# Определение всех признаков (соответствует обучению модели)
all_columns = ['Oscar Nominations', 'BAFTA Nominations', 'Golden Globe Nominations', 'BAFTA', 'Golden Globes']

# Заголовок приложения
st.title("Предсказание количества Оскаров для актёров")

# Ввод данных пользователя
oscars_nominations = st.number_input("Номинации на Оскар", min_value=0)
bafta_nominations = st.number_input("Номинации на BAFTA", min_value=0)
golden_globe_nominations = st.number_input("Номинации на Золотые глобусы", min_value=0)
bafta = st.number_input("BAFTA награды", min_value=0)
golden_globes = st.number_input("Золотые глобусы", min_value=0)

# Преобразование введенных данных в формат, который понимает модель
input_data = {
    'Oscar Nominations': oscars_nominations,
    'BAFTA Nominations': bafta_nominations,
    'Golden Globe Nominations': golden_globe_nominations,
    'BAFTA': bafta,
    'Golden Globes': golden_globes
}

# Заполняем отсутствующие признаки нулями
input_df = pd.DataFrame([input_data], columns=all_columns).fillna(0)

# Вывод входных данных для отладки
st.write("Входные данные для предсказания:")
st.write(input_df)

# Предсказание количества Оскаров
if st.button("Предсказать"):
    prediction = loaded_model.predict(input_df)

    # Выводим предсказанное значение
    st.success(f"Предсказанное количество Оскаров: {int(prediction[0])}")

    # Для дополнительной диагностики
    st.write("Предсказание:", prediction)

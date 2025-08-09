
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title("Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.2)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

Sex_male = 1 if sex == "male" else 0
Embarked_Q = 1 if embarked == "Q" else 0
Embarked_S = 1 if embarked == "S" else 0

input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [Sex_male],
    'Embarked_Q': [Embarked_Q],
    'Embarked_S': [Embarked_S]
})


def load_model():
    data = pd.read_csv("Titanic_train.csv")
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

model = load_model()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.subheader(f"Prediction: {result}")

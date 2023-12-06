import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv")


def clean_sm(x):
    x=np.where(x==1,1,0)
    return x



ss = s[["web1h","income","educ2","par","marital","gender","age"]]
ss = ss.dropna()

ss = ss[
    (ss["income"] < 10) &
    (ss["educ2"] < 9) &
    (ss["age"] < 98) ]

ss["par"] = np.where(ss["par"]==1,1,0)
ss["marital"] = np.where(ss["marital"]==1,1,0)
ss["female"] = np.where(ss["gender"]==2,1,0)
ss.drop("gender",inplace=True,axis=1)

ss["sm_li"] = clean_sm(ss["web1h"])
ss.drop("web1h",inplace=True,axis=1)

y = ss["sm_li"]
x = ss[["income","educ2","par","marital","age","female"]]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify=y, #check this; add class thing maybe remove
                                                   test_size=0.2,
                                                   random_state=1923)

lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)



st.markdown('''
    :rainbow[Linkedin User or Not?]:computer::smile::computer:''')
st.markdown('''
    :violet[Input income level, education level, parental status, marital status, age, and gender information to predict if a person is a Linkedin user or not.]''')


st.markdown("Income Level Options:")
st.markdown("1: 10k - under 20k")
st.markdown("2: 20k - under 30k")
st.markdown("3: 30k - under 40k")
st.markdown("4: 40k - under 50k")
st.markdown("5: 50k - under 75k")
st.markdown("6: 75k - under 100k")
st.markdown("7: 100k - under 150k")
st.markdown("8: 150k or more")
option1 = st.number_input("Income Level?", 1, 8, 1, 1)

st.markdown("Education Level Options:")
st.markdown("1: Less than high school (Grades 1-8 or no formal schooling)")
st.markdown("2: High school incomplete (Grades 9-11 or Grade 12 with NO diploma")
st.markdown("3: High school graduate (Grade 12 with diploma or GED certificate")
st.markdown("4: Some college, no degree (includes some community college")
st.markdown("5: Two-year associate degree from a college or university")
st.markdown("6: Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)")
st.markdown("7: Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)")
st.markdown("8: Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)")
option2 = st.number_input("Education Level?", 1, 8, 1, 1)

option3 = st.number_input("Parent? (1-Yes, 0-No)",0, 1, 1, 1)

option4 = st.number_input("Married? (1-Yes, 0-No)",0, 1, 1, 1)

option5 = st.number_input("Age? (Input any value between 18 and 97)", 18, 97, 18, 1)

option6 = st.number_input("Female? (1-Yes, 0-No)",0, 1, 1, 1)

person = [option1,option2,option3,option4,option5,option6]
predicted_class = lr.predict([person])
probs = lr.predict_proba([person])

if person:
    st.write(f"Predicted class (0 = Not a user; 1 = Linkedin user): {predicted_class[0]}")

if person:
    st.write(f"Probability that this person uses Linkedin: {probs[0][1]}")

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Life expectancy.csv")

df_value = df["Entity"].value_counts().keys()

for num, var in enumerate(df_value):
    num+=1
    df["Entity"].replace(var,num, inplace=True)

X = df.drop("Life expectancy", axis=1)
y = df["Life expectancy"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=123)

sc=StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("Life_expetancy_prediction.pkl")

def life_insurance_prediction(model,Entity, Year):
    for num,var in enumerate(df_value):
        if var == Entity:
            entity = num
    x = np.zeros(len(X.columns))
    x[0] = entity
    x[1] = Year
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Entity = request.form["Entity"]
    Year = request.form["Year"]
    
    predicated_price1 =life_insurance_prediction(model,Entity, Year)
    predicated_price = round(predicated_price1, 2)

    return render_template("index.html", prediction_text="Predicated price of bangalore House is {} RS".format(predicated_price))


if __name__ == "__main__":
    app.run()    
    
import numpy as np

from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
scaler= pickle.load(open("scalar.pkl",'rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
  float_features = [float(x) for x in request.form.values()]
  pre_final_features = [np.array(float_features)]
  final_features = scaler.transform(pre_final_features)
  prediction = model.predict(final_features)
  print('prediction value is ',prediction[0])
  if prediction[0]==0:
    output = "Water is not safe for consumption!!"
  else:
    output = "Water is safe for consumption!!"

  return render_template('index.html',prediction_text='Prediction Result:- {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

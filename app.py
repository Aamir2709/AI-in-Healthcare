from flask import Flask, render_template, url_for, request
import tensorflow as tf
from keras.models import load_model
import cv2
from PIL import Image
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("main.html")


# Stroke Disease
@app.route("/stroke")
def stroke():
    return render_template("stroke.html")


@app.route('/predictStroke', methods=["POST"])
def predictStroke():

    loaded_model = joblib.load('models/stroke/stroke_model.pkl')
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))


# Hepatitis
@app.route("/hepatitis")
def hepatitis():
    return render_template("hepatitis.html")


@app.route('/predictHepatitis', methods=["POST"])
def predictHepatitis():

    loaded_model = joblib.load('models/hepatitis/hepatitis_model.pkl')
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))


# ct scan images lung cancer detection
def predict_label(img_path):
	model = load_model("models/lung cancer/ct-scan-model.h5")

	img = cv2.imread(img_path)
	img = Image.fromarray(img)
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	return pred[0]

@app.route("/lung")
def lung():
	return render_template("lung.html")

@app.route("/preditLC", methods = ['GET', 'POST'])
def get_output():
	dic ={ 0:"Adenocarcinoma", 1:"Carcinoma", 2:"Normal", 3:"Squamous"}

	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/ct scan images/" + img.filename
		img.save(img_path)

		p = np.argmax(predict_label(img_path))
		# print(p)

	return render_template("lung.html", prediction = dic[p], img_path = img_path)


# Ocular disease
def predict_label1(img_path):
	model = load_model("models/eye disease/model.h5")

	img = cv2.imread(img_path)
	img = Image.fromarray(img)
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	return pred[0]


@app.route("/ocular")
def ocular():
	return render_template("ocular.html")


@app.route("/predictOcular", methods = ['GET', 'POST'])
def predictOcular():
	dic ={ 0:"No chance of disease", 1:"chance Of Ocular disease!"}

	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/ocular images/" + img.filename
		img.save(img_path)

		p = predict_label1(img_path)
		print(round(p[0]))

	return render_template("ocular.html", prediction = dic[round(p[0])], img_path = img_path)


# Skin cancer
def predict_label2(img_path):
	model = load_model("models/skin cancer/skin_model.h5")

	img = cv2.imread(img_path)
	img = Image.fromarray(img)
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	return pred[0]


@app.route("/skin")
def skin():
	return render_template("skin.html")


@app.route("/predictSkinC", methods = ['GET', 'POST'])
def predictSkinC():
	dic ={ 0:"Benign", 1:"Malignant!"}

	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/skin images/" + img.filename
		img.save(img_path)

		p = predict_label2(img_path)[0]
		print(np.round(p))

	return render_template("skin.html", prediction = dic[np.round(p)], img_path = img_path)

if __name__ == "__main__":
    app.run(debug=True)

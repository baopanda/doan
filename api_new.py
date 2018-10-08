import pickle
from os.path import join

import flask
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return flask.render_template('index2.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        text = request.form.get("text")
        pre = []
        pre.append(text)
        prediction = clf.predict(pre)
        for i in range(0,len(prediction)):
            prediction[i] = prediction[i].strip('\n')

        print(prediction)
        return render_template('index.html', label=prediction)



if __name__ == '__main__':
    load_file = open(join("models_new", 'LOCATION_new.pkl'), 'rb')
    clf = pickle.load(load_file)
    app.run(host='localhost', port=8000, debug=True)
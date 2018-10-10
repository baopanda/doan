import pickle
from os.path import join

from envs.ScrapyEnviroment.Lib import os
from flask import Flask, render_template, session, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

import PreProcessing_valid

MESSAGE = ["I'm sitting behind the screen!"
          ""
          ""          
          ""
          ""
          ""]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'abc'

bootstrap = Bootstrap(app)
moment = Moment(app)


class NameForm(FlaskForm):
    name = StringField('Input something here!', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    form = NameForm()
    if form.validate_on_submit():
        old_name = session.get('name')

        # if old_name is not None and old_name != form.name.data:
        #     flash('Do you thing there is a human sitting behind the screen!')
        session['name'] = PreProcessing_valid.PreProcessing(form.name.data)
        prediction = []
        pre = []
        pre1 = []
        predict_on_screen =""
        pre.append(session['name'])
        list_file = os.listdir("models_new")
        # print(list_file)
        for i in list_file:
            load_file = open(join("models_new", i), 'rb')
            clf = pickle.load(load_file)
            t = clf.predict(pre)
            prediction.append(t)
            print(t)
        for j in prediction:
            if(j != "None\n"):
                j = str(j).strip("\[]'n,")
                predict_on_screen = predict_on_screen+ j+", "
        # print(predict_on_screen)
        # predict_on_screen = predict_on_screen.strip('\n')
        pre1.append(predict_on_screen)
        session['name'] =str(pre1[0])
        session['name'] = session['name'].strip("\n")
        print(session['name'])
        return redirect(url_for('index'))
    return render_template('index.html', form=form, name=session.get('name'))

if __name__ == "__main__":
    # load_file = open(join("models_new", 'LOCATION_new.pkl'), 'rb')
    # print("1")
    # clf = pickle.load(load_file)
    app.run(host="localhost", port=7000, debug=True)

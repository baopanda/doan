import flask
from flask import Flask, render_template
app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"
    # return flask.render_template('index.html')
if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
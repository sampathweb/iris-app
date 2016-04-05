from flask import Flask
import pickle


app = Flask(__name__)
app.config.from_object("app.config")

# unpickle my model
estimator = pickle.load(open('models/iris_model.pkl'))
target_names = ['setosa', 'versicolor', 'virginica']

from .views import *


# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    """Page Not Found"""
    return render_template('404.html'), 404

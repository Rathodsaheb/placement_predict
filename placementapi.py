from flask import Flask, render_template
# from flask import Flask
from flask import request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)


@app.route('/')
def home():
    #    return 'Hello'
    return render_template('placement_html.html')

@app.route('/home1', methods=['Get', 'Post'])
def index():
    errors = []
    results = {}
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form.to_dict()
        b = pd.DataFrame(form_data, index=[0])
        filename = 'RF_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        predict = loaded_model.predict(b)
        p = float(predict)
        if p == 1.0:
            f = 'The candidate is placed'
        else:
            f = 'The candidate is not placed'

        return render_template("presult.html", prediction=f)


if __name__ == '__main__':
    app.run(debug=True)

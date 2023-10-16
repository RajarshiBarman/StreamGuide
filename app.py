
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/homepage')
def homepage():
    return render_template("homepage.html")


@app.route('/about')
def about():
    return render_template("aboutus.html")


@app.route('/test')
def test():
    return render_template("test.html")


@app.route('/contact')
def contact():
    return render_template("contacts.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        i = 0
        print(result)
        res = result.to_dict(flat=True)
        print("res:", res)
        arr1 = res.values()
        arr = ([int(value) for value in arr1])

        data = np.array(arr)

        data = data.reshape(1, -1)
        print(data)
        loaded_model = pickle.load(open("model.pkl", 'rb'))
        predictions = loaded_model.predict(data)
       # return render_template('testafter.html',a=predictions)

        print(predictions)

        

        return render_template("results.html",final=predictions[0])


if __name__ == "__main__":
    app.run(debug=True)

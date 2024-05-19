from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import pickle
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

def create_app():
    
    # app = Flask(__name__)

    # @app.route('/')

    # def greeting():
    #    return 'Flask is Awesome!'
    
    model = pickle.load(open("example_weigth_knn.pkl", "rb"))

    @app.route('/')

    def use_template():
        return render_template("index.html")
    
    @app.route('/predict', methods=["POST", "GET"])

    def predict():
        input_one = request.form['1']
        input_two = request.form['2']
        input_three = request.form['3']
        input_four = request.form['4']
        input_five = request.form['5']
        input_six = request.form['6']
        input_seven = request.form['7']
        input_eight = request.form['8']

        setup_df = pd.DataFrame([pd.Series ([input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight])])

        diabetes_prediction = model.predict_proba(setup_df)
        output = '{0:.{1}f}'.format(diabetes_prediction[0][1],2)
        output = str(float(output) * 100) + '%'

        if output > str(0.5):
            return render_template('result.html', pred=f'you have following chance of having diabeties based on our KNN model. \nProbability of having Diabetes is {output}')
        else:
            return render_template('result.html', pred=f'you have a low chance of diabeties which is currently considered safe (this in only example, please consult a certified doctor for any medical advice).\n Probability of having Diabetes is {output}')
    
    if __name__ == '__main__':
        app.run(debug = True)
        # app.run(host = '0.0.0.0', port=80)

 
    return app

app = create_app()
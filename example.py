import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# df = pd.read_csv('/Users/tushar/abc/flask/diabetes.csv')
df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

#knn_score = knn.score(x_test, y_test)

pickle.dump(knn, open('example_weigth_knn.pkl', "wb"))

"""
Debug code

 app = Flask(__name__)

    @app.route('/')

    def index():
    
        if __name__ == '__main__':
            app.run(debug=True)

"""

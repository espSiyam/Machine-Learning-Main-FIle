from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Importing the dataset
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

    # Cleaning the texts
    import re
    import nltk

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

        # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    clf.score(X_test,y_test)



    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        review = re.sub('[^a-zA-Z]', ' ', str(data))
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        review = [review]
        vect = cv.transform(review).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
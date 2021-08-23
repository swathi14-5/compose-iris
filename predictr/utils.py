import pickle
from sklearn.naive_bayes import GaussianNB

# define a Gaussain NB classifier
clf = GaussianNB()

# define the class encodings and reverse encodings
classes = {0: "Astrometry", 1: "imaging", 2: "transit"}
r_classes = {y: x for x, y in classes.items()}


# function to load the model
def load_model():
    global clf
    clf = pickle.load(open("models/planets_nb.pkl", "rb"))


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    return classes[prediction]

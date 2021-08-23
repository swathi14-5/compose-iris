import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "Astrometry", 1: "imaging", 2: "transit"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "number": d.number,
            "orbital_period": d.orbital_period,
            "mass": d.mass,
            "distance": d.distance,
            "method_class": d.method_class,
        }
         
    
        for d in data
    ]

    return processed

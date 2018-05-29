from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer
import bisect
import numpy as np
import pandas as pd 

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_bin = {}
    
    def fit(self, X, y=None):
        self.cats = list(X.select_dtypes(include='object').columns)
        for c in self.cats:
            self.label_bin[c] = LabelBinarizer()
            self.label_bin[c].fit(X[c])
            
            lb_classes = self.label_bin[c].classes_.tolist()
            bisect.insort_left(lb_classes, 'other')
            self.label_bin[c].classes_ = lb_classes
            
        return self
            
    def transform(self, X):
        X = X.reset_index(drop=True).copy()
        new_dfs = []

        for c in self.cats:
            names = [''.join([c + "==", j]) for j in self.label_bin[c].classes_]
                
            repl = list(set(list(X[c])) - set(list(self.label_bin[c].classes_)))
            
            if repl:
                X[c] = X[c].replace(repl, "other")

            arr = self.label_bin[c].transform(X[c])

            if len(names) == 2:
                arr = np.concatenate(((~arr.astype(bool)).astype(int), arr), axis=1)

            df = pd.DataFrame(arr, columns=names)
            new_dfs.append(df)
                
        dummies = pd.concat(new_dfs, axis=1)
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(self.cats, axis=1)
                          
        return X
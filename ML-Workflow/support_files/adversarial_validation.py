class adversarial_validation():
    def __init__(self, sample_1, sample_2, target, id, clf):
        self.sample_1 = sample_1
        self.sample_2 = sample_2
        self.target = target
        self.id = id
        self.clf = clf

    def check_dist(self):
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        # Check if variable type
        categorical_vars =[]
        numerical_vars=[]
        for i in list(self.sample_1.drop([self.target, self.id], axis=1)):
            if self.sample_1[i].dtype == 'O':
                categorical_vars.append(i)
            else:
                numerical_vars.append(i)
        
        # One-Hot-Encode categorical variables
        # Train on sample_1, apply on sample_2
        for i in categorical_vars:
            lb_ = LabelBinarizer()
            lb_.fit(self.sample_1[i]) # Learn encoder on training data

            self.sample_1[lb_.classes_] = pd.DataFrame(lb_.transform(self.sample_1[i]), columns=lb_.classes_)
            self.sample_2[lb_.classes_] = pd.DataFrame(lb_.transform(self.sample_2[i]), columns=lb_.classes_)

            self.sample_1 = self.sample_1.drop(i, axis=1)
            self.sample_2 = self.sample_2.drop(i, axis=1)                
        
        # Identify features
        features = [f for f in list(self.sample_1) if f not in [self.target, self.id]]        
        
        # Split between sample 1 and sample 2 datasets and prepare datasets for training
        sample_1_check = self.sample_1[features + [self.target, self.id]].copy()  # Train dataset
        sample_2_check = self.sample_2[features + [self.target, self.id]].copy()  # Test dataset

        # Mark sample_1 observations as 1 and sample_2 observations as 0
        sample_1_check['TARGET_adv'] = 0
        sample_2_check['TARGET_adv'] = 1

        # Concatenate sample_1 and sample_2 datasets into one dataset
        adv_val_set = pd.concat((sample_1_check, sample_2_check))
        adv_val_set.reset_index(inplace=True, drop=True)

        # Split features and target
        adv_val_x = adv_val_set.drop(["TARGET_adv", self.target], axis=1)
        adv_val_y = adv_val_set.TARGET_adv

        # Predict observations to be sample_1 or sample_2 - based on model built when observation was in cv test set

        adv_val_clf = self.clf

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=55555)  # stratified k-fold CV

        predictions = cross_val_predict(adv_val_clf, X=adv_val_x.drop(self.id, axis=1), y=adv_val_y,
                                        cv=skf, n_jobs=-1, method='predict_proba')

        print("AUC is: " + str(roc_auc_score(adv_val_y, predictions[:, 1])))

        return adv_val_set, predictions

    def get_scored_obs(self, adv_val_set, predictions, get_plot=False):
        import pandas as pd
        import matplotlib.pyplot as plt

        # Sort the sample_1 points by their estimated probability of being sample_2 examples

        # Get adversarial prediction into the adversarial dataset
        adv_val_set['prob_being_sample_2'] = predictions[:, 1]

        # Sort observations by the adversarial prediction
        adv_val_set_sorted = adv_val_set.sort_values(by=["TARGET_adv", "prob_being_sample_2"])

        # Get plot
        if get_plot:
            adv_val_set_sorted.prob_being_sample_2.hist()
            plt.axvline(x=0.5, color="red")
            plt.title("Distribution of probability of being an observation from sample 2")
            plt.show()

        return adv_val_set_sorted

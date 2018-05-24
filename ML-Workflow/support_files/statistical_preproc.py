def cramersv(feature1, feature2):
    from scipy.stats import chi2_contingency
    import pandas as pd
    
    # Get the cross tab frequencies
    confusion_matrix = pd.crosstab(feature1, feature2)

    # Calculate the chi2 from that contigency table
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
  
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def correlation_analysis(df, corr_limit=0.8):
    
    """ Return a table with variables that have correlation above the given limit (corr_limit) and the number of 
    correlations with other variables they have above this threshold. """
    
    import pandas as pd
    
    corr = df.corr().stack().reset_index()
    corr.columns = ['variable 1', 'variable 2','correlation']
 
    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    corr=corr.loc[ (corr['correlation'] > corr_limit) & 
                  (corr['variable 1'] != corr['variable 2']) ].sort_values("correlation", ascending=False)
    
    corr_net= corr['variable 1'].value_counts().reset_index()
    corr_net = corr_net[corr_net['variable 1'] >= 1]
    corr_net.columns = ['Variable', '# Correlations > limit']
    
    return corr, corr_net


def univ_auc(df, features, target):
    
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score    

    model_ = LogisticRegression()
    
    save=[]
    for i in features:
        model_.fit(df[[i]].fillna(df[i].median()), 
                         df[target])
        save.append([i,roc_auc_score(df[target],
                                     model_.predict_proba(df[[i]].fillna(df[i].median()))[:, 1])])

    auc_df = pd.DataFrame(save, columns=["variable", "AUC"])
    
    return auc_df
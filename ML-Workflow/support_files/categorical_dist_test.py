def categorical_dist_test(sample_1, sample_2, vars, p_value=0.05):
    
    """Null hypothesis : the two distributions are the same

    p_value above threshold: don't reject - assume they have the same distribution"""
    
    import scipy.stats as stats
    import pandas as pd
    
    results = []
    
    for i in vars:        
        observed = pd.crosstab(index=sample_1[i], columns="count")

        _ratios = pd.crosstab(index=sample_2[i], columns="count")/len(sample_2)  # Get population ratios

        expected = _ratios * len(sample_1) 

        crit, pvalue = stats.chisquare(f_obs= observed, f_exp= expected)
        
        results.append([i,pvalue[0]])
        
    results = pd.DataFrame(results, columns=["variable","p_value"])
    print("Variables with unequal distribution: ")
    
    return results[results["p_value"]<p_value]
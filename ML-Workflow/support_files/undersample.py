def undersample(df, target, odds, seed):
    
    """
    Creates a sample of Non-Events from the original table with the given odds, keeping all the target events.
    
    Parameters:
    ----------
    df : pandas.core.frame.DataFrame
        A pandas dataframe
    target : string
        A string with the name of the target variable
    odds : int
        The number corresponding to the odds of Non-Events / Events
    seed : int
        Random seed for sampling reproducibility        

    
    Attributes:
    -------
    
    Returns:
    -------
    DataFrame (pandas.core.frame.DataFrame)
        A dataframe sampled from the original DataFrame with the given odds ratio proportion regarding the target variable

    ----------
    References:
    ----------
    
    """    
    
    # Undersample majority class "0" (non-defaults)
    # Calculate % value for sample, based on exact number of observations in class "1"
    sample_number = df[df[target]==1].shape[0]
    sample_ratio = (sample_number) / float(df[df[target]==0].shape[0])

    class_1 = df[df[target]==1]
    class_0 = df[df[target]==0].sample(frac=sample_ratio*odds, random_state=seed)

    dataset_sampled = class_0.append(class_1, ignore_index=True)
    del class_1, class_0

    # Shuffle dataset rows
    dataset_sampled = dataset_sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return dataset_sampled
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_nulls(df):
    nulls_dataset = df.copy()            
    numerical_vars = list(nulls_dataset.select_dtypes(include=["int","float"]).columns)

    null_class = []
    for i in nulls_dataset[numerical_vars].columns.tolist():
        if nulls_dataset[nulls_dataset[i].isnull()].shape[0] > 0:
            null_class.append(i)

    return null_class


def _optimal_binner(series, targets, min_bin_percent=0.05):
    assert not targets.isnull().sum(), "targets can't have nulls"

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree._tree import TREE_UNDEFINED

    # in case of a null series
    n = series.nunique(dropna=True)  # nan -> 1
    if n == 0:
        return []
    # in case of a unique series
    if n == 1:
        return 1

    mask_dropna = series.notnull()
    series, y_train = series[mask_dropna], targets[mask_dropna]
    min_samples_leaf = max(int(series.size * min_bin_percent), 1)

    tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf)
    tree.fit(series.values.reshape(-1, 1), y_train)

    thresholds = sorted(filter(lambda t: t != TREE_UNDEFINED, tree.tree_.threshold))

    return [series.min()] + thresholds + [series.max()]


def get_optimized_bins(df, target_var, id_var, vars_to_bin, min_bin_percent=0.1, min_bin_percent_low=0.01,
                       get_plots=False, y_axis="% of observations", max_classes_plot=30, save_plots=False, save_plots_path=None):
    """
    Creates a table with every feature passed optimally binned, with statistics for each bin.
    
    Parameters:
    ----------
    df : pandas.core.frame.DataFrame
        A pandas dataframe
    target_var : string
        A string with the name of the Event variable
    id_var : string or int
        A string or int which uniquely identifies a customer/loan
    vars_to_bin : list
        A list with the variable(s) to be binned
    min_bin_percent : float (default=0.1)
        A float corresponding to the minimum percentage of observations to include in each bin (for numericals only)
    min_bin_percent_low : float (default=0.01)
        A float corresponding to the minimum percentage of observations to include in each bin (for numericals only), 
        when all the observations get grouped into a single bin an additional drill-down for those variables is necessary       
    get_plots : bool, optional (default=False)
        Whether to plot vars_to_bin or not
    y_axis : string    
        A string which allows to choose between having absolute values (pass '# of observations') or 
        relative values (pass '% of observations') on the y axis of the plot
    max_classes_plot : int (default=30)
        An int for the maximum number of classes a categorical feature can have to be plotted        
    save_plots : bool, optional (default=False)
        Whether to save the plots in a given directory or not    
    save_plots_path: string, optional (default=None)
        Thee path where the images should be saved

    Attributes:
    -------

    Returns:
    -------
    DataFrame (pandas.core.frame.DataFrame)
        A dataframe with all the bins created, respective statistics and optionally the plots

    ----------
    References:
    ----------
    *Author* *Hugo Veiga*

    """

    # Identify numerical and categorical variables
    discard = ["target", "id"]
    categorical_vars = [x for x in list(df[vars_to_bin].select_dtypes(include="object").columns) if x not in discard]
    bool_vars = [x for x in list(df[vars_to_bin].select_dtypes(include="bool").columns) if x not in discard]
    numerical_vars = [x for x in list(df[vars_to_bin].select_dtypes(include=["int","float"]).columns) if x not in discard]

    binned_dfs_num = []
    binned_dfs_cat = []
    null_class = _get_nulls(df[numerical_vars])
    bin_dataset = df.copy()

    for i in list(bin_dataset):
        bin_dataset[i] = bin_dataset[i].replace([np.inf, -np.inf], 0)

    # If feature only has missing values, don't analyze it
    vars_to_bin_no_nulls = [i for i in numerical_vars + categorical_vars + bool_vars if df[i].isnull().sum() != df.shape[0]]

    for i in vars_to_bin_no_nulls:
            
        # Calculate Bins, different for numerical vs. categorical
        if i in numerical_vars:            
            _pre_bins = _optimal_binner(bin_dataset[i],bin_dataset[target_var], min_bin_percent=min_bin_percent)

            if type(_pre_bins) == list:
                length = len(_pre_bins)
            else:
                length = _pre_bins
                
            if length < 3:
                bins=_optimal_binner(bin_dataset[i],bin_dataset[target_var], min_bin_percent=min_bin_percent_low)
            else:
                bins=_pre_bins
                
            bin_dataset[i+"_Binned"] = pd.cut(bin_dataset[i], bins=bins, include_lowest=True, retbins=True, precision=1)[0] 

        elif i in categorical_vars:
            bin_dataset[i+"_Binned"] = bin_dataset[i].fillna("Missing")
            
        elif i in bool_vars:            
            bin_dataset[i] = bin_dataset[i].astype(int)
            bin_dataset[i+"_Binned"] = bin_dataset[i].fillna("Missing")

        # Create grouped table
        grp = bin_dataset.groupby(i + "_Binned").agg({target_var: ["mean", "sum"], id_var: "count"}).reset_index()
        grp.columns = ["_".join(x) for x in grp.columns.ravel()]
        grp = grp.rename(columns={target_var + "_mean": 'Event Rate', target_var + "_sum": '# Events in group',
                                  id_var + "_count": id_var, i + "_Binned_": i + "_Binned"})

        grp["Variable"] = i
        grp["% of observations"] = grp[id_var] / bin_dataset.shape[0]
        grp = grp.rename(columns={i + "_Binned": "Bin", id_var: "# of observations"})
        grp = grp[["Bin", "Event Rate", "# Events in group", "# of observations", "Variable", "% of observations"]]

        if i in numerical_vars:

            # If the variable has NaNs calculate metrics for the NaN group
            if i in null_class:
                # Calculate % of NaNs
                nan_perc = pd.DataFrame(bin_dataset[i].value_counts(normalize=False, dropna=False)).reset_index()
                add_nans_row = ["Missing",
                                bin_dataset[bin_dataset[i].isnull()][target_var].mean(),
                                bin_dataset[bin_dataset[i].isnull()][target_var].sum(),
                                nan_perc[nan_perc["index"].isnull()][i].tolist()[0],
                                i,
                                nan_perc[nan_perc["index"].isnull()][i].tolist()[0] / bin_dataset.shape[0]]
                grp.loc[len(grp)] = add_nans_row

        grp = grp.reset_index()
        grp = grp.rename(columns={"index": "Bin order"})
        grp["Bin order"] = grp["Bin order"].astype(int)

        grp["# Total events"] = bin_dataset[target_var].sum()
        grp["# Total non-events"] = bin_dataset.shape[0] - bin_dataset[target_var].sum()
        grp["# Non-events in group"] = grp["# of observations"] - grp['# Events in group']

        # Append bins table for numeric and categorical features

        if i in numerical_vars:
            binned_dfs_num.append(grp)

        else :
            binned_dfs_cat.append(grp)

        bin_dataset = bin_dataset.drop(i, axis=1)

    # Sort numeric and categorical tables (categoricals sorted to give a better looking plot - linear trend)

    if (len(binned_dfs_num) != 0) & (len(binned_dfs_cat) != 0):
        Bins_table_num = pd.concat(binned_dfs_num)
        Bins_table_num = Bins_table_num.sort_values(["Variable", "Bin order"])
        Bins_table_cat = pd.concat(binned_dfs_cat)
        Bins_table_cat = Bins_table_cat.sort_values(["Variable", "Event Rate"], ascending=False)

        Bins_table = pd.concat([Bins_table_num, Bins_table_cat])

    elif (len(binned_dfs_num) != 0) & (len(binned_dfs_cat) == 0):
        Bins_table_num = pd.concat(binned_dfs_num)
        Bins_table_num = Bins_table_num.sort_values(["Variable", "Bin order"])

        Bins_table = pd.concat(binned_dfs_num)

    elif (len(binned_dfs_num) == 0) & (len(binned_dfs_cat) != 0):
        Bins_table_cat = pd.concat(binned_dfs_cat)
        Bins_table_cat = Bins_table_cat.sort_values(["Variable", "Event Rate"], ascending=False)
        
        Bins_table = pd.concat(binned_dfs_cat)

    Bins_table = Bins_table[["Bin order", "Variable", "Bin", "# of observations", "% of observations", "Event Rate"]]

    # Create plots
    if get_plots:

        for i in Bins_table.Variable.unique().tolist():

            if Bins_table[Bins_table["Variable"] == i].shape[0] <= max_classes_plot:

                ax = Bins_table[Bins_table["Variable"] == i][y_axis].plot(alpha=.75,
                                                                          kind='bar', color="#FF2000")
                ax2 = ax.twinx()
                ax2.plot(ax.get_xticks(), Bins_table[Bins_table["Variable"] == i]["Event Rate"].values, alpha=.75,
                         color='#0C0C0C')
                ax2.grid(False)
                ax.set_xlabel('Bins')
                ax.set_ylabel('Population')
                ax2.set_ylabel("Event Rate")
                ax.set_title(i)
                ax.xaxis.grid(False)
                ax.set_xticklabels([j for j in np.array(Bins_table[Bins_table["Variable"] == i]["Bin"].map(str))])

                if save_plots:
                    plt.savefig(save_plots_path + i + ".png", bbox_inches='tight')

                plt.show()

    return Bins_table

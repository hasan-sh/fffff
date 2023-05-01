import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd
import seaborn as sns

def make_cm(clf, pred, y_test, target_names, to_labels=None):
    
    if to_labels:
        target_names = [f'{to_labels[i]} ({i})' for i in target_names]
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    ax.xaxis.set_ticklabels(target_names, rotation = 90)
    ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(
        f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
    )
    
    
    
def create_dt_decisions(tree, feature_names, target_names, save=False, to_labels=None):
    
    if to_labels:
        target_names = [f'{to_labels[i]} ({i})' for i in target_names]
        
    fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (64, 64), dpi=300)
    plot_tree(tree,
                   feature_names = feature_names, 
                   class_names= target_names,
                   filled = True);
    if save:
        fig.savefig('result.png')
    
    
def plot_feature_importance(importance, names, model_type, top_n=20):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names, 'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    fi_df = fi_df[:top_n]
    
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.barh(x=fi_df['feature_importance'], y=fi_df['feature_names'],  alpha=0.5)

#     ax.set_xlabel('Average Feature Importance')
#     ax.legend()
#     ax.set_title(f'Top {top_k} Feature Importance per Target Class')
    
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.legend()
    plt.show()
    
def features_importance_rf(clf, feature_names, top_n=20):

    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(clf.feature_importances_, index=feature_names)
    forest_importances.sort_values(ascending=False, inplace=True)

    forest_importances = forest_importances[:top_n]
    std = std[:top_n]
    fig, ax = plt.subplots()

    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
    
    

def plot_feature_effects(clf, X_train, feature_names, target_names, verbose=False, to_labels=None):        
    # learned coefficients weighted by frequency of appearance
    average_feature_effects = clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
    
    target_names = np.sort(target_names)
    
    if to_labels:
        target_names = [f'{to_labels[i]} ({i})' for i in target_names]
    
    for i, label in enumerate(target_names):
        top5 = np.argsort(average_feature_effects[i])[-5:][::-1]
        if i == 0:
            top = pd.DataFrame(feature_names[top5], columns=[label])
            top_indices = top5
        else:
            top[label] = feature_names[top5]
            top_indices = np.concatenate((top_indices, top5), axis=None)
    top_indices = np.unique(top_indices)
    predictive_words = feature_names[top_indices]

    # plot feature effects
    bar_size = 0.25
    padding = 0.75
    y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(target_names):
        ax.barh(
            y_locs + (i - 2) * bar_size,
            average_feature_effects[i, top_indices],
            height=bar_size,
            label=label,
        )
    ax.set(
        yticks=y_locs,
        yticklabels=predictive_words,
        ylim=[
            0 - 4 * bar_size,
            len(top_indices) * (4 * bar_size + padding) - 4 * bar_size,
        ],
    )
    ax.legend(loc="lower right")
    
    if verbose:
        print("top 5 keywords per class:")
        print(top)

    return ax


    # _ = plot_feature_effects().set_title("Average feature effect on the original data")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import plotly.graph_objects as go


def make_cms(clf_performances, target_names, to_labels=None, iteration=0):
    """
    Generate and display multiple confusion matrices for each classifier's performance.

    Args:
        clf_performances (list): List of tuples containing the classifier, predicted labels, and true labels.
        target_names (list): List of target names or labels.
        to_labels (list, optional): List of labels to replace the original target names. Defaults to None.
        iteration (int, optional): Iteration number. Defaults to 0.
    """
    fig_title = f'output/confusion_matrix/{iteration}_{"_".join(target_names)}.png'
    
    fig, ax = plt.subplots(1, len(clf_performances), figsize=(12, 6), sharey=True, sharex=True)
    
    for i, (clf, pred, y_test) in enumerate(clf_performances):
        if to_labels:
            target_names = [f'{to_labels[i]} ({i})' for i in clf.classes_]
        else:
            target_names = clf.classes_

        cf_matrix = confusion_matrix(y_test, pred)
        sns.heatmap(cf_matrix, ax=ax[i], cbar=i==len(clf_performances)-1, annot=True, fmt=".0f")
        
        ax[i].xaxis.set_ticklabels(target_names, rotation=90)
        ax[i].yaxis.set_ticklabels(target_names, rotation=0)
        
        if i == 0:
            ax[i].set_ylabel('True Label')
        if i == 1:
            ax[i].set_xlabel('Predicted Label')
            
        _ = ax[i].set_title(f"{clf.__class__.__name__}")

    fig.tight_layout()
    fig.show()
    plt.savefig(fig_title)


def make_cm(clf, pred, y_test, target_names, to_labels=None, iteration=0):
    """
    Generate and display a confusion matrix for a single classifier's performance.

    Args:
        clf (Classifier): The classifier model.
        pred (array-like): Predicted labels.
        y_test (array-like): True labels.
        target_names (list): List of target names or labels.
        to_labels (list, optional): List of labels to replace the original target names. Defaults to None.
        iteration (int, optional): Iteration number. Defaults to 0.
    """
    fig_title = f'output/{iteration}_{clf.__class__.__name__}_{"_".join(target_names)}.png'
    
    if to_labels:
        target_names = [f'{to_labels[i]} ({i})' for i in clf.classes_]
    else:
        target_names = clf.classes_
        
    fig, ax = plt.subplots(figsize=(10, 5))
    
    cf_matrix = confusion_matrix(y_test, pred)
    sns.heatmap(cf_matrix, ax=ax, annot=True, fmt=".0f")
    
    ax.xaxis.set_ticklabels(target_names, rotation=90)
    ax.yaxis.set_ticklabels(target_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    _ = ax.set_title(f"{clf.__class__.__name__}")

    fig.show()
    plt.savefig(fig_title)    
    

def plot_per_culture(df, model, with_incorrect=False, top_k=5):
    """
    Plot the distribution of correct and incorrect classifications per culture.

    Args:
        df (DataFrame): The data frame containing the classification results.
        model (str): The model name.
        with_incorrect (bool, optional): Include incorrect classifications. Defaults to False.
        top_k (int, optional): Number of top cultures to display. Defaults to 5.

    Returns:
        fig (Figure): The generated figure.
    """
    correct = df[df[f'predicted_{model}'] == df['ocms']]

    fig = go.Figure(
        layout_height=400,
        layout_title=f'Top {top_k} Correct Cultures.',
        layout_title_x=0.5,
        layout_xaxis_title='Correctly Classified',
        layout_yaxis_title='Cultures',
    )
    culture_dict = defaultdict(int)
    for cul in correct['culture'].items():
        culture_dict[cul[1]] += 1
    culture_dict = {k: v for k, v in culture_dict.items() if v >= top_k}

    culture_dict = sorted(culture_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    fig.add_bar(x=[a for a, _ in culture_dict], y=[b for _, b in culture_dict], name='correct')
    
    if with_incorrect:
        culture_dict = defaultdict(int)
        incorrect = df[df[f'predicted_{model}'] != df['ocms']]
        for cul in incorrect['culture'].iteritems():
            culture_dict[cul[1]] += 1
        culture_dict = {k: v for k, v in culture_dict.items() if v >= top_k}
        fig.add_bar(x=[a for a, _ in culture_dict], y=[b for _, b in culture_dict], name='incorrect')
    
    if not culture_dict:
        return fig
        
    fig.show()
    return fig
    
    

def plot_feature_importance(importance, names, model_type, top_n=20):
    """
    Plot the feature importance of a model.

    Args:
        importance (array-like): Array of feature importances.
        names (array-like): Array of feature names.
        model_type (str): The type or name of the model.
        top_n (int, optional): Number of top features to display. Defaults to 20.

    Returns:
        fi_df['feature_names'] (array-like): Array of top feature names.
    """
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi_df = fi_df[:top_n]
    
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.legend()
    plt.show()
    
    return fi_df['feature_names']


def features_importance_rf(clf, feature_names, top_n=10):
    """
    Plot the feature importance using Random Forest model.

    Args:
        clf (Classifier): The trained Random Forest classifier.
        feature_names (array-like): Array of feature names.
        top_n (int, optional): Number of top features to display. Defaults to 10.
    """
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(clf.feature_importances_, index=feature_names)
    forest_importances.sort_values(ascending=False, inplace=True)

    forest_importances = forest_importances[:top_n]
    std = std[:top_n]
    fig, ax = plt.subplots()

    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title(f"{clf.__class__.__name__}: Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()    


def plot_feature_effects(clf, X_train, feature_names, target_names, to_labels=False, top_k=10, verbose=False, iteration=0):
    """
    Plot the feature effects per target class for a given classifier.

    Args:
        clf (Classifier): The trained classifier.
        X_train (array-like): The training data.
        feature_names (array-like): Array of feature names.
        target_names (array-like): Array of target names.
        to_labels (bool or array-like, optional): Mapping of target labels. Defaults to False.
        top_k (int, optional): Number of top features to display. Defaults to 10.
        verbose (bool, optional): Whether to display top features per class. Defaults to False.
        iteration (int, optional): Iteration number. Defaults to 0.

    Returns:
        top_k_words (array-like): Array of top feature names.
    """
    fig_title = f'output/feature_effect/{clf.__class__.__name__}_{iteration}_{"_".join(target_names)}.png'
    
    average_feature_effects = abs(clf.coef_) * np.asarray(X_train.mean(axis=0)).ravel()

    target_names = np.sort(target_names)
    if to_labels:
        target_names = [to_labels[i] for i in target_names]
        
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(target_names):
        top_indices = np.argsort(average_feature_effects[i if len(average_feature_effects) > 1 else 0])[::-1][:top_k]
        top_k_words = feature_names[top_indices]
        top_k_effects = average_feature_effects[i if len(average_feature_effects) > 1 else 0, top_indices]
        ax.barh(top_k_words, top_k_effects, label=label, alpha=0.5)

    ax.set_xlabel('Average Feature Effects')
    if len(average_feature_effects) > 1:
        ax.legend()
    ax.set_title(f'{clf.__class__.__name__}: Feature Effects per Target Class (iter.={iteration})')

    if verbose:
        top_k_keywords = pd.DataFrame()
        for i, label in enumerate(target_names):
            top_indices = np.argsort(average_feature_effects[i if len(average_feature_effects) > 1 else 0])[::-1][:top_k]
            top_k_words = feature_names[top_indices]
            top_k_keywords[label] = top_k_words
        print(f"Top {top_k} Keywords per Class:\n{top_k_keywords}")
    plt.show()
    fig.tight_layout()
    fig.savefig(fig_title)
    
    return top_k_words
     
   

def plot_feature_effects_detailed(clf, X_train, feature_names, target_names, verbose=False, to_labels=None):        
    """
    Plot the feature effects per target class for a given classifier. A more detailed plot.

    Args:
        clf (Classifier): The trained classifier.
        X_train (array-like): The training data.
        feature_names (array-like): Array of feature names.
        target_names (array-like): Array of target names.
        to_labels (bool or array-like, optional): Mapping of target labels. Defaults to False.
        verbose (bool, optional): Whether to display top features per class. Defaults to False.

    Returns:
        predictive_words (array-like): Array of top feature names.
    """
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
    y_locs = np.arange(len(top_indices)) * (6 * bar_size + padding)

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
    plt.show()
    return predictive_words

    
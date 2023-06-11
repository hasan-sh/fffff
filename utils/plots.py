import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import plotly.graph_objects as go




def make_cms(clf_performances, target_names, to_labels=None, iteration=0):
    fig_title = f'output/confusion_matrix/{iteration}_{"_".join(target_names)}.png'
    

        
    fig, ax = plt.subplots(1, len(clf_performances), figsize=(12, 6),  sharey=True, sharex=True)
    
    # print(target_names, clf.classes_)

    for i, (clf, pred, y_test) in enumerate(clf_performances):
        if to_labels:
            target_names = [f'{to_labels[i]} ({i})' for i in clf.classes_]
        else:
            target_names = clf.classes_


        cf_matrix = confusion_matrix(y_test, pred)
        # print(cf_matrix)
        sns.heatmap(cf_matrix, ax=ax[i], cbar=i==len(clf_performances)-1, annot=True, fmt=".0f")

        # disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax[i], display_labels=target_names)
        # # disp.ax_.set_title(key)
        # disp.im_.colorbar.remove()
        # disp.ax_.set_xlabel('')
        # if i != 0:
        #     disp.ax_.set_ylabel('')
        
        ax[i].xaxis.set_ticklabels(target_names, rotation = 90)
        ax[i].yaxis.set_ticklabels(target_names, rotation = 0)
        
        if i == 0:
            ax[i].set_ylabel('True Label')
        if i == 1:
            ax[i].set_xlabel('Predicted Label')
            
        _ = ax[i].set_title(
            f"{clf.__class__.__name__}"
        )

    # fig.colorbar(disp.im_, ax=ax)
    fig.tight_layout()

    fig.show()
    # plt.savefig(f'{clf.__class__.__name__}_{"_".join(target_names)}.png')
    plt.savefig(fig_title)

    
def make_cm(clf, pred, y_test, target_names, to_labels=None, iteration=0):
    fig_title = f'output/{iteration}_{clf.__class__.__name__}_{"_".join(target_names)}.png'
    
    if to_labels:
        target_names = [f'{to_labels[i]} ({i})' for i in clf.classes_]
    else:
        target_names = clf.classes_
        
    fig, ax = plt.subplots(figsize=(10, 5))
    # print(target_names, clf.classes_)

    
    cf_matrix = confusion_matrix(y_test, pred)
    # print(cf_matrix)
    sns.heatmap(cf_matrix, ax=ax, annot=True, fmt=".0f")

    # disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax[i], display_labels=target_names)
    # # disp.ax_.set_title(key)
    # disp.im_.colorbar.remove()
    # disp.ax_.set_xlabel('')
    # if i != 0:
    #     disp.ax_.set_ylabel('')
    
    ax.xaxis.set_ticklabels(target_names, rotation = 90)
    ax.yaxis.set_ticklabels(target_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    _ = ax.set_title(
        f"{clf.__class__.__name__}"
    )

    # disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    
    
    # ax.xaxis.set_ticklabels(target_names, rotation = 90)
    # ax.yaxis.set_ticklabels(target_names)
    # _ = ax.set_title(
    #     f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
    # )
    fig.show()
    # plt.savefig(f'{clf.__class__.__name__}_{"_".join(target_names)}.png')
    plt.savefig(fig_title)
    

def plot_per_culture(df, model, with_incorrect=False, top_k=5):
    correct = df[df[f'predicted_{model}'] == df['ocms']]
#     per_culture = correct.groupby('culture').count('ocms')
    
    fig = go.Figure(
        layout_height=400,
        layout_title=f'Top {top_k} Correct Cultures .',
        layout_title_x=0.5,
        layout_xaxis_title='Correctly Classified',
        layout_yaxis_title='Cultures',
    )
    culture_dict = defaultdict(int)
    for cul in correct['culture'].items():
        culture_dict[cul[1]] += 1
    culture_dict = {k: v for k, v in culture_dict.items() if v >= top_k}

    culture_dict = sorted(culture_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    # print(culture_dict)
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
    fig = plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.legend()
    plt.show()
    return fi_df['feature_names']


# def plot_feature_importance(rf_model, feature_names, class_labels, top_k=10):
#     """
#     Plot the class-specific feature importance for a Random Forest classifier.

#     Parameters:
#         rf_model (RandomForestClassifier): Trained Random Forest classifier.
#         feature_names (list): List of feature names (words).
#         class_labels (list): List of class labels.
#         top_k (int): Number of top features to display (default: 10).
#     """
#     importances = rf_model.feature_importances_
#     num_classes = rf_model.n_classes_

#     plt.figure(figsize=(10, 6))

#     for i in range(num_classes):
#         class_importances = importances[i::num_classes]
#         top_indices = np.argsort(class_importances)[-top_k:]
#         top_importances = class_importances[top_indices]

#         plt.barh(np.arange(top_k), top_importances, label=class_labels[i])
#         plt.yticks(np.arange(top_k), [feature_names[j] for j in top_indices])
#         plt.xlabel('Importance')
#         plt.title('Random Forest Class-Specific Feature Importance')
#         plt.legend()

#     plt.show()
    
def features_importance_rf(clf, feature_names, top_n=20):

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
    


def plot_feature_effects(clf, 
                         X_train,
                         feature_names,
                         target_names,
                         to_labels=False,
                         top_k=10,
                         verbose=False,
                        iteration=0,):
    fig_title = f'output/feature_effect/{iteration}_{"_".join(target_names)}.png'
    # learned coefficients weighted by frequency of appearance
    average_feature_effects = abs(clf.coef_) * np.asarray(X_train.mean(axis=0)).ravel()

    target_names = np.sort(target_names)
    if to_labels:
        target_names = [to_labels[i] for i in target_names]
        
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get top k features for each target class
    for i, label in enumerate(target_names):
        # if len(average_feature_effects) < 1:
        #     print(i, 'label', label, average_feature_effects)
        top_indices = np.argsort(average_feature_effects[i if len(average_feature_effects) > 1 else 0])[::-1][:top_k]
        top_k_words = feature_names[top_indices]
        top_k_effects = average_feature_effects[i if len(average_feature_effects) > 1 else 0, top_indices]
        ax.barh(top_k_words, top_k_effects, label=label, alpha=0.5)

    ax.set_xlabel('Average Feature Effects')
    if len(average_feature_effects) > 1:
        ax.legend()
    ax.set_title(f'{clf.__class__.__name__}: Feature Effects per Target Class (iter.={iteration})')

    if verbose:
        # Display top k keywords for each class in a table
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
                             


# def plot_feature_effects(coef, names, target_names, iteration=0, top=-1):
#     fig_title = f'output/feature_effect/{iteration}_{"_".join(target_names)}.png'
#     fig, ax = plt.subplots(figsize=(10, 8))
#     imp = coef
#     imp, names = zip(*sorted(list(zip(imp, names))))

#     # Show all features
#     if top == -1:
#         top = len(names)

#     ax.barh(range(top), imp[::-1][0:top], align='center', alpha=0.5)
#     plt.yticks(range(top), names[::-1][0:top])
#     plt.show()
#     fig.savefig(fig_title)
#     return names[::-1][0:top]
    

def plot_feature_effects_detailed(clf, X_train, feature_names, target_names, verbose=False, to_labels=None):        
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

    
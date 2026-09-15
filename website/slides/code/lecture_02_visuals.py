"""Data tables and fitted-model plots for Lecture 2's seven-person example."""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.tree import plot_tree

CLASS_COLORS = {'Happy': "#B4D6AC", 'Unhappy': '#187fbd'}
HEADINGS = {
    'supportive_colleagues': 'Supportive<br>colleagues',
    'salary': 'Salary ($)',
    'free_coffee': 'Free<br>coffee',
    'boss_vegan': 'Boss<br>vegan',
    'happy?': 'Happy?',
}


def show_jobs(frame, annotate=False):
    """Style a DataFrame for slides, colouring Happy and Unhappy cells."""
    css = 'jobs-table annotated' if annotate else 'jobs-table'
    label_classes = frame.map(
        lambda value: {'Happy': 'happy-label', 'Unhappy': 'unhappy-label'}.get(value, '')
    )
    display(
        frame.style
        .hide(axis='index')
        .format({'salary': '{:,.0f}'}, escape='html')
        .relabel_index([HEADINGS.get(c, c) for c in frame.columns], axis='columns')
        .set_td_classes(label_classes)
        .set_table_attributes(f'class="{css}"')
    )


def show_tree(model, feature_names):
    """Plot the actual fitted tree, with class colours shared by other slides."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    artists = plot_tree(
        model, feature_names=list(feature_names), class_names=list(model.classes_),
        impurity=False, label='none', filled=True, rounded=True, fontsize=22, ax=ax,
    )
    nodes = [artist for artist in artists if artist.get_bbox_patch() is not None]
    for node, artist in enumerate(nodes):
        predicted_class = model.classes_[model.tree_.value[node][0].argmax()]
        if model.tree_.children_left[node] == -1:
            count = model.tree_.n_node_samples[node]
            artist.set_text(f'{predicted_class}\n{count} {"person" if count == 1 else "people"}')
        else:
            feature = feature_names[model.tree_.feature[node]]
            threshold = model.tree_.threshold[node]
            if feature == 'salary':
                artist.set_text(f'Salary ≤ ${threshold:,.0f}?')
            else:
                label = feature.replace('_', ' ').capitalize()
                artist.set_text(f'{label}\n≤ {threshold:g}?')
        artist.get_bbox_patch().set_facecolor(CLASS_COLORS[predicted_class])
        artist.set_color('white' if predicted_class == 'Unhappy' else '#17212b')
    fig.tight_layout()
    plt.show()


def show_boundary(model, features, targets):
    """Show fitted predictions on identical axes at each depth."""
    fig, ax = plt.subplots(figsize=(11, 4.7))
    horizontal, vertical = np.meshgrid(np.linspace(-0.2, 1.2, 281),
                                       np.linspace(50000, 160000, 301))
    grid = pd.DataFrame({'supportive_colleagues': horizontal.ravel(),
                         'salary': vertical.ravel()})
    predicted = model.predict(grid).reshape(horizontal.shape)
    encoded = (predicted == 'Happy').astype(int)
    ax.pcolormesh(horizontal, vertical / 1000, encoded,
                  cmap=ListedColormap(['#c6e2f5', '#fff0b3']),
                  vmin=0, vmax=1, shading='nearest', rasterized=True)
    ax.contour(horizontal, vertical / 1000, encoded, levels=[0.5],
               colors=['#17212b'], linewidths=2)
    for label, marker in [('Happy', 'o'), ('Unhappy', 'X')]:
        rows = targets == label
        ax.scatter(features.loc[rows, 'supportive_colleagues'],
                   features.loc[rows, 'salary'] / 1000,
                   c=CLASS_COLORS[label], marker=marker, s=150,
                   edgecolors='#17212b', linewidths=1.2, label=f'Actual: {label}', zorder=3)
    ax.set(xlim=(-0.2, 1.2), ylim=(50, 160),
           xlabel='Supportive colleagues', ylabel='Salary ($ thousands)')
    ax.set_xticks([0, 1], ['No (0)', 'Yes (1)'])
    ax.set_yticks([60, 75, 100, 125, 150])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    fig.tight_layout()
    plt.show()

from empath import Empath
import spacy
import pandas as pd
import random
from src.utils.data_utils import get_movie_plots
import seaborn as sns
import matplotlib.pyplot as plt

class EmpathModel:
    def __init__(self):
        self.lexicon = Empath()
        self.nlp = spacy.load("en_core_web_sm")

    def get_topk_empath_features(self, text, topk=10):
        doc = self.nlp(text)
        empath_features = self.lexicon.analyze(doc.text, normalize=True)
        if topk is not None and empath_features is not None:
            return {k: v for k, v in sorted(empath_features.items(), key=lambda item: item[1], reverse=True)[:topk]}
        return empath_features

    def empath_feature_extraction(self, df, genre, prod_type=None, topk=10):
        results = []
        top_features = set()
        plots = []
        for era in df.dvd_era.unique():
            plots_era = get_movie_plots(df, genre, era) if prod_type is None else get_movie_plots(
                df[df['prod_type'] == prod_type], genre, era)
            random.shuffle(plots_era)
            text = ";".join(plots_era)
            if len(text) > 1e6:
                text = text[:1000000]
                plots_era = text.split(';')[:-1]
                text = ';'.join(plots_era)
            plots.append(text)
            top_k_features = self.get_topk_empath_features(text, topk=topk)
            results.append(top_k_features)
            top_features.update(set(results[-1].keys()))

        for i, era in enumerate(df.dvd_era.unique()):
            if len(set(results[i].keys())) != len(top_features):
                doc = self.nlp(plots[i])
                empath_features = self.lexicon.analyze(doc.text, normalize=True)
                for feature in top_features:
                    if feature not in results[i].keys():
                        results[i][feature] = empath_features[feature]

        words = []
        for d in results:
            words = words + list(d.keys())
        words = list(set(words))

        prop_dict = {'prod_type': [], 'genre': [], 'word': [], 'era': [], 'factor': []}
        for i, era in enumerate(df.dvd_era.unique()):
            for word in words:
                prop_dict['prod_type'].append(prod_type)
                prop_dict['genre'].append(genre)
                prop_dict['era'].append(era)
                prop_dict['word'].append(word)
                if word in results[i]:
                    prop_dict['factor'].append(results[i][word])
                else:
                    prop_dict['factor'].append(0)

        return pd.DataFrame(data=prop_dict)

    def get_features_genres_prods(self, df, selected_genres, prod_types, topk=10):
        results = pd.DataFrame(data={'prod_type': [], 'genre': [], 'word': [], 'era': [], 'factor': []})
        for selected_genre in selected_genres:
            for prod_type in prod_types:
                print(f'Extracting features for {prod_type} {selected_genre} movies')
                result = self.empath_feature_extraction(df, selected_genre, prod_type, topk)
                results = pd.concat([results, result], axis=0)

        return results


    def plot_features_single_prod(self, df, prod_type=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Create a normalized colormap for continuous coloring using Viridis
        unique_words = df['word'].unique()
        word_to_color = {word: i / len(unique_words) for i, word in enumerate(unique_words)}
        colors = [plt.cm.viridis(word_to_color[word]) for word in df['word']]

        # Plot with continuous colors based on 'word'
        sns.lineplot(data=df, x='era', y='factor', hue='word', palette=colors, marker='o', legend='full', ax=ax)
        sns.move_legend(ax, bbox_to_anchor=(1.45, 1), loc='upper right')

        ax.set_title(f'{prod_type} production')
        ax.set_xlabel('DVD era')
        ax.set_ylabel('Importance coefficient (normalized)')
        plt.tight_layout(pad=1)

        print("WARNING / DIFF FROM MAIN. RETURNS INSTEAD OF PLOTTING")
        
        return ax

    def plot_all_features(self, df, genre, topk=10):
        prod_types = df.prod_type.unique()
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
        for j, prod_type in enumerate(prod_types):
            subset = df[(df['genre'] == genre) & (df['prod_type'] == prod_type)]
            self.plot_features_single_prod(subset, prod_type, ax=axes.flatten()[j])

        print("WARNING / DIFF FROM MAIN. RETURNS INSTEAD OF PLOTTING")
        
        return fig
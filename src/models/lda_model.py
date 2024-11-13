from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import math

stop_words = set(stopwords.words('english'))
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]


class LDAModel:
    def __init__(self, text, num_topics):
        self.model = None
        self.texts = [[word for word in word_tokenize(document.lower()) if word not in stop_words] for document in text]
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.num_topics = num_topics

    def train(self):
        self.model = models.LdaModel(self.corpus, num_topics=self.num_topics, id2word=self.dictionary,
                                     per_word_topics=True, random_state=42)

    def get_topics(self):
        return self.model.show_topics(formatted=False)

    def plot_topics(self):
        cloud = WordCloud(stopwords=stop_words,
                          background_color='white',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)

        topics = self.get_topics()
        max_col = 3
        num_columns = min(self.num_topics, max_col)
        num_rows = int(math.ceil(float(self.num_topics) / max_col))
        fig, axes = plt.subplots(num_rows, num_columns,
                                 figsize=(8,8), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            if i >= self.num_topics:
                break
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i))
            plt.gca().axis('off')

        # Remove any extra subplots in the grid
        for j in range(self.num_topics, len(axes.flatten())):
            fig.delaxes(axes.flatten()[j])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()
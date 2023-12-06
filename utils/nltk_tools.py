from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def get_lemma(word):
    return lemmatizer.lemmatize(word)


def get_lemma_toks(text):
    lemmas = []
    toks = []

    for w in word_tokenize(text):
        toks.append(w)
        lemmas.append(get_lemma(w.lower()))
    return lemmas, toks
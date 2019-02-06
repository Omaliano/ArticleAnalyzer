#!/usr/bin/python

import sys
import requests
import nltk
import spacy
from spacy.tokens import Doc
import urllib
from bs4 import BeautifulSoup
from collections import Counter
import re
import heapq
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pprint import pprint
from datetime import datetime


def polarity_scores(doc):
    return sentiment_analyzer.polarity_scores(doc.text)


def sentiment_recommendation(score):

    if-1 <= score <= -0.5:
        return "Article is fairly negative, a response with rebuttal, appology or explanation is recommended as soon" \
               " as possible"
    elif -0.5 <= score <= 0:
        return "Article is somewhat negative, content needs to be re-represented in a more positive light "
    elif 0 <= score <= 0.5:
        return "Article is somewhat positive, positive parts of the article needs to be focused on and re-shared"
    else:
        return "Article is fairly positive, needs to be capitalized on and spread across forms of media"


sentiment_analyzer = SentimentIntensityAnalyzer()
Doc.set_extension('polarity_scores', getter=polarity_scores)
nlp = spacy.load('en_core_web_sm')
nlp2 = spacy.load('en')


def analyze(keyword):

    keyword = keyword.strip(" ")
    keyword = '+' + keyword.replace(" ", "+")

    url = ('https://newsapi.org/v2/everything?'
           'q=' + keyword + '&'
           'sortBy=relevancy&'
           'sortBy=popularity&'
           'sortBy=publishedAt&'
           'pageSize=10&'
           'apiKey=fc62e343c87544bfa4971e94115165e6')

    response = requests.get(url)
    json = dict(response.json())
    i = 1

    for article in json['articles']:

        pprint("Article No.: " + str(i))
        i += 1
        datetime_object = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        print("Date publised: ")
        pprint(str(datetime_object))

        article = urllib.request.urlopen(article['url']).read()

        parsed_article = BeautifulSoup(article, 'lxml')

        paragraphs = parsed_article.find_all('p')

        article_text = ""

        for p in paragraphs:
            article_text += p.text

        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        # Sentiment
        spacySenti = nlp2(formatted_article_text)
        print("Sentiment: ")
        pprint( sentiment_recommendation(spacySenti._.polarity_scores['compound']))

        # Summary
        sentence_list = nltk.sent_tokenize(article_text)
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

        word_frequencies = {}
        for word in nlp(formatted_article_text):

            if word not in spacy_stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word.lemma_] = 1
                else:
                    word_frequencies[word.lemma_] += 1

        if len(word_frequencies) > 0:
            maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

        sentence_scores = {}
        for sent in sentence_list:
            for word in nlp(sent.lower()):
                if word.lemma_ in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word.lemma_]
                        else:
                            sentence_scores[sent] += word_frequencies[word.lemma_]

        if len(sentence_scores) > 3:
            summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)

            summary = ' '.join(summary_sentences)
            print("Summary: ")
            if len(summary.strip(" ")) > 0:

                pprint(str(summary))
        else:
            pprint("Article was too short to produce a coherent summary")

        text = formatted_article_text
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        doc = nlp(text)
        items = [x.text for x in doc.ents]
        ent_dict = {}
        for item in doc.ents:

            ent_dict[item.text] = item.label_

        keys = Counter(items).most_common(3)
        print("Keywords:")
        keys = [key[0] for key in keys]
        for key in keys:
            outstr = str(key) + " : " + str(ent_dict[key])
            pprint(outstr)
        print("")
        print("")
        print("")
        print("")
        print("")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        pprint("Too many arguments")
        exit(1)
    elif len(sys.argv) < 2:
        pprint("keyword to query is missing")
        exit(1)

    analyze(sys.argv[1])
    exit(0)



import matplotlib.pyplot as plt
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Read text
def readText(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

#Plot POS tagging distribution
def posDistributionPlot(file_path):
    text = readText(file_path)
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in pos_tags)

    tags, counts = zip(*pos_counts.items())

    plt.figure(figsize=(14, 7))
    plt.plot(tags, counts, marker='o', linestyle='-', color='skyblue')
    plt.xlabel('Part of Speech')
    plt.ylabel('Frequency')
    plt.title('POS Tagging Distribution')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
       
# Example usage
posDistributionPlot('sample.txt')

import string
import re
import sys

MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
UNK = "<UNK>"
PAD = "<PAD>"

vocabulary = open("data/vocabulary.txt").read().split("\n")
inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))

def words_to_indices(inverse_vocabulary, words):
    return [inverse_vocabulary.get(word, inverse_vocabulary[UNK]) for word in words]

def clean(text):
    text = text.lower()
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation + " ")
    for i in range(10):
       text = text.replace(str(i), " " + str(i) + " ")
    text = MATCH_MULTIPLE_SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n"))

cleanedString = (clean("French presidential candidate Emmanuel Macrons anti-system angle is a sham | Philippe Marlire"))
print(cleanedString)

print(cleanedString.split())

indices = words_to_indices(inverse_vocabulary, cleanedString.split()) 
print(indices)

print(cleanedString.count)
print(indices.count)

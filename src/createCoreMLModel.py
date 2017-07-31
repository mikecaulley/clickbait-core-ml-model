from models.convnets import ConvolutionalNet
from keras.models import load_model
from keras.preprocessing import sequence
from preprocessors.preprocess_text import clean
import sys
import string 
import re
import coremltools

SEQUENCE_LENGTH = 20
EMBEDDING_DIMENSION = 30

vocabulary = open("../data/vocabulary.txt").read().split("\n")
inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))

model = ConvolutionalNet(vocabulary_size=len(vocabulary), embedding_dimension=EMBEDDING_DIMENSION, input_length=SEQUENCE_LENGTH)
model.load_weights("../models/detector.h5")

coreml_model = coremltools.converters.keras.convert(model,["headline"], "clickbaityness")
coreml_model.author = 'Saurabh Mathur, Core ML conversion by Mike Caulley'
coreml_model.license = 'GNU GPL'
coreml_model.short_description = 'Article headline clickbait detector.'
coreml_model.input_description['headline'] = 'An article headline as a tokenized array.'
coreml_model.output_description['clickbaityness'] = 'Probability that the headline is clickbait.'

coreml_model.save('clickbait_model.mlmodel')

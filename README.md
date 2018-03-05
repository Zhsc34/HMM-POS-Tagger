# HMM-POS-Tagger
A simple POS tagger using hidden markov model. Trains HMM with tagged and tokenized data from any language. Takes an input of untagged tokenized text and output word/tags sequences for the text.

### Note
* hmmlearn3.py takes a tagged tokenized text in a the form word/tag sequences as command line input and outputs hmmmodel.txt, containing trained parameters of the model.
* hmmdecode3.py takes a tagged untokenized text as command line input and outputs hmmoutput.txt, containing the result of the tagger in the form of word/tag sequences.
* The current model is not language-specific, meaning a tagged and tokenized text of any language could be learned. I am planning to implement some language-specific feature engineering methods in the future.

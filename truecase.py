import stanfordnlp
import re
from nltk.tokenize import sent_tokenize
import nltk
# nltk.download('punkt')
# stanfordnlp.download('en')

# List of manual regex replacements
replacements=[(r" i ", r" I "), (r" i\'ll ", r" I\'ll "), (r' i\'ve ', r' I\'ve '), (r' i\'m ', r' I\'m ')]
# Sample text
text = "i think that john stone is a nice guy. there is a stone on the grass. i'm fat. are you welcome and smart in london? is this martin's dog?"

def truecase(text):
    # Tokenize the text by sentences
    sentences = sent_tokenize(text, language='english')
    # Capitalize each sentence
    sentences_capitalized = [s.capitalize() for s in sentences]
    # Join the sentences back
    capitalized_text = re.sub(" (?=[\.,'!?:;])", "", ' '.join(sentences_capitalized))
    # Create a stanfordnlp pipeline by the following processors
    stf_nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')
    # Process the text
    doc = stf_nlp(capitalized_text)
    # Capitalize the words if the word is the following parts of speech
    lst = [w.text.capitalize() if w.upos in ["PROPN","NNS"] else w.text for sent in doc.sentences for w in sent.words]
    # Join the list of words
    pos_capitalized = ' '.join(lst)
    # Replace i, i'll, i'm, i've with the capitalized variants
    for pat, repl in replacements:
        pos_capitalized = re.sub(pat, repl, pos_capitalized)
    # Remove the spaces between the punctuation
    result = re.sub(r'\s+([?.!"\'])', r'\1', pos_capitalized)
    return result

print(truecase(text))
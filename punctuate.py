from punctuator import Punctuator

p = Punctuator('Models/model.pcl')

def auto_punctuation(text):
    return p.punctuate(text)

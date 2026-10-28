import numpy as np 
from transformers import pipeline

print("\n loading  lang model")
generator = pipeline('text-generation', model='google/flan-t5-small', device=-1)
print("lang model loaded")

documents = [
        "students can apply scholarship",
       "java is a evergreen pl",
        "the uni li is open now",
        "students can run ",
        "ml allows computers to learn from data",
        "students are good they complete every req fastt",
        "students complete every exams",
        "the univerrsity provides examinations"
]

document_embeddings = model.encode(documents)
print("embediing_shape",document_embeddings.shape)

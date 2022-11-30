import spacy  # version 3.0.6'
from spacy.kb import KnowledgeBase


# initialize language model
nlp = spacy.load("en_core_web_md")

# add pipeline (declared through entry_points in setup.py)
#nlp.create_pipe("entityLinker")
nlp.add_pipe("entityLinker", last=True)

doc = nlp("I watched the Pirates of the Caribbean last silvester")

# returns all entities in the whole document
all_linked_entities = doc._.linkedEntities
# iterates over sentences and prints linked entities
for sent in doc.sents:
    sent._.linkedEntities.pretty_print()

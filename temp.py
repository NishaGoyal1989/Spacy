import spacy  # version 3.0.6'

# initialize language model
nlp = spacy.load("en_core_web_md")

# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)

doc1 = nlp("I watched the Pirates of the Caribbean last silvester")

# returns all entities in the whole document
all_linked_entities = doc1._.linkedEntities
print(all_linked_entities)
# iterates over sentences and prints linked entities
#for sent in doc1.sents:
   # sent._.linkedEntities.pretty_print()

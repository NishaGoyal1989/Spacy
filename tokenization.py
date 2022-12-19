import nltk
import spacy
import classy_classification
import pytextrank
from spacy.tokens import SpanGroup
from flask import render_template,request,Flask
from spacy.language import Language
from spacy_transformers import Transformer
#from spacy.tokens.span_group import SpanGroup
from spacy.tokens import Span
#nltk.download('omw-1.4')
from collections import Counter
from string import punctuation

nlp = spacy.load("en_core_web_md")
app = Flask(__name__)

text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
#doc = nlp(text)

@app.route('/attribute_ruler', methods = ['POST'])
def attribute_ruler():
    d=[]
    ruler = nlp.get_pipe("attribute_ruler")
    patterns =[[{"LOWER": "the"}, {"TEXT": "Who"}]]
    attrs = {"TAG": "NNP", "POS": "PROPN"}
    ruler.add(patterns=patterns,attrs=attrs, index=1)
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.pos_,token.tag_))
    return d

@app.route('/entity_ruler', methods = ['POST'])
def entity_ruler():
    d=[]
    patterns = [
                {"label": "Time-Period", "pattern": "digital era"}
            ]  
    ruler = nlp.add_pipe("entity_ruler")  
    ruler.add_patterns(patterns)
     
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for word in doc.ents: 
            d.append((word.text,word.label_))
    return d

@app.route('/textCategorizer', methods = ['POST'])
def textCategorizer():
    if request.method == 'POST':
        text = request.form['text']
    data = {
        "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
        "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
           }    
    nlp.add_pipe("text_categorizer", 
        config={
            "data": data,
            "model": "spacy"
               }
                )
    doc=nlp(text)
    return(doc._.cats)

@app.route('/spanCategorizer', methods = ['POST'])
def spanCategorizer():
    d=[]
    if request.method == 'POST':
        text = request.form['text'] 
    sent_types=['inf','decl','frag','imp']
    doc=nlp(text)
    sent_group = SpanGroup(doc=doc, name="sentences",spans=list(doc.sents))
    doc.spans['sentences']=sent_group
    Span.set_extension('mood',default=None,force=True)
    for mood,span in zip(sent_types,doc.spans['sentences']):
        span._.mood=mood
    for mood,span in zip(doc.spans['sentences'],doc.spans['sentences']):
        d.append((span.text,span._.mood))
    return d

@app.route('/spanruler', methods = ['POST'])
def spanruler():
    d=[]
    if request.method == 'POST':
        text = request.form['text'] 
    ruler = nlp.add_pipe("span_ruler")
    patterns = [{"label": "ORG", "pattern": "Apple"},
            {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}]
    ruler.add_patterns(patterns)
    doc = nlp(text)
    d.append([(span.text, span.label_) for span in doc.spans["ruler"]])
    return d

@app.route('/sentence', methods = ['POST'])
def sentence(): 
    d={}
    sents_list = []
    token_list=[]
    ent_list=[]
    ent_link=[]
    phrase_list=[]
    dep_list=[]
    morph_list=[]   
    nlp.add_pipe("entityLinker", last=True)
    nlp.add_pipe("textrank")
    if request.method == 'POST':
        text = request.form['text']
        doc=nlp(text)
        for sent in doc.sents:
            sents_list.append(sent.text)   
        for token in doc:
            token_list.append(str(token))
            dep_list.append((str(token),token.dep_))
            morph_list.append((str(token),str(token.morph)))
        for word in doc.ents:
            ent_list.append((word.text,word.label_))
        all_linked_entities = doc._.linkedEntities        
        for i in doc._.linkedEntities:        
            ent_link.append((i.get_url(),i.get_label(),i.get_description()))
        for phrase in doc._.phrases:
            phrase_list.append((str(phrase.count),str(phrase.text)))
            phrase_list.sort(reverse=True)    
        d['tokenizer']=token_list
        d['sentencizer']=sents_list
        d['entity_recognizer']=ent_list
        d['entity_linker']=ent_link
        d['key_phrases']=phrase_list
        d['morphologizer']=morph_list
        d['dependency_parser']=dep_list 
    return d

@app.route('/word', methods = ['POST'])    
def word():
    d={} 
    if request.method == 'POST':
        word = request.form['word']
        doc = nlp(word)
        for token in doc : 
            d['word']=word         
            d['lemmatizer']=token.lemma_
            d['tagger']=token.tag_
            d['dependency_parser']=token.dep_
            d['morphologizer']=str(token.morph)        

    return d
#to run the app in debug mode
if __name__ == "__main__":
    app.run(debug = True)
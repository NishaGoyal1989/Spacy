from flask import render_template,request,Flask
import nltk
import spacy
from spacy.tokens import SpanGroup
import classy_classification
from spacy.language import Language
from spacy_transformers import Transformer
import pytextrank
from spacy.tokens.span_group import SpanGroup
from spacy.tokens import Span
#nltk.download('omw-1.4')
from spacy.pipeline.spancat import DEFAULT_SPANCAT_MODEL
config = {
    "threshold": 0.5,
    "spans_key": "labeled_spans",
    "max_positive": None,
    "model": DEFAULT_SPANCAT_MODEL,
    "suggester": {"@misc": "spacy.ngram_suggester.v1", "sizes": [1, 2, 3]},
}
nlp = spacy.load("en_core_web_md")
app = Flask(__name__)

text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
#doc = nlp(text)

@app.route('/attribute_ruler', methods = ['POST'])
def attribute_ruler():
    d=[]
    ruler = nlp.get_pipe("attribute_ruler")
    patterns = [[
                {"text": "each"}, {"text": "player"}
            ]]
    attrs = {"TAG": "NNP", "POS": "PROPN"}
    ruler.add(patterns=patterns,attrs=attrs, index=1)
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.pos_,token.tag_))
    return d

@app.route('/lemma', methods = ['POST'])    
def lemmatizer():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.lemma_))
    return d 
    
@app.route('/morph', methods = ['POST'])    
def morphologizer():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),str(token.morph)))
    return d 

@app.route('/tagger', methods = ['POST'])    
def tagger():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.tag_,token.pos_))
    return d 

@app.route('/dependency_parser', methods = ['POST'])    
def parser():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for token in doc :
           d.append((str(token),token.dep_))
    return d 

@app.route('/entity_r', methods = ['POST'])    
def entity_recognizer():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for word in doc.ents:
            d.append((word.text,word.label_))
    return d 

@app.route('/entity_ruler', methods = ['POST'])
def entity_ruler():
    d=[]
    patterns = [
                {"label": "Game", "pattern": "Backgammon"}
            ]  
    ruler = nlp.add_pipe("entity_ruler")  
    ruler.add_patterns(patterns)
     
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for word in doc.ents: 
            d.append((word.text,word.label_))
    return d

@app.route('/entity_linker', methods = ['POST'])
def entity_linker():
    d=[]
    nlp.add_pipe("entityLinker", last=True)
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        all_linked_entities = doc._.linkedEntities
        for sent in doc.sents:
            d.append(str(sent._.linkedEntities))
    return d

@app.route('/sentencizer', methods = ['POST'])
def sentencizer():
    sents_list = []
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        for sent in doc.sents:
            sents_list.append(sent.text)
    return sents_list   

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

@app.route('/tok2vec', methods = ['POST'])
def tok2vec(): 
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc=nlp(text)
        d.append(str(doc.tensor))
    return d

@app.route('/tokenizer', methods = ['POST'])
def tokenizer(): 
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        doc=nlp(text)
        for token in doc:
            d.append(str(token))
    return d

@app.route('/transformer', methods = ['POST'])
def transformer(): 
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        #nlp.add_pipe("transformer")
        nlp = spacy.load("en_core_web_trf")
        doc=nlp(text)
        d.append(str(doc._.trf_data))
    return d

def spanCategorizer(text):
    #nlp_spancat = spacy.load("spancat_model")
    #nlp.add_pipe("spancat",config=config)
    doc=nlp(text)
    sent_group = SpanGroup(doc=doc, name="sentences",spans=list(doc.sents))
    doc.spans['sentences']=sent_group
    Span.set_extension('mood',default=None)
    for mood,span in zip(doc,doc.spans['sentences']):
        span._.mood=mood
    print(doc.spans)
    
    #spans = doc.spans["sc"]
    #for span, confidence in zip(spans, spans.attrs["scores"]):
       # print(span.label_, confidence)

@app.route('/keyphrases', methods = ['POST'])
def phrases():
    d=[]
    if request.method == 'POST':
        text = request.form['text']
        nlp.add_pipe("textrank")
        doc=nlp(text)
        for phrase in doc._.phrases:
            d.append(str(phrase.text))
            d.append((str(phrase.rank), str(phrase.count)))
            d.append(str(phrase.chunks))
    return d

#lemmatizer(doc)
#morphologizer(doc)
#tagger(doc)
#dependency_parser(doc)
#entity_recognizer(doc)
#entity_ruler(doc)
#attribute_ruler(doc)
#entity_linker(doc)
#sentencizer(doc)
#spanCategorizer(text)
#textCategorizer(text)
#tok2vec(text)
#tokenizer(text)
#transformer()
#phrases()
#to run the app in debug mode
if __name__ == "__main__":
    app.run(debug = True)
## Hugging Face
Introduction to Transformers for NLP: With the Hugging Face Library and Models to Solve Problems
#### app_5-1.py - Question and Answering
Successfully run against model deepset/roberta-base-squad2
```
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import gradio as grad
import ast
mdl_name = "deepset/roberta-base-squad2"
my_pipeline = pipeline('question-answering', model=mdl_name, tokenizer=mdl_name)
def answer_question(question,context):
    text= "{"+"'question': '"+question+"','context': '"+context+"'}"
    di=ast.literal_eval(text)
    response = my_pipeline(di)
    return response
grad.Interface(answer_question, inputs=["text","text"], outputs="text").launch()
```
#### app_5-2.py - Question and Answering
Succesfully run against model distilbert-base-cased-distilled-squad
```
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import gradio as grad
import ast

mdl_name = "distilbert-base-cased-distilled-squad"
my_pipeline = pipeline('question-answering', model=mdl_name, tokenizer=mdl_name)
def answer_question(question,context):
    text= "{"+"'question': '"+question+"','context': '"+context+"'}"
    di=ast.literal_eval(text)
    response = my_pipeline(di)
    return response
grad.Interface(answer_question, inputs=["text","text"], outputs="text").launch()
```
#### app_5-3.py - Translation (English to German)
Succfully run agains model Helsinki-NLP/opus-mt-en-de
```
from transformers import pipeline
import gradio as grad
mdl_name = "Helsinki-NLP/opus-mt-en-de"
opus_translator = pipeline("translation", model=mdl_name)
def translate(text):
    response = opus_translator(text)
    return response
grad.Interface(translate, inputs=["text",], outputs="text").launch()
```
#### app_5-4.py - Translation (English to German) without pipeline abstraction
Succesfully run against model Helsinki-NLP/opus-mt-en-de
```
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import gradio as grad
mdl_name = "Helsinki-NLP/opus-mt-en-de"
mdl = AutoModelForSeq2SeqLM.from_pretrained(mdl_name)
my_tkn = AutoTokenizer.from_pretrained(mdl_name)
#opus_translator = pipeline("translation", model=mdl_name)
def translate(text):
    inputs = my_tkn(text, return_tensors="pt")
    trans_output = mdl.generate(**inputs)
    response = my_tkn.decode(trans_output[0], skip_special_tokens=True)
    #response = opus_translator(text)
    return response
grad.Interface(translate, inputs=["text",], outputs="text").launch()
```
#### app_5-5.py - Translation (English to French)
Succesfully run agains model Helsinki-NLP/opus-mt-en-fr
```
from transformers import pipeline
import gradio as grad
mdl_name = "Helsinki-NLP/opus-mt-en-fr"
opus_translator = pipeline("translation", model=mdl_name)
def translate(text):
    response = opus_translator(text)
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="French")
grad.Interface(translate, inputs=txt, outputs=out).launch()
```
#### app_5-6.py - Translation (English to French) without pipeline abstraction
Succesfully run against model Helsinki-NLP/opus-mt-en-fr
```
from transformers import AutoModel,AutoTokenizer,AutoModelForSeq2SeqLM
import gradio as grad
mdl_name = "Helsinki-NLP/opus-mt-en-fr"
mdl = AutoModelForSeq2SeqLM.from_pretrained(mdl_name)
my_tkn = AutoTokenizer.from_pretrained(mdl_name)
#opus_translator = pipeline("translation", model=mdl_name)
def translate(text):
    inputs = my_tkn(text, return_tensors="pt")
    trans_output = mdl.generate(**inputs)
    response = my_tkn.decode(trans_output[0], skip_special_tokens=True)
    #response = opus_translator(text)
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="French")
grad.Interface(translate, inputs=txt, outputs=out).launch()
```
#### app_5-7.py - Text Summmary
Succesfully run against model google/pegasus-xsum 
```
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import gradio as grad
mdl_name = "google/pegasus-xsum"
pegasus_tkn = PegasusTokenizer.from_pretrained(mdl_name)
mdl = PegasusForConditionalGeneration.from_pretrained(mdl_name)
def summarize(text):
    tokens = pegasus_tkn(text, truncation=True, padding="longest", return_tensors="pt")
    txt_summary = mdl.generate(**tokens)
    response = pegasus_tkn.batch_decode(txt_summary, skip_special_tokens=True)
    return response
txt=grad.Textbox(lines=10, label="English", placeholder="English Text here")
out=grad.Textbox(lines=10, label="Summary")
grad.Interface(summarize, inputs=txt, outputs=out).launch()
```
#### app_5-8.py - Text Summary using parameters
Succesfully run against model google/pegasus-xsum
```
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import gradio as grad
mdl_name = "google/pegasus-xsum"
pegasus_tkn = PegasusTokenizer.from_pretrained(mdl_name)
mdl = PegasusForConditionalGeneration.from_pretrained(mdl_name)
def summarize(text):
    tokens = pegasus_tkn(text, truncation=True, padding="longest", return_tensors="pt")
    translated_txt = mdl.generate(**tokens,num_return_sequences=5,max_length=200,temperature=1.5,num_beams=10)
    response = pegasus_tkn.batch_decode(translated_txt, skip_special_tokens=True)
    return response
txt=grad.Textbox(lines=10, label="English", placeholder="English Text here")
out=grad.Textbox(lines=10, label="Summary")
grad.Interface(summarize, inputs=txt, outputs=out).launch()
```
#### app_5-9.py - Zero-shot learning
Successfully run -- no model specified
```
from transformers import pipeline
import gradio as grad
zero_shot_classifier = pipeline("zero-shot-classification")
def classify(text,labels):
    classifer_labels = labels.split(",")
    #["software", "politics", "love", "movies", "emergency", "advertisment","sports"]
    response = zero_shot_classifier(text,classifer_labels)
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="text to be classified")
labels=grad.Textbox(lines=1, label="Labels", placeholder="comma separated labels")
out=grad.Textbox(lines=1, label="Classification")
grad.Interface(classify, inputs=[txt,labels], outputs=out).launch()
```
#### app_5-10.py - Zero-shot without pipeline API
Successfully run -- against model specified facebook/bart-large-mnli
```
from transformers import BartForSequenceClassification, BartTokenizer
import gradio as grad
bart_tkn = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
mdl = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
def classify(text,label):
    tkn_ids = bart_tkn.encode(text, label, return_tensors='pt')
    tkn_lgts = mdl(tkn_ids)[0]
    entail_contra_tkn_lgts = tkn_lgts[:,[0,2]]
    probab = entail_contra_tkn_lgts.softmax(dim=1)
    response =  probab[:,1].item() * 100
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="text to be classified")
labels=grad.Textbox(lines=1, label="Label", placeholder="Input a Label")
out=grad.Textbox(lines=1, label="Probablity of label being true is")
grad.Interface(classify, inputs=[txt,labels], outputs=out).launch()
```
#### app_5-11.py - Text Generation
Using HuggingFace API model to GPT2 model
```
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import gradio as grad
mdl = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tkn=GPT2Tokenizer.from_pretrained('gpt2')
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids)
    response = gpt2_tensors
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="Generated Tensors")
grad.Interface(generate, inputs=txt, outputs=out).launch()
```
#### app_5-12.py - Decode tensors in the generate function
```
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response
```
#### app_5-13.py - Changed parameter
```
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids,max_length=100)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response
```
#### app_5-14.py - Added parameter to eliminate repitition
```
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids,max_length=100,no_repeat_ngram_size=True)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response
```
#### app_5-15.py - Add "num_beams" parameters
```
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids,max_length=100,no_repeat_ngram_size=True,num_beams=3)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response
```
#### app_5-16.py - Sampling approach
```
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids,max_length=100,no_repeat_ngram_size=True,num_beams=3,do_sample=True)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response
```
#### app_5-17.py - Change the temperature
```
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids,max_length=100,no_repeat_ngram_size=True,num_beams=3,do_sample=True,temperatue=1.5)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response
```
#### app_5-18.py
```
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import gradio as grad
mdl = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tkn=GPT2Tokenizer.from_pretrained('gpt2')
def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids,max_length=100,no_repeat_ngram_size=True,num_beams=3,do_sample=True,temperatue=0.1)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="Generated Text")
grad.Interface(generate, inputs=txt, outputs=out).launch()
```
#### app_5-19.py
```
from transformers import pipeline, set_seed
import gradio as grad
gpt2_pipe = pipeline('text-generation', model='distilgpt2')
set_seed(42)
def generate(starting_text):
    response= gpt2_pipe(starting_text, max_length=20, num_return_sequences=5)
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="Generated Text")
grad.Interface(generate, inputs=txt, outputs=out).launch()
```
#### app_5-20.py
```
from transformers import AutoModelWithLMHead, AutoTokenizer
import gradio as grad
text2text_tkn = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
mdl = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
def text2text(context,answer):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = text2text_tkn ([input_text], return_tensors='pt')
    output = mdl.generate(input_ids=features['input_ids'],
               attention_mask=features['attention_mask'],
               max_length=64)
    response=text2text_tkn.decode(output[0])
    return response
context=grad.Textbox(lines=10, label="English", placeholder="Context")
ans=grad.Textbox(lines=1, label="Answer")
out=grad.Textbox(lines=1, label="Genereated Question")
grad.Interface(text2text, inputs=[context,ans], outputs=out).launch()
```
#### app_5-21.py
```
from transformers import AutoTokenizer, AutoModelWithLMHead
import gradio as grad
text2text_tkn = AutoTokenizer.from_pretrained("deep-learning-analytics/wikihow-t5-small")
mdl = AutoModelWithLMHead.from_pretrained("deep-learning-analytics/wikihow-t5-small")
def text2text_summary(para):
    initial_txt = para.strip().replace("\n","")
    tkn_text = text2text_tkn.encode(initial_txt, return_tensors="pt")
    tkn_ids = mdl.generate(
            tkn_text,
            max_length=250,
            num_beams=5,
            repetition_penalty=2.5,
            early_stopping=True
        )
    response = text2text_tkn.decode(tkn_ids[0], skip_special_tokens=True)
    return response
para=grad.Textbox(lines=10, label="Paragraph", placeholder="Copy paragraph")
out=grad.Textbox(lines=1, label="Summary")
grad.Interface(text2text_summary, inputs=para, outputs=out).launch()
```
#### app_5-22.py
```
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad
text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")
def text2text_translation(text):
     inp = "translate English to German:: "+text
     enc = text2text_tkn(inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response
para=grad.Textbox(lines=1, label="English Text", placeholder="Text in English")
out=grad.Textbox(lines=1, label="German Translation")
grad.Interface(text2text_translation, inputs=para, outputs=out).launch()
```
#### app_5-23.py
```
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad
text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")
def text2text_translation(text):
     inp = "translate English to French:: "+text
     enc = text2text_tkn(inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response
para=grad.Textbox(lines=1, label="English Text", placeholder="Text in English")
out=grad.Textbox(lines=1, label="French Translation")
grad.Interface(text2text_translation, inputs=para, outputs=out).launch()
```
#### app_5-24.py
```
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad
text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")
def text2text_sentiment(text):
     inp = "sst2 sentence: "+text
     enc = text2text_tkn(inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response
para=grad.Textbox(lines=1, label="English Text", placeholder="Text in English")
out=grad.Textbox(lines=1, label="Sentiment")
grad.Interface(text2text_sentiment, inputs=para, outputs=out).launch()
```
#### app_5-25.py
```
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad
text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")
def text2text_acceptable_sentence(text):
     inp = "cola sentence: "+text
     enc = text2text_tkn(inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response
para=grad.Textbox(lines=1, label="English Text", placeholder="Text in English")
out=grad.Textbox(lines=1, label="Whether the sentence is acceptable or not")
grad.Interface(text2text_acceptable_sentence, inputs=para, outputs=out).launch()
```
#### app_5-26.py
```
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad
text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")
def text2text_paraphrase(sentence1,sentence2):
     inp1 = "mrpc sentence1: "+sentence1
     inp2 = "sentence2: "+sentence2
     combined_inp=inp1+" "+inp2
     enc = text2text_tkn(combined_inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response
sent1=grad.Textbox(lines=1, label="Sentence1", placeholder="Text in English")
sent2=grad.Textbox(lines=1, label="Sentence2", placeholder="Text in English")
out=grad.Textbox(lines=1, label="Whether the sentence is acceptable or not")
grad.Interface(text2text_paraphrase, inputs=[sent1,sent2], outputs=out).launch()
```
#### app_5-27.py
```
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad
text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")
def text2text_deductible(sentence1,sentence2):
     inp1 = "rte sentence1: "+sentence1
     inp2 = "sentence2: "+sentence2
     combined_inp=inp1+" "+inp2
     enc = text2text_tkn(combined_inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response
sent1=grad.Textbox(lines=1, label="Sentence1", placeholder="Text in English")
sent2=grad.Textbox(lines=1, label="Sentence2", placeholder="Text in English")
out=grad.Textbox(lines=1, label="Whether sentence2 is deductible from sentence1")
grad.Interface(text2text_ deductible, inputs=[sent1,sent2], outputs=out).launch()
```


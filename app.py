import config
import torch
import flask
from flask import Flask,render_template,url_for,request,flash,session
from model import BERTBaseUncased
import torch.nn as nn
import transformers
import logging
import sys

app = Flask(__name__)
app.secret_key= 'b\xdb\x87:\xdf/\x89D\xd6\x04|\x97j\x04\xfbh\xd3\x96\x8aj\x90\xe50Q'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/')
def homepage():
    return render_template('index.html')

def sentence_prediction(sentence, model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    article = str(sentence)
    article = " ".join(article.split())

    inputs = tokenizer.encode_plus(
        article,
        None,
        add_special_tokens=True,
        max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    #sigmoid for Binary Classification while softmax for Multi-class
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route("/predict", methods=['POST'])
def predict():
    sentence = request.form.get("sentence")
    positive_prediction = sentence_prediction(sentence, model=MODEL)
    negative_prediction = 1 - positive_prediction
    response = {}
    response = {
        'positive': positive_prediction,
        'negative': negative_prediction,
        'sentence': str(sentence)
    }
    if not sentence:
        flash("Form is blank!")
        return render_template('index.html')
    else:
        return render_template("result.html", sentence=response['sentence'],
                    positive=response['positive'], negative=response['negative'])



if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    #MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True, host="127.0.0.1")

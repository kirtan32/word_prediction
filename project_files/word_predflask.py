from flask import Flask,render_template,request
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break

		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

in_filename = 'sequences_of_alice.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

model = load_model('model_alice.h5')
tokenizer = load(open('tokenizer_alice.pkl', 'rb'))
predicted='None'
model._make_predict_function() 
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('search_word2.html',svar=predicted)

@app.route('/search' , methods=['GET','POST'])
def search():
    try:
        if request.method=='POST':
            svar = request.form['search']
            generated = generate_seq(model, tokenizer, seq_length, svar, 10)
            total = svar + ' ' + generated
            total = np.array(total)
            total = total.reshape(-1,1)
            return render_template('search_word2.html',svar=total)
    except Exception as e:
        e = np.array(e)
        e = e.reshape(-1,1)
        return render_template('search_word2.html',svar=e)

if __name__=='__main__':
    app.run(debug=True, port=4000)

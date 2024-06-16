from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_complete_text(input_ids):
    for _ in range(3): 
        output = model.generate(
            input_ids,
            max_length=250, 
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response_text)
        if sentences:
            response_text = sentences[0]
            for sentence in sentences[1:]:
                if response_text.endswith(('.', '?', '!')):
                    break
                response_text += ' ' + sentence

        if response_text.endswith(('.', '?', '!')):
            return response_text

    return response_text + '.'

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        speech_text = request.form.get('speech_text')
        print(f"Received text: {speech_text}")

        if not speech_text.strip():
            return render_template("home.html", name1="No input received.")

        inputs = tokenizer.encode_plus(speech_text, return_tensors='pt', add_special_tokens=True, max_length=512, truncation=True)

        if inputs['input_ids'].size(0) == 0:
            return render_template("home.html", name1="Tokenization resulted in an empty tensor.")

        input_ids = inputs['input_ids']
        response_text = generate_complete_text(input_ids)
        print(f"Generated text: {response_text}")
        return render_template("home.html", name1=response_text)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)

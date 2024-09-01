from speach import transcribir
from nlp.elbert import load_model, clasifySentiment
from argparse import Namespace

def main():
    model_sentiment = load_model('default')
    trans_args = Namespace(model="base", non_english= True, energy_threshold= 1000, record_timeout=3, phrase_timeout= 3, default_microphone= "pulse")
    
    msg = ''
    for text in transcribir(trans_args):
        print(f'\r{text}', end='')
        msg += text
    
    print(msg)
    sentiment = clasifySentiment(model_sentiment, msg)
    model_sentiment.label_to_text(sentiment)
    print(sentiment)

if __name__ == '__main__':
    main()
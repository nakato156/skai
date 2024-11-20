from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Cargar el modelo entrenado y el tokenizer
model_path = "./reward_model/final"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configurar el modelo para evaluación
model.eval()

def preprocess_for_inference(prompt, response, tokenizer):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    
    prompt_plus_response = f"{prompt}: {response}"
    tokens = tokenizer.encode_plus(prompt_plus_response, **kwargs)
    
    return {
        "input_ids": tokens["input_ids"].squeeze(0),
        "attention_mask": tokens["attention_mask"].squeeze(0)
    }

def predict(prompt, response, model, tokenizer):
    inputs = preprocess_for_inference(prompt, response, tokenizer)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    return probabilities


# Ejemplo de prompt y respuesta para probar
prompt = "Hoy gane un premio en una competencia de ajedrez!"
rejected_response = "afadjlaad .a.bd.adsÑP2L"
chosen_response = "Que felicidad ganar un premio es bastante bueno, que alegría"

probs = predict(prompt, chosen_response, model, tokenizer) # 0 para correcto y 1 para incorrecto
print(probs)
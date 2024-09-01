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
prompt = "sintió un calor reconfortante en su pecho cuando sus miradas se cruzaron"
rejected_response = "El compromiso y la motivación puede ser una señal de que necesitas un cambio. Aprecia la serenidad y busca aplicar ese nuevas experiencias en tu vida"
chosen_response = "Que felicidad ganar un premio es bastante bueno, que alegría"

probs = predict(prompt, rejected_response, model, tokenizer)
print(probs)

exit()

# Preparar y tokenizar las entradas
inputs_chosen = tokenizer(prompt + ": " + chosen_response, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
inputs_rejected = tokenizer(prompt + ": " + rejected_response, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

# Evaluar el modelo en las respuestas tokenizadas
with torch.no_grad():
    chosen_score = model(**inputs_chosen).logits
    rejected_score = model(**inputs_rejected).logits

# Convertir logits a probabilidades (si es necesario)
chosen_prob = torch.softmax(chosen_score, dim=-1)
rejected_prob = torch.softmax(rejected_score, dim=-1)

print(chosen_score)

# Mostrar los resultados
print("Chosen response score:", chosen_score.item())
print("Rejected response score:", rejected_score.item())

print("Chosen response probability:", chosen_prob.item())
print("Rejected response probability:", rejected_prob.item())

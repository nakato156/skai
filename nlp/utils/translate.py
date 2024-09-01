import requests
import json
import pandas as pd

def api_translate(text: str):
	url = "http://localhost:5000//translate"

	data = {
		"q": text,
		"source": "en",
		"target": "es",
		"format": "text",
		"alternatives": 3,
	}

	headers = { "Content-Type": "application/json" }

	response = requests.post(url, headers=headers, data=json.dumps(data))
	return response.json()["translatedText"]


def translate_csv(path: str):
	df = pd.read_csv(path)
	df["text"] = df["text"].apply(api_translate)
	df.to_csv(path.replace(".csv", "_translated.csv"), index=False)
	check(df)

def check(df: pd.DataFrame):
	no_traducido = df[df["text"].str.startswith("#")]
	print(f"Hay {len(no_traducido)} registros que no se puedieron traducir")
	r = input("Desea reemplazarlos manualmente? (s/n)")
	if r.lower() == 's':
		for idx, row in no_traducido.iterrows():
			print(f"Traduccion de {row['text']} a:")
			new_text = input(">>> ")
			df.at[idx, "text"] = new_text
		df.to_csv("./datasets/validation_es.csv", index=False)

if __name__ == "__main__":
	translate_csv("./datasets/validation_en.csv")
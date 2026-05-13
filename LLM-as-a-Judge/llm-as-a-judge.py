import requests
import json
import sys
import os

if len(sys.argv) < 2:
    print("Error: You must provide a file name as an argument.")
    print("Usage: python llm-as-a-judge.py words.txt")
    sys.exit(1)

file_name = sys.argv[1]

try:
    with open(file_name, "r", encoding="utf-8") as file:
        words = file.read()
except FileNotFoundError:
    print(f"Błąd: Nie znaleziono pliku {file_name}")
    words = ""

SYSTEM_PROMPT = """
Rola: Działaj jako ekspert lingwista i specjalista ds. przetwarzania języka naturalnego.
Zadanie: Przeanalizuj poniższą listę słów i pogrupuj je w zbiory synonimów oraz wyrazów bliskoznacznych.
Zasady grupowania:
Korzystaj tylko i wyłącznie ze słów z podanej listy, nie dodawaj słów których na niej nie ma.
Minimalny rozmiar grupy to 2 słowa - nie twórz grup złożonych z tylko jednego słowa.
Łącz słowa w grupy tylko wtedy, gdy faktycznie niosą ze sobą zbliżone znaczenie lub są bliskoznacznymi zamiennikami w kontekście językowym.
Nie próbuj na siłę przypisywać każdego słowa z listy do jakiejś grupy. Jeśli słowo nie ma na liście swojego odpowiednika, pomiń je.
W obrębie jednej grupy mogą występować tylko słowa będące tą samą częścią mowy (np. rzeczowniki)
Format wyjściowy:
Wygeneruj wyłącznie poprawny obiekt JSON.
Nie dodawaj żadnego wstępu, komentarzy ani podsumowań, wyjście musi być czystym kodem gotowym do parsowania.
Schemat wyjścia JSON: {"groups": [["słowo1", "słowo2"], ["słowo3", "słowo4", "słowo5"]]}.
"""

full_input = f"{SYSTEM_PROMPT}\n\nOto lista słów do analizy:\n{words}"

response = requests.post(
    url="http://llm-router-service.llm/v1/responses",
    json={
        "model": "speakleash/Bielik-11B-v2.2-Instruct",
        "input": full_input,
        "max_output_tokens": 2048,
    },
    headers={"Content-Type": "application/json"},
)

try:
    data = response.json()
    raw_text = data['output'][0]['content'][0]['text']

    print("Model response:", raw_text)
    
    start_index = raw_text.find('{')
    end_index = raw_text.rfind('}') + 1
    
    if start_index != -1 and end_index != 0:
        clean_json = raw_text[start_index:end_index]
        json_data = json.loads(clean_json)
        
        os.makedirs("output", exist_ok=True)
        output_path = f"output/response_{file_name}.json"
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(json_data, outfile, indent=4, ensure_ascii=False)
        
        print(f"Success! Saved to {output_path}")
    else:
        print("Could not find JSON brackets in the response.")
    

except json.JSONDecodeError as e:
    print(f"JSON parsing failed: {e}")
    print("Model response was:", raw_text)
import os
import json
import sys

def verify_lemmas(file_prefix, output_dir="output"):
    
    lemmas_file = os.path.join(output_dir, f"{file_prefix}_lemmas.txt")
    key_file = os.path.join(output_dir, f"{file_prefix}_key.json")
    
    if not os.path.exists(lemmas_file):
        print(f"❌ Brak pliku: {lemmas_file}")
        return
    
    if not os.path.exists(key_file):
        print(f"❌ Brak pliku: {key_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"Plik: {file_prefix}")
    print('='*60)
    
    with open(lemmas_file, 'r', encoding='utf-8') as f:
        lemmas = set(line.strip() for line in f if line.strip())
    
    with open(key_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_words_in_json = set()
    cleaned_groups = []
    original_groups_count = len(data.get("groups", []))
    
    for group in data.get("groups", []):
        for word in group:
            all_words_in_json.add(word)
        cleaned_group = [word for word in group if word in lemmas]
        if len(cleaned_group) >= 2:
            cleaned_groups.append(cleaned_group)
    
    original_words_count = sum(len(group) for group in data.get("groups", []))
    
    missing_words = all_words_in_json - lemmas
    
    print(f"Słów w lemmas: {len(lemmas)}")
    print(f"Słów w JSON groups: {len(all_words_in_json)}")
    
    if missing_words:
        print(f"\n❌ Brakujące słowa ({len(missing_words)}):")
        for word in sorted(missing_words):
            print(f"  - {word}")
    else:
        print("\n✅ Wszystkie słowa z JSON-a znajdują się w pliku lemmas!")
    
    data["groups"] = cleaned_groups
    
    final_groups_count = len(cleaned_groups)
    final_words_count = sum(len(group) for group in cleaned_groups)
    
    removed_groups = original_groups_count - final_groups_count
    removed_words = original_words_count - final_words_count
    
    print(f"\n{'='*60}")
    print("🔧 Czyszczenie danych...")
    print('='*60)
    print(f"Grup przed czyszczeniem: {original_groups_count}")
    print(f"Grup po czyszczeniu: {final_groups_count}")
    print(f"Usunięto grup: {removed_groups}")
    print(f"\nSlów przed czyszczeniem: {original_words_count}")
    print(f"Slów po czyszczeniu: {final_words_count}")
    print(f"Usunięto słów: {removed_words}")

    print("\n✅ Czyszczenie zakończone!")
    input("\nNaciśnij Enter, aby zapisać oczyszczony JSON...")
    
    with open(key_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Plik {key_file} został zaktualizowany!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python verify_groups.py <file_prefix>")
        print("Przykład: python verify_groups.py orzeczenie_1.pdf")
        sys.exit(1)
    
    file_prefix = sys.argv[1]
    verify_lemmas(file_prefix)

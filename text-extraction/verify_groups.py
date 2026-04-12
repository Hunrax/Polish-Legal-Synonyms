import os
import json
import sys

def verify_lemmas(file_prefix, output_dir="output"):
    
    lemmas_file = os.path.join(output_dir, file_prefix, f"{file_prefix}_lemmas.txt")
    key_file = os.path.join(output_dir, file_prefix, f"{file_prefix}_key.json")
    
    if not os.path.exists(lemmas_file):
        print(f"❌ No file: {lemmas_file}")
        return
    
    if not os.path.exists(key_file):
        print(f"❌ No file: {key_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"File: {file_prefix}")
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
    
    print(f"Words in lemmas: {len(lemmas)}")
    print(f"Words in JSON groups: {len(all_words_in_json)}")
    
    if missing_words:
        print(f"\n❌ Missing words ({len(missing_words)}):")
        for word in sorted(missing_words):
            print(f"  - {word}")
    else:
        print("\n✅ All words from JSON are present in the lemmas file!")
    
    data["groups"] = cleaned_groups
    
    final_groups_count = len(cleaned_groups)
    final_words_count = sum(len(group) for group in cleaned_groups)
    
    removed_groups = original_groups_count - final_groups_count
    removed_words = original_words_count - final_words_count
    
    print(f"\n{'='*60}")
    print("🔧 Data cleaning...")
    print('='*60)
    print(f"Groups before cleaning: {original_groups_count}")
    print(f"Groups after cleaning: {final_groups_count}")
    print(f"Removed groups: {removed_groups}")
    print(f"\nWords before cleaning: {original_words_count}")
    print(f"Words after cleaning: {final_words_count}")
    print(f"Removed words: {removed_words}")

    print("\n✅ Data cleaning completed!")
    input("\nPress Enter to save the cleaned JSON...")
    
    with open(key_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ File {key_file} has been updated!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_groups.py <file_prefix>")
        print("Example: python verify_groups.py orzeczenie_1")
        sys.exit(1)
    
    file_prefix = sys.argv[1]
    verify_lemmas(file_prefix)

#!/usr/bin/env python3
"""
Script to find all hardcoded English text and translate using Gemini API
"""
import os
import re
import json
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict, Set

# Initialize Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def find_hardcoded_strings(file_path: Path) -> List[Dict]:
    """Find all hardcoded English strings in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    strings = []
    
    # Pattern 1: JSX text content: >Text<
    pattern1 = r'>([A-Z][a-zA-Z\s]{3,})<'
    for match in re.finditer(pattern1, content):
        text = match.group(1).strip()
        if text and not text.startswith('{') and not 't(' in text:
            strings.append({
                'text': text,
                'line': content[:match.start()].count('\n') + 1,
                'type': 'jsx_text'
            })
    
    # Pattern 2: String literals in quotes
    pattern2 = r'["\']([A-Z][a-zA-Z\s]{5,})["\']'
    for match in re.finditer(pattern2, content):
        text = match.group(1).strip()
        # Skip if it's already a translation key or variable
        if text and not 't(' in content[max(0, match.start()-50):match.end()]:
            strings.append({
                'text': text,
                'line': content[:match.start()].count('\n') + 1,
                'type': 'string_literal'
            })
    
    # Pattern 3: Toast messages
    pattern3 = r'toast\.(success|error|info|warning)\(["\']([^"\']+)["\']'
    for match in re.finditer(pattern3, content):
        text = match.group(2).strip()
        strings.append({
            'text': text,
            'line': content[:match.start()].count('\n') + 1,
            'type': 'toast_message'
        })
    
    return strings

def translate_text(text: str, target_lang: str) -> str:
    """Translate text using Gemini API"""
    prompt = f"Translate the following text to {target_lang}. Only return the translation, no explanations:\n\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text

def extract_all_text_from_components():
    """Extract all hardcoded text from React components"""
    frontend_dir = Path('src')
    all_strings = {}
    
    # Files to check
    files_to_check = [
        'pages/**/*.tsx',
        'components/**/*.tsx',
    ]
    
    for pattern in files_to_check:
        for file_path in frontend_dir.glob(pattern):
            if file_path.is_file():
                strings = find_hardcoded_strings(file_path)
                if strings:
                    all_strings[str(file_path)] = strings
    
    return all_strings

def generate_translation_keys(texts: Set[str]) -> Dict[str, str]:
    """Generate translation keys for texts"""
    keys = {}
    for text in sorted(texts):
        # Generate a key from the text
        key = text.lower().replace(' ', '_').replace("'", '').replace('"', '').replace('.', '').replace('!', '').replace('?', '')
        key = re.sub(r'[^a-z0-9_]', '', key)
        key = key[:50]  # Limit length
        keys[text] = key
    return keys

def main():
    print("🔍 Finding all hardcoded English text...")
    all_strings = extract_all_text_from_components()
    
    # Collect all unique texts
    unique_texts = set()
    for file_strings in all_strings.values():
        for item in file_strings:
            unique_texts.add(item['text'])
    
    print(f"📝 Found {len(unique_texts)} unique English strings")
    
    # Load existing translations
    en_file = Path('src/i18n/locales/en.json')
    with open(en_file, 'r', encoding='utf-8') as f:
        en_translations = json.load(f)
    
    # Find missing translations
    existing_keys = set()
    for section in en_translations.values():
        if isinstance(section, dict):
            existing_keys.update(section.values())
    
    missing_texts = [t for t in unique_texts if t not in existing_keys]
    print(f"🆕 Found {len(missing_texts)} new texts to translate")
    
    if not missing_texts:
        print("✅ All texts are already translated!")
        return
    
    # Generate translation keys
    translation_keys = generate_translation_keys(set(missing_texts))
    
    # Translate to Hindi and Malayalam
    print("🌐 Translating to Hindi and Malayalam using Gemini API...")
    
    hi_translations = {}
    ml_translations = {}
    
    for i, text in enumerate(missing_texts, 1):
        print(f"  [{i}/{len(missing_texts)}] Translating: {text[:50]}...")
        
        key = translation_keys[text]
        
        # Translate to Hindi
        hi_text = translate_text(text, "Hindi")
        hi_translations[key] = hi_text
        
        # Translate to Malayalam
        ml_text = translate_text(text, "Malayalam")
        ml_translations[key] = ml_text
        
        # Add to English translations
        # Determine section based on context
        section = "common"
        if any(word in text.lower() for word in ['assignment', 'approve', 'reject', 'ranked']):
            section = "assignments"
        elif any(word in text.lower() for word in ['train', 'trainset', 'fleet']):
            section = "trainsets"
        elif any(word in text.lower() for word in ['setting', 'notification', 'security']):
            section = "settings"
        elif any(word in text.lower() for word in ['dashboard', 'overview']):
            section = "dashboard"
        elif any(word in text.lower() for word in ['login', 'sign', 'auth']):
            section = "auth"
        
        if section not in en_translations:
            en_translations[section] = {}
        
        en_translations[section][key] = text
    
    # Update translation files
    print("💾 Updating translation files...")
    
    # Update English
    with open(en_file, 'w', encoding='utf-8') as f:
        json.dump(en_translations, f, indent=2, ensure_ascii=False)
    
    # Update Hindi
    hi_file = Path('src/i18n/locales/hi.json')
    with open(hi_file, 'r', encoding='utf-8') as f:
        hi_data = json.load(f)
    
    for section, keys in en_translations.items():
        if section not in hi_data:
            hi_data[section] = {}
        for key, en_text in keys.items():
            if key in translation_keys.values() and key in hi_translations:
                hi_data[section][key] = hi_translations[key]
    
    with open(hi_file, 'w', encoding='utf-8') as f:
        json.dump(hi_data, f, indent=2, ensure_ascii=False)
    
    # Update Malayalam
    ml_file = Path('src/i18n/locales/ml.json')
    with open(ml_file, 'r', encoding='utf-8') as f:
        ml_data = json.load(f)
    
    for section, keys in en_translations.items():
        if section not in ml_data:
            ml_data[section] = {}
        for key, en_text in keys.items():
            if key in translation_keys.values() and key in ml_translations:
                ml_data[section][key] = ml_translations[key]
    
    with open(ml_file, 'w', encoding='utf-8') as f:
        json.dump(ml_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Translation complete!")
    print(f"📊 Added {len(missing_texts)} new translations")
    print("\n📋 Translation keys generated:")
    for text, key in list(translation_keys.items())[:10]:
        print(f"  {key}: {text[:50]}")

if __name__ == '__main__':
    main()


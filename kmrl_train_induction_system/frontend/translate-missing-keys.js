/**
 * Translate missing keys using Gemini API
 * Run: GEMINI_API_KEY=your_key node translate-missing-keys.js
 */

const fs = require('fs');
const path = require('path');

// Check if @google/generative-ai is installed
let genAI;
try {
  const { GoogleGenerativeAI } = require('@google/generative-ai');
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  if (!GEMINI_API_KEY) {
    console.error('❌ GEMINI_API_KEY environment variable not set');
    console.log('💡 Set it with: export GEMINI_API_KEY=your_key');
    process.exit(1);
  }
  genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
} catch (e) {
  console.error('❌ @google/generative-ai not installed. Install with: npm install @google/generative-ai');
  process.exit(1);
}

const model = genAI.getGenerativeModel({ model: 'gemini-pro' });

async function translateText(text, targetLang) {
  try {
    const prompt = `Translate the following text to ${targetLang}. Return only the translation, no explanations, no quotes, no markdown:\n\n${text}`;
    const result = await model.generateContent(prompt);
    const response = await result.response;
    let translated = response.text().trim();
    // Remove quotes if present
    translated = translated.replace(/^["']|["']$/g, '');
    return translated;
  } catch (error) {
    console.error(`Error translating "${text}":`, error.message);
    return text;
  }
}

async function main() {
  console.log('🌐 Translating missing keys using Gemini API...\n');

  const enPath = path.join(__dirname, 'src/i18n/locales/en.json');
  const hiPath = path.join(__dirname, 'src/i18n/locales/hi.json');
  const mlPath = path.join(__dirname, 'src/i18n/locales/ml.json');

  const enData = JSON.parse(fs.readFileSync(enPath, 'utf8'));
  const hiData = JSON.parse(fs.readFileSync(hiPath, 'utf8'));
  const mlData = JSON.parse(fs.readFileSync(mlPath, 'utf8'));

  // Find missing translations
  const missing = [];
  for (const [section, keys] of Object.entries(enData)) {
    if (typeof keys !== 'object') continue;
    
    if (!hiData[section]) hiData[section] = {};
    if (!mlData[section]) mlData[section] = {};
    
    for (const [key, enText] of Object.entries(keys)) {
      if (typeof enText !== 'string') continue;
      
      if (!hiData[section][key] || hiData[section][key] === enText) {
        missing.push({ section, key, text: enText, lang: 'hi' });
      }
      if (!mlData[section][key] || mlData[section][key] === enText) {
        missing.push({ section, key, text: enText, lang: 'ml' });
      }
    }
  }

  if (missing.length === 0) {
    console.log('✅ All translations are complete!');
    return;
  }

  console.log(`📝 Found ${missing.length} missing translations\n`);

  // Translate
  for (let i = 0; i < missing.length; i++) {
    const { section, key, text, lang } = missing[i];
    const langName = lang === 'hi' ? 'Hindi' : 'Malayalam';
    
    console.log(`[${i + 1}/${missing.length}] ${langName}: ${text.substring(0, 50)}...`);
    
    const translated = await translateText(text, langName);
    
    if (lang === 'hi') {
      hiData[section][key] = translated;
    } else {
      mlData[section][key] = translated;
    }
    
    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  // Save files
  console.log('\n💾 Saving translation files...');
  fs.writeFileSync(hiPath, JSON.stringify(hiData, null, 2), 'utf8');
  fs.writeFileSync(mlPath, JSON.stringify(mlData, null, 2), 'utf8');

  console.log('✅ Translation complete!');
}

main().catch(console.error);


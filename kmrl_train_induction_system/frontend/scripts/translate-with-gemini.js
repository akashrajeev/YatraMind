/**
 * Script to find all hardcoded English text and translate using Gemini API
 * Run: node scripts/translate-with-gemini.js
 */

const fs = require('fs');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Get API key from environment
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.error('❌ GEMINI_API_KEY environment variable not set');
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-pro' });

// Hardcoded strings found in components
const hardcodedStrings = new Set([
  // Dashboard
  'SOS ALERT',
  'Sign Out',
  'Notifications',
  'No approved trains',
  'Approved trains will appear here',
  'Loading ranked induction list...',
  'No ranked induction data available',
  'AI-Ranked Induction List',
  'ML Optimized',
  'Edit Mode',
  'Adjust Order',
  'Running...',
  'Run Optimization',
  'Export',
  'Failed to update ranked list',
  'Failed to run optimization',
  'Assignment Approved',
  'Approval Failed',
  'Assignment Overridden',
  'Override Failed',
  'Failed to approve assignment',
  'Failed to override assignment',
  'Approved by supervisor',
  'Failed to create assignment',
  'Rolling Stock',
  'Signalling',
  'Telecom',
  'Maintenance',
  'EMERGENCY SOS SIGNAL SENT! Admins and Supervisors Notified.',
  'Failed to send SOS signal',
  'Emergency signal cancelled.',
  'SOS ACTIVE - TAP TO CANCEL',
  'EMERGENCY SOS',
  'Signed out successfully',
  'Failed to sign out',
  'Please select a train and rating',
  'Failed to submit review',
  'Submitting...',
  'Submit Review',
  'Failed to load pending users',
  'User approved successfully',
  'Failed to approve user',
  'User rejected successfully',
  'Failed to reject user',
  'Ingesting...',
  'Ingest from Google Sheets',
  'Daily Operations Report',
  'Summary of daily train induction activities',
  'Performance Analytics',
  'System performance and efficiency metrics',
  'Compliance Report',
  'Safety and regulatory compliance status',
  'Required Service',
  'Decided Service',
  'Stabled Service',
  'Induction Shortfall',
  'Capacity Shortfall',
  'Effective Shortfall',
  'Total Stabled',
  'Total Capacity',
  'Aluva Terminal',
  'Petta Terminal',
  'Muttom Depot',
  'System Information',
  'System Status',
  'Online - All systems operational',
  'Database',
  'Connected - Last backup: 2 hours ago',
  'Active Users',
  'System operational',
  'API Endpoint',
  'Data Refresh Interval',
  'seconds',
  'Error details',
  'Toggle sidebar',
  'Toggle theme',
  'File Upload',
  'Detailed Results',
  'Explanation',
]);

async function translateText(text, targetLang) {
  try {
    const prompt = `Translate the following text to ${targetLang}. Return only the translation, no explanations or quotes:\n\n${text}`;
    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text().trim().replace(/^["']|["']$/g, '');
  } catch (error) {
    console.error(`Error translating "${text}":`, error.message);
    return text;
  }
}

async function main() {
  console.log('🔍 Found', hardcodedStrings.size, 'hardcoded strings to translate\n');

  // Load existing translations
  const enPath = path.join(__dirname, '../src/i18n/locales/en.json');
  const hiPath = path.join(__dirname, '../src/i18n/locales/hi.json');
  const mlPath = path.join(__dirname, '../src/i18n/locales/ml.json');

  const enData = JSON.parse(fs.readFileSync(enPath, 'utf8'));
  const hiData = JSON.parse(fs.readFileSync(hiPath, 'utf8'));
  const mlData = JSON.parse(fs.readFileSync(mlPath, 'utf8'));

  // Find missing translations
  const existingTexts = new Set();
  Object.values(enData).forEach(section => {
    if (typeof section === 'object') {
      Object.values(section).forEach(text => {
        if (typeof text === 'string') {
          existingTexts.add(text);
        }
      });
    }
  });

  const missingTexts = Array.from(hardcodedStrings).filter(text => !existingTexts.has(text));
  
  if (missingTexts.length === 0) {
    console.log('✅ All texts are already in translation files!');
    return;
  }

  console.log(`🆕 Found ${missingTexts.length} new texts to translate\n`);
  console.log('🌐 Translating using Gemini API...\n');

  // Generate keys and translate
  const newTranslations = {
    en: {},
    hi: {},
    ml: {}
  };

  for (let i = 0; i < missingTexts.length; i++) {
    const text = missingTexts[i];
    const key = text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, '_')
      .substring(0, 50);

    console.log(`[${i + 1}/${missingTexts.length}] Translating: ${text.substring(0, 50)}...`);

    // Determine section
    let section = 'common';
    const lowerText = text.toLowerCase();
    if (lowerText.includes('assignment') || lowerText.includes('approve') || lowerText.includes('ranked')) {
      section = 'assignments';
    } else if (lowerText.includes('train') || lowerText.includes('fleet')) {
      section = 'trainsets';
    } else if (lowerText.includes('setting') || lowerText.includes('notification') || lowerText.includes('security')) {
      section = 'settings';
    } else if (lowerText.includes('dashboard') || lowerText.includes('overview')) {
      section = 'dashboard';
    } else if (lowerText.includes('login') || lowerText.includes('sign') || lowerText.includes('auth')) {
      section = 'auth';
    } else if (lowerText.includes('alert') || lowerText.includes('sos') || lowerText.includes('emergency')) {
      section = 'alerts';
    } else if (lowerText.includes('optimization') || lowerText.includes('run')) {
      section = 'optimization';
    } else if (lowerText.includes('report')) {
      section = 'reports';
    }

    // Initialize sections
    if (!enData[section]) enData[section] = {};
    if (!hiData[section]) hiData[section] = {};
    if (!mlData[section]) mlData[section] = {};

    // Translate
    const hiText = await translateText(text, 'Hindi');
    const mlText = await translateText(text, 'Malayalam');

    // Add translations
    enData[section][key] = text;
    hiData[section][key] = hiText;
    mlData[section][key] = mlText;

    // Small delay to avoid rate limits
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  // Save files
  console.log('\n💾 Saving translation files...');
  fs.writeFileSync(enPath, JSON.stringify(enData, null, 2), 'utf8');
  fs.writeFileSync(hiPath, JSON.stringify(hiData, null, 2), 'utf8');
  fs.writeFileSync(mlPath, JSON.stringify(mlData, null, 2), 'utf8');

  console.log('✅ Translation complete!');
  console.log(`📊 Added ${missingTexts.length} new translations\n`);
}

main().catch(console.error);


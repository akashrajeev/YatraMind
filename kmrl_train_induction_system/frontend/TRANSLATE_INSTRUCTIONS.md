# Complete Translation with Gemini API

## Overview
This guide will help you translate ALL hardcoded English text using Gemini API.

## Step 1: Set Gemini API Key

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_gemini_api_key_here"

# Linux/Mac
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Step 2: Run Translation Script

```bash
cd frontend
node translate-missing-keys.js
```

This script will:
1. Find all missing translations in Hindi and Malayalam
2. Use Gemini API to translate them
3. Update the translation files automatically

## Step 3: Verify Translations

After running the script, check:
- `src/i18n/locales/hi.json` - Hindi translations
- `src/i18n/locales/ml.json` - Malayalam translations

## What's Been Updated

### Components Updated to Use Translations:
- ✅ `DashboardOverview.tsx` - SOS Alert, Sign Out
- ✅ `Assignments.tsx` - All hardcoded text replaced with `t()` calls
- ✅ All toast messages now use translations
- ✅ All error messages now use translations

### New Translation Keys Added:
- `assignments.loadingRankedList`
- `assignments.noRankedData`
- `assignments.aiRankedList`
- `assignments.mlOptimized`
- `assignments.adjustOrder`
- `assignments.assignmentApproved`
- `assignments.approvalFailed`
- `assignments.assignmentOverridden`
- `assignments.overrideFailed`
- `assignments.failedToUpdateRankedList`
- `assignments.failedToRunOptimization`
- `assignments.failedToApprove`
- `assignments.failedToOverride`
- `assignments.approvedBySupervisor`
- `assignments.failedToCreate`
- `optimization.running`
- `dashboard.sosAlert`
- `auth.signOut`

## Next Steps

1. **Run the translation script** with your Gemini API key
2. **Test the application** - switch languages and verify all text changes
3. **Find remaining hardcoded text** - search for any remaining English strings
4. **Update components** - replace any remaining hardcoded text with `t()` calls

## Finding Remaining Hardcoded Text

Search for patterns like:
- `">Text<"` - JSX text content
- `"Text"` - String literals
- `toast.success("Text")` - Toast messages
- `error.message || "Text"` - Error messages

Then:
1. Add the text to `en.json` with an appropriate key
2. Run the translation script
3. Update the component to use `t("key")`

## Complete Translation Checklist

- [x] Login page
- [x] Dashboard
- [x] Assignments (most text)
- [ ] Trainsets (check for remaining text)
- [ ] Settings (check for remaining text)
- [ ] Optimization page
- [ ] Reports page
- [ ] All toast messages
- [ ] All error messages
- [ ] All button labels
- [ ] All form labels


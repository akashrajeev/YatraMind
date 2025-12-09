import { createContext, useContext, useMemo, useState, ReactNode, useEffect } from "react";

type SupportedLanguage = "en" | "ml";

type TranslationDictionary = Record<SupportedLanguage, Record<string, string>>;

const TRANSLATIONS: TranslationDictionary = {
  en: {
    operations: "Operations",
    analytics: "Analytics",
    system: "System",
    dashboard: "Dashboard",
    assignments: "Assignments",
    trainsets: "Trainsets",
    optimization: "Optimization",
    multiDepotSimulation: "Multi-Depot Simulation",
    dataIngestion: "Data Ingestion",
    reports: "Reports",
    userManagement: "User Management",
    settings: "Settings",
    kmrlTitle: "KMRL Operations Center",
    kmrlSubtitle: "Train Induction Management System",
    signOut: "Sign Out",
    language: "Language",
    english: "English",
    malayalam: "Malayalam",
  },
  ml: {
    operations: "ഓപ്പറേഷൻസ്",
    analytics: "വിശകലനം",
    system: "സിസ്റ്റം",
    dashboard: "ഡാഷ്ബോർഡ്",
    assignments: "അസൈൻമെന്റുകൾ",
    trainsets: "ട്രെയിൻസെറ്റുകൾ",
    optimization: "ഒപ്റ്റിമൈസേഷൻ",
    multiDepotSimulation: "മൾട്ടി-ഡിപ്പോ സിമുലേഷൻ",
    dataIngestion: "ഡാറ്റ ഇൻജെക്ഷൻ",
    reports: "റിപ്പോർട്ടുകൾ",
    userManagement: "യൂസർ മാനേജ്‌മെന്റ്",
    settings: "സെറ്റിങ്സ്",
    kmrlTitle: "കെഎംആർഎൽ ഓപ്പറേഷൻസ് സെന്റർ",
    kmrlSubtitle: "ട്രെയിൻ ഇൻഡക്ഷൻ മാനേജ്മെന്റ് സിസ്റ്റം",
    signOut: "സൈൻ ഔട്ട്",
    language: "ഭാഷ",
    english: "ഇംഗ്ലീഷ്",
    malayalam: "മലയാളം",
  },
};

type LanguageContextState = {
  language: SupportedLanguage;
  setLanguage: (lang: SupportedLanguage) => void;
  t: (key: string) => string;
};

const LanguageContext = createContext<LanguageContextState | undefined>(undefined);

const STORAGE_KEY = "kmrl-language";

export const LanguageProvider = ({ children }: { children: ReactNode }) => {
  const [language, setLanguage] = useState<SupportedLanguage>("en");

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as SupportedLanguage | null;
    if (stored === "en" || stored === "ml") {
      setLanguage(stored);
    }
  }, []);

  const setLanguageAndPersist = (lang: SupportedLanguage) => {
    setLanguage(lang);
    localStorage.setItem(STORAGE_KEY, lang);
  };

  const t = (key: string) => TRANSLATIONS[language][key] ?? TRANSLATIONS.en[key] ?? key;

  const value = useMemo(
    () => ({
      language,
      setLanguage: setLanguageAndPersist,
      t,
    }),
    [language],
  );

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
};

export const useLanguage = () => {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return ctx;
};

export const useTranslate = () => {
  const { t } = useLanguage();
  return t;
};


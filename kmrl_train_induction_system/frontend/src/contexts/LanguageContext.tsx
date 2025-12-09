import { createContext, useContext, useMemo, useState, ReactNode, useEffect } from "react";

type SupportedLanguage = "en" | "ml";

type TranslationDictionary = Record<SupportedLanguage, Record<string, string>>;

const TRANSLATIONS: TranslationDictionary = {
  en: {
    operations: "Operations",
    analytics: "Analytics",
    system: "System",
    systemOnline: "System Online",
    generateReport: "Generate Report",
    notifications: "Notifications",
    operationsDashboard: "Operations Dashboard",
    systemOverview: "System Overview",
    activeTrainsets: "Active Trainsets",
    pendingAssignments: "Pending Assignments",
    fleetEfficiency: "Fleet Efficiency",
    activeConflicts: "Active Conflicts",
    currentlyInService: "Currently in service",
    awaitingApproval: "Awaiting approval",
    operationalEfficiency: "Operational efficiency",
    requiringAttention: "Requiring attention",
    tomorrowsPlan: "Tomorrow's Service Plan",
    runningInService: "Running in Service",
    scheduledPassenger: "Trains scheduled for passenger service",
    onStandby: "On Standby",
    readyDeployment: "Trains ready for deployment",
    inInspectionBay: "In Inspection Bay",
    underMaintenance: "Trains under maintenance/inspection",
    approvedTrainsList: "Approved Trains List",
    noApprovedTrains: "No approved trains",
    approvedTrainsAppearHere: "Approved trains will appear here",
    approvedStatus: "Approved",
    quickActions: "Quick Actions",
    noNewNotifications: "No new notifications",
    unreadCount: "unread",
    generating: "Generating...",
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
    systemOnline: "സിസ്റ്റം ഓൺലൈൻ",
    generateReport: "റിപ്പോർട്ട് ഉണ്ടാക്കുക",
    notifications: "അറിയിപ്പുകൾ",
    operationsDashboard: "ഓപ്പറേഷൻസ് ഡാഷ്ബോർഡ്",
    systemOverview: "സിസ്റ്റം അവലോകനം",
    activeTrainsets: "സജീവ ട്രെയിൻസെറ്റുകൾ",
    pendingAssignments: "തീർച്ചപ്പെടുത്താനുള്ള അസൈൻമെന്റുകൾ",
    fleetEfficiency: "ഫ്ലീറ്റിന്റെ കാര്യക്ഷമത",
    activeConflicts: "സജീവ സംഘർഷങ്ങൾ",
    currentlyInService: "സേവനത്തിൽ",
    awaitingApproval: "അംഗീകാരം കാത്തിരിക്കുന്നു",
    operationalEfficiency: "ഓപ്പറേഷണൽ കാര്യക്ഷമത",
    requiringAttention: "ശ്രദ്ധ ആവശ്യമാണ്",
    tomorrowsPlan: "നാളെയുടെ സേവന പദ്ധതി",
    runningInService: "സേവനത്തിൽ",
    scheduledPassenger: "യാത്രാ സേവനത്തിന് നിയോഗിച്ച ട്രെയിനുകൾ",
    onStandby: "സ്റ്റാൻഡ്ബൈയിൽ",
    readyDeployment: "വിന്യാസത്തിന് തയ്യാറായി",
    inInspectionBay: "ഇൻസ്പെക്ഷൻ ബേയിൽ",
    underMaintenance: "പരിപാലന/പരിശോധനയിൽ",
    approvedTrainsList: "അംഗീകരിച്ച ട്രെയിൻ പട്ടിക",
    noApprovedTrains: "അംഗീകരിച്ച ട്രെയിനുകളില്ല",
    approvedTrainsAppearHere: "അംഗീകരിച്ച ട്രെയിനുകൾ ഇവിടെ കാണാം",
    approvedStatus: "അംഗീകരിച്ചു",
    quickActions: "ക്വിക്ക് ആക്ഷനുകൾ",
    noNewNotifications: "പുതിയ അറിയിപ്പുകളില്ല",
    unreadCount: "വായിക്കാനുള്ളത്",
    generating: "സൃഷ്ടിക്കുന്നു...",
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


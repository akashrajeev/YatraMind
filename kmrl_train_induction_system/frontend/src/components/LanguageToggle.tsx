import { useLanguage } from "@/contexts/LanguageContext";
import { Button } from "./ui/button";
import { Globe2 } from "lucide-react";

export const LanguageToggle = () => {
  const { language, setLanguage, t } = useLanguage();

  const handleToggle = () => {
    setLanguage(language === "en" ? "ml" : "en");
  };

  return (
    <Button variant="ghost" size="sm" onClick={handleToggle} className="flex items-center gap-2">
      <Globe2 className="h-4 w-4" />
      <span className="text-sm">
        {language === "en" ? t("english") : t("malayalam")}
      </span>
    </Button>
  );
};


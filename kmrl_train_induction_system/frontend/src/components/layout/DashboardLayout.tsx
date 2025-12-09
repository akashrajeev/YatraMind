import { useNavigate } from "react-router-dom";

import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "./AppSidebar";
import { StatusBar } from "./StatusBar";
import { ThemeToggle } from "@/components/theme-toggle";

import { Settings, User, LogOut } from "lucide-react";

import { Button } from "@/components/ui/button";

import { useAuth } from "@/contexts/AuthContext";
import { LanguageToggle } from "@/components/LanguageToggle";
import { useTranslate } from "@/contexts/LanguageContext";


import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuItem
} from "@/components/ui/dropdown-menu";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const t = useTranslate();

  const displayName = user?.name || user?.username || "User";
  const displayUsername = user?.username || "unknown";
  const displayRole = user?.role?.toLowerCase().replace("_", " ") || "member";

  const handleLogout = () => {
    logout();
    navigate("/login", { replace: true });
  };

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background h-screen">
        
        <AppSidebar />
        
        <div className="flex-1 flex flex-col">
          
          {/* Header */}
          <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
            <div className="flex items-center justify-between h-full px-6">
              
              <div className="flex items-center gap-4">
                <SidebarTrigger className="text-foreground" />
                
                <div>
                  <h1 className="font-semibold text-lg text-foreground">
                    {t("kmrlTitle")}
                  </h1>
                  <p className="text-sm text-muted-foreground">
                    {t("kmrlSubtitle")}
                  </p>
                </div>
              </div>

              {/* Right Header Section */}
              <div className="flex items-center gap-3">
                <LanguageToggle />
                <ThemeToggle />

                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => navigate("/settings")}
                >
                  <Settings className="h-5 w-5" />
                </Button>

                {/* User Menu */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="border-2 border-primary">
                      <User className="h-5 w-5" />
                    </Button>
                  </DropdownMenuTrigger>

                  <DropdownMenuContent align="end" className="w-56">

                    <DropdownMenuLabel>
                      <div className="flex flex-col space-y-1">
                        <p className="text-sm font-medium leading-none">
                          {displayName}
                        </p>
                        <p className="text-xs leading-none text-muted-foreground">
                          {displayUsername}
                        </p>
                        <p className="text-xs leading-none text-muted-foreground mt-1">
                          {displayRole}
                        </p>
                      </div>
                    </DropdownMenuLabel>

                    <DropdownMenuSeparator />

                    <DropdownMenuItem
                      onClick={handleLogout}
                      className="cursor-pointer"
                    >
                      <LogOut className="mr-2 h-5 w-5" />
                      <span>{t("signOut")}</span>
                    </DropdownMenuItem>

                  </DropdownMenuContent>
                </DropdownMenu>

              </div>
            </div>
          </header>

          {/* Status Bar */}
          <StatusBar />

          {/* Main Content */}
          <main className="flex-1 p-6">
            {children}
          </main>

        </div>
      </div>
    </SidebarProvider>
  );
}

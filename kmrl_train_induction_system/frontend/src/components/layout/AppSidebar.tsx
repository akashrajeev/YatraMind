import { NavLink, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { dashboardApi } from "@/services/api";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  LayoutDashboard,
  ClipboardList,
  FileText,
  Bell,
  Shield,
  BarChart3,
  Settings,
  Users,
  CheckCircle,
  AlertTriangle,
  Train,
  Brain,
  Database,
  Upload,
  Target,
  Wifi,
  Activity,
} from "lucide-react";

import { useAuth } from "@/contexts/AuthContext";
import { UserRole } from "@/types/auth";

export function AppSidebar() {
  const { state } = useSidebar();
  const location = useLocation();
  const { user } = useAuth();
  const isCollapsed = state === "collapsed";

  // Fetch pending assignments count for the badge
  const { data: overview } = useQuery({
    queryKey: ['dashboard-overview'],
    queryFn: () => dashboardApi.getOverview().then(res => res.data),
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  const pendingCount = overview?.pending_assignments || 0;

  const navigationItems = [
    {
      title: "Dashboard",
      url: "/",
      icon: LayoutDashboard,
      badge: null,
    },
    {
      title: "Assignments",
      url: "/assignments",
      icon: ClipboardList,
      badge: pendingCount > 0 ? pendingCount.toString() : null,
      hidden: user?.role === UserRole.STATION_SUPERVISOR
    },
    {
      title: "Trainsets",
      url: "/trainsets",
      icon: Train,
      badge: null,
    },
    {
      title: "Optimization",
      url: "/optimization",
      icon: Brain,
      badge: null,
      hidden: user?.role === UserRole.STATION_SUPERVISOR
    },
    {
      title: "Data Ingestion",
      url: "/data-ingestion",
      icon: Database,
      badge: null,
      hidden: user?.role === UserRole.STATION_SUPERVISOR
    },
  ].filter(item => !item.hidden);

  const reportItems = [
    {
      title: "Reports",
      url: "/reports",
      icon: BarChart3,
    },
  ];

  const systemItems = [
    {
      title: "User Management",
      url: "/users",
      icon: Users,
      hidden: user?.role === UserRole.STATION_SUPERVISOR
    },
    {
      title: "Settings",
      url: "/settings",
      icon: Settings,
    },
  ].filter(item => !item.hidden);

  const isActive = (path: string) => {
    return location.pathname === path || (path !== "/" && location.pathname.startsWith(path));
  };

  const getNavClass = (path: string) => {
    const active = isActive(path);
    return active
      ? "bg-primary/20 text-primary border-r-2 border-primary font-medium"
      : "hover:bg-muted/50 text-muted-foreground hover:text-foreground";
  };

  return (
    <Sidebar className="border-r border-border bg-card/30 backdrop-blur-sm">
      <SidebarContent className="p-4">
        {/* Main Navigation */}
        <SidebarGroup>
          <SidebarGroupLabel className={`text-xs font-semibold text-muted-foreground uppercase tracking-wider ${isCollapsed ? "sr-only" : ""}`}>
            Operations
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      className={`${getNavClass(item.url)} transition-all duration-200 ${isCollapsed ? "h-12 w-12 px-0 justify-center" : ""
                        }`}
                    >
                      <item.icon className={isCollapsed ? "!h-5 !w-5 flex-shrink-0" : "h-5 w-5 flex-shrink-0"} />
                      {!isCollapsed && (
                        <>
                          <span>{item.title}</span>
                          {item.badge && (
                            <span className="ml-auto bg-accent text-accent-foreground text-xs px-2 py-0.5 rounded-full font-medium">
                              {item.badge}
                            </span>
                          )}
                        </>
                      )}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Reports Section */}
        <SidebarGroup>
          <SidebarGroupLabel className={`text-xs font-semibold text-muted-foreground uppercase tracking-wider ${isCollapsed ? "sr-only" : ""}`}>
            Analytics
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {reportItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      className={`${getNavClass(item.url)} transition-all duration-200 ${isCollapsed ? "h-12 w-12 px-0 justify-center" : ""
                        }`}
                    >
                      <item.icon className={isCollapsed ? "!h-5 !w-5 flex-shrink-0" : "h-5 w-5 flex-shrink-0"} />
                      {!isCollapsed && <span>{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* System Section */}
        <SidebarGroup>
          <SidebarGroupLabel className={`text-xs font-semibold text-muted-foreground uppercase tracking-wider ${isCollapsed ? "sr-only" : ""}`}>
            System
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {systemItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      className={`${getNavClass(item.url)} transition-all duration-200 ${isCollapsed ? "h-12 w-12 px-0 justify-center" : ""
                        }`}
                    >
                      <item.icon className={isCollapsed ? "!h-5 !w-5 flex-shrink-0" : "h-5 w-5 flex-shrink-0"} />
                      {!isCollapsed && <span>{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}

import { Badge } from "@/components/ui/badge";
import { Activity, Wifi, Shield, Clock } from "lucide-react";

export function StatusBar() {
  const systemStatus = {
    online: true,
    lastUpdate: "2 min ago",
    activeUsers: 12,
    systemLoad: 68,
  };

  return (
    <div className="h-8 bg-muted/30 border-b border-border flex items-center justify-between px-6 text-xs">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1">
          <Wifi className="h-3 w-3 text-success" />
          <span className="text-muted-foreground">System</span>
          <Badge variant="success" className="text-xs">
            Online
          </Badge>
        </div>
        
        <div className="flex items-center gap-1">
          <Activity className="h-3 w-3 text-primary" />
          <span className="text-muted-foreground">Load</span>
          <span className="font-mono">{systemStatus.systemLoad}%</span>
        </div>
        
        <div className="flex items-center gap-1">
          <Shield className="h-3 w-3 text-accent" />
          <span className="text-muted-foreground">Users</span>
          <span className="font-mono">{systemStatus.activeUsers}</span>
        </div>
      </div>
      
      <div className="flex items-center gap-1 text-muted-foreground">
        <Clock className="h-3 w-3" />
        <span>Last update: {systemStatus.lastUpdate}</span>
      </div>
    </div>
  );
}

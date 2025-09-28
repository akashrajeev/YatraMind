import { Badge } from "@/components/ui/badge";
import { Activity, Wifi, Clock } from "lucide-react";

export function StatusBar() {
  const currentTime = new Date().toLocaleTimeString();

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
          <span className="text-muted-foreground">KMRL Operations</span>
        </div>
      </div>
      
      <div className="flex items-center gap-1 text-muted-foreground">
        <Clock className="h-3 w-3" />
        <span>{currentTime}</span>
      </div>
    </div>
  );
}

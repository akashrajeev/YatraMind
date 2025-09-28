import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { 
  Settings as SettingsIcon, 
  Bell, 
  Shield, 
  Database, 
  Users, 
  Wifi,
  Save
} from "lucide-react";

const Settings = () => {
  const settings = [
    {
      category: "Notifications",
      icon: Bell,
      items: [
        { name: "Email Alerts", description: "Receive email notifications for critical events", enabled: true },
        { name: "System Alerts", description: "Desktop notifications for system updates", enabled: false },
        { name: "Mobile Push", description: "Push notifications to mobile devices", enabled: true }
      ]
    },
    {
      category: "Security",
      icon: Shield,
      items: [
        { name: "Two-Factor Auth", description: "Require 2FA for all user accounts", enabled: true },
        { name: "Session Timeout", description: "Auto-logout after 30 minutes of inactivity", enabled: false },
        { name: "Audit Logging", description: "Log all user actions and system changes", enabled: true }
      ]
    },
    {
      category: "System",
      icon: Database,
      items: [
        { name: "Auto-Backup", description: "Automatically backup data every 6 hours", enabled: true },
        { name: "Performance Monitoring", description: "Monitor system performance metrics", enabled: true },
        { name: "Maintenance Mode", description: "Enable maintenance mode for updates", enabled: false }
      ]
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">System Settings</h2>
          <p className="text-muted-foreground">Configure system preferences and security settings</p>
        </div>
        <Button variant="industrial">
          <Save className="h-4 w-4 mr-2" />
          Save Changes
        </Button>
      </div>

      {/* Settings Categories */}
      <div className="space-y-6">
        {settings.map((category, categoryIndex) => (
          <Card key={categoryIndex}>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <category.icon className="h-5 w-5 text-primary" />
                </div>
                {category.category}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {category.items.map((item, itemIndex) => (
                <div key={itemIndex} className="flex items-center justify-between p-4 border border-border rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Label htmlFor={`setting-${categoryIndex}-${itemIndex}`} className="font-medium">
                        {item.name}
                      </Label>
                      {item.enabled && <Badge variant="success" className="text-xs">Active</Badge>}
                    </div>
                    <p className="text-sm text-muted-foreground">{item.description}</p>
                  </div>
                  <Switch 
                    id={`setting-${categoryIndex}-${itemIndex}`}
                    checked={item.enabled}
                    onCheckedChange={(checked) => {
                      // Handle setting change
                      console.log(`${item.name} changed to ${checked}`);
                    }}
                  />
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* System Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-accent/10 rounded-lg">
              <SettingsIcon className="h-5 w-5 text-accent" />
            </div>
            System Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Wifi className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">System Status</p>
                  <p className="text-sm text-muted-foreground">Online - All systems operational</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Database className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Database</p>
                  <p className="text-sm text-muted-foreground">Connected - Last backup: 2 hours ago</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Users className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Active Users</p>
                  <p className="text-sm text-muted-foreground">System operational</p>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <Label htmlFor="api-endpoint" className="text-sm font-medium">API Endpoint</Label>
                <Input 
                  id="api-endpoint"
                  defaultValue="http://localhost:8000/api"
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor="refresh-interval" className="text-sm font-medium">Data Refresh Interval</Label>
                <Input 
                  id="refresh-interval"
                  defaultValue="30"
                  type="number"
                  className="mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">seconds</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Settings;
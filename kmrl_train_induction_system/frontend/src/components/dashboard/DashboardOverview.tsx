import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useQuery } from "@tanstack/react-query";
import { dashboardApi } from "@/services/api";
import { DashboardOverview as DashboardOverviewType, AlertsResponse, PerformanceMetrics } from "@/types/api";
import { 
  Activity, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  TrendingUp, 
  Users,
  Zap,
  Shield,
  Train
} from "lucide-react";

export function DashboardOverview() {
  // Fetch dashboard data
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ['dashboard-overview'],
    queryFn: () => dashboardApi.getOverview().then(res => res.data),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const { data: alerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['dashboard-alerts'],
    queryFn: () => dashboardApi.getAlerts().then(res => res.data),
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  const { data: performance, isLoading: performanceLoading } = useQuery({
    queryKey: ['dashboard-performance'],
    queryFn: () => dashboardApi.getPerformance().then(res => res.data),
    refetchInterval: 60000, // Refetch every minute
  });

  // Transform API data to component format
  const stats = overview ? [
    {
      title: "Active Trainsets",
      value: overview.fleet_status.active.toString(),
      change: "+2",
      changeType: "increase" as const,
      icon: Train,
      description: "Currently in service"
    },
    {
      title: "Pending Assignments",
      value: "8", // This would come from assignments API
      change: "-1",
      changeType: "decrease" as const,
      icon: Clock,
      description: "Awaiting assignment"
    },
    {
      title: "Conflicts Detected",
      value: alerts?.critical_count?.toString() || "0",
      change: "+1",
      changeType: "increase" as const,
      icon: AlertTriangle,
      description: "Require manual review"
    },
    {
      title: "Completed Today",
      value: "15", // This would come from assignments API
      change: "+5",
      changeType: "increase" as const,
      icon: CheckCircle,
      description: "Successfully processed"
    }
  ] : [];

  const systemMetrics = performance ? [
    { 
      label: "System Load", 
      value: performance.system_health?.api_response_time_ms || 0, 
      status: "normal" 
    },
    { 
      label: "Queue Efficiency", 
      value: Math.round(performance.operational_metrics?.fleet_availability || 0), 
      status: "excellent" 
    },
    { 
      label: "Risk Assessment", 
      value: Math.round(performance.operational_metrics?.energy_efficiency || 0), 
      status: "good" 
    },
    { 
      label: "Compliance Rate", 
      value: Math.round(performance.operational_metrics?.punctuality_rate || 0), 
      status: "excellent" 
    }
  ] : [];

  const recentActivity = alerts?.alerts?.slice(0, 4).map((alert, index) => ({
    id: index + 1,
    action: alert.message,
    user: "System",
    time: new Date(alert.timestamp).toLocaleTimeString(),
    type: alert.type.toLowerCase()
  })) || [];
  const getProgressColor = (value: number) => {
    if (value >= 90) return "bg-success";
    if (value >= 70) return "bg-warning";
    return "bg-destructive";
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'approval':
        return <CheckCircle className="h-4 w-4 text-success" />;
      case 'alert':
        return <AlertTriangle className="h-4 w-4 text-destructive" />;
      case 'override':
        return <Shield className="h-4 w-4 text-accent" />;
      default:
        return <Activity className="h-4 w-4 text-primary" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
          <p className="text-muted-foreground">Real-time train induction monitoring and control</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-success border-success/20">
            <Zap className="h-3 w-3 mr-1" />
            System Online
          </Badge>
          <Button variant="industrial">
            Generate Report
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <Card key={stat.title} className="relative overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {stat.title}
              </CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold text-foreground">{stat.value}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stat.description}
                  </p>
                </div>
                <Badge 
                  variant={stat.changeType === 'increase' ? 'default' : 'secondary'}
                  className="text-xs"
                >
                  {stat.change}
                </Badge>
              </div>
            </CardContent>
            <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-primary opacity-20" />
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Performance */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              System Performance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {systemMetrics.map((metric) => (
              <div key={metric.label} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">{metric.label}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-mono">{metric.value}%</span>
                    <Badge 
                      variant="outline" 
                      className={
                        metric.status === 'excellent' ? 'text-success border-success/20' :
                        metric.status === 'good' ? 'text-warning border-warning/20' :
                        'text-muted-foreground'
                      }
                    >
                      {metric.status}
                    </Badge>
                  </div>
                </div>
                <Progress 
                  value={metric.value} 
                  className="h-2"
                />
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {recentActivity.map((activity) => (
              <div key={activity.id} className="flex items-start gap-3 p-3 rounded-lg border border-border/50 bg-muted/20">
                {getActivityIcon(activity.type)}
                <div className="flex-1 space-y-1">
                  <p className="text-sm text-foreground leading-tight">
                    {activity.action}
                  </p>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{activity.user}</span>
                    <span>{activity.time}</span>
                  </div>
                </div>
              </div>
            ))}
            <Button variant="ghost" className="w-full text-sm">
              View All Activity
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button variant="industrial">
              <CheckCircle className="h-4 w-4 mr-2" />
              Approve Pending Queue
            </Button>
            <Button variant="outline">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Review Conflicts
            </Button>
            <Button variant="secondary">
              <Activity className="h-4 w-4 mr-2" />
              Generate Daily Brief
            </Button>
            <Button variant="ghost">
              <Users className="h-4 w-4 mr-2" />
              Manage Assignments
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

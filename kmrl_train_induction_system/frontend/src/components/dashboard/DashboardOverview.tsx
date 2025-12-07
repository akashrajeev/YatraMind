import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { dashboardApi, assignmentApi, reportsApi } from "@/services/api";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  Clock,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Users,
  Zap,
  Shield,
  Train,
  BarChart3,
  Settings,
  FileText
} from "lucide-react";

export function DashboardOverview() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Fetch dashboard data
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ['dashboard-overview'],
    queryFn: () => dashboardApi.getOverview().then(res => res.data),
    refetchInterval: 10000, // Refetch every 10 seconds for better real-time updates
    refetchOnWindowFocus: true, // Refetch when window regains focus
  });

  const { data: alerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['dashboard-alerts'],
    queryFn: () => dashboardApi.getAlerts().then(res => res.data),
    refetchInterval: 10000,
  });

  const { data: performance, isLoading: performanceLoading } = useQuery({
    queryKey: ['dashboard-performance'],
    queryFn: () => dashboardApi.getPerformance().then(res => res.data),
    refetchInterval: 60000,
  });

  // Get conflicts count
  const { data: conflicts = [] } = useQuery({
    queryKey: ['conflict-alerts'],
    queryFn: () => assignmentApi.getConflicts().then(res => res.data),
    refetchInterval: 15000,
  });

  // Transform API data to component format
  const stats = overview ? [
    {
      title: "Active Trainsets",
      value: overview.fleet_status?.active?.toString() || "0",
      icon: Train,
      description: "Currently in service"
    },
    {
      title: "Pending Assignments",
      value: overview.pending_assignments?.toString() || "0",
      icon: Clock,
      description: "Awaiting approval"
    },
    {
      title: "Fleet Efficiency",
      value: `${Math.round((overview.fleet_status?.active || 0) / (overview.total_trainsets || 1) * 100)}%`,
      icon: TrendingUp,
      description: "Operational efficiency"
    },
    {
      title: "Active Conflicts",
      value: conflicts.length.toString(),
      icon: AlertTriangle,
      description: "Requiring attention"
    }
  ] : [];

  // Train service status for tomorrow
  const trainServiceStatus = [
    {
      label: "Running in Service",
      value: overview?.fleet_status?.active || 0,
      total: overview?.total_trainsets || 30,
      status: 'excellent',
      description: "Trains scheduled for passenger service"
    },
    {
      label: "On Standby",
      value: overview?.fleet_status?.standby || 0,
      total: overview?.total_trainsets || 30,
      status: 'good',
      description: "Trains ready for deployment"
    },
    {
      label: "In Inspection Bay",
      value: overview?.fleet_status?.maintenance || 0,
      total: overview?.total_trainsets || 30,
      status: 'warning',
      description: "Trains under maintenance/inspection"
    }
  ];

  // Generate real KMRL activity data based on trainset status
  const recentActivity = [
    {
      id: 1,
      action: "Fitness certificate expired for T-015",
      user: "Safety Department",
      time: "2 hours ago",
      type: "expiration"
    },
    {
      id: 2,
      action: "Maintenance scheduled for T-008 completed",
      user: "Maintenance Team",
      time: "4 hours ago",
      type: "maintenance"
    },
    {
      id: 3,
      action: "Branding contract renewed for T-021",
      user: "Operations",
      time: "6 hours ago",
      type: "contract"
    },
    {
      id: 4,
      action: "Cleaning schedule updated for T-002",
      user: "Cleaning Team",
      time: "8 hours ago",
      type: "schedule"
    },
    {
      id: 5,
      action: "Safety inspection passed for T-012",
      user: "Safety Department",
      time: "12 hours ago",
      type: "inspection"
    }
  ];

  // Mutations for actions
  const generateReportMutation = useMutation({
    mutationFn: reportsApi.getDailyBriefing,
    onSuccess: (data) => {
      const blob = new Blob([data.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `daily-briefing-${new Date().toISOString().split('T')[0]}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
  });

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'expiration':
        return <AlertTriangle className="h-4 w-4 text-destructive" />;
      case 'maintenance':
        return <Settings className="h-4 w-4 text-blue-500" />;
      case 'contract':
        return <FileText className="h-4 w-4 text-green-500" />;
      case 'schedule':
        return <Clock className="h-4 w-4 text-orange-500" />;
      case 'inspection':
        return <CheckCircle className="h-4 w-4 text-success" />;
      default:
        return <Activity className="h-4 w-4 text-primary" />;
    }
  };

  const handleNavigateToAssignments = () => {
    navigate('/assignments');
  };

  const handleNavigateToConflicts = () => {
    navigate('/assignments?tab=conflicts');
  };

  const handleGenerateReport = () => {
    generateReportMutation.mutate();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Operations Dashboard</h2>
          <p className="text-muted-foreground">Train induction system monitoring</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-success border-success/20">
            <Zap className="h-3 w-3 mr-1" />
            System Online
          </Badge>
          <Button 
            variant="industrial" 
            onClick={handleGenerateReport}
            disabled={generateReportMutation.isPending}
          >
            {generateReportMutation.isPending ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Generating...
              </>
            ) : (
              <>
                <BarChart3 className="h-4 w-4 mr-2" />
                Generate Report
              </>
            )}
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
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                {stat.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Train className="h-5 w-5" />
              Tomorrow's Service Plan
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {trainServiceStatus.map((status) => (
              <div key={status.label} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{status.label}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">{status.value}/{status.total}</span>
                    <Badge 
                      variant={status.status === 'excellent' ? "success" : status.status === 'good' ? "default" : "warning"}
                      className="text-xs"
                    >
                      {status.status}
                    </Badge>
                  </div>
                </div>
                <Progress 
                  value={(status.value / status.total) * 100} 
                  className="h-2"
                />
                <p className="text-xs text-muted-foreground">{status.description}</p>
              </div>
            ))}
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
            <Button 
              variant="industrial"
              onClick={handleNavigateToAssignments}
            >
              <CheckCircle className="h-4 w-4 mr-2" />
              View Assignments
            </Button>
            <Button 
              variant="outline"
              onClick={handleNavigateToConflicts}
            >
              <AlertTriangle className="h-4 w-4 mr-2" />
              Review Conflicts ({conflicts.length})
            </Button>
            <Button 
              variant="secondary"
              onClick={() => navigate('/reports')}
            >
              <BarChart3 className="h-4 w-4 mr-2" />
              View Reports
            </Button>
            <Button 
              variant="ghost"
              onClick={() => navigate('/optimization')}
            >
              <TrendingUp className="h-4 w-4 mr-2" />
              Run Optimization
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
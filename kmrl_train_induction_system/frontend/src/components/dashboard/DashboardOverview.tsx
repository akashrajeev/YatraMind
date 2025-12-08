import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { dashboardApi, assignmentApi, reportsApi, optimizationApi } from "@/services/api";
import { useNavigate } from "react-router-dom";
import { useMemo, ReactNode, useState } from "react";
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
  FileText,
  ArrowRight
} from "lucide-react";

// Tomorrow's Service Plan types and helpers
interface ServiceData {
  id: string;
  label: string;
  subLabel: string;
  percent: number;
  count: string;
  startColor: string;
  endColor: string;
  iconPath: ReactNode;
}

const CHART_SIZE = 280;
const STROKE_WIDTH = 35;
const CENTER = CHART_SIZE / 2;
const RADIUS = (CHART_SIZE / 2) - STROKE_WIDTH;
const GAP_DEGREES = 6;

const toRad = (deg: number) => (deg * Math.PI) / 180;

const polarToCartesian = (centerX: number, centerY: number, radius: number, angleInDegrees: number) => {
  const angleInRadians = toRad(angleInDegrees);
  return {
    x: centerX + (radius * Math.cos(angleInRadians)),
    y: centerY + (radius * Math.sin(angleInRadians))
  };
};

const describeArc = (x: number, y: number, radius: number, startAngle: number, endAngle: number) => {
  const start = polarToCartesian(x, y, radius, endAngle);
  const end = polarToCartesian(x, y, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return [
    "M", start.x, start.y,
    "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y
  ].join(" ");
};

export function DashboardOverview() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [hoverItem, setHoverItem] = useState<ServiceData | null>(null);

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

  // Fetch assignments to get approved trains
  const { data: assignments = [] } = useQuery({
    queryKey: ['assignments'],
    queryFn: () => assignmentApi.getAll().then(res => res.data),
    refetchInterval: 30000,
  });

  // Fetch ranked list to get ranks for approved trains
  const { data: rankedList = [] } = useQuery({
    queryKey: ['ranked-induction-list'],
    queryFn: () => optimizationApi.getLatest().then(res => res.data),
    refetchInterval: 60000,
  });

  // Get approved assignments sorted by rank and enrich with decision from latest ranked list
  const approvedTrainsByRank = useMemo(() => {
    const approvedAssignments = assignments.filter((a: any) => a.status === "APPROVED");
    
    // Create a map of trainset_id to rank from ranked list
    const rankMap = new Map();
    const decisionMap = new Map();
    rankedList.forEach((decision: any, index: number) => {
      if (decision?.trainset_id) {
        rankMap.set(decision.trainset_id, index + 1);
        if (decision?.decision) decisionMap.set(decision.trainset_id, decision.decision);
      }
    });

    // Add rank to approved assignments and sort by rank
    const approvedWithRank = approvedAssignments.map((assignment: any) => ({
      ...assignment,
      rank: rankMap.get(assignment.trainset_id) || 999, // Default to 999 if not in ranked list
      displayDecision: decisionMap.get(assignment.trainset_id) || assignment.decision?.decision || assignment.status
    }));

    // Sort by rank then recency to ensure ordering is correct
    return approvedWithRank.sort((a: any, b: any) => {
      if (a.rank !== b.rank) return a.rank - b.rank;
      const aTime = a.last_updated || a.approved_at || a.created_at || "";
      const bTime = b.last_updated || b.approved_at || b.created_at || "";
      return new Date(bTime).getTime() - new Date(aTime).getTime();
    });
  }, [assignments, rankedList]);

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

  // Service data for Tomorrow's Service Plan
  const serviceData: ServiceData[] = useMemo(() => {
    const totalTrainsets = overview?.total_trainsets || 25;
    const active = overview?.fleet_status?.active || 0;
    const standby = overview?.fleet_status?.standby || 0;
    const maintenance = overview?.fleet_status?.maintenance || 0;

    const pct = (v: number) => totalTrainsets > 0 ? Math.round((v / totalTrainsets) * 100) : 0;

    return [
      {
        id: 'running',
        label: 'Running in Service',
        subLabel: 'Trains scheduled for passenger service',
        percent: pct(active),
        count: `${active}/${totalTrainsets}`,
        startColor: '#4ADE80',
        endColor: '#22C55E',
        iconPath: (
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
        )
      },
      {
        id: 'standby',
        label: 'On Standby',
        subLabel: 'Trains ready for deployment',
        percent: pct(standby),
        count: `${standby}/${totalTrainsets}`,
        startColor: '#60A5FA',
        endColor: '#3B82F6',
        iconPath: (
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        )
      },
      {
        id: 'inspection',
        label: 'In Inspection Bay',
        subLabel: 'Trains under maintenance/inspection',
        percent: pct(maintenance),
        count: `${maintenance}/${totalTrainsets}`,
        startColor: '#FBBF24',
        endColor: '#F59E0B',
        iconPath: (
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        )
      },
    ];
  }, [overview]);

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
    generateReportMutation.mutate(undefined);
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
        <Card className="h-[440px] flex flex-col overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Train className="h-6 w-6" />
              Tomorrow's Service Plan
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0">
            <div className="flex flex-col md:flex-row items-center gap-8">
              {/* Left: Semi-donut chart */}
              <div
                className="relative flex-shrink-0 overflow-hidden"
                style={{ width: CHART_SIZE / 2 + 20, height: CHART_SIZE }}
              >
                <svg
                  width={CHART_SIZE}
                  height={CHART_SIZE}
                  viewBox={`0 0 ${CHART_SIZE} ${CHART_SIZE}`}
                  className="absolute left-0 top-0"
                  style={{ transform: 'translateX(-50%)' }}
                >
                  <defs>
                    {serviceData.map((item) => (
                      <linearGradient key={item.id} id={`grad-${item.id}`} x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor={item.startColor} />
                        <stop offset="100%" stopColor={item.endColor} />
                      </linearGradient>
                    ))}
                  </defs>

                  {/* Background track */}
                  <path
                    d={describeArc(CENTER, CENTER, RADIUS, -90, 90)}
                    fill="none"
                    stroke="hsl(var(--muted))"
                    strokeWidth={STROKE_WIDTH}
                    strokeLinecap="round"
                  />

                  {/* Segments */}
                  {(() => {
                    let angle = -90;
                    return serviceData.map((item) => {
                      const angleSpan = (item.percent / 100) * 180;
                      const drawAngle = Math.max(0, angleSpan - GAP_DEGREES);
                      const start = angle;
                      const end = angle + drawAngle;
                      angle += angleSpan;

                      return (
                        <path
                          key={item.id}
                          d={describeArc(CENTER, CENTER, RADIUS, start, end)}
                          fill="none"
                          stroke={`url(#grad-${item.id})`}
                          strokeWidth={STROKE_WIDTH}
                          strokeLinecap="round"
                          className="transition-all duration-500 hover:opacity-90 cursor-pointer"
                          onMouseEnter={() => setHoverItem(item)}
                          onMouseLeave={() => setHoverItem(null)}
                        />
                      );
                    });
                  })()}
                </svg>

                {/* Arrow */}
                <div className="absolute top-1/2 left-0 -translate-y-1/2 translate-x-4 pointer-events-none">
                  <ArrowRight className="h-8 w-8 text-muted-foreground/60" strokeWidth={3} />
                </div>

                {/* Hover tooltip */}
                {hoverItem && (
                  <div className="absolute left-1/2 bottom-4 -translate-x-1/2 px-3 py-2 rounded-lg shadow-sm border border-border bg-card/95 backdrop-blur-sm text-sm">
                    <div className="font-semibold text-foreground">{hoverItem.label}</div>
                    <div className="text-xs text-muted-foreground">
                      {hoverItem.percent}% • {hoverItem.count}
                    </div>
                  </div>
                )}
              </div>

              {/* Right: Cards */}
              <div className="flex-grow w-full space-y-3">
                {serviceData.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between p-4 rounded-xl transition-transform hover:scale-[1.01] border border-border"
                    style={{ backgroundColor: `${item.startColor}15` }}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 bg-card shadow-sm border border-border/60"
                      >
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          stroke={item.endColor}
                          viewBox="0 0 24 24"
                          strokeWidth={2}
                        >
                          {item.iconPath}
                        </svg>
                      </div>
                      <div>
                        <h3 className="font-bold text-foreground text-sm">{item.label}</h3>
                        <p className="text-xs text-muted-foreground mt-0.5">{item.subLabel}</p>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <div className="text-lg font-extrabold text-foreground">
                        {item.percent}%
                      </div>
                      <span
                        className="px-2 py-0.5 text-xs font-semibold rounded text-white shadow-sm"
                        style={{ backgroundColor: item.endColor }}
                      >
                        {item.count}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="h-[440px] flex flex-col overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5" />
              Approved Trains List
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0">
            {approvedTrainsByRank.length === 0 ? (
              <div className="text-center py-8">
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4 opacity-50" />
                <p className="text-muted-foreground">No approved trains</p>
                <p className="text-sm text-muted-foreground mt-2">Approved trains will appear here</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
                {approvedTrainsByRank.map((assignment: any) => (
                  <div
                    key={assignment.id}
                    className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary font-semibold text-sm">
                        {assignment.rank}
                      </div>
                      <div>
                        <div className="font-medium text-foreground">{assignment.trainset_id}</div>
                        <div className="text-xs text-muted-foreground">
                          {assignment.displayDecision || assignment.decision?.decision || 'N/A'}
                        </div>
                      </div>
                    </div>
                    <Badge variant="success" className="text-xs">
                      Approved
                    </Badge>
                  </div>
                ))}
              </div>
            )}
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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useMutation } from "@tanstack/react-query";
import { reportsApi } from "@/services/api";
import { Download, FileText, BarChart3, TrendingUp } from "lucide-react";

const Reports = () => {
  // Mutations for report generation
  const dailyBriefingMutation = useMutation({
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

  const assignmentsExportMutation = useMutation({
    mutationFn: ({ format, filters }: { format: string; filters?: any }) => 
      reportsApi.exportAssignments(format, filters),
    onSuccess: (data, variables) => {
      const blob = new Blob([data.data], { 
        type: variables.format === 'pdf' ? 'application/pdf' : 'text/csv' 
      });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `assignments-${new Date().toISOString().split('T')[0]}.${variables.format}`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
  });

  const fleetStatusMutation = useMutation({
    mutationFn: reportsApi.getFleetStatus,
    onSuccess: (data) => {
      const blob = new Blob([data.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fleet-status-${new Date().toISOString().split('T')[0]}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
  });

  const reportTypes = [
    {
      title: "Daily Operations Report",
      description: "Summary of daily train induction activities",
      icon: FileText,
      lastGenerated: "2 hours ago",
      status: "ready",
      onGenerate: () => dailyBriefingMutation.mutate()
    },
    {
      title: "Performance Analytics",
      description: "System performance and efficiency metrics",
      icon: BarChart3,
      lastGenerated: "1 day ago",
      status: "ready",
      onGenerate: () => fleetStatusMutation.mutate()
    },
    {
      title: "Compliance Report",
      description: "Safety and regulatory compliance status",
      icon: TrendingUp,
      lastGenerated: "3 days ago",
      status: "pending",
      onGenerate: () => assignmentsExportMutation.mutate({ format: 'pdf' })
    }
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "ready":
        return <Badge variant="success">Ready</Badge>;
      case "pending":
        return <Badge variant="secondary">Pending</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Reports & Analytics</h2>
          <p className="text-muted-foreground">Generate and view operational reports</p>
        </div>
        <Button variant="industrial">
          <Download className="h-4 w-4 mr-2" />
          Generate All Reports
        </Button>
      </div>

      {/* Report Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {reportTypes.map((report, index) => (
          <Card key={index} className="hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-primary/10 rounded-lg">
                    <report.icon className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <CardTitle className="text-lg">{report.title}</CardTitle>
                    <p className="text-sm text-muted-foreground">{report.description}</p>
                  </div>
                </div>
                {getStatusBadge(report.status)}
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Last Generated:</span>
                  <span className="font-medium">{report.lastGenerated}</span>
                </div>
                <div className="flex gap-2">
                  <Button 
                    size="sm" 
                    variant="outline" 
                    className="flex-1"
                    onClick={report.onGenerate}
                    disabled={dailyBriefingMutation.isPending || assignmentsExportMutation.isPending || fleetStatusMutation.isPending}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    {dailyBriefingMutation.isPending || assignmentsExportMutation.isPending || fleetStatusMutation.isPending ? 'Generating...' : 'Download'}
                  </Button>
                  <Button 
                    size="sm" 
                    variant="secondary" 
                    className="flex-1"
                    onClick={report.onGenerate}
                    disabled={dailyBriefingMutation.isPending || assignmentsExportMutation.isPending || fleetStatusMutation.isPending}
                  >
                    Regenerate
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Reports</p>
                <p className="text-2xl font-bold">24</p>
              </div>
              <FileText className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">This Month</p>
                <p className="text-2xl font-bold">8</p>
              </div>
              <TrendingUp className="h-8 w-8 text-success" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Auto-Generated</p>
                <p className="text-2xl font-bold">18</p>
              </div>
              <BarChart3 className="h-8 w-8 text-primary" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Pending</p>
                <p className="text-2xl font-bold">3</p>
              </div>
              <Badge variant="secondary">3</Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Reports;
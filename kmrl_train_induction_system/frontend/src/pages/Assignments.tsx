import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { assignmentApi } from "@/services/api";
import { Assignment, AssignmentSummary } from "@/types/api";
import { RefreshCw, Download, Plus, CheckCircle, AlertTriangle, Clock } from "lucide-react";

const Assignments = () => {
  const queryClient = useQueryClient();

  // Fetch assignments data
  const { data: assignments = [], isLoading, refetch } = useQuery({
    queryKey: ['assignments'],
    queryFn: () => assignmentApi.getAll().then(res => res.data),
    refetchInterval: 30000,
  });

  const { data: summary } = useQuery({
    queryKey: ['assignments-summary'],
    queryFn: () => assignmentApi.getSummary().then(res => res.data),
  });

  // Mutations for actions
  const approveMutation = useMutation({
    mutationFn: assignmentApi.approve,
    onSuccess: () => {
      queryClient.invalidateQueries(['assignments']);
      queryClient.invalidateQueries(['assignments-summary']);
    },
  });

  const overrideMutation = useMutation({
    mutationFn: assignmentApi.override,
    onSuccess: () => {
      queryClient.invalidateQueries(['assignments']);
      queryClient.invalidateQueries(['assignments-summary']);
    },
  });

  // Filter assignments by status
  const pendingAssignments = assignments.filter(a => a.status === "PENDING");
  const approvedAssignments = assignments.filter(a => a.status === "APPROVED");
  const overriddenAssignments = assignments.filter(a => a.status === "OVERRIDDEN");

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "PENDING":
        return <Badge variant="secondary" className="text-warning"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
      case "APPROVED":
        return <Badge variant="success"><CheckCircle className="h-3 w-3 mr-1" />Approved</Badge>;
      case "OVERRIDDEN":
        return <Badge variant="outline" className="text-accent"><AlertTriangle className="h-3 w-3 mr-1" />Overridden</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "HIGH":
        return "text-destructive";
      case "MEDIUM":
        return "text-warning";
      case "LOW":
        return "text-success";
      default:
        return "text-muted-foreground";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Train Induction Assignments</h2>
          <p className="text-muted-foreground">Manage and monitor train induction assignments</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="industrial">
            <Plus className="h-4 w-4 mr-2" />
            Run Optimization
          </Button>
        </div>
      </div>

      {/* Assignments Tabs */}
      <Tabs defaultValue="pending" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="pending">
            Pending ({pendingAssignments.length})
          </TabsTrigger>
          <TabsTrigger value="approved">
            Approved ({approvedAssignments.length})
          </TabsTrigger>
          <TabsTrigger value="overridden">
            Overridden ({overriddenAssignments.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="pending" className="space-y-4">
          {isLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
              <p className="text-muted-foreground mt-2">Loading assignments...</p>
            </div>
          ) : pendingAssignments.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-muted-foreground">No pending assignments</p>
            </div>
          ) : (
            pendingAssignments.map((assignment) => (
              <Card key={assignment.id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{assignment.trainset_id}</CardTitle>
                    <div className="flex items-center gap-2">
                      {getStatusBadge(assignment.status)}
                      <span className={`text-sm font-medium ${getPriorityColor(assignment.priority.toString())}`}>
                        Priority {assignment.priority}
                      </span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Decision:</span>
                      <p className="font-medium">{assignment.decision.decision}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Confidence:</span>
                      <p className="font-medium">{Math.round(assignment.decision.confidence_score * 100)}%</p>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mt-2">{assignment.decision.reasoning}</p>
                  <div className="flex gap-2 mt-4">
                    <Button size="sm" variant="outline">View Details</Button>
                    <Button 
                      size="sm" 
                      variant="secondary"
                      onClick={() => overrideMutation.mutate({
                        assignment_id: assignment.id,
                        override_decision: "OVERRIDE",
                        reason: "Manual override"
                      })}
                      disabled={overrideMutation.isPending}
                    >
                      Override
                    </Button>
                    <Button 
                      size="sm" 
                      variant="industrial"
                      onClick={() => approveMutation.mutate({
                        assignment_ids: [assignment.id],
                        comments: "Approved by supervisor"
                      })}
                      disabled={approveMutation.isPending}
                    >
                      Approve
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="approved" className="space-y-4">
          {approvedAssignments.map((assignment) => (
            <Card key={assignment.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">{assignment.trainset}</CardTitle>
                  <div className="flex items-center gap-2">
                    {getStatusBadge(assignment.status)}
                    <span className={`text-sm font-medium ${getPriorityColor(assignment.priority)}`}>
                      {assignment.priority}
                    </span>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Assigned to:</span>
                    <p className="font-medium">{assignment.assignedTo}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Scheduled:</span>
                    <p className="font-medium">{assignment.scheduledDate}</p>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground mt-2">{assignment.description}</p>
                <div className="flex gap-2 mt-4">
                  <Button size="sm" variant="outline">View Details</Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="overridden" className="space-y-4">
          {overriddenAssignments.map((assignment) => (
            <Card key={assignment.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">{assignment.trainset}</CardTitle>
                  <div className="flex items-center gap-2">
                    {getStatusBadge(assignment.status)}
                    <span className={`text-sm font-medium ${getPriorityColor(assignment.priority)}`}>
                      {assignment.priority}
                    </span>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Assigned to:</span>
                    <p className="font-medium">{assignment.assignedTo}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Scheduled:</span>
                    <p className="font-medium">{assignment.scheduledDate}</p>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground mt-2">{assignment.description}</p>
                <div className="flex gap-2 mt-4">
                  <Button size="sm" variant="outline">View Details</Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Assignments;
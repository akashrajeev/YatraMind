import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { assignmentApi, optimizationApi, trainsetsApi } from "@/services/api";
import { Assignment, AssignmentSummary } from "@/types/api";
import { RefreshCw, Download, Plus, CheckCircle, AlertTriangle, Clock, Brain, Target, BarChart3, TrendingUp, Eye, Info } from "lucide-react";
import { useState } from "react";

const Assignments = () => {
  const queryClient = useQueryClient();
  const [selectedTrainset, setSelectedTrainset] = useState<string | null>(null);
  const [explanationData, setExplanationData] = useState<any>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [trainsetDetails, setTrainsetDetails] = useState<any>(null);
  const [showTrainsetDetails, setShowTrainsetDetails] = useState(false);

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

  // Fetch ranked induction list
  const { data: rankedList = [], isLoading: loadingRankedList } = useQuery({
    queryKey: ['ranked-induction-list'],
    queryFn: () => optimizationApi.getLatest().then(res => res.data),
    refetchInterval: 60000,
  });

  // Fetch conflict alerts
  const { data: conflictAlerts = [] } = useQuery({
    queryKey: ['conflict-alerts'],
    queryFn: () => assignmentApi.getAll({ status: 'PENDING' }).then(res => 
      res.data.filter((a: any) => a.decision?.violations?.length > 0)
    ),
    refetchInterval: 15000,
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

  // Explanation mutation
  const explanationMutation = useMutation({
    mutationFn: ({ trainsetId, decision }: { trainsetId: string; decision: string }) => 
      optimizationApi.explainAssignment(trainsetId, decision),
    onSuccess: (data) => {
      setExplanationData(data.data);
      setShowExplanation(true);
    },
  });

  const handleExplainDecision = (trainsetId: string, decision: string) => {
    setSelectedTrainset(trainsetId);
    explanationMutation.mutate({ trainsetId, decision });
  };

  const handleViewDetails = async (trainsetId: string) => {
    try {
      const response = await trainsetsApi.getDetails(trainsetId);
      setTrainsetDetails(response.data);
      setShowTrainsetDetails(true);
    } catch (error) {
      console.error('Error fetching trainset details:', error);
    }
  };

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
      <Tabs defaultValue="ranked" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="ranked">
            <Target className="h-4 w-4 mr-2" />
            Ranked ({rankedList.length})
          </TabsTrigger>
          <TabsTrigger value="conflicts">
            <AlertTriangle className="h-4 w-4 mr-2" />
            Conflicts ({conflictAlerts.length})
          </TabsTrigger>
          <TabsTrigger value="pending">
            <Clock className="h-4 w-4 mr-2" />
            Pending ({pendingAssignments.length})
          </TabsTrigger>
          <TabsTrigger value="approved">
            <CheckCircle className="h-4 w-4 mr-2" />
            Approved ({approvedAssignments.length})
          </TabsTrigger>
          <TabsTrigger value="overridden">
            <AlertTriangle className="h-4 w-4 mr-2" />
            Overridden ({overriddenAssignments.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="ranked" className="space-y-4">
          {loadingRankedList ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
              <p className="text-muted-foreground mt-2">Loading ranked induction list...</p>
            </div>
          ) : rankedList.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-muted-foreground">No ranked induction data available</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">AI-Ranked Induction List</h3>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Brain className="h-3 w-3" />
                  ML Optimized
                </Badge>
              </div>
              {rankedList.map((decision: any, index: number) => (
                <Card key={decision.trainset_id} className="hover:shadow-md transition-shadow">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-bold">
                          {index + 1}
                        </div>
                        <CardTitle className="text-lg">{decision.trainset_id}</CardTitle>
                        <Badge variant={decision.decision === "INDUCT" ? "success" : "secondary"}>
                          {decision.decision}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-muted-foreground">
                          Score: {Math.round(decision.score * 100)}%
                        </span>
                        <span className="text-sm font-medium text-muted-foreground">
                          Confidence: {Math.round(decision.confidence_score * 100)}%
                        </span>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-2">Top Reasons:</h4>
                        <ul className="list-disc list-inside text-sm space-y-1">
                          {decision.top_reasons?.slice(0, 3).map((reason: string, i: number) => (
                            <li key={i} className="text-green-600">{reason}</li>
                          ))}
                        </ul>
                      </div>
                      {decision.top_risks && decision.top_risks.length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm text-muted-foreground mb-2">Risks:</h4>
                          <ul className="list-disc list-inside text-sm space-y-1">
                            {decision.top_risks.slice(0, 2).map((risk: string, i: number) => (
                              <li key={i} className="text-red-600">{risk}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div className="flex gap-2">
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={() => handleExplainDecision(decision.trainset_id, decision.decision)}
                          disabled={explanationMutation.isPending}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          Explain Decision
                        </Button>
                            <Button 
                              size="sm" 
                              variant="secondary"
                              onClick={() => handleViewDetails(decision.trainset_id)}
                            >
                              <Info className="h-4 w-4 mr-2" />
                              View Details
                            </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="conflicts" className="space-y-4">
          {conflictAlerts.length === 0 ? (
            <div className="text-center py-8">
              <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <p className="text-muted-foreground">No conflict alerts detected</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Conflict Alerts</h3>
                <Badge variant="destructive" className="flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  {conflictAlerts.length} Conflicts
                </Badge>
              </div>
              {conflictAlerts.map((assignment: any) => (
                <Card key={assignment.id} className="border-red-200 bg-red-50/50">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg text-red-800">{assignment.trainset_id}</CardTitle>
                      <Badge variant="destructive">Conflict Detected</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-2">Violations:</h4>
                        <ul className="list-disc list-inside text-sm space-y-1">
                          {assignment.decision?.violations?.map((violation: string, i: number) => (
                            <li key={i} className="text-red-600">{violation}</li>
                          ))}
                        </ul>
                      </div>
                      <div className="flex gap-2">
                        <Button 
                          size="sm" 
                          variant="destructive"
                          onClick={() => handleExplainDecision(assignment.trainset_id, assignment.decision?.decision)}
                          disabled={explanationMutation.isPending}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          Explain Conflict
                        </Button>
                        <Button size="sm" variant="outline">
                          <Info className="h-4 w-4 mr-2" />
                          Resolve
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

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

      {/* Explanation Modal */}
      {showExplanation && explanationData && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">
                  AI Decision Explanation - {selectedTrainset}
                </CardTitle>
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowExplanation(false)}
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-lg mb-3 text-green-600">Top Reasons</h3>
                  <ul className="space-y-2">
                    {explanationData.top_reasons?.map((reason: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold text-lg mb-3 text-red-600">Risks & Violations</h3>
                  <ul className="space-y-2">
                    {explanationData.top_risks?.map((risk: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{risk}</span>
                      </li>
                    ))}
                    {explanationData.violations?.map((violation: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-red-600">{violation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {explanationData.shap_values && explanationData.shap_values.length > 0 && (
                <div>
                  <h3 className="font-semibold text-lg mb-3">Feature Impact Analysis</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {explanationData.shap_values.map((feature: any, i: number) => (
                      <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm font-medium">{feature.name}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{feature.value}</span>
                          <span className={`text-xs px-2 py-1 rounded ${
                            feature.impact === 'positive' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {feature.impact === 'positive' ? '↑' : '↓'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setShowExplanation(false)}>
                  Close
                </Button>
                <Button onClick={() => window.print()}>
                  Print Explanation
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
          )}

          {/* Trainset Details Modal */}
          {showTrainsetDetails && trainsetDetails && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
              <Card className="max-w-6xl w-full max-h-[90vh] overflow-y-auto">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-xl">
                      Trainset Details - {trainsetDetails.trainset_id}
                    </CardTitle>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => setShowTrainsetDetails(false)}
                    >
                      ×
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="font-semibold text-lg mb-3">Basic Information</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Status:</span>
                          <Badge variant={trainsetDetails.basic_info.status === 'ACTIVE' ? 'success' : 'secondary'}>
                            {trainsetDetails.basic_info.status}
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Model:</span>
                          <span>{trainsetDetails.basic_info.model}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Manufacturer:</span>
                          <span>{trainsetDetails.basic_info.manufacturer}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Commission Date:</span>
                          <span>{trainsetDetails.basic_info.commission_date}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Last Inspection:</span>
                          <span>{trainsetDetails.basic_info.last_inspection}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-lg mb-3">Operational Metrics</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Assignments:</span>
                          <span>{trainsetDetails.operational_metrics.total_assignments}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Success Rate:</span>
                          <span className="text-green-600">{trainsetDetails.operational_metrics.success_rate}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Fitness Score:</span>
                          <span className="text-blue-600">{Math.round(trainsetDetails.operational_metrics.current_fitness_score * 100)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Failure Risk:</span>
                          <span className="text-red-600">{Math.round(trainsetDetails.operational_metrics.predicted_failure_risk * 100)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-semibold text-lg mb-3">Performance Data</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Operational Hours</div>
                        <div className="text-lg font-semibold">{trainsetDetails.performance_data.operational_hours.toLocaleString()}</div>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Total Mileage</div>
                        <div className="text-lg font-semibold">{trainsetDetails.performance_data.mileage.toLocaleString()} km</div>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Avg Sensor Health</div>
                        <div className="text-lg font-semibold">{Math.round(trainsetDetails.performance_data.avg_sensor_health * 100)}%</div>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm text-muted-foreground">Next Maintenance</div>
                        <div className="text-lg font-semibold">{trainsetDetails.performance_data.next_scheduled_maintenance?.split('T')[0] || 'N/A'}</div>
                      </div>
                    </div>
                  </div>

                  {trainsetDetails.recommendations && trainsetDetails.recommendations.filter(r => r).length > 0 && (
                    <div>
                      <h3 className="font-semibold text-lg mb-3">Recommendations</h3>
                      <ul className="space-y-2">
                        {trainsetDetails.recommendations.filter(r => r).map((rec: string, i: number) => (
                          <li key={i} className="flex items-start gap-2">
                            <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm">{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowTrainsetDetails(false)}>
                      Close
                    </Button>
                    <Button onClick={() => window.print()}>
                      Print Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      );
    };

    export default Assignments;
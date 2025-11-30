import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { assignmentApi, optimizationApi, trainsetsApi } from "@/services/api";
import { Assignment, AssignmentSummary } from "@/types/api";
import { RefreshCw, Download, Plus, CheckCircle, AlertTriangle, Clock, Brain, Target, BarChart3, TrendingUp, Eye, Info, Edit2, Save, X, ArrowUp, ArrowDown, GripVertical, Calculator } from "lucide-react";
import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { DragDropContext, Droppable, Draggable, DropResult } from "react-beautiful-dnd";

const Assignments = () => {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const [searchParams] = useSearchParams();
  const [selectedTrainset, setSelectedTrainset] = useState<string | null>(null);
  const [explanationData, setExplanationData] = useState<any>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [trainsetDetails, setTrainsetDetails] = useState<any>(null);
  const [showTrainsetDetails, setShowTrainsetDetails] = useState(false);
  const [defaultTab, setDefaultTab] = useState("ranked");
  const [isEditMode, setIsEditMode] = useState(false);
  const [editedRankedList, setEditedRankedList] = useState<any[]>([]);
  const [reorderReason, setReorderReason] = useState("");

  // Check for tab parameter from URL
  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab === 'conflicts') {
      setDefaultTab('conflicts');
    }
  }, [searchParams]);

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

  // Initialize edited list when entering edit mode
  useEffect(() => {
    if (isEditMode && rankedList.length > 0) {
      setEditedRankedList([...rankedList]);
    }
  }, [isEditMode, rankedList]);

  // Check if user is admin (OPERATIONS_MANAGER)
  const isAdmin = user?.role === 'OPERATIONS_MANAGER';

  // Reorder mutation
  const reorderMutation = useMutation({
    mutationFn: (data: { trainset_ids: string[]; reason?: string }) =>
      optimizationApi.reorderRankedList(data),
    onSuccess: () => {
      toast.success("Ranked list updated", {
        description: "The AI-ranked induction list has been manually adjusted.",
      });
      setIsEditMode(false);
      setReorderReason("");
      queryClient.invalidateQueries({ queryKey: ['ranked-induction-list'] });
    },
    onError: (error: any) => {
      const errorMessage = error.response?.data?.detail || error.message || "Failed to update ranked list";
      toast.error("Update Failed", {
        description: errorMessage,
      });
      console.error("Reorder error:", error);
    },
  });

  // Handle drag end
  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return;

    const items = Array.from(editedRankedList);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setEditedRankedList(items);
  };

  // Move item up
  const moveItemUp = (index: number) => {
    if (index === 0) return;
    const items = [...editedRankedList];
    [items[index - 1], items[index]] = [items[index], items[index - 1]];
    setEditedRankedList(items);
  };

  // Move item down
  const moveItemDown = (index: number) => {
    if (index === editedRankedList.length - 1) return;
    const items = [...editedRankedList];
    [items[index], items[index + 1]] = [items[index + 1], items[index]];
    setEditedRankedList(items);
  };

  // Save reordered list
  const handleSaveReorder = () => {
    const trainsetIds = editedRankedList.map(item => item.trainset_id);
    reorderMutation.mutate({
      trainset_ids: trainsetIds,
      reason: reorderReason || undefined,
    });
  };

  // Fetch conflict alerts
  const { data: conflictAlerts = [] } = useQuery({
    queryKey: ['conflict-alerts'],
    queryFn: () => assignmentApi.getConflicts().then(res => res.data),
    refetchInterval: 15000,
  });

  // Mutations for actions
  const optimizationMutation = useMutation({
    mutationFn: optimizationApi.runOptimization,
    onSuccess: () => {
      toast.success("Optimization completed", {
        description: "AI-ranked induction list has been refreshed for tomorrow's service window.",
      });
      queryClient.invalidateQueries({ queryKey: ['ranked-induction-list'] });
    },
    onError: (error: any) => {
      const errorMessage = error.response?.data?.detail || error.message || "Failed to run optimization";
      toast.error("Optimization Failed", {
        description: errorMessage,
      });
      console.error("Optimization error:", error);
    },
  });

  const approveMutation = useMutation({
    mutationFn: assignmentApi.approve,
    onSuccess: (data) => {
      toast.success("Assignment Approved", {
        description: `Successfully approved ${data.data?.approved_count || 1} assignment(s)`,
      });
      queryClient.invalidateQueries({ queryKey: ['assignments'] });
      queryClient.invalidateQueries({ queryKey: ['assignments-summary'] });
    },
    onError: (error: any) => {
      const errorMessage = error.response?.data?.detail || error.message || "Failed to approve assignment";
      toast.error("Approval Failed", {
        description: errorMessage,
      });
      console.error("Approve error:", error);
    },
  });

  const overrideMutation = useMutation({
    mutationFn: assignmentApi.override,
    onSuccess: (data) => {
      toast.success("Assignment Overridden", {
        description: data.data?.message || "Assignment decision has been overridden",
      });
      queryClient.invalidateQueries({ queryKey: ['assignments'] });
      queryClient.invalidateQueries({ queryKey: ['assignments-summary'] });
    },
    onError: (error: any) => {
      const errorMessage = error.response?.data?.detail || error.message || "Failed to override assignment";
      toast.error("Override Failed", {
        description: errorMessage,
      });
      console.error("Override error:", error);
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
          <Button
            variant="industrial"
            disabled={optimizationMutation.isPending}
            onClick={() => {
              optimizationMutation.mutate({});
            }}
          >
            <Plus className="h-4 w-4 mr-2" />
            {optimizationMutation.isPending ? "Running..." : "Run Optimization"}
          </Button>
        </div>
      </div>

      {/* Assignments Tabs */}
      <Tabs defaultValue={defaultTab} className="w-full">
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
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold">AI-Ranked Induction List</h3>
                  <Badge variant="outline" className="flex items-center gap-1">
                    <Brain className="h-3 w-3" />
                    ML Optimized
                  </Badge>
                  {isEditMode && (
                    <Badge variant="destructive" className="flex items-center gap-1">
                      <Edit2 className="h-3 w-3" />
                      Edit Mode
                    </Badge>
                  )}
                </div>
                {isAdmin && (
                  <div className="flex items-center gap-2">
                    {!isEditMode ? (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setIsEditMode(true)}
                      >
                        <Edit2 className="h-4 w-4 mr-2" />
                        Adjust Order
                      </Button>
                    ) : (
                      <>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setIsEditMode(false);
                            setEditedRankedList([...rankedList]);
                            setReorderReason("");
                          }}
                        >
                          <X className="h-4 w-4 mr-2" />
                          Cancel
                        </Button>
                        <Button
                          variant="industrial"
                          size="sm"
                          onClick={handleSaveReorder}
                          disabled={reorderMutation.isPending}
                        >
                          <Save className="h-4 w-4 mr-2" />
                          {reorderMutation.isPending ? "Saving..." : "Save Order"}
                        </Button>
                      </>
                    )}
                  </div>
                )}
              </div>
              {isEditMode && (
                <Card className="p-4 bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800">
                  <div className="space-y-3">
                    <Label htmlFor="reorder-reason">Reason for adjustment (optional):</Label>
                    <Input
                      id="reorder-reason"
                      placeholder="e.g., Operational priority, maintenance schedule conflict..."
                      value={reorderReason}
                      onChange={(e) => setReorderReason(e.target.value)}
                    />
                    <p className="text-sm text-muted-foreground">
                      Drag items to reorder or use the arrow buttons. Click "Save Order" to apply changes.
                    </p>
                  </div>
                </Card>
              )}
              <DragDropContext onDragEnd={handleDragEnd}>
                <Droppable droppableId="ranked-list">
                  {(provided) => (
                    <div {...provided.droppableProps} ref={provided.innerRef} className="space-y-4">
                      {(isEditMode ? editedRankedList : rankedList).map((decision: any, index: number) => (
                        <Draggable
                          key={decision.trainset_id}
                          draggableId={decision.trainset_id}
                          index={index}
                          isDragDisabled={!isEditMode}
                        >
                          {(provided, snapshot) => (
                            <Card
                              ref={provided.innerRef}
                              {...provided.draggableProps}
                              className={`hover:shadow-md transition-shadow ${snapshot.isDragging ? 'shadow-lg ring-2 ring-primary' : ''
                                } ${isEditMode ? 'cursor-move' : ''}`}
                            >
                              <CardHeader>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-3">
                                    {isEditMode && (
                                      <div {...provided.dragHandleProps} className="cursor-grab active:cursor-grabbing">
                                        <GripVertical className="h-5 w-5 text-muted-foreground" />
                                      </div>
                                    )}
                                    <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-bold">
                                      {index + 1}
                                    </div>
                                    <CardTitle className="text-lg">{decision.trainset_id}</CardTitle>
                                    <Badge variant={decision.decision === "INDUCT" ? "success" : "secondary"}>
                                      {decision.decision}
                                    </Badge>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    {isEditMode && (
                                      <div className="flex flex-col gap-1 mr-2">
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          className="h-6 w-6 p-0"
                                          onClick={() => moveItemUp(index)}
                                          disabled={index === 0}
                                        >
                                          <ArrowUp className="h-3 w-3" />
                                        </Button>
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          className="h-6 w-6 p-0"
                                          onClick={() => moveItemDown(index)}
                                          disabled={index === editedRankedList.length - 1}
                                        >
                                          <ArrowDown className="h-3 w-3" />
                                        </Button>
                                      </div>
                                    )}
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
                          )}
                        </Draggable>
                      ))}
                      {provided.placeholder}
                    </div>
                  )}
                </Droppable>
              </DragDropContext>
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
                <Card
                  key={assignment.id}
                  className="border border-red-200 bg-red-50/70 text-red-900 dark:border-red-900/60 dark:bg-red-950/70 dark:text-red-100"
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg text-red-800 dark:text-red-100">
                        {assignment.trainset_id}
                      </CardTitle>
                      <Badge className="bg-red-100 text-red-900 dark:bg-red-900 dark:text-white" variant="destructive">
                        Conflict Detected
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-2">
                          Violations:
                        </h4>
                        <ul className="list-disc list-inside text-sm space-y-1">
                          {assignment.decision?.violations?.map((violation: string, i: number) => (
                            <li key={i} className="text-red-700 dark:text-red-200">
                              {violation}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() =>
                            handleExplainDecision(assignment.trainset_id, assignment.decision?.decision)
                          }
                          disabled={explanationMutation.isPending}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          Explain Conflict
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-red-900 border-red-200 hover:bg-red-100 dark:text-red-100 dark:border-red-800 dark:hover:bg-red-900/40"
                        >
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
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleViewDetails(assignment.trainset_id)}
                    >
                      <Info className="h-4 w-4 mr-2" />
                      View Details
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (!assignment.id) {
                          toast.error("Error", { description: "Assignment ID is missing" });
                          return;
                        }
                        console.log("Override clicked - Assignment:", assignment);
                        const newDecision = assignment.decision.decision === "INDUCT" ? "STANDBY" :
                          assignment.decision.decision === "STANDBY" ? "MAINTENANCE" : "INDUCT";
                        overrideMutation.mutate({
                          assignment_id: assignment.id,
                          user_id: "system",
                          override_decision: newDecision,
                          reason: `Manual override: Changed from ${assignment.decision.decision} to ${newDecision}`
                        });
                      }}
                      disabled={overrideMutation.isPending || assignment.status !== "PENDING"}
                    >
                      {overrideMutation.isPending ? "Overriding..." : "Override"}
                    </Button>
                    <Button
                      size="sm"
                      variant="industrial"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (!assignment.id) {
                          toast.error("Error", { description: "Assignment ID is missing" });
                          return;
                        }
                        console.log("Approve clicked - Assignment:", assignment);
                        approveMutation.mutate({
                          assignment_ids: [assignment.id],
                          user_id: "system",
                          comments: "Approved by supervisor"
                        });
                      }}
                      disabled={approveMutation.isPending || assignment.status !== "PENDING"}
                    >
                      {approveMutation.isPending ? "Approving..." : "Approve"}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="approved" className="space-y-4">
          {approvedAssignments.length === 0 ? (
            <div className="text-center py-8">
              <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <p className="text-muted-foreground">No approved assignments</p>
              <p className="text-sm text-muted-foreground mt-2">Approved assignments will appear here after you approve them from the Pending tab</p>
            </div>
          ) : (
            approvedAssignments.map((assignment) => (
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
                    {assignment.approved_by && (
                      <div>
                        <span className="text-muted-foreground">Approved by:</span>
                        <p className="font-medium">{assignment.approved_by}</p>
                      </div>
                    )}
                    {assignment.approved_at && (
                      <div>
                        <span className="text-muted-foreground">Approved at:</span>
                        <p className="font-medium">{new Date(assignment.approved_at).toLocaleString()}</p>
                      </div>
                    )}
                  </div>
                  {assignment.decision.reasons && assignment.decision.reasons.length > 0 && (
                    <p className="text-sm text-muted-foreground mt-2">{assignment.decision.reasons[0]}</p>
                  )}
                  <div className="flex gap-2 mt-4">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleViewDetails(assignment.trainset_id)}
                    >
                      <Info className="h-4 w-4 mr-2" />
                      View Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="overridden" className="space-y-4">
          {overriddenAssignments.length === 0 ? (
            <div className="text-center py-8">
              <AlertTriangle className="h-12 w-12 text-orange-500 mx-auto mb-4" />
              <p className="text-muted-foreground">No overridden assignments</p>
              <p className="text-sm text-muted-foreground mt-2">Overridden assignments will appear here after you override them from the Pending tab</p>
            </div>
          ) : (
            overriddenAssignments.map((assignment) => (
              <Card key={assignment.id} className="hover:shadow-md transition-shadow border-orange-200">
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
                      <span className="text-muted-foreground">Original Decision:</span>
                      <p className="font-medium">{assignment.decision.decision}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Override Decision:</span>
                      <p className="font-medium text-orange-600">{assignment.override_decision || 'N/A'}</p>
                    </div>
                    {assignment.override_by && (
                      <div>
                        <span className="text-muted-foreground">Overridden by:</span>
                        <p className="font-medium">{assignment.override_by}</p>
                      </div>
                    )}
                    {assignment.override_at && (
                      <div>
                        <span className="text-muted-foreground">Overridden at:</span>
                        <p className="font-medium">{new Date(assignment.override_at).toLocaleString()}</p>
                      </div>
                    )}
                  </div>
                  {assignment.override_reason && (
                    <div className="mt-2 p-2 bg-orange-50 rounded">
                      <span className="text-sm font-medium text-orange-800">Override Reason:</span>
                      <p className="text-sm text-orange-700">{assignment.override_reason}</p>
                    </div>
                  )}
                  <div className="flex gap-2 mt-4">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleViewDetails(assignment.trainset_id)}
                    >
                      <Info className="h-4 w-4 mr-2" />
                      View Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>
      </Tabs>

      {/* Explanation Modal */}
      {showExplanation && explanationData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-background border border-border rounded-lg shadow-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-lg font-semibold">Induction Decision Explanation</h3>
              <Button variant="ghost" size="sm" onClick={() => setShowExplanation(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="p-6 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-medium text-muted-foreground">Trainset</h4>
                  <p className="text-2xl font-bold">{selectedTrainset}</p>
                </div>
                <div className="text-right">
                  <h4 className="text-sm font-medium text-muted-foreground">Composite Score</h4>
                  <p className="text-2xl font-bold text-primary">
                    {Math.round(explanationData.score * 100)}%
                  </p>
                </div>
              </div>

              {explanationData.score_details && (
                <div className="p-4 bg-muted/50 rounded-lg border border-border">
                  <h4 className="font-medium mb-3 flex items-center gap-2 text-sm">
                    <Calculator className="h-4 w-4" />
                    Score Calculation
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground block text-xs">Tier 2 Score (Branding/Defects)</span>
                      <span className="font-mono font-medium">{explanationData.score_details.tier2_score?.toFixed(1)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block text-xs">Tier 3 Score (Mileage/Ops)</span>
                      <span className="font-mono font-medium">{explanationData.score_details.tier3_score?.toFixed(1)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block text-xs">Formula</span>
                      <span className="font-mono text-xs text-muted-foreground">{explanationData.score_details.formula}</span>
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h4 className="font-medium flex items-center gap-2 text-green-600">
                    <CheckCircle className="h-4 w-4" />
                    Top Reasons
                  </h4>
                  <ul className="space-y-2">
                    {explanationData.top_reasons?.map((reason: string, i: number) => (
                      <li key={i} className="text-sm p-2 bg-green-50 dark:bg-green-900/20 rounded border border-green-100 dark:border-green-900">
                        {reason}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium flex items-center gap-2 text-red-600">
                    <AlertTriangle className="h-4 w-4" />
                    Risks & Violations
                  </h4>
                  <ul className="space-y-2">
                    {explanationData.top_risks?.map((risk: string, i: number) => (
                      <li key={i} className="text-sm p-2 bg-red-50 dark:bg-red-900/20 rounded border border-red-100 dark:border-red-900">
                        {risk}
                      </li>
                    ))}
                    {explanationData.violations?.map((violation: string, i: number) => (
                      <li key={i} className="text-sm p-2 bg-red-50 dark:bg-red-900/20 rounded border border-red-100 dark:border-red-900 font-medium">
                        {violation}
                      </li>
                    ))}
                    {(!explanationData.top_risks?.length && !explanationData.violations?.length) && (
                      <li className="text-sm text-muted-foreground italic">No significant risks detected</li>
                    )}
                  </ul>
                </div>
              </div>

              {explanationData.shap_values && explanationData.shap_values.length > 0 && (
                <div className="space-y-3">
                  <h4 className="font-medium flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" />
                    Key Contributing Factors
                  </h4>
                  <div className="space-y-2">
                    {explanationData.shap_values.map((feature: any, i: number) => (
                      <div key={i} className="flex items-center justify-between text-sm p-2 border border-border rounded">
                        <span>{feature.name}</span>
                        <div className="flex items-center gap-2">
                          <div className={`h-2 w-24 rounded-full bg-secondary overflow-hidden`}>
                            <div
                              className={`h-full ${feature.impact === 'positive' ? 'bg-green-500' : feature.impact === 'negative' ? 'bg-red-500' : 'bg-gray-500'}`}
                              style={{ width: `${Math.abs(feature.value) * 100}%` }}
                            />
                          </div>
                          <span className={`font-mono text-xs ${feature.impact === 'positive' ? 'text-green-600' : feature.impact === 'negative' ? 'text-red-600' : 'text-muted-foreground'}`}>
                            {feature.impact === 'positive' ? '+' : ''}{feature.value.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div className="p-4 border-t border-border flex justify-end">
              <Button onClick={() => setShowExplanation(false)}>Close</Button>
            </div>
          </div>
        </div>
      )}

      {/* Trainset Details Modal */}
      {showTrainsetDetails && trainsetDetails && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-background border border-border rounded-lg shadow-lg w-full max-w-3xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-lg font-semibold">Trainset Details: {trainsetDetails.trainset_id}</h3>
              <Button variant="ghost" size="sm" onClick={() => setShowTrainsetDetails(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="p-6 space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-3 bg-secondary/10 rounded-lg">
                  <div className="text-sm text-muted-foreground">Status</div>
                  <div className="font-medium">{trainsetDetails.status}</div>
                </div>
                <div className="p-3 bg-secondary/10 rounded-lg">
                  <div className="text-sm text-muted-foreground">Mileage</div>
                  <div className="font-medium">{trainsetDetails.current_mileage?.toLocaleString()} km</div>
                </div>
                <div className="p-3 bg-secondary/10 rounded-lg">
                  <div className="text-sm text-muted-foreground">Location</div>
                  <div className="font-medium">{trainsetDetails.location || 'Depot'}</div>
                </div>
                <div className="p-3 bg-secondary/10 rounded-lg">
                  <div className="text-sm text-muted-foreground">Commissioned</div>
                  <div className="font-medium">{new Date(trainsetDetails.commission_date).toLocaleDateString()}</div>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-medium">Fitness Certificates</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {Object.entries(trainsetDetails.fitness_certificates || {}).map(([key, cert]: [string, any]) => (
                    <div key={key} className="flex items-center justify-between p-2 border border-border rounded text-sm">
                      <span className="capitalize">{key.replace('_', ' ')}</span>
                      <Badge variant={cert.status === 'VALID' ? 'success' : 'destructive'}>
                        {cert.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-medium">Recent Job Cards</h4>
                {trainsetDetails.job_cards?.open_cards > 0 ? (
                  <div className="text-sm p-3 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 rounded border border-yellow-200 dark:border-yellow-800">
                    {trainsetDetails.job_cards.open_cards} open cards ({trainsetDetails.job_cards.critical_cards} critical)
                  </div>
                ) : (
                  <div className="text-sm p-3 bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-200 rounded border border-green-200 dark:border-green-800">
                    No open job cards
                  </div>
                )}
              </div>

              {trainsetDetails.branding && (
                <div className="space-y-3">
                  <h4 className="font-medium">Branding</h4>
                  <div className="p-3 border border-border rounded text-sm">
                    <div className="grid grid-cols-2 gap-2">
                      <div><span className="text-muted-foreground">Advertiser:</span> {trainsetDetails.branding.current_advertiser || 'None'}</div>
                      <div><span className="text-muted-foreground">Priority:</span> {trainsetDetails.branding.priority || 'N/A'}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="p-4 border-t border-border flex justify-end">
              <Button onClick={() => setShowTrainsetDetails(false)}>Close</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Assignments;
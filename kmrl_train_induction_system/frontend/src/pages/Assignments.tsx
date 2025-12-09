import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { assignmentApi, optimizationApi, trainsetsApi } from "@/services/api";
import { Assignment, AssignmentSummary } from "@/types/api";
import { RefreshCw, Download, Plus, CheckCircle, AlertTriangle, Clock, Brain, Target, BarChart3, TrendingUp, Eye, Info, Edit2, Save, X, ArrowUp, ArrowDown, GripVertical, Activity } from "lucide-react";
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
    onSuccess: async (data) => {
      toast.success("Assignment Approved", {
        description: `Successfully approved ${data.data?.approved_count || 1} assignment(s)`,
      });
      // Invalidate and immediately refetch queries for real-time updates
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['assignments'] }),
        queryClient.invalidateQueries({ queryKey: ['assignments-summary'] }),
        queryClient.refetchQueries({ queryKey: ['dashboard-overview'] }), // Immediate refetch for dashboard
      ]);
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
    onSuccess: async (data) => {
      toast.success("Assignment Overridden", {
        description: data.data?.message || "Assignment decision has been overridden",
      });
      // Invalidate and immediately refetch queries for real-time updates
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['assignments'] }),
        queryClient.invalidateQueries({ queryKey: ['assignments-summary'] }),
        queryClient.refetchQueries({ queryKey: ['dashboard-overview'] }), // Immediate refetch for dashboard
      ]);
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

  // Create a map of trainset_id to assignment for quick lookup
  const assignmentsMap = new Map(assignments.map(a => [a.trainset_id, a]));
  
  // Get pending assignments from ranked list (sorted by rank) or from assignments
  // Priority: ranked list order (if exists) > assignment priority
  const pendingAssignments = rankedList.length > 0
    ? rankedList
        .map((decision: any, index: number) => {
          const assignment = assignmentsMap.get(decision.trainset_id);
          // If assignment exists and is pending, use it; otherwise create a virtual one
          if (assignment && assignment.status === "PENDING") {
            return {
              ...assignment,
              rank: index + 1,
              decision: decision // Use decision from ranked list
            };
          } else if (!assignment) {
            // Create virtual assignment for trains in ranked list but not in assignments
            return {
              id: `virtual-${decision.trainset_id}`,
              trainset_id: decision.trainset_id,
              decision: decision,
              status: "PENDING" as const,
              priority: 5 - index, // Higher rank = higher priority
              rank: index + 1,
              created_at: new Date().toISOString(),
              created_by: "system",
              last_updated: new Date().toISOString()
            };
          }
          return null;
        })
        .filter((a): a is NonNullable<typeof a> => a !== null && a.status === "PENDING")
    : assignments
        .filter(a => a.status === "PENDING")
        .sort((a, b) => (b.priority || 0) - (a.priority || 0)); // Sort by priority descending
  
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
          <h2 className="text-3xl font-bold text-foreground">Assignments</h2>
          <p className="text-muted-foreground">All Assignments</p>
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
              const tomorrow = new Date();
              tomorrow.setDate(tomorrow.getDate() + 1);
              optimizationMutation.mutate({
                target_date: tomorrow.toISOString(),
                required_service_hours: 14,
              });
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
            AI-Ranked Induction List ({rankedList.length})
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
            Approved Assignments ({approvedAssignments.length})
          </TabsTrigger>
          <TabsTrigger value="overridden">
            <AlertTriangle className="h-4 w-4 mr-2" />
            Rejected Assignments ({overriddenAssignments.length})
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
              <p className="text-muted-foreground">No trains requiring maintenance</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Maintenance Required</h3>
                <Badge variant="destructive" className="flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  {conflictAlerts.length} Trains Need Maintenance
                </Badge>
              </div>
              {conflictAlerts.map((assignment: any, index: number) => (
                <Card
                  key={assignment.id || assignment.trainset_id}
                  className="hover:shadow-md transition-shadow"
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-bold">
                          {index + 1}
                        </div>
                        <CardTitle className="text-lg">{assignment.trainset_id}</CardTitle>
                        <Badge variant="destructive" className="bg-orange-500 hover:bg-orange-600">
                          MAINTENANCE
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-muted-foreground">
                          Score: {Math.round((assignment.decision?.score || 0) * 100)}%
                        </span>
                        <span className="text-sm font-medium text-muted-foreground">
                          Confidence: {Math.round((assignment.decision?.confidence_score || 1.0) * 100)}%
                        </span>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-2">Top Reasons:</h4>
                        <ul className="list-disc list-inside text-sm space-y-1">
                          {assignment.decision?.top_reasons && assignment.decision.top_reasons.length > 0 ? (
                            assignment.decision.top_reasons.slice(0, 3).map((reason: string, i: number) => (
                              <li key={i} className="text-orange-600 dark:text-orange-400">
                                {reason}
                              </li>
                            ))
                          ) : (
                            <li className="text-orange-600 dark:text-orange-400">
                              Maintenance required based on system analysis
                            </li>
                          )}
                        </ul>
                      </div>
                      {assignment.decision?.top_risks && assignment.decision.top_risks.length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm text-muted-foreground mb-2">Risks:</h4>
                          <ul className="list-disc list-inside text-sm space-y-1">
                            {assignment.decision.top_risks.slice(0, 2).map((risk: string, i: number) => (
                              <li key={i} className="text-red-600 dark:text-red-400">
                                {risk}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            handleExplainDecision(assignment.trainset_id, assignment.decision?.decision || "MAINTENANCE")
                          }
                          disabled={explanationMutation.isPending}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          Explain Decision
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => handleViewDetails(assignment.trainset_id)}
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
            pendingAssignments.map((assignment, index) => (
              <Card key={assignment.id || assignment.trainset_id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-bold">
                        {assignment.rank || (index + 1)}
                      </div>
                      <CardTitle className="text-lg">{assignment.trainset_id}</CardTitle>
                      <Badge variant={assignment.decision.decision === "INDUCT" ? "success" : assignment.decision.decision === "MAINTENANCE" ? "destructive" : "secondary"}>
                        {assignment.decision.decision}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      {getStatusBadge(assignment.status)}
                      <span className="text-sm font-medium text-muted-foreground">
                        Score: {Math.round((assignment.decision?.score || 0) * 100)}%
                      </span>
                      <span className="text-sm font-medium text-muted-foreground">
                        Confidence: {Math.round((assignment.decision?.confidence_score || 0.8) * 100)}%
                      </span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {assignment.decision?.top_reasons && assignment.decision.top_reasons.length > 0 && (
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-2">Top Reasons:</h4>
                        <ul className="list-disc list-inside text-sm space-y-1">
                          {assignment.decision.top_reasons.slice(0, 3).map((reason: string, i: number) => (
                            <li key={i} className="text-green-600 dark:text-green-400">{reason}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {assignment.decision?.top_risks && assignment.decision.top_risks.length > 0 && (
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-2">Risks:</h4>
                        <ul className="list-disc list-inside text-sm space-y-1">
                          {assignment.decision.top_risks.slice(0, 2).map((risk: string, i: number) => (
                            <li key={i} className="text-red-600 dark:text-red-400">{risk}</li>
                          ))}
                        </ul>
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
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          if (!assignment.id || assignment.id.startsWith('virtual-')) {
                            toast.error("Error", { description: "Cannot override: Assignment not yet created. Please approve first." });
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
                        disabled={overrideMutation.isPending || assignment.status !== "PENDING" || assignment.id?.startsWith('virtual-')}
                      >
                        {overrideMutation.isPending ? "Overriding..." : "Override"}
                      </Button>
                      <Button
                        size="sm"
                        variant="industrial"
                        onClick={async (e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          
                          // If virtual assignment, create it first
                          if (!assignment.id || assignment.id.startsWith('virtual-')) {
                            try {
                              // Create assignment first
                              const createResponse = await assignmentApi.create({
                                trainset_id: assignment.trainset_id,
                                decision: assignment.decision,
                                created_by: "system",
                                priority: assignment.priority || 5
                              });
                              
                              // Invalidate queries to refresh the list
                              queryClient.invalidateQueries({ queryKey: ['assignments'] });
                              
                              // Then approve it
                              approveMutation.mutate({
                                assignment_ids: [createResponse.data.id],
                                user_id: "system",
                                comments: "Approved by supervisor"
                              });
                            } catch (error: any) {
                              toast.error("Error", { 
                                description: error.response?.data?.detail || "Failed to create assignment" 
                              });
                            }
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
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
          <Card className="max-w-4xl w-full max-h-[90vh] overflow-y-auto bg-white dark:bg-slate-900 text-foreground dark:text-slate-100 shadow-2xl border border-slate-200 dark:border-slate-800">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl text-foreground dark:text-white">
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
                  <h3 className="font-semibold text-lg mb-3 text-green-600 dark:text-green-400">Top Reasons</h3>
                  <ul className="space-y-2">
                    {explanationData.top_reasons?.map((reason: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 dark:text-green-400 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-foreground dark:text-slate-100">{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h3 className="font-semibold text-lg mb-3 text-red-600 dark:text-red-400">Risks & Violations</h3>
                  <ul className="space-y-2">
                    {explanationData.top_risks?.map((risk: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-red-500 dark:text-red-400 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-foreground dark:text-slate-100">{risk}</span>
                      </li>
                    ))}
                    {explanationData.violations?.map((violation: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-red-500 dark:text-red-400 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-red-600 dark:text-red-300">{violation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {explanationData.shap_values && explanationData.shap_values.length > 0 && (
                <div>
                  <h3 className="font-semibold text-lg mb-3 text-foreground dark:text-white">Feature Impact Analysis</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {explanationData.shap_values.map((feature: any, i: number) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-slate-800/70 border border-gray-100 dark:border-slate-700"
                      >
                        <span className="text-sm font-medium text-foreground dark:text-slate-100">{feature.name}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-foreground dark:text-slate-200">{feature.value}</span>
                          <span className={`text-xs px-2 py-1 rounded ${feature.impact === 'positive'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200'
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
        </div >
      )
      }

      {/* Trainset Details Modal */}
      {
        showTrainsetDetails && trainsetDetails && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
            <Card className="max-w-4xl w-full max-h-[90vh] overflow-y-auto bg-white dark:bg-slate-900 shadow-2xl">
              <CardHeader className="border-b">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-2xl font-bold">
                      Trainset Details - {trainsetDetails.trainset_id}
                    </CardTitle>
                    <p className="text-muted-foreground text-sm mt-1">Comprehensive fleet status and health metrics</p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowTrainsetDetails(false)}
                  >
                    <X className="h-5 w-5" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-6 space-y-8">

                {/* Top Row: Basic Info & Score */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="md:col-span-2 space-y-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <Info className="h-5 w-5 text-primary" />
                      Basic Information
                    </h3>
                    <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm">
                      <div className="flex justify-between border-b pb-2">
                        <span className="text-muted-foreground">Trainset ID:</span>
                        <span className="font-medium">{trainsetDetails.trainset_id}</span>
                      </div>
                      <div className="flex justify-between border-b pb-2">
                        <span className="text-muted-foreground">Manufacturer:</span>
                        <span className="font-medium">{trainsetDetails.basic_info.manufacturer}</span>
                      </div>
                      <div className="flex justify-between border-b pb-2">
                        <span className="text-muted-foreground">Model:</span>
                        <span className="font-medium">{trainsetDetails.basic_info.model}</span>
                      </div>
                      <div className="flex justify-between border-b pb-2">
                        <span className="text-muted-foreground">Year of Manufacture:</span>
                        <span className="font-medium">{trainsetDetails.basic_info.year_of_manufacture}</span>
                      </div>
                      <div className="flex justify-between border-b pb-2">
                        <span className="text-muted-foreground">Status:</span>
                        <Badge variant={trainsetDetails.basic_info.status === 'ACTIVE' ? 'success' : 'secondary'}>
                          {trainsetDetails.basic_info.status}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  {/* Score Card */}
                  <Card className="bg-slate-50 dark:bg-slate-800 border-none shadow-inner flex flex-col items-center justify-center p-6">
                    <div className="relative flex items-center justify-center">
                      <svg className="w-32 h-32 transform -rotate-90">
                        <circle
                          className="text-gray-200"
                          strokeWidth="8"
                          stroke="currentColor"
                          fill="transparent"
                          r="58"
                          cx="64"
                          cy="64"
                        />
                        <circle
                          className="text-primary"
                          strokeWidth="8"
                          strokeDasharray={365}
                          strokeDashoffset={365 - (365 * (trainsetDetails.operational_metrics.optimization_score || 0.85))}
                          strokeLinecap="round"
                          stroke="currentColor"
                          fill="transparent"
                          r="58"
                          cx="64"
                          cy="64"
                        />
                      </svg>
                      <div className="absolute flex flex-col items-center">
                        <span className="text-3xl font-bold text-primary">
                          {Math.round((trainsetDetails.operational_metrics.optimization_score || 0.85) * 100)}%
                        </span>
                        <span className="text-xs text-muted-foreground uppercase">Score</span>
                      </div>
                    </div>
                    <div className="mt-4 text-center">
                      <p className="text-sm font-medium">Optimization Score</p>
                      <p className="text-xs text-muted-foreground">Based on health & priority</p>
                    </div>
                  </Card>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Mileage */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <Activity className="h-5 w-5 text-blue-500" />
                      Mileage Statistics
                    </h3>
                    <div className="bg-slate-50 dark:bg-slate-800/50 p-4 rounded-lg space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Current Mileage</span>
                        <span className="text-lg font-bold">{trainsetDetails.performance_data.current_mileage?.toLocaleString()} km</span>
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Maintenance Limit</span>
                          <span>{trainsetDetails.performance_data.max_mileage?.toLocaleString()} km</span>
                        </div>
                        <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${Math.min(((trainsetDetails.performance_data.current_mileage || 0) / (trainsetDetails.performance_data.max_mileage || 1)) * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Certificates */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      Certificates
                    </h3>
                    <div className="grid grid-cols-1 gap-3">
                      {['Rolling Stock', 'Signalling', 'Telecom'].map((cert) => {
                        const key = cert.toLowerCase().replace(' ', '_');
                        const certData = trainsetDetails.certificates?.[key];
                        // Handle both string (legacy) and object (real data) formats
                        const status = typeof certData === 'object' ? certData?.status : certData || 'VALID';

                        return (
                          <div key={cert} className="flex items-center justify-between p-3 bg-white dark:bg-slate-800 border rounded-lg shadow-sm">
                            <span className="text-sm font-medium">{cert} Certificate</span>
                            <Badge variant={status === 'VALID' ? 'success' : 'destructive'}>
                              {status || 'VALID'}
                            </Badge>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Job Cards */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5 text-orange-500" />
                      Job Cards
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-100 dark:border-orange-800 rounded-lg text-center">
                        <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                          {trainsetDetails.job_cards?.open || 0}
                        </div>
                        <div className="text-sm text-orange-800 dark:text-orange-300">Open Job Cards</div>
                      </div>
                      <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-800 rounded-lg text-center">
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                          {trainsetDetails.job_cards?.critical || 0}
                        </div>
                        <div className="text-sm text-red-800 dark:text-red-300">Critical Issues</div>
                      </div>
                    </div>
                  </div>

                  {/* Branding */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <Target className="h-5 w-5 text-purple-500" />
                      Branding Status
                    </h3>
                    <div className="bg-purple-50 dark:bg-purple-900/10 border border-purple-100 dark:border-purple-800 rounded-lg p-4 space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Advertiser</span>
                        <span className="font-semibold text-purple-900 dark:text-purple-100">
                          {trainsetDetails.branding?.advertiser || 'None'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Priority</span>
                        <Badge variant="outline" className="border-purple-200 text-purple-700">
                          {trainsetDetails.branding?.priority || 'LOW'}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Status</span>
                        <span className="text-sm font-medium">
                          {trainsetDetails.branding?.status || 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end gap-3 pt-4 border-t">
                  <Button variant="outline" onClick={() => setShowTrainsetDetails(false)}>
                    Close
                  </Button>
                  <Button onClick={() => window.print()}>
                    <Download className="h-4 w-4 mr-2" />
                    Export Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )
      }
    </div >
  );
};

export default Assignments;
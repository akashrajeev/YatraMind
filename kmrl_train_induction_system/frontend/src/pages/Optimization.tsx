import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { optimizationApi } from "@/services/api";
import { getBaselineResult, getScenarioResult } from "@/utils/simulation";
import { DetailedResults } from "@/components/simulation/DetailedResults";
import {
  Brain,
  Play,
  Settings,
  BarChart3,
  Target,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  Download,
  RefreshCw,
  Train
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";

type OptimizationParamsState = {
  target_date: string;
  required_service_hours: number;
};

type SimulationParamsState = {
  exclude_trainsets: string;
  force_induct: string;
  required_service_count: number;
  w_readiness: number;
  w_reliability: number;
  w_branding: number;
  w_shunt: number;
  w_mileage_balance: number;
};

type OptimizationDiagnostics = {
  requested_service_hours?: number;
  requested_train_count?: number;
  eligible_train_count?: number;
  granted_train_count?: number;
  avg_hours_per_train?: number;
};

const DEFAULT_HOURS_PER_TRAIN = 12;

const parseNumericInput = (value: string, fallback: number) => {
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const splitCommaList = (value: string) =>
  value
    .split(",")
    .map((token) => token.trim())
    .filter(Boolean);

const Optimization = () => {
  const [optimizationParams, setOptimizationParams] = useState<OptimizationParamsState>({
    target_date: new Date().toISOString().split('T')[0],
    required_service_hours: 0 // Default to 0 to trigger timetable logic
  });

  const [simulationParams, setSimulationParams] = useState<SimulationParamsState>({
    exclude_trainsets: "",
    force_induct: "",
    required_service_count: 14,
    w_readiness: 0.35,
    w_reliability: 0.30,
    w_branding: 0.20,
    w_shunt: 0.10,
    w_mileage_balance: 0.05
  });
  const [simulationResults, setSimulationResults] = useState<any>(null);
  const [simulationParamsUsed, setSimulationParamsUsed] = useState<any>(null);
  const [showSimulationResults, setShowSimulationResults] = useState(false);
  const [optimizationDiagnostics, setOptimizationDiagnostics] = useState<OptimizationDiagnostics | null>(null);
  const [optimizationNote, setOptimizationNote] = useState<string | null>(null);

  const queryClient = useQueryClient();

  // Fetch optimization data
  const { data: constraints, isLoading: constraintsLoading } = useQuery({
    queryKey: ['optimization-constraints'],
    queryFn: () => optimizationApi.checkConstraints().then(res => res.data),
    refetchInterval: 60000,
  });

  const { data: latestOptimization } = useQuery({
    queryKey: ['optimization-latest'],
    queryFn: () => optimizationApi.getLatest().then(res => res.data),
  });

  const { data: stablingGeometry, error: stablingError } = useQuery({
    queryKey: ['optimization-stabling'],
    queryFn: () => optimizationApi.getStablingGeometry().then(res => res.data),
    retry: 0,
  });

  const { data: shuntingSchedule, error: shuntingError } = useQuery({
    queryKey: ['optimization-shunting'],
    queryFn: () => optimizationApi.getShuntingSchedule().then(res => res.data),
    retry: 0,
  });

  const safeLatestDecisions = useMemo(() => {
    if (!latestOptimization) {
      return [];
    }
    if (Array.isArray(latestOptimization)) {
      return latestOptimization;
    }
    if (Array.isArray(latestOptimization?.decisions)) {
      return latestOptimization.decisions;
    }
    return [];
  }, [latestOptimization]);

  const handleRefresh = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['optimization-constraints'] }),
      queryClient.invalidateQueries({ queryKey: ['optimization-latest'] }),
      queryClient.invalidateQueries({ queryKey: ['optimization-stabling'] }),
      queryClient.invalidateQueries({ queryKey: ['optimization-shunting'] }),
    ]);
  }, [queryClient]);

  // Transform frontend parameters to backend scenario format
  const transformSimulationParams = useCallback((params: SimulationParamsState) => {
    const scenario: Record<string, any> = {
      required_service_hours: params.required_service_count,
    };

    const forced = splitCommaList(params.force_induct);
    if (forced.length) {
      scenario.force_decisions = forced.reduce((acc: Record<string, string>, id: string) => {
        acc[id] = "INDUCT";
        return acc;
      }, {});
    }

    const excluded = splitCommaList(params.exclude_trainsets);
    if (excluded.length) {
      scenario.override_train_attributes = excluded.reduce((acc: Record<string, any>, id: string) => {
        acc[id] = {
          "fitness_certificates.rolling_stock.status": "EXPIRED"
        };
        return acc;
      }, {});
    }

    scenario.weights = {
      readiness: params.w_readiness || 0.35,
      reliability: params.w_reliability || 0.30,
      branding: params.w_branding || 0.20,
      shunt: params.w_shunt || 0.10,
      mileage_balance: params.w_mileage_balance || 0.05
    };

    return scenario;
  }, []);

  // Mutations
  const runOptimizationMutation = useMutation({
    mutationFn: async () => {
      const hours = Number(optimizationParams.required_service_hours);
      if (Number.isNaN(hours)) {
        throw new Error("Required Service Hours must be a valid number.");
      }
      if (hours < 0 || hours > 10000) {
        throw new Error("Required Service Hours must be between 0 and 10\u202f000.");
      }

      const payload = {
        target_date: new Date(optimizationParams.target_date).toISOString(),
        required_service_hours: hours || undefined,
      };

      const res = await optimizationApi.runOptimization(payload);
      return res.data;
    },
    onSuccess: async (data) => {
      const normalizedDiagnostics: OptimizationDiagnostics | null = (() => {
        const diagSource = data?.diagnostics ?? data;
        if (!diagSource || typeof diagSource !== "object") {
          return null;
        }
        const fallback: OptimizationDiagnostics = {
          requested_service_hours: diagSource.requested_service_hours,
          requested_train_count: diagSource.requested_train_count,
          eligible_train_count: diagSource.eligible_train_count,
          granted_train_count: diagSource.granted_train_count,
          avg_hours_per_train: diagSource.avg_hours_per_train ?? DEFAULT_HOURS_PER_TRAIN,
        };
        const hasValues = Object.values(fallback).some(
          (value) => typeof value === "number" && Number.isFinite(value)
        );
        return hasValues ? fallback : null;
      })();

      setOptimizationDiagnostics(normalizedDiagnostics);
      setOptimizationNote(data?.note || data?.diagnostics?.note || null);
      await queryClient.invalidateQueries({ queryKey: ['optimization-latest'] });
      await queryClient.invalidateQueries({ queryKey: ['optimization-constraints'] });
      await queryClient.invalidateQueries({ queryKey: ['optimization-stabling'] });
      await queryClient.invalidateQueries({ queryKey: ['optimization-shunting'] });

      try {
        const [stablingRes, shuntingRes] = await Promise.all([
          optimizationApi.getStablingGeometry(),
          optimizationApi.getShuntingSchedule(),
        ]);
        queryClient.setQueryData(['optimization-stabling'], stablingRes.data);
        queryClient.setQueryData(['optimization-shunting'], shuntingRes.data);
      } catch (err) {
        console.error("Failed to refresh stabling/shunting after optimization", err);
      }
    },
    onError: (error: any) => {
      const message =
        error?.message ||
        error?.response?.data?.detail ||
        "Optimization failed. Please check inputs and try again.";
      alert(message);
    },
  });

  const runSimulationMutation = useMutation({
    mutationFn: (params: SimulationParamsState) => {
      const scenario = transformSimulationParams(params);
      return optimizationApi.runSimulation(scenario);
    },
    onSuccess: (response, variables) => {
      const data = response?.data || response;
      if (!data) {
        alert("Simulation completed but returned no data.");
        return;
      }
      setSimulationResults(data);
      setSimulationParamsUsed(variables);
      setShowSimulationResults(true);
    },
    onError: (error: any) => {
      const errorMessage = error?.response?.data?.detail || error?.message || 'Simulation failed';
      alert(errorMessage);
    },
  });

  const [explanationData, setExplanationData] = useState<any>(null);
  const [showExplanation, setShowExplanation] = useState(false);

  const explainMutation = useMutation({
    mutationFn: ({ trainsetId, decision }: { trainsetId: string; decision: string }) =>
      optimizationApi.explainAssignment(trainsetId, decision, 'json'),
    onSuccess: (data) => {
      setExplanationData(data.data);
      setShowExplanation(true);
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">AI/ML Optimization</h2>
          <p className="text-muted-foreground">Advanced optimization and simulation tools</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={handleRefresh} disabled={runOptimizationMutation.isPending}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="industrial" onClick={() => window.print()}>
            <Download className="h-4 w-4 mr-2" />
            Export Results
          </Button>
        </div>
      </div>

      {/* Constraint Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            Constraint Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          {constraintsLoading ? (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mx-auto"></div>
            </div>
          ) : constraints ? (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-foreground">{constraints.total_trainsets}</div>
                <div className="text-sm text-muted-foreground">Total Trainsets</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-success">{constraints.valid_trainsets}</div>
                <div className="text-sm text-muted-foreground">Valid</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-destructive">{constraints.trainsets_with_violations}</div>
                <div className="text-sm text-muted-foreground">With Violations</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">
                  {Math.round((constraints.valid_trainsets / constraints.total_trainsets) * 100)}%
                </div>
                <div className="text-sm text-muted-foreground">Compliance Rate</div>
              </div>
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-muted-foreground">Unable to load constraint status</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Optimization Tabs */}
      <Tabs defaultValue="optimization" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="optimization">Run Optimization</TabsTrigger>
          <TabsTrigger value="simulation">What-If Simulation</TabsTrigger>
          <TabsTrigger value="stabling">Stabling Geometry</TabsTrigger>
          <TabsTrigger value="shunting">Shunting Schedule</TabsTrigger>
        </TabsList>

        {/* Run Optimization */}
        <TabsContent value="optimization" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI/ML Optimization Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="target-date">Target Date</Label>
                  <Input
                    id="target-date"
                    type="date"
                    value={optimizationParams.target_date}
                    onChange={(e) => setOptimizationParams(prev => ({
                      ...prev,
                      target_date: e.target.value
                    }))}
                  />
                </div>

              </div>
              {/* Hidden by default, can be re-enabled if manual override is needed */}
              {/* 
                <div>
                  <Label htmlFor="service-hours">Required Service Hours</Label>
                  <Input
                    id="service-hours"
                    type="number"
                    value={optimizationParams.required_service_hours}
                    onChange={(e) => {
                      const value = parseNumericInput(e.target.value, 0);
                      setOptimizationParams(prev => ({
                        ...prev,
                        required_service_hours: value
                      }));
                    }}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    Helper: hours are converted to trains using a configurable average (default {DEFAULT_HOURS_PER_TRAIN} hrs/train).
                    The backend returns the exact conversion and granted trains.
                  </p>
                </div>
                */}


              <div className="text-sm text-muted-foreground bg-secondary/20 p-3 rounded-md">
                <p>
                  <strong>Note:</strong> Fleet requirements are now automatically calculated based on the timetable for the selected date.
                </p>
              </div>
              <Button
                onClick={() => runOptimizationMutation.mutate()}
                disabled={runOptimizationMutation.isPending}
                className="w-full"
                size="lg"
              >
                {runOptimizationMutation.isPending ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Running Optimization...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Run AI/ML Optimization
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {optimizationDiagnostics && (() => {
            const d = optimizationDiagnostics;
            const requested = Number(d.requested_train_count ?? 0);
            const granted = Number(d.granted_train_count ?? 0);
            const eligible = Number(d.eligible_train_count ?? 0);

            if (!requested || granted >= requested) {
              return null;
            }

            const requestedHours =
              typeof d.requested_service_hours === "number"
                ? d.requested_service_hours.toFixed(1)
                : d.requested_service_hours;

            return (
              <Card className="mt-4 border-blue-300 bg-blue-50">
                <CardContent className="py-3 text-sm text-blue-900 space-y-1">
                  <div>
                    Requested{" "}
                    <span className="font-semibold">
                      {requestedHours ?? "—"}
                    </span>{" "}
                    service hours → approximately{" "}
                    <span className="font-semibold">{requested}</span> trains
                    (avg{" "}
                    <span className="font-semibold">
                      {d.avg_hours_per_train ?? DEFAULT_HOURS_PER_TRAIN}
                    </span>{" "}
                    hrs/train).
                  </div>
                  <div>
                    Only{" "}
                    <span className="font-semibold">{eligible}</span> trains
                    were eligible; optimization granted{" "}
                    <span className="font-semibold">{granted}</span> trains.
                  </div>
                </CardContent>
              </Card>
            );
          })()}

          {optimizationNote && (
            <Card className="mt-4 border-yellow-300 bg-yellow-50">
              <CardContent className="py-3 text-sm text-yellow-900">
                {optimizationNote}
              </CardContent>
            </Card>
          )}

          {/* Latest Results */}
          {!!safeLatestDecisions.length && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Latest Optimization Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {safeLatestDecisions.map((decision: any, index: number) => {
                    const confidence =
                      typeof decision?.confidence_score === "number"
                        ? Math.round(decision.confidence_score * 100)
                        : null;
                    const badgeVariant =
                      decision?.decision === "INDUCT"
                        ? "success"
                        : decision?.decision === "STANDBY"
                          ? "secondary"
                          : "destructive";

                    return (
                      <div key={index} className="flex items-center justify-between p-3 border border-border rounded-lg">
                        <div className="flex items-center gap-3">
                          <Train className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium">{decision.trainset_id || `Train ${index + 1}`}</span>
                          <Badge variant={badgeVariant as any}>
                            {decision.decision ?? "—"}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">
                            Confidence: {confidence !== null ? `${confidence}%` : "N/A"}
                          </span>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => explainMutation.mutate({
                              trainsetId: decision.trainset_id,
                              decision: decision.decision
                            })}
                          >
                            Explain
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* What-If Simulation */}
        <TabsContent value="simulation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                What-If Simulation Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="exclude">Exclude Trainsets (comma-separated)</Label>
                  <Input
                    id="exclude"
                    placeholder="RK-001, RK-002"
                    value={simulationParams.exclude_trainsets}
                    onChange={(e) => setSimulationParams(prev => ({
                      ...prev,
                      exclude_trainsets: e.target.value
                    }))}
                  />
                </div>
                <div>
                  <Label htmlFor="force">Force Induct (comma-separated)</Label>
                  <Input
                    id="force"
                    placeholder="RK-003, RK-004"
                    value={simulationParams.force_induct}
                    onChange={(e) => setSimulationParams(prev => ({
                      ...prev,
                      force_induct: e.target.value
                    }))}
                  />
                </div>
                <div>
                  <Label htmlFor="service-count">Required Service Count</Label>
                  <Input
                    id="service-count"
                    type="number"
                    value={simulationParams.required_service_count}
                    onChange={(e) => setSimulationParams(prev => ({
                      ...prev,
                      required_service_count: Math.max(0, parseNumericInput(e.target.value, 14))
                    }))}
                  />
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-medium">Optimization Weights</h4>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div>
                    <Label htmlFor="readiness">Readiness</Label>
                    <Input
                      id="readiness"
                      type="number"
                      step="0.01"
                      value={simulationParams.w_readiness}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_readiness: parseFloat(e.target.value) || 0.35
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="reliability">Reliability</Label>
                    <Input
                      id="reliability"
                      type="number"
                      step="0.01"
                      value={simulationParams.w_reliability}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_reliability: parseFloat(e.target.value) || 0.30
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="branding">Branding</Label>
                    <Input
                      id="branding"
                      type="number"
                      step="0.01"
                      value={simulationParams.w_branding}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_branding: parseFloat(e.target.value) || 0.20
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="shunt">Shunt</Label>
                    <Input
                      id="shunt"
                      type="number"
                      step="0.01"
                      value={simulationParams.w_shunt}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_shunt: parseFloat(e.target.value) || 0.10
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="mileage">Mileage Balance</Label>
                    <Input
                      id="mileage"
                      type="number"
                      step="0.01"
                      value={simulationParams.w_mileage_balance}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_mileage_balance: parseFloat(e.target.value) || 0.05
                      }))}
                    />
                  </div>
                </div>
              </div>

              <Button
                onClick={() => runSimulationMutation.mutate(simulationParams)}
                disabled={runSimulationMutation.isPending}
                className="w-full"
                size="lg"
              >
                {runSimulationMutation.isPending ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Running Simulation...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Run What-If Simulation
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {simulationResults && (() => {
            const simulationData = simulationResults;
            const baselineResult = getBaselineResult(simulationData);
            const scenarioResult = getScenarioResult(simulationData);

            return (
              <Card>
                <CardHeader>
                  <CardTitle>Simulation Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {baselineResult && scenarioResult && (
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="border rounded-lg p-4">
                          <h3 className="font-semibold mb-2">Baseline</h3>
                          <div className="space-y-2 text-sm">
                            <div>Inducted: {baselineResult.kpis?.num_inducted_trains || 0}</div>
                            <div>Shunting Time: {baselineResult.kpis?.total_shunting_time || 0} min</div>
                            <div>Efficiency: {baselineResult.kpis?.efficiency_improvement || 0}%</div>
                          </div>
                        </div>
                        <div className="border rounded-lg p-4">
                          <h3 className="font-semibold mb-2">Scenario</h3>
                          <div className="space-y-2 text-sm">
                            <div>Inducted: {scenarioResult.kpis?.num_inducted_trains || 0}</div>
                            <div>Shunting Time: {scenarioResult.kpis?.total_shunting_time || 0} min</div>
                            <div>Efficiency: {scenarioResult.kpis?.efficiency_improvement || 0}%</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {simulationData.deltas && (
                      <div className="border rounded-lg p-4">
                        <h3 className="font-semibold mb-2">Changes (Scenario - Baseline)</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center">
                            <div className={`text-2xl font-bold ${simulationData.deltas.num_inducted_trains >= 0 ? 'text-success' : 'text-destructive'}`}>
                              {simulationData.deltas.num_inducted_trains >= 0 ? '+' : ''}{simulationData.deltas.num_inducted_trains || 0}
                            </div>
                            <div className="text-sm text-muted-foreground">Inducted</div>
                          </div>
                          <div className="text-center">
                            <div className={`text-2xl font-bold ${simulationData.deltas.total_shunting_time <= 0 ? 'text-success' : 'text-destructive'}`}>
                              {simulationData.deltas.total_shunting_time >= 0 ? '+' : ''}{simulationData.deltas.total_shunting_time || 0}
                            </div>
                            <div className="text-sm text-muted-foreground">Shunting Time (min)</div>
                          </div>
                          <div className="text-center">
                            <div className={`text-2xl font-bold ${simulationData.deltas.efficiency_improvement >= 0 ? 'text-success' : 'text-destructive'}`}>
                              {simulationData.deltas.efficiency_improvement >= 0 ? '+' : ''}{(simulationData.deltas.efficiency_improvement || 0).toFixed(2)}%
                            </div>
                            <div className="text-sm text-muted-foreground">Efficiency</div>
                          </div>
                          <div className="text-center">
                            <div className={`text-2xl font-bold ${simulationData.deltas.num_unassigned <= 0 ? 'text-success' : 'text-destructive'}`}>
                              {simulationData.deltas.num_unassigned >= 0 ? '+' : ''}{simulationData.deltas.num_unassigned || 0}
                            </div>
                            <div className="text-sm text-muted-foreground">Unassigned</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {simulationData.explain_log && simulationData.explain_log.length > 0 && (
                      <div className="border rounded-lg p-4">
                        <h3 className="font-semibold mb-2">Explanation</h3>
                        <ul className="list-disc list-inside space-y-1 text-sm">
                          {simulationData.explain_log.map((log: string, idx: number) => (
                            <li key={idx}>{log}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })()}
        </TabsContent>

        {/* Stabling Geometry */}
        <TabsContent value="stabling" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Stabling Geometry Optimization
              </CardTitle>
            </CardHeader>
            <CardContent>
              {(() => {
                const detail = (stablingError as any)?.response?.data?.detail || (stablingError as any)?.response?.data;
                const noDecisions = detail?.code === 'no_induction_decisions';

                if (noDecisions) {
                  return (
                    <div className="text-center py-8">
                      <Settings className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        No optimization decisions found. Please run AI/ML Optimization first.
                      </p>
                    </div>
                  );
                }

                if (!stablingGeometry) {
                  return (
                    <div className="text-center py-8">
                      <Settings className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">No stabling geometry data available</p>
                    </div>
                  );
                }

                const fleetSummary = stablingGeometry.fleet_summary;
                const depotAllocation = stablingGeometry.depot_allocation || [];
                const bayLayout = stablingGeometry.bay_layout || {};
                const kpis = stablingGeometry.optimization_kpis;
                const warnings = stablingGeometry.warnings || [];

                return (
                  <div className="space-y-6">
                    {/* Fleet Summary */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Fleet Summary</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                          <div className="text-center">
                            <div className="text-xl font-bold text-foreground">{fleetSummary?.total_trainsets || 0}</div>
                            <div className="text-xs text-muted-foreground">Total Trainsets</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xl font-bold text-success">{fleetSummary?.actual_induct_count || 0}</div>
                            <div className="text-xs text-muted-foreground">Service Trains</div>
                            <div className="text-xs text-muted-foreground">(Req: {fleetSummary?.required_service_trains || 0})</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xl font-bold text-warning">{fleetSummary?.actual_standby_count || 0}</div>
                            <div className="text-xs text-muted-foreground">Standby</div>
                            <div className="text-xs text-muted-foreground">(Buffer: {fleetSummary?.standby_buffer || 0})</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xl font-bold text-destructive">{fleetSummary?.maintenance_count || 0}</div>
                            <div className="text-xs text-muted-foreground">Maintenance</div>
                          </div>
                          <div className="text-center">
                            <div className={`text-xl font-bold ${fleetSummary?.service_shortfall > 0 ? 'text-destructive' : 'text-success'}`}>
                              {fleetSummary?.service_shortfall || 0}
                            </div>
                            <div className="text-xs text-muted-foreground">Service Shortfall</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Warnings */}
                    {warnings.length > 0 && (
                      <Card className="border-yellow-300 bg-yellow-50">
                        <CardHeader>
                          <CardTitle className="text-lg flex items-center gap-2">
                            <AlertTriangle className="h-5 w-5 text-yellow-600" />
                            Operational Warnings
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ul className="list-disc list-inside space-y-1 text-sm text-yellow-900">
                            {warnings.map((warning: string, idx: number) => (
                              <li key={idx}>{warning}</li>
                            ))}
                          </ul>
                        </CardContent>
                      </Card>
                    )}

                    {/* Depot Allocation */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Depot Allocation</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {depotAllocation.map((depot: any) => (
                            <div key={depot.depot_name} className="border rounded-lg p-4">
                              <div className="flex items-center justify-between mb-3">
                                <h3 className="font-semibold text-lg">{depot.depot_name}</h3>
                                {depot.capacity_warning && (
                                  <Badge variant="destructive">Capacity Warning</Badge>
                                )}
                              </div>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div>
                                  <div className="text-sm text-muted-foreground">Service</div>
                                  <div className="text-lg font-bold text-success">{depot.service_trains}</div>
                                  <div className="text-xs text-muted-foreground">Capacity: {depot.service_bay_capacity}</div>
                                </div>
                                <div>
                                  <div className="text-sm text-muted-foreground">Standby</div>
                                  <div className="text-lg font-bold text-warning">{depot.standby_trains}</div>
                                </div>
                                <div>
                                  <div className="text-sm text-muted-foreground">Maintenance</div>
                                  <div className="text-lg font-bold text-destructive">{depot.maintenance_trains}</div>
                                  <div className="text-xs text-muted-foreground">Capacity: {depot.maintenance_bay_capacity}</div>
                                </div>
                                <div>
                                  <div className="text-sm text-muted-foreground">Total</div>
                                  <div className="text-lg font-bold">{depot.total_trains}</div>
                                  <div className="text-xs text-muted-foreground">Total Capacity: {depot.total_bay_capacity}</div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>

                    {/* Bay Layout Grid */}
                    {Object.keys(bayLayout).length > 0 && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Bay Layout</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-6">
                            {Object.entries(bayLayout).map(([depotName, bays]: [string, any]) => (
                              <div key={depotName}>
                                <h3 className="font-semibold mb-3">{depotName}</h3>
                                <div className="grid grid-cols-4 md:grid-cols-8 lg:grid-cols-12 gap-2">
                                  {bays.map((bay: any) => {
                                    const roleColors: Record<string, string> = {
                                      SERVICE: "bg-green-100 border-green-300 text-green-900",
                                      STANDBY: "bg-yellow-100 border-yellow-300 text-yellow-900",
                                      MAINTENANCE: "bg-red-100 border-red-300 text-red-900",
                                      EMPTY: "bg-gray-100 border-gray-300 text-gray-500"
                                    };
                                    return (
                                      <div
                                        key={bay.bay_id}
                                        className={`border rounded p-2 text-center text-xs ${roleColors[bay.role] || roleColors.EMPTY}`}
                                        title={bay.notes || `${bay.role} - ${bay.trainset_id || 'Empty'}`}
                                      >
                                        <div className="font-bold">Bay {bay.bay_id}</div>
                                        <div className="text-xs">{bay.trainset_id || 'Empty'}</div>
                                        <div className="text-xs opacity-75">{bay.role}</div>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Optimization KPIs */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Optimization Performance</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center">
                            <div className="text-xl font-bold text-foreground">{kpis?.optimized_positions || 0}</div>
                            <div className="text-xs text-muted-foreground">Optimized Positions</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xl font-bold text-primary">{kpis?.total_shunting_time_min || 0} min</div>
                            <div className="text-xs text-muted-foreground">Shunting Time</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xl font-bold text-success">{kpis?.total_turnout_time_min || 0} min</div>
                            <div className="text-xs text-muted-foreground">Turnout Time</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xl font-bold text-primary">{kpis?.efficiency_improvement_pct?.toFixed(1) || 0}%</div>
                            <div className="text-xs text-muted-foreground">Efficiency Improvement</div>
                          </div>
                        </div>
                        {kpis?.energy_savings_kwh && (
                          <div className="mt-4 text-center text-sm text-muted-foreground">
                            Estimated Energy Savings: <span className="font-semibold">{kpis.energy_savings_kwh.toFixed(1)} kWh</span>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                );
              })()}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Shunting Schedule */}
        <TabsContent value="shunting" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Shunting Schedule
              </CardTitle>
            </CardHeader>
            <CardContent>
              {(() => {
                const detail = (shuntingError as any)?.response?.data?.detail || (shuntingError as any)?.response?.data;
                const noDecisions = detail?.code === 'no_induction_decisions';

                if (noDecisions) {
                  return (
                    <div className="text-center py-8">
                      <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        No optimization decisions found. Please run AI/ML Optimization first to generate a shunting schedule.
                      </p>
                    </div>
                  );
                }

                if (!shuntingSchedule) {
                  return (
                    <div className="text-center py-8">
                      <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">No shunting schedule available</p>
                    </div>
                  );
                }

                const schedule = shuntingSchedule.shunting_schedule || [];
                const scheduleByDepot = shuntingSchedule.schedule_by_depot || {};
                const depotSummaries = shuntingSchedule.depot_summaries || {};
                const operationalWindow = shuntingSchedule.operational_window;

                return (
                  <div className="space-y-6">
                    {/* Summary Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <Card>
                        <CardContent className="pt-6">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-foreground">
                              {shuntingSchedule.total_operations || 0}
                            </div>
                            <div className="text-sm text-muted-foreground">Total Operations</div>
                          </div>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent className="pt-6">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-primary">
                              {shuntingSchedule.estimated_total_time || 0} min
                            </div>
                            <div className="text-sm text-muted-foreground">Estimated Time</div>
                            {operationalWindow && (
                              <div className="text-xs text-muted-foreground mt-1">
                                Window: {operationalWindow.start_time} - {operationalWindow.end_time}
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent className="pt-6">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-destructive">
                              {shuntingSchedule.crew_requirements?.high_complexity || 0}
                            </div>
                            <div className="text-sm text-muted-foreground">High Complexity</div>
                          </div>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent className="pt-6">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-warning">
                              {shuntingSchedule.crew_requirements?.medium_complexity || 0}
                            </div>
                            <div className="text-sm text-muted-foreground">Medium Complexity</div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {/* Depot Summaries */}
                    {Object.keys(depotSummaries).length > 0 && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Depot Summaries</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {Object.entries(depotSummaries).map(([depot, summary]: [string, any]) => (
                              <div key={depot} className="border rounded-lg p-4">
                                <h3 className="font-semibold mb-2">{depot}</h3>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                  <div>
                                    <span className="text-muted-foreground">Operations:</span>
                                    <span className="ml-2 font-semibold">{summary.total_operations}</span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">Time:</span>
                                    <span className="ml-2 font-semibold">{summary.estimated_time_min} min</span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">High:</span>
                                    <span className="ml-2 font-semibold text-destructive">{summary.high_complexity}</span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">Medium:</span>
                                    <span className="ml-2 font-semibold text-warning">{summary.medium_complexity}</span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Ordered Schedule by Depot */}
                    {Object.keys(scheduleByDepot).length > 0 && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Ordered Shunting Schedule</CardTitle>
                          <p className="text-sm text-muted-foreground">
                            Operations ordered by priority (Service → Maintenance → Standby) and time
                          </p>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-6">
                            {Object.entries(scheduleByDepot).map(([depot, ops]: [string, any]) => (
                              <div key={depot}>
                                <h3 className="font-semibold text-lg mb-3">{depot}</h3>
                                <div className="space-y-2">
                                  {ops.map((op: any) => {
                                    const complexityColors: Record<string, string> = {
                                      HIGH: "border-red-300 bg-red-50",
                                      MEDIUM: "border-yellow-300 bg-yellow-50",
                                      LOW: "border-green-300 bg-green-50"
                                    };
                                    const decisionColors: Record<string, string> = {
                                      INDUCT: "text-green-600",
                                      MAINTENANCE: "text-red-600",
                                      STANDBY: "text-yellow-600"
                                    };
                                    return (
                                      <div
                                        key={op.sequence}
                                        className={`border rounded-lg p-3 ${complexityColors[op.complexity] || complexityColors.LOW}`}
                                      >
                                        <div className="flex items-center justify-between">
                                          <div className="flex items-center gap-3">
                                            <div className="font-bold text-lg w-8">#{op.sequence}</div>
                                            <div>
                                              <div className="font-semibold">{op.trainset_id}</div>
                                              <div className="text-sm text-muted-foreground">{op.operation}</div>
                                            </div>
                                          </div>
                                          <div className="text-right">
                                            <div className="font-semibold">{op.estimated_time} min</div>
                                            <div className={`text-xs font-medium ${decisionColors[op.decision] || ''}`}>
                                              {op.decision}
                                            </div>
                                            <div className="text-xs text-muted-foreground">{op.complexity}</div>
                                          </div>
                                        </div>
                                        {op.crew_required && (
                                          <div className="mt-2 text-xs text-muted-foreground">
                                            Crew: {op.crew_required}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Operational Window Info */}
                    {operationalWindow && (
                      <Card className="border-blue-300 bg-blue-50">
                        <CardContent className="pt-6">
                          <div className="text-sm">
                            <div className="font-semibold mb-2">Operational Window</div>
                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <span className="text-muted-foreground">Window:</span>
                                <span className="ml-2 font-semibold">
                                  {operationalWindow.start_time} - {operationalWindow.end_time}
                                </span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Buffer:</span>
                                <span className={`ml-2 font-semibold ${operationalWindow.buffer_minutes < 30 ? 'text-destructive' : 'text-success'}`}>
                                  {operationalWindow.buffer_minutes} minutes
                                </span>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                );
              })()}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Simulation Results Modal */}
      {
        showSimulationResults && simulationResults && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <Card className="max-w-6xl w-full max-h-[90vh] overflow-y-auto">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xl">What-If Simulation Results</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowSimulationResults(false)}
                  >
                    ×
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold text-lg mb-3">Simulation Parameters</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Excluded Trainsets:</span>
                        <span>{simulationParamsUsed?.exclude_trainsets || 'None'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Forced Inductions:</span>
                        <span>{simulationParamsUsed?.force_induct || 'None'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Required Service Count:</span>
                        <span>{simulationParamsUsed?.required_service_count || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Weights:</span>
                        <span className="text-xs">
                          {simulationParamsUsed
                            ? <>R:{simulationParamsUsed.w_readiness?.toFixed(2)} R:{simulationParamsUsed.w_reliability?.toFixed(2)} B:{simulationParamsUsed.w_branding?.toFixed(2)} S:{simulationParamsUsed.w_shunt?.toFixed(2)} M:{simulationParamsUsed.w_mileage_balance?.toFixed(2)}</>
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-semibold text-lg mb-3">Simulation Summary</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Decisions:</span>
                        <span>{getScenarioResult(simulationResults)?.decisions?.length || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Inducted:</span>
                        <span className="text-green-600">
                          {getScenarioResult(simulationResults)?.kpis?.num_inducted_trains || 0}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Standby:</span>
                        <span className="text-yellow-600">
                          {getScenarioResult(simulationResults)?.kpis?.num_standby_trains || 0}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Maintenance:</span>
                        <span className="text-red-600">
                          {getScenarioResult(simulationResults)?.kpis?.num_maintenance_trains || 0}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <DetailedResults
                  results={getScenarioResult(simulationResults)?.decisions || []}
                  className="w-full"
                />

                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setShowSimulationResults(false)}>
                    Close
                  </Button>
                  <Button onClick={() => window.print()}>
                    Print Results
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )
      }

      {/* Explanation Modal */}
      {
        showExplanation && explanationData && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <Card className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xl">
                    AI Decision Explanation - {explanationData.trainset_id}
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
                            <span className={`text-xs px-2 py-1 rounded ${feature.impact === 'positive'
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
        )
      }
    </div >
  );
};

export default Optimization;

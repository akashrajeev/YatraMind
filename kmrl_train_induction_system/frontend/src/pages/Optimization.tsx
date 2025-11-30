import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { optimizationApi } from "@/services/api";
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
  TrendingUp,
  Download,
  Upload,
  RefreshCw,
  Train,
  Eye
} from "lucide-react";
import { useState } from "react";

const Optimization = () => {
  const [optimizationParams, setOptimizationParams] = useState({
    constraints: {}
  });

  const [simulationParams, setSimulationParams] = useState({
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
  const [showSimulationResults, setShowSimulationResults] = useState(false);

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

  const { data: stablingGeometry } = useQuery({
    queryKey: ['optimization-stabling'],
    queryFn: () => optimizationApi.getStablingGeometry().then(res => res.data),
  });

  const { data: shuntingSchedule } = useQuery({
    queryKey: ['optimization-shunting'],
    queryFn: () => optimizationApi.getShuntingSchedule().then(res => res.data),
  });

  // Mutations
  const runOptimizationMutation = useMutation({
    mutationFn: optimizationApi.runOptimization,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['optimization-latest'] });
      queryClient.invalidateQueries({ queryKey: ['optimization-stabling'] });
      queryClient.invalidateQueries({ queryKey: ['optimization-shunting'] });
    },
  });

  const runSimulationMutation = useMutation({
    mutationFn: optimizationApi.simulate,
    onSuccess: (data) => {
      setSimulationResults(data.data);
      setShowSimulationResults(true);
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
          <Button variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="industrial">
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
              <div className="text-sm text-muted-foreground mb-4">
                Click the button below to run the AI/ML optimization engine. The system will automatically select the best trainsets for induction based on current health, maintenance schedules, and operational constraints.
              </div>
              <Button
                onClick={() => runOptimizationMutation.mutate(optimizationParams)}
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

          {/* Latest Results */}
          {latestOptimization && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Latest Optimization Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {latestOptimization.map((decision: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-3 border border-border rounded-lg">
                      <div className="flex items-center gap-3">
                        <Train className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium">{decision.trainset_id}</span>
                        <Badge variant={decision.decision === "INDUCT" ? "success" : "secondary"}>
                          {decision.decision}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          Confidence: {Math.round(decision.confidence_score * 100)}%
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
                  ))}
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
                      required_service_count: parseInt(e.target.value) || 14
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

          {/* Simulation Results */}
          {runSimulationMutation.data && (
            <Card>
              <CardHeader>
                <CardTitle>Simulation Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-foreground">
                        {runSimulationMutation.data.results?.total_score || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Total Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-success">
                        {runSimulationMutation.data.results?.inducted_count || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Inducted</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-warning">
                        {runSimulationMutation.data.results?.standby_count || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Standby</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-destructive">
                        {runSimulationMutation.data.results?.maintenance_count || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Maintenance</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
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
              {stablingGeometry ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-foreground">
                        {stablingGeometry.optimized_layout?.length || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Optimized Positions</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-success">
                        {stablingGeometry.total_shunting_time || 0} min
                      </div>
                      <div className="text-sm text-muted-foreground">Total Shunting Time</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">
                        {stablingGeometry.efficiency_improvement || 0}%
                      </div>
                      <div className="text-sm text-muted-foreground">Efficiency Improvement</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Settings className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">No stabling geometry data available</p>
                </div>
              )}
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
              {shuntingSchedule ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-foreground">
                        {shuntingSchedule.total_operations || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Total Operations</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">
                        {shuntingSchedule.estimated_total_time || 0} min
                      </div>
                      <div className="text-sm text-muted-foreground">Estimated Time</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-destructive">
                        {shuntingSchedule.crew_requirements?.high_complexity || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">High Complexity</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-warning">
                        {shuntingSchedule.crew_requirements?.medium_complexity || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Medium Complexity</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">No shunting schedule available</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Simulation Results Modal */}
      {showSimulationResults && simulationResults && (
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
                      <span>{simulationResults.scenario?.excluded_trainsets?.join(', ') || 'None'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Forced Inductions:</span>
                      <span>{simulationResults.scenario?.forced_inductions?.join(', ') || 'None'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Required Service Count:</span>
                      <span>{simulationResults.scenario?.required_service_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Weights:</span>
                      <span className="text-xs">
                        R:{simulationResults.scenario?.weights?.readiness}
                        R:{simulationResults.scenario?.weights?.reliability}
                        B:{simulationResults.scenario?.weights?.branding}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold text-lg mb-3">Simulation Summary</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Total Decisions:</span>
                      <span>{simulationResults.results?.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Inducted:</span>
                      <span className="text-green-600">
                        {simulationResults.results?.filter((r: any) => r.decision === 'INDUCT').length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Standby:</span>
                      <span className="text-yellow-600">
                        {simulationResults.results?.filter((r: any) => r.decision === 'STANDBY').length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Maintenance:</span>
                      <span className="text-red-600">
                        {simulationResults.results?.filter((r: any) => r.decision === 'MAINTENANCE').length || 0}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold text-lg mb-3">Detailed Results</h3>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {simulationResults.results?.map((result: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-xs font-bold">
                          {index + 1}
                        </div>
                        <span className="font-medium">{result.trainset_id}</span>
                        <Badge variant={result.decision === 'INDUCT' ? 'success' : 'secondary'}>
                          {result.decision}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <span>Score: {Math.round((result.score || 0) * 100)}%</span>
                        <span>Confidence: {Math.round((result.confidence_score || 0) * 100)}%</span>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => explainMutation.mutate({
                            trainsetId: result.trainset_id,
                            decision: result.decision
                          })}
                        >
                          <Eye className="h-3 w-3 mr-1" />
                          Explain
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

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
      )}

      {/* Explanation Modal */}
      {showExplanation && explanationData && (
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
      )}
    </div>
  );
};

export default Optimization;

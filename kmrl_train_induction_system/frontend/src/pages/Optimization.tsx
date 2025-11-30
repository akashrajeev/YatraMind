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
  Eye,
  X
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
                    <Label htmlFor="w-readiness">Readiness</Label>
                    <Input
                      id="w-readiness"
                      type="number"
                      step="0.05"
                      value={simulationParams.w_readiness}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_readiness: parseFloat(e.target.value)
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="w-reliability">Reliability</Label>
                    <Input
                      id="w-reliability"
                      type="number"
                      step="0.05"
                      value={simulationParams.w_reliability}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_reliability: parseFloat(e.target.value)
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="w-branding">Branding</Label>
                    <Input
                      id="w-branding"
                      type="number"
                      step="0.05"
                      value={simulationParams.w_branding}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_branding: parseFloat(e.target.value)
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="w-shunt">Shunting</Label>
                    <Input
                      id="w-shunt"
                      type="number"
                      step="0.05"
                      value={simulationParams.w_shunt}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_shunt: parseFloat(e.target.value)
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="w-mileage">Mileage</Label>
                    <Input
                      id="w-mileage"
                      type="number"
                      step="0.05"
                      value={simulationParams.w_mileage_balance}
                      onChange={(e) => setSimulationParams(prev => ({
                        ...prev,
                        w_mileage_balance: parseFloat(e.target.value)
                      }))}
                    />
                  </div>
                </div>
              </div>

              <Button
                onClick={() => runSimulationMutation.mutate(simulationParams)}
                disabled={runSimulationMutation.isPending}
                className="w-full"
              >
                {runSimulationMutation.isPending ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Running Simulation...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Run Simulation
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Simulation Results */}
          {showSimulationResults && simulationResults && (
            <Card>
              <CardHeader>
                <CardTitle>Simulation Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-secondary/10 rounded-lg">
                      <div className="text-sm text-muted-foreground">Total Inducted</div>
                      <div className="text-2xl font-bold">{simulationResults.filter((r: any) => r.decision === "INDUCT").length}</div>
                    </div>
                    <div className="p-4 bg-secondary/10 rounded-lg">
                      <div className="text-sm text-muted-foreground">Average Confidence</div>
                      <div className="text-2xl font-bold">
                        {Math.round(simulationResults.reduce((acc: number, curr: any) => acc + curr.confidence_score, 0) / simulationResults.length * 100)}%
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {simulationResults.map((result: any, index: number) => (
                      <div key={index} className="flex items-center justify-between p-3 border border-border rounded-lg">
                        <div className="flex items-center gap-3">
                          <Train className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium">{result.trainset_id}</span>
                          <Badge variant={result.decision === "INDUCT" ? "success" : "secondary"}>
                            {result.decision}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Score: {result.score.toFixed(2)}
                        </div>
                      </div>
                    ))}
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
                <Target className="h-5 w-5" />
                Stabling Optimization
              </CardTitle>
            </CardHeader>
            <CardContent>
              {stablingGeometry ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {stablingGeometry.bays.map((bay: any, index: number) => (
                      <div key={index} className="p-4 border border-border rounded-lg">
                        <div className="font-medium mb-2">Bay {bay.id}</div>
                        <div className="flex gap-2">
                          {bay.slots.map((slot: any, sIndex: number) => (
                            <div
                              key={sIndex}
                              className={`h-12 w-full rounded flex items-center justify-center text-xs font-medium ${slot.occupied ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
                                }`}
                            >
                              {slot.trainset_id || "Empty"}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No stabling data available
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
                  {shuntingSchedule.moves.map((move: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-3 border border-border rounded-lg">
                      <div className="flex items-center gap-4">
                        <div className="font-mono text-sm">{move.time}</div>
                        <div className="font-medium">{move.trainset_id}</div>
                        <div className="text-sm text-muted-foreground">
                          {move.from_track} → {move.to_track}
                        </div>
                      </div>
                      <Badge variant="outline">{move.status}</Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No shunting schedule available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Explanation Modal */}
      {
        showExplanation && explanationData && (
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
                    <p className="text-2xl font-bold">{explanationData.trainset_details?.trainset_id || 'Unknown'}</p>
                  </div>
                  <div className="text-right">
                    <h4 className="text-sm font-medium text-muted-foreground">Composite Score</h4>
                    <p className="text-2xl font-bold text-primary">
                      {Math.round(explanationData.score * 100)}%
                    </p>
                  </div>
                </div>

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
        )
      }
    </div >
  );
};

export default Optimization;

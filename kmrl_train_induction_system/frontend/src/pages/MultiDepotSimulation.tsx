/**
 * Multi-Depot Simulation Control Panel
 * Allows operators to simulate 1..N depots with configurable fleet sizes
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
// Select component - using button dropdown instead
import { useMutation, useQuery } from "@tanstack/react-query";
import { multiDepotSimulationApi } from "@/services/api";
import { 
  Play, 
  Download, 
  AlertTriangle, 
  CheckCircle, 
  Train, 
  Building2,
  ArrowRight,
  Loader2,
  Settings,
  TrendingUp
} from "lucide-react";
import { toast } from "sonner";

interface DepotConfig {
  name: string;
  location_type: string;
  service_bays: number;
  maintenance_bays: number;
  standby_bays: number;
}

interface SimulationRequest {
  depots: DepotConfig[];
  fleet: number;
  seed?: number;
  service_requirement?: number;
  ai_mode: boolean;
  sim_days: number;
}

interface SimulationResult {
  run_id: string;
  used_ai: boolean;
  per_depot: Record<string, any>;
  inter_depot_transfers: any[];
  global_summary: {
    service_trains?: number;
    required_service?: number;
    stabled_service?: number;
    service_shortfall?: number;
    shunting_time?: number;
    turnout_time?: number;
    total_capacity?: number;
    fleet?: number;
    transfers_recommended?: number;
    [key: string]: any;
  };
  warnings: string[];
  infrastructure_recommendations: any[];
}

const FLEET_PRESETS = [25, 40, 60, 100];
const DEPOT_PRESETS = [
  { name: "Muttom", location_type: "FULL_DEPOT", service_bays: 6, maintenance_bays: 4, standby_bays: 2 },
  { name: "Kakkanad", location_type: "FULL_DEPOT", service_bays: 6, maintenance_bays: 3, standby_bays: 2 },
  { name: "Aluva Terminal", location_type: "TERMINAL_YARD", service_bays: 0, maintenance_bays: 0, standby_bays: 6 },
  { name: "Petta Terminal", location_type: "TERMINAL_YARD", service_bays: 0, maintenance_bays: 0, standby_bays: 6 },
];

export default function MultiDepotSimulation() {
  const [depots, setDepots] = useState<DepotConfig[]>([DEPOT_PRESETS[0]]);
  const [fleetSize, setFleetSize] = useState<number>(40);
  const [serviceRequirement, setServiceRequirement] = useState<number | undefined>(undefined);
  const [seed, setSeed] = useState<number | undefined>(undefined);
  const [aiMode, setAiMode] = useState<boolean>(true);
  const [simDays, setSimDays] = useState<number>(1);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);

  // Fetch depot presets (optional - using hardcoded presets for now)
  // const { data: depotPresets } = useQuery({
  //   queryKey: ['depot-presets'],
  //   queryFn: () => multiDepotSimulationApi.getDepotPresets().then(res => res.data),
  // });

  // Normalize simulation response to ensure all required keys exist
  const normalizeSimulationResponse = (data: any): SimulationResult => {
    const requiredKeys = [
      'service_trains', 'required_service', 'stabled_service', 'service_shortfall',
      'shunting_time', 'turnout_time', 'total_capacity', 'fleet', 'transfers_recommended'
    ];
    const missingKeys: string[] = [];
    
    if (!data.global_summary) {
      data.global_summary = {};
    }
    
    const defaults: Record<string, number> = {
      service_trains: 0,
      required_service: 13,
      stabled_service: 0,
      service_shortfall: 0,
      shunting_time: 0,
      turnout_time: 0,
      total_capacity: 0,
      fleet: 0,
      transfers_recommended: 0
    };
    
    requiredKeys.forEach(key => {
      if (!(key in data.global_summary)) {
        data.global_summary[key] = defaults[key];
        missingKeys.push(key);
        console.warn(`Missing response key: ${key}; using fallback ${defaults[key]}`);
      }
    });
    
    if (missingKeys.length > 0 && !data.warnings) {
      data.warnings = [];
    }
    if (missingKeys.length > 0) {
      data.warnings.push(`Missing response keys: ${missingKeys.join(', ')}; using fallback values`);
    }
    
    return data as SimulationResult;
  };

  // Simulation mutation
  const simulateMutation = useMutation({
    mutationFn: async (request: SimulationRequest) => {
      // Pre-run validation: check capacity
      const totalCapacity = depots.reduce((sum, d) => 
        sum + d.service_bays + d.maintenance_bays + d.standby_bays, 0
      );
      const terminalCapacity = 12; // Aluva + Petta default
      const totalAvailable = totalCapacity + terminalCapacity;
      
      if (request.fleet > totalAvailable) {
        const overflow = request.fleet - totalAvailable;
        // Show modal (will be handled by UI component)
        const proceed = window.confirm(
          `Fleet (${request.fleet}) exceeds total capacity (${totalAvailable}) by ${overflow} trains.\n\n` +
          `Options:\n` +
          `1. Add Terminal Presets (Aluva/Petta) - Click OK then use "Add Terminal" button\n` +
          `2. Reduce Fleet - Cancel and adjust fleet size\n` +
          `3. Proceed Anyway - Click OK to continue\n\n` +
          `Click OK to proceed anyway, or Cancel to adjust.`
        );
        if (!proceed) {
          throw new Error('Simulation cancelled by user');
        }
      }
      
      const response = await multiDepotSimulationApi.simulate(request);
      return normalizeSimulationResponse(response.data);
    },
    onSuccess: (data) => {
      setSimulationResult(data);
      toast.success("Simulation completed", {
        description: `Run ID: ${data.run_id}`,
      });
    },
    onError: (error: any) => {
      toast.error("Simulation failed", {
        description: error.message,
      });
    },
  });

  const handleAddDepot = (presetName?: string) => {
    const preset = presetName 
      ? DEPOT_PRESETS.find(d => d.name === presetName)
      : DEPOT_PRESETS[0];
    
    if (preset) {
      setDepots([...depots, { ...preset }]);
    }
  };

  const handleAddTerminals = () => {
    const aluva = DEPOT_PRESETS.find(d => d.name === "Aluva Terminal");
    const petta = DEPOT_PRESETS.find(d => d.name === "Petta Terminal");
    const newDepots = [...depots];
    if (aluva && !newDepots.find(d => d.name === "Aluva Terminal")) {
      newDepots.push({ ...aluva });
    }
    if (petta && !newDepots.find(d => d.name === "Petta Terminal")) {
      newDepots.push({ ...petta });
    }
    setDepots(newDepots);
  };

  const handleRemoveDepot = (index: number) => {
    setDepots(depots.filter((_, i) => i !== index));
  };

  const handleUpdateDepot = (index: number, field: keyof DepotConfig, value: any) => {
    const updated = [...depots];
    updated[index] = { ...updated[index], [field]: value };
    setDepots(updated);
  };

  const handleRunSimulation = () => {
    const request: SimulationRequest = {
      depots,
      fleet: fleetSize,
      seed: seed || undefined,
      service_requirement: serviceRequirement || undefined,
      ai_mode: aiMode,
      sim_days: simDays,
    };
    simulateMutation.mutate(request);
  };

  const handleStressTest = (fleet: number, numDepots: number) => {
    const selectedDepots = DEPOT_PRESETS.slice(0, numDepots);
    setDepots(selectedDepots);
    setFleetSize(fleet);
    setServiceRequirement(undefined); // Auto-compute
    handleRunSimulation();
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Multi-Depot Simulation</h2>
          <p className="text-muted-foreground">Simulate fleet operations across multiple depots</p>
        </div>
      </div>

      {/* Simulation Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Simulation Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Fleet Size */}
          <div>
            <Label>Fleet Size</Label>
            <div className="flex gap-2 mt-2">
              {FLEET_PRESETS.map(size => (
                <Button
                  key={size}
                  variant={fleetSize === size ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFleetSize(size)}
                >
                  {size}
                </Button>
              ))}
              <Input
                type="number"
                value={fleetSize}
                onChange={(e) => setFleetSize(parseInt(e.target.value) || 40)}
                className="w-24"
                min={1}
              />
            </div>
          </div>

          {/* Depots */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <Label>Depots ({depots.length})</Label>
              <div className="flex gap-2">
                {DEPOT_PRESETS.map(preset => (
                  <Button
                    key={preset.name}
                    variant="outline"
                    size="sm"
                    onClick={() => handleAddDepot(preset.name)}
                  >
                    + {preset.name}
                  </Button>
                ))}
              </div>
            </div>
            <div className="space-y-3">
              {depots.map((depot, index) => (
                <Card key={index}>
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-2">
                        <Building2 className="h-4 w-4" />
                        <span className="font-medium">{depot.name}</span>
                        <Badge variant="outline">{depot.location_type}</Badge>
                      </div>
                      {depots.length > 1 && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveDepot(index)}
                        >
                          Remove
                        </Button>
                      )}
                    </div>
                    <div className="grid grid-cols-4 gap-2">
                      <div>
                        <Label className="text-xs">Service Bays</Label>
                        <Input
                          type="number"
                          value={depot.service_bays}
                          onChange={(e) => handleUpdateDepot(index, 'service_bays', parseInt(e.target.value) || 0)}
                          min={0}
                        />
                      </div>
                      <div>
                        <Label className="text-xs">Maintenance Bays</Label>
                        <Input
                          type="number"
                          value={depot.maintenance_bays}
                          onChange={(e) => handleUpdateDepot(index, 'maintenance_bays', parseInt(e.target.value) || 0)}
                          min={0}
                        />
                      </div>
                      <div>
                        <Label className="text-xs">Standby Bays</Label>
                        <Input
                          type="number"
                          value={depot.standby_bays}
                          onChange={(e) => handleUpdateDepot(index, 'standby_bays', parseInt(e.target.value) || 0)}
                          min={0}
                        />
                      </div>
                      <div>
                        <Label className="text-xs">Total</Label>
                        <Input
                          value={depot.service_bays + depot.maintenance_bays + depot.standby_bays}
                          disabled
                          className="bg-muted"
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Capacity Warning */}
          {(() => {
            const totalCapacity = depots.reduce((sum, d) => 
              sum + d.service_bays + d.maintenance_bays + d.standby_bays, 0
            );
            const terminalCapacity = 12;
            const totalAvailable = totalCapacity + terminalCapacity;
            const overflow = fleetSize > totalAvailable ? fleetSize - totalAvailable : 0;
            
            if (overflow > 0) {
              return (
                <Card className="border-warning">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-semibold text-warning">Capacity Warning</div>
                        <div className="text-sm text-muted-foreground">
                          Fleet ({fleetSize}) exceeds capacity ({totalAvailable}) by {overflow} trains
                        </div>
                      </div>
                      <Button variant="outline" size="sm" onClick={handleAddTerminals}>
                        Add Terminal Presets
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            }
            return null;
          })()}

          {/* Advanced Options */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Service Requirement (auto if empty)</Label>
              <Input
                type="number"
                value={serviceRequirement || ''}
                onChange={(e) => setServiceRequirement(e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Auto"
                min={0}
              />
            </div>
            <div>
              <Label>Random Seed (optional)</Label>
              <Input
                type="number"
                value={seed || ''}
                onChange={(e) => setSeed(e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Random"
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Switch
                checked={aiMode}
                onCheckedChange={setAiMode}
              />
              <Label>AI Mode</Label>
            </div>
            <Button
              onClick={handleRunSimulation}
              disabled={simulateMutation.isPending || depots.length === 0}
              size="lg"
            >
              {simulateMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Simulation
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Stress Test Presets */}
      <Card>
        <CardHeader>
          <CardTitle>Stress Test Presets</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant="outline"
              onClick={() => handleStressTest(40, 1)}
            >
              40 trains, 1 depot
            </Button>
            <Button
              variant="outline"
              onClick={() => handleStressTest(40, 2)}
            >
              40 trains, 2 depots
            </Button>
            <Button
              variant="outline"
              onClick={() => handleStressTest(60, 2)}
            >
              60 trains, 2 depots
            </Button>
            <Button
              variant="outline"
              onClick={() => handleStressTest(100, 3)}
            >
              100 trains, 3 depots
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {simulationResult && (
        <Card>
          <CardHeader>
            <CardTitle>Simulation Results</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="summary">
              <TabsList>
                <TabsTrigger value="summary">Global Summary</TabsTrigger>
                <TabsTrigger value="depots">Per-Depot</TabsTrigger>
                <TabsTrigger value="transfers">Transfers</TabsTrigger>
                <TabsTrigger value="infrastructure">Infrastructure</TabsTrigger>
              </TabsList>

              <TabsContent value="summary" className="space-y-4">
                {/* Used AI Badge and Note */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Badge variant={simulationResult.used_ai ? "default" : "secondary"} className="text-sm">
                      Used AI: {simulationResult.used_ai ? "Yes" : "No"}
                    </Badge>
                    {!simulationResult.used_ai && (
                      <span className="text-xs text-muted-foreground" title="AI used: shows whether ML inference was used; if No, result used deterministic fallback">
                        (Deterministic fallback)
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Note: Results may use AI inference or a deterministic fallback. If fallback used, a warning is shown.
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4">
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{simulationResult.global_summary?.service_trains ?? 0}</div>
                      <div className="text-sm text-muted-foreground">Service Trains</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{simulationResult.global_summary?.required_service ?? 13}</div>
                      <div className="text-sm text-muted-foreground">Required Service</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold text-destructive">
                        {simulationResult.global_summary?.service_shortfall ?? 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Service Shortfall</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{simulationResult.global_summary?.shunting_time ?? 0}</div>
                      <div className="text-sm text-muted-foreground">Shunting Time (min)</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{simulationResult.global_summary?.turnout_time ?? 0}</div>
                      <div className="text-sm text-muted-foreground">Turnout Time (min)</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{simulationResult.global_summary?.total_capacity ?? 0}</div>
                      <div className="text-sm text-muted-foreground">Total Capacity</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{simulationResult.global_summary?.fleet ?? 0}</div>
                      <div className="text-sm text-muted-foreground">Fleet Size</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <div className="text-2xl font-bold">{simulationResult.global_summary?.transfers_recommended ?? 0}</div>
                      <div className="text-sm text-muted-foreground">Transfers Recommended</div>
                    </CardContent>
                  </Card>
                </div>

                {simulationResult.warnings.length > 0 && (
                  <Card className="border-warning">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-warning">
                        <AlertTriangle className="h-5 w-5" />
                        Warnings
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="list-disc list-inside space-y-1">
                        {simulationResult.warnings.map((w, i) => (
                          <li key={i} className="text-sm">{w}</li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="depots" className="space-y-4">
                {Object.entries(simulationResult.per_depot).map(([depotId, depotResult]) => (
                  <Card key={depotId}>
                    <CardHeader>
                      <CardTitle>{depotResult.depot_name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-4 gap-4">
                        <div>
                          <div className="text-lg font-bold">{depotResult.stabling_summary.service_trains}</div>
                          <div className="text-xs text-muted-foreground">Service Trains</div>
                        </div>
                        <div>
                          <div className="text-lg font-bold">{depotResult.stabling_summary.maintenance_trains}</div>
                          <div className="text-xs text-muted-foreground">Maintenance</div>
                        </div>
                        <div>
                          <div className="text-lg font-bold">{depotResult.stabling_summary.standby_trains}</div>
                          <div className="text-xs text-muted-foreground">Standby</div>
                        </div>
                        <div>
                          <div className="text-lg font-bold text-destructive">
                            {depotResult.stabling_summary.service_shortfall}
                          </div>
                          <div className="text-xs text-muted-foreground">Shortfall</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>

              <TabsContent value="transfers" className="space-y-2">
                {simulationResult.inter_depot_transfers.length === 0 ? (
                  <p className="text-muted-foreground">No transfer recommendations</p>
                ) : (
                  simulationResult.inter_depot_transfers.map((transfer, i) => (
                    <Card key={i} className={transfer.recommended ? "border-success" : ""}>
                      <CardContent className="p-4">
                        <div className="flex justify-between items-center">
                          <div className="flex items-center gap-2">
                            <Train className="h-4 w-4" />
                            <span className="font-medium">{transfer.train_id}</span>
                            <ArrowRight className="h-4 w-4" />
                            <span>{transfer.from_depot} → {transfer.to_depot}</span>
                            {transfer.recommended && (
                              <Badge variant="success">Recommended</Badge>
                            )}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Cost: ₹{transfer.cost_estimate.toLocaleString()} | 
                            Benefit: ₹{transfer.benefit_estimate.toLocaleString()}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))
                )}
              </TabsContent>

              <TabsContent value="infrastructure" className="space-y-2">
                {simulationResult.infrastructure_recommendations.length === 0 ? (
                  <p className="text-muted-foreground">No infrastructure recommendations</p>
                ) : (
                  simulationResult.infrastructure_recommendations.map((rec, i) => (
                    <Card key={i}>
                      <CardContent className="p-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium">
                              Add {rec.bays_to_add} {rec.bay_type} bays to {rec.depot_name}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              Reduces shortfall by {rec.shortfall_reduction} trains
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-bold">₹{rec.estimated_cost.toLocaleString()}</div>
                            <div className="text-xs text-muted-foreground">
                              Payback: {rec.payback_days.toFixed(1)} days | ROI: {(rec.roi * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
}


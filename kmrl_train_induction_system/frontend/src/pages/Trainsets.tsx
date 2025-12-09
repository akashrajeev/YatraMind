import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { trainsetsApi, optimizationApi } from "@/services/api";
import { Trainset } from "@/types/api";
import {
  Train,
  Search,
  Filter,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Clock,
  MapPin,
  Wrench,
  Activity,
  Upload,
  Download
} from "lucide-react";
import { useState, useMemo } from "react";

const Trainsets = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [depotFilter, setDepotFilter] = useState<string>("all");
  const queryClient = useQueryClient();

  // Fetch trainsets data
  const { data: trainsets = [], isLoading, refetch } = useQuery({
    queryKey: ['trainsets', statusFilter, depotFilter],
    queryFn: () => trainsetsApi.getAll({
      status: statusFilter !== 'all' ? statusFilter : undefined,
      depot: depotFilter !== 'all' ? depotFilter : undefined
    }).then(res => res.data),
    refetchInterval: 30000,
  });

  // Fetch latest induction decisions so we can display INDUCT/STANDBY/MAINTENANCE
  const { data: latestDecisions = [] } = useQuery({
    queryKey: ['latest-induction-for-trainsets'],
    queryFn: () => optimizationApi.getLatest().then(res => res.data),
    refetchInterval: 60000,
  });

  // Map trainset_id -> decision for quick lookup
  const decisionMap = useMemo(() => {
    const map: Record<string, string> = {};
    (latestDecisions as any[]).forEach((d) => {
      if (d?.trainset_id && d?.decision) {
        map[d.trainset_id] = d.decision;
      }
    });
    return map;
  }, [latestDecisions]);

  // Update trainset mutation
  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) =>
      trainsetsApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['trainsets']);
    },
  });

  // Filter trainsets based on search and filters
  const filteredTrainsets = trainsets.filter((trainset: Trainset) => {
    const matchesSearch = trainset.trainset_id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || trainset.status === statusFilter;
    const matchesDepot = depotFilter === 'all' || trainset.current_location?.depot === depotFilter;
    return matchesSearch && matchesStatus && matchesDepot;
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "ACTIVE":
        return <Badge variant="success"><CheckCircle className="h-3 w-3 mr-1" />Active</Badge>;
      case "MAINTENANCE":
        return <Badge variant="warning"><Wrench className="h-3 w-3 mr-1" />Maintenance</Badge>;
      case "STANDBY":
        return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Standby</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getDecisionBadge = (decision?: string) => {
    if (!decision) return null;
    switch (decision) {
      case "INDUCT":
        return <Badge variant="success"><CheckCircle className="h-3 w-3 mr-1" />Induct</Badge>;
      case "STANDBY":
        return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Standby</Badge>;
      case "MAINTENANCE":
        return <Badge variant="warning"><Wrench className="h-3 w-3 mr-1" />Maintenance</Badge>;
      default:
        return <Badge variant="outline">{decision}</Badge>;
    }
  };

  const getFitnessStatus = (fitness: any) => {
    const validCount = Object.values(fitness || {}).filter((cert: any) => cert.status === "VALID").length;
    const totalCount = Object.keys(fitness || {}).length;
    if (validCount === totalCount) return "success";
    if (validCount > 0) return "warning";
    return "destructive";
  };

  const getJobCardsStatus = (jobCards: any) => {
    if (jobCards?.critical_cards > 0) return "destructive";
    if (jobCards?.open_cards > 0) return "warning";
    return "success";
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Fleet Management</h2>
          <p className="text-muted-foreground">Monitor and manage trainset fleet operations</p>
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
            <Upload className="h-4 w-4 mr-2" />
            Upload Data
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Filters & Search</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="search">Search Trainsets</Label>
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="search"
                  placeholder="Search by ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <div>
              <Label htmlFor="status">Status</Label>
              <select
                id="status"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="w-full p-2 border border-input rounded-md bg-background"
              >
                <option value="all">All Status</option>
                <option value="ACTIVE">Active</option>
                <option value="MAINTENANCE">Maintenance</option>
                <option value="STANDBY">Standby</option>
              </select>
            </div>
            <div>
              <Label htmlFor="depot">Depot</Label>
              <select
                id="depot"
                value={depotFilter}
                onChange={(e) => setDepotFilter(e.target.value)}
                className="w-full p-2 border border-input rounded-md bg-background"
              >
                <option value="all">All Depots</option>
                <option value="Depot A">Depot A</option>
                <option value="Depot B">Depot B</option>
                <option value="Depot C">Depot C</option>
              </select>
            </div>
            <div className="flex items-end">
              <Button
                variant="outline"
                onClick={() => {
                  setSearchTerm("");
                  setStatusFilter("all");
                  setDepotFilter("all");
                }}
                className="w-full"
              >
                <Filter className="h-4 w-4 mr-2" />
                Clear Filters
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trainsets Grid */}
      {isLoading ? (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
          <p className="text-muted-foreground mt-2">Loading trainsets...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredTrainsets.map((trainset: Trainset) => (
            <Card key={trainset.trainset_id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Train className="h-5 w-5" />
                    {trainset.trainset_id}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    {getStatusBadge(trainset.status)}
                    {getDecisionBadge(decisionMap[trainset.trainset_id])}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Location */}
                <div className="flex items-center gap-2 text-sm">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">Location:</span>
                  <span className="font-medium">{trainset.current_location?.depot}</span>
                </div>

                {/* Fitness Certificates */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Fitness Certificates</span>
                    <Badge
                      variant={getFitnessStatus(trainset.fitness_certificates) as any}
                      className="text-xs"
                    >
                      {Object.values(trainset.fitness_certificates || {}).filter((cert: any) => cert.status === "VALID").length}/
                      {Object.keys(trainset.fitness_certificates || {}).length} Valid
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(trainset.fitness_certificates || {}).map(([type, cert]: [string, any]) => (
                      <div key={type} className="flex items-center justify-between">
                        <span className="text-muted-foreground">{type}</span>
                        <Badge
                          variant={cert.status === "VALID" ? "success" : "destructive"}
                          className="text-xs"
                        >
                          {cert.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Job Cards */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Job Cards</span>
                    <Badge
                      variant={getJobCardsStatus(trainset.job_cards) as any}
                      className="text-xs"
                    >
                      {trainset.job_cards?.open_cards || 0} Open
                    </Badge>
                  </div>
                  {trainset.job_cards?.critical_cards > 0 && (
                    <div className="flex items-center gap-2 text-xs text-destructive">
                      <AlertTriangle className="h-3 w-3" />
                      <span>{trainset.job_cards.critical_cards} Critical</span>
                    </div>
                  )}
                </div>

                {/* Mileage */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Current Mileage</span>
                    <span className="font-medium">{trainset.current_mileage || 0} km</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Max Before Maintenance</span>
                    <span className="font-medium">{trainset.max_mileage_before_maintenance || 0} km</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full"
                      style={{
                        width: `${Math.min(100, ((trainset.current_mileage || 0) / (trainset.max_mileage_before_maintenance || 1)) * 100)}%`
                      }}
                    />
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 pt-2">
                  <Button size="sm" variant="outline" className="flex-1">
                    View Details
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => updateMutation.mutate({
                      id: trainset.trainset_id,
                      data: { status: trainset.status === "ACTIVE" ? "STANDBY" : "ACTIVE" }
                    })}
                    disabled={updateMutation.isPending}
                  >
                    {trainset.status === "ACTIVE" ? "Standby" : "Activate"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {filteredTrainsets.length === 0 && !isLoading && (
        <div className="text-center py-8">
          <Train className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">No trainsets found matching your criteria</p>
        </div>
      )}
    </div>
  );
};

export default Trainsets;

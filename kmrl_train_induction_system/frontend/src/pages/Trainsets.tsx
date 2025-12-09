import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
  Download,
  X,
  Info,
  Target
} from "lucide-react";
import { useState, useMemo } from "react";


const Trainsets = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [depotFilter, setDepotFilter] = useState<string>("all");
  const [trainsetDetails, setTrainsetDetails] = useState<any>(null);
  const [showTrainsetDetails, setShowTrainsetDetails] = useState(false);
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

  const handleViewDetails = async (trainsetId: string) => {
    try {
      const response = await trainsetsApi.getDetails(trainsetId);
      setTrainsetDetails(response.data);
      setShowTrainsetDetails(true);
    } catch (error) {
      console.error('Error fetching trainset details:', error);
    }
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
                    {decisionMap[trainset.trainset_id] !== trainset.status && getDecisionBadge(decisionMap[trainset.trainset_id])}
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
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1"
                    onClick={() => handleViewDetails(trainset.trainset_id)}
                  >
                    View Details
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


      {/* Trainset Details Modal */}
      {showTrainsetDetails && trainsetDetails && (
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
      )}
    </div>
  );
};

export default Trainsets;

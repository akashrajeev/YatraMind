import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileUpload } from "@/components/ui/file-upload";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { ingestionApi } from "@/services/api";
import {
  Upload,
  Download,
  Database,
  Wifi,
  WifiOff,
  FileText,
  BarChart3,
  Settings,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  Clock,
  Activity
} from "lucide-react";
import { useState } from "react";

const DataIngestion = () => {
  const [selectedFiles, setSelectedFiles] = useState<{ [key: string]: File | File[] | null }>({});
  const [googleSheetUrl, setGoogleSheetUrl] = useState("");
  const queryClient = useQueryClient();

  // Fetch ingestion status
  const { data: ingestionStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['ingestion-status'],
    queryFn: () => ingestionApi.getStatus().then(res => res.data),
    refetchInterval: 30000,
  });

  const { data: mqttStatus } = useQuery({
    queryKey: ['mqtt-status'],
    queryFn: () => ingestionApi.getMQTTStatus().then(res => res.data),
    refetchInterval: 10000,
  });

  // Mutations
  const ingestAllMutation = useMutation({
    mutationFn: ingestionApi.ingestAll,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
    },
  });

  const ingestMaximoMutation = useMutation({
    mutationFn: ingestionApi.ingestMaximo,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
    },
  });

  const ingestIoTMutation = useMutation({
    mutationFn: ingestionApi.ingestIoT,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
    },
  });

  const uploadTimeseriesMutation = useMutation({
    mutationFn: ingestionApi.uploadTimeseries,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
      queryClient.invalidateQueries({ queryKey: ['trainsets'] });
      queryClient.invalidateQueries({ queryKey: ['optimization'] });
      queryClient.invalidateQueries({ queryKey: ['assignments'] });
    },
  });

  const uploadFitnessMutation = useMutation({
    mutationFn: ingestionApi.uploadFitness,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
      queryClient.invalidateQueries({ queryKey: ['trainsets'] });
      queryClient.invalidateQueries({ queryKey: ['optimization'] });
      queryClient.invalidateQueries({ queryKey: ['assignments'] });
    },
  });

  const uploadBrandingMutation = useMutation({
    mutationFn: ingestionApi.uploadBranding,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
      queryClient.invalidateQueries({ queryKey: ['trainsets'] });
      queryClient.invalidateQueries({ queryKey: ['optimization'] });
      queryClient.invalidateQueries({ queryKey: ['assignments'] });
    },
  });

  const uploadDepotMutation = useMutation({
    mutationFn: ingestionApi.uploadDepot,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
      queryClient.invalidateQueries({ queryKey: ['trainsets'] });
      queryClient.invalidateQueries({ queryKey: ['optimization'] });
      queryClient.invalidateQueries({ queryKey: ['assignments'] });
    },
  });

  const startMQTTMutation = useMutation({
    mutationFn: ingestionApi.startMQTT,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mqtt-status'] });
    },
  });

  const stopMQTTMutation = useMutation({
    mutationFn: ingestionApi.stopMQTT,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mqtt-status'] });
    },
  });

  const ingestCleaningMutation = useMutation({
    mutationFn: ingestionApi.ingestCleaningGoogle,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ingestion-status'] });
      queryClient.invalidateQueries({ queryKey: ['trainsets'] });
      queryClient.invalidateQueries({ queryKey: ['optimization'] });
      queryClient.invalidateQueries({ queryKey: ['assignments'] });
    },
  });

  const uploadN8NMutation = useMutation({
    mutationFn: ingestionApi.uploadN8N,
    onSuccess: (data) => {
      // You might want to show a toast here with data.n8n_response
      console.log("N8N Response:", data);
    },
  });

  const handleFileSelect = (type: string, file: File | File[]) => {
    setSelectedFiles(prev => ({
      ...prev,
      [type]: file
    }));
  };

  const handleUpload = (type: string) => {
    const files = selectedFiles[type];
    if (!files) return;

    switch (type) {
      case 'timeseries':
        if (files instanceof File) uploadTimeseriesMutation.mutate(files);
        break;
      case 'fitness':
        if (files instanceof File) uploadFitnessMutation.mutate(files);
        break;
      case 'branding':
        if (files instanceof File) uploadBrandingMutation.mutate(files);
        break;
      case 'depot':
        if (files instanceof File) uploadDepotMutation.mutate(files);
        break;
      case 'n8n':
        // Ensure we handle both single file and array of files
        const fileArray = Array.isArray(files) ? files : [files];
        uploadN8NMutation.mutate(fileArray);
        break;
    }
  };

  // ... (existing render logic) ...



  const getStatusBadge = (status: string) => {
    switch (status) {
      case "available":
      case "streaming":
      case "connected":
        return <Badge variant="success"><CheckCircle className="h-3 w-3 mr-1" />{status}</Badge>;
      case "disconnected":
      case "stopped":
        return <Badge variant="destructive"><WifiOff className="h-3 w-3 mr-1" />{status}</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Data Ingestion</h2>
          <p className="text-muted-foreground">Manage data sources and uploads</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="industrial">
            <Download className="h-4 w-4 mr-2" />
            Export Data
          </Button>
        </div>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Data Sources Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <div className="text-center py-4">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mx-auto"></div>
              </div>
            ) : ingestionStatus ? (
              <div className="space-y-3">
                {Object.entries(ingestionStatus.sources || {}).map(([source, status]) => (
                  <div key={source} className="flex items-center justify-between">
                    <span className="capitalize">{source.replace('_', ' ')}</span>
                    {getStatusBadge(status as string)}
                  </div>
                ))}
                <div className="pt-2 border-t">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Last Ingestion</span>
                    <span className="font-medium">{ingestionStatus.last_ingestion}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-4">
                <p className="text-muted-foreground">Unable to load status</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wifi className="h-5 w-5" />
              MQTT IoT Streaming
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span>Connection Status</span>
                {mqttStatus ? getStatusBadge(mqttStatus.status) : <Badge variant="secondary">Unknown</Badge>}
              </div>
              {mqttStatus?.topics && (
                <div>
                  <span className="text-sm text-muted-foreground">Active Topics:</span>
                  <div className="mt-1 space-y-1">
                    {mqttStatus.topics.map((topic: string, index: number) => (
                      <div key={index} className="text-xs bg-muted px-2 py-1 rounded">
                        {topic}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => startMQTTMutation.mutate()}
                  disabled={startMQTTMutation.isPending || mqttStatus?.status === 'connected'}
                >
                  Start Streaming
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => stopMQTTMutation.mutate()}
                  disabled={stopMQTTMutation.isPending || mqttStatus?.status === 'disconnected'}
                >
                  Stop Streaming
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Ingestion Tabs */}
      <Tabs defaultValue="automated" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="automated">Automated Ingestion</TabsTrigger>
          <TabsTrigger value="uploads">File Uploads</TabsTrigger>
          <TabsTrigger value="google">Google Sheets</TabsTrigger>
          <TabsTrigger value="universal">Universal Ingestion</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>

        {/* Automated Ingestion */}
        <TabsContent value="automated" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  All Sources
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Ingest data from all configured sources including Maximo, IoT sensors, and manual overrides.
                </p>
                <Button
                  onClick={() => ingestAllMutation.mutate()}
                  disabled={ingestAllMutation.isPending}
                  className="w-full"
                >
                  {ingestAllMutation.isPending ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Ingesting...
                    </>
                  ) : (
                    <>
                      <Activity className="h-4 w-4 mr-2" />
                      Ingest All Sources
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Maximo Data
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Ingest job card data from IBM Maximo system.
                </p>
                <Button
                  onClick={() => ingestMaximoMutation.mutate()}
                  disabled={ingestMaximoMutation.isPending}
                  className="w-full"
                  variant="outline"
                >
                  {ingestMaximoMutation.isPending ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary mr-2"></div>
                      Ingesting...
                    </>
                  ) : (
                    <>
                      <Database className="h-4 w-4 mr-2" />
                      Ingest Maximo
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wifi className="h-5 w-5" />
                  IoT Sensors
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Ingest real-time IoT sensor data from trainsets.
                </p>
                <Button
                  onClick={() => ingestIoTMutation.mutate()}
                  disabled={ingestIoTMutation.isPending}
                  className="w-full"
                  variant="outline"
                >
                  {ingestIoTMutation.isPending ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary mr-2"></div>
                      Ingesting...
                    </>
                  ) : (
                    <>
                      <Wifi className="h-4 w-4 mr-2" />
                      Ingest IoT Data
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* File Uploads */}
        <TabsContent value="uploads" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Time Series Data
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <FileUpload
                    onFileSelect={(file) => handleFileSelect('timeseries', file)}
                    accept=".csv"
                    maxSize={50}
                    disabled={uploadTimeseriesMutation.isPending}
                  />
                  <Button
                    onClick={() => handleUpload('timeseries')}
                    disabled={!selectedFiles.timeseries || uploadTimeseriesMutation.isPending}
                    className="w-full"
                  >
                    {uploadTimeseriesMutation.isPending ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="h-4 w-4 mr-2" />
                        Upload Time Series
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Fitness Certificates
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <FileUpload
                    onFileSelect={(file) => handleFileSelect('fitness', file)}
                    accept=".csv,.xlsx"
                    maxSize={20}
                    disabled={uploadFitnessMutation.isPending}
                  />
                  <Button
                    onClick={() => handleUpload('fitness')}
                    disabled={!selectedFiles.fitness || uploadFitnessMutation.isPending}
                    className="w-full"
                  >
                    {uploadFitnessMutation.isPending ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="h-4 w-4 mr-2" />
                        Upload Fitness Data
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Branding Contracts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <FileUpload
                    onFileSelect={(file) => handleFileSelect('branding', file)}
                    accept=".csv,.xlsx"
                    maxSize={20}
                    disabled={uploadBrandingMutation.isPending}
                  />
                  <Button
                    onClick={() => handleUpload('branding')}
                    disabled={!selectedFiles.branding || uploadBrandingMutation.isPending}
                    className="w-full"
                  >
                    {uploadBrandingMutation.isPending ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="h-4 w-4 mr-2" />
                        Upload Branding Data
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Depot Layout
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <FileUpload
                    onFileSelect={(file) => handleFileSelect('depot', file)}
                    accept=".geojson,.json"
                    maxSize={10}
                    disabled={uploadDepotMutation.isPending}
                  />
                  <Button
                    onClick={() => handleUpload('depot')}
                    disabled={!selectedFiles.depot || uploadDepotMutation.isPending}
                    className="w-full"
                  >
                    {uploadDepotMutation.isPending ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="h-4 w-4 mr-2" />
                        Upload Depot Layout
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Google Sheets */}
        <TabsContent value="google" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Google Sheets Integration
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="sheet-url">Google Sheets URL</Label>
                  <Input
                    id="sheet-url"
                    placeholder="https://docs.google.com/spreadsheets/d/..."
                    value={googleSheetUrl}
                    onChange={(e) => setGoogleSheetUrl(e.target.value)}
                  />
                </div>
                <div className="text-sm text-muted-foreground">
                  Make sure the Google Sheet is published as CSV/TSV and the URL is publicly accessible.
                </div>
                <Button
                  onClick={() => {
                    if (googleSheetUrl) {
                      ingestCleaningMutation.mutate(googleSheetUrl);
                    }
                  }}
                  disabled={!googleSheetUrl || ingestCleaningMutation.isPending}
                  className="w-full"
                >
                  <FileText className="h-4 w-4 mr-2" />
                  {ingestCleaningMutation.isPending ? 'Ingesting...' : 'Ingest from Google Sheets'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Monitoring */}
        <TabsContent value="monitoring" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Ingestion Monitoring
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-foreground">
                      {ingestionStatus?.sources ? Object.keys(ingestionStatus.sources).length : 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Data Sources</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-success">
                      {mqttStatus?.streaming ? 'ON' : 'OFF'}
                    </div>
                    <div className="text-sm text-muted-foreground">IoT Streaming</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">
                      {mqttStatus?.topics?.length || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Active Topics</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Universal Ingestion */}
        <TabsContent value="universal" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Universal Ingestion (n8n)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Upload multiple files (PDF, Excel, JSON, etc.) to be processed by the universal ingestion pipeline via n8n.
                </p>
                <FileUpload
                  onFileSelect={(files) => handleFileSelect('n8n', files)}
                  accept="*"
                  maxSize={50}
                  multiple={true}
                  disabled={uploadN8NMutation.isPending}
                />
                <Button
                  onClick={() => handleUpload('n8n')}
                  disabled={!selectedFiles.n8n || uploadN8NMutation.isPending}
                  className="w-full"
                >
                  {uploadN8NMutation.isPending ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Uploading to n8n...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4 mr-2" />
                      Upload to n8n
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DataIngestion;

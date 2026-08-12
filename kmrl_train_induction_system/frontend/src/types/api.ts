// Dashboard Types
export interface DashboardOverview {
  total_trainsets: number;
  fleet_status: { active: number; maintenance: number; standby: number };
  fitness_certificates: { valid: number; expired: number; expiring_soon: number };
  job_cards: { total_open: number; critical: number };
  depot_distribution: Record<string, number>;
  sensor_health: { average_health_score: number; sensors_online: number; sensors_critical: number };
  last_updated: string;
}

export interface Alert { type: 'CRITICAL' | 'HIGH' | 'WARNING'; category: string; trainset_id: string; message: string; timestamp: string; }
export interface AlertsResponse { total_alerts: number; critical_count: number; high_count: number; warning_count: number; alerts: Alert[]; }
export interface PerformanceMetrics {
  optimization_performance: { total_runs: number; average_confidence_score: number; recent_history: any[] };
  operational_metrics: { punctuality_rate: number; fleet_availability: number; energy_efficiency: number; maintenance_cost_reduction: number };
  sensor_analytics: any;
  system_health: { api_response_time_ms: number; optimization_time_seconds: number; database_performance: string; mqtt_connectivity: string };
}

// Assignment Types
export interface Assignment {
  id: string; trainset_id: string;
  decision: { decision: string; confidence_score: number; reasoning: string; top_reasons?: string[]; top_risks?: string[]; violations?: string[] };
  status: 'PENDING' | 'APPROVED' | 'REJECTED' | 'OVERRIDDEN'; priority: number; created_by: string; created_at: string;
  approved_by?: string; approved_at?: string; override_reason?: string; override_by?: string; override_at?: string;
  execution_date: string; last_updated: string;
}
export interface AssignmentCreate { trainset_id: string; decision: { decision: string; confidence_score: number; reasoning: string; violations?: string[] }; priority: number; execution_date: string; }
export interface AssignmentUpdate { status?: string; priority?: number; execution_date?: string; }
export interface ApprovalRequest { assignment_ids: string[]; comments: string; }
export interface OverrideRequest { assignment_id: string; override_decision: string; reason: string; }
export interface AssignmentSummary { total_assignments: number; pending_count: number; approved_count: number; rejected_count: number; overridden_count: number; high_priority_count: number; critical_risks_count: number; avg_confidence_score: number; last_updated: string; }

// Trainset Types
export interface Trainset {
  trainset_id: string;
  status: 'ACTIVE' | 'MAINTENANCE' | 'STANDBY';
  current_location: { depot: string; platform?: string; lane?: string | number };
  fitness_certificates: Record<string, { status: 'VALID' | 'EXPIRED' | 'EXPIRING_SOON'; expiry_date: string; issued_date: string; }>; 
  job_cards: { open_cards: number; critical_cards: number; cards: any[] };
  branding: { current_advertiser: string | null; priority: 'HIGH' | 'MEDIUM' | 'LOW'; contract_end_date?: string; };
  cleaning: { requires_cleaning: boolean; cleaning_type?: string; last_cleaned?: string; next_cleaning_due?: string; };
  current_mileage: number; max_mileage_before_maintenance: number;
  mileage_today?: number; mileage_this_month?: number;
  maintenance_severity?: 'NONE' | 'LIGHT' | 'HEAVY';
  sensor_health_score?: number; predicted_failure_risk?: number;
  stabling?: { bay_id: string; depot: string; position: number; departure_time?: string; };
  last_updated?: string;
}

export interface TrainsetResponse { trainsets: Trainset[]; total: number; page: number; page_size: number; }
export interface FitnessCertificate { status: string; expiry_date: string; days_until_expiry: number; is_valid: boolean; certificate_type?: string; issued_date?: string; }
export interface JobCard { id: string; description: string; status: string; priority: string; due_date?: string; estimated_hours?: number; }
export interface BrandingContract { advertiser: string; priority: 'HIGH' | 'MEDIUM' | 'LOW'; start_date: string; end_date: string; is_active: boolean; }

// Other API types
export interface OptimizationRequest { target_date?: string; service_date?: string; required_service_count?: number; override_constraints?: any; weights?: any; }
export interface OptimizationDecision { trainset_id: string; decision: 'INDUCT' | 'STANDBY' | 'MAINTENANCE'; confidence_score: number; reasons: string[]; score?: number; top_reasons?: string[]; top_risks?: string[]; violations?: string[]; }
export interface OptimizationResponse { required_service_trains: number; standby_buffer: number; total_required_trains: number; calculation_method: string; eligible_train_count: number; granted_train_count: number; actual_induct_count: number; service_shortfall: number; note?: string; diagnostics?: any; decisions: OptimizationDecision[]; stabling_geometry?: any; }

// Dashboard Types
export interface DashboardOverview {
  total_trainsets: number;
  fleet_status: {
    active: number;
    maintenance: number;
    standby: number;
  };
  fitness_certificates: {
    valid: number;
    expired: number;
    expiring_soon: number;
  };
  job_cards: {
    total_open: number;
    critical: number;
  };
  depot_distribution: Record<string, number>;
  sensor_health: {
    average_health_score: number;
    sensors_online: number;
    sensors_critical: number;
  };
  last_updated: string;
}

export interface Alert {
  type: 'CRITICAL' | 'HIGH' | 'WARNING';
  category: string;
  trainset_id: string;
  message: string;
  timestamp: string;
}

export interface AlertsResponse {
  total_alerts: number;
  critical_count: number;
  high_count: number;
  warning_count: number;
  alerts: Alert[];
}

export interface PerformanceMetrics {
  optimization_performance: {
    total_runs: number;
    average_confidence_score: number;
    recent_history: any[];
  };
  operational_metrics: {
    punctuality_rate: number;
    fleet_availability: number;
    energy_efficiency: number;
    maintenance_cost_reduction: number;
  };
  sensor_analytics: any;
  system_health: {
    api_response_time_ms: number;
    optimization_time_seconds: number;
    database_performance: string;
    mqtt_connectivity: string;
  };
}

// Assignment Types
export interface Assignment {
  id: string;
  trainset_id: string;
  decision: {
    decision: string;
    confidence_score: number;
    reasoning: string;
    violations?: string[];
  };
  status: 'PENDING' | 'APPROVED' | 'REJECTED' | 'OVERRIDDEN';
  priority: number;
  created_by: string;
  created_at: string;
  approved_by?: string;
  approved_at?: string;
  override_reason?: string;
  override_by?: string;
  override_at?: string;
  execution_date: string;
  last_updated: string;
}

export interface AssignmentCreate {
  trainset_id: string;
  decision: {
    decision: string;
    confidence_score: number;
    reasoning: string;
    violations?: string[];
  };
  priority: number;
  execution_date: string;
}

export interface AssignmentUpdate {
  status?: string;
  priority?: number;
  execution_date?: string;
}

export interface ApprovalRequest {
  assignment_ids: string[];
  comments: string;
}

export interface OverrideRequest {
  assignment_id: string;
  override_decision: string;
  reason: string;
}

export interface AssignmentSummary {
  total_assignments: number;
  pending_count: number;
  approved_count: number;
  rejected_count: number;
  overridden_count: number;
  high_priority_count: number;
  critical_risks_count: number;
  avg_confidence_score: number;
  last_updated: string;
}

// Trainset Types
export interface Trainset {
  trainset_id: string;
  status: 'ACTIVE' | 'MAINTENANCE' | 'STANDBY';
  current_location: {
    depot: string;
    platform?: string;
  };
  fitness_certificates: Record<string, {
    status: 'VALID' | 'EXPIRED' | 'EXPIRING_SOON';
    expiry_date: string;
  }>;
  job_cards: {
    open_cards: number;
    critical_cards: number;
  };
  current_mileage: number;
  max_mileage_before_maintenance: number;
  last_maintenance_date: string;
  next_maintenance_due: string;
}

// Notification Types
export interface Notification {
  id: string;
  type: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  title: string;
  message: string;
  data?: any;
  read: boolean;
  created_at: string;
}

// Optimization Types
export interface OptimizationRequest {
  target_date: string;
  required_service_hours: number;
  constraints?: any;
}

export interface OptimizationResult {
  id: string;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  assignments: Assignment[];
  confidence_score: number;
  created_at: string;
  completed_at?: string;
}

// Report Types
export interface ReportRequest {
  format: 'pdf' | 'csv';
  start_date?: string;
  end_date?: string;
  filters?: any;
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

// Error Types
export interface ApiError {
  detail: string;
  status_code: number;
  timestamp: string;
}

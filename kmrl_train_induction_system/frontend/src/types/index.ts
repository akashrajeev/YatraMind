export interface Trainset {
  trainset_id: string
  status: 'ACTIVE' | 'STANDBY' | 'MAINTENANCE'
  current_location: {
    depot: string
    bay: string
  }
  current_mileage: number
  max_mileage_before_maintenance: number
  fitness_certificates: Record<string, {
    status: 'VALID' | 'EXPIRED' | 'EXPIRING_SOON'
    expiry_date: string
    issued_by: string
  }>
  job_cards: {
    open_cards: number
    critical_cards: number
  }
  branding_priority: number
  sensor_health_score: number
}

export interface InductionDecision {
  trainset_id: string
  decision: 'INDUCT' | 'STANDBY' | 'MAINTENANCE'
  confidence_score: number
  reasons: string[]
  score: number
  top_reasons: string[]
  top_risks: string[]
  violations: string[]
  shap_values: Array<{
    name: string
    value: number
    impact: 'positive' | 'negative' | 'neutral'
  }>
}

export interface Assignment {
  id: string
  trainset_id: string
  decision: InductionDecision
  status: 'PENDING' | 'APPROVED' | 'REJECTED' | 'OVERRIDDEN'
  created_at: string
  approved_by?: string
  approved_at?: string
  override_reason?: string
  override_by?: string
  override_at?: string
}

export interface OverrideRequest {
  assignment_id: string
  reason: string
  user_id: string
  override_decision: 'INDUCT' | 'STANDBY' | 'MAINTENANCE'
}

export interface ApprovalRequest {
  assignment_ids: string[]
  user_id: string
  comments?: string
}

export interface Alert {
  id: string
  type: 'CRITICAL' | 'HIGH' | 'WARNING'
  category: 'CERTIFICATE' | 'MAINTENANCE' | 'MILEAGE' | 'SENSOR'
  trainset_id: string
  message: string
  timestamp: string
  acknowledged: boolean
  acknowledged_by?: string
  acknowledged_at?: string
}

export interface User {
  id: string
  username: string
  name: string
  email?: string
  role: 'SUPERVISOR' | 'MAINTENANCE_ENGINEER' | 'OPERATIONS_MANAGER' | 'READONLY_VIEWER'
  permissions: string[]
}

export interface AuditLog {
  id: string
  user_id: string
  action: string
  resource_type: string
  resource_id: string
  timestamp: string
  details: Record<string, any>
  ip_address?: string
}

export interface DashboardOverview {
  total_trainsets: number
  fleet_status: {
    active: number
    maintenance: number
    standby: number
  }
  fitness_certificates: {
    valid: number
    expired: number
    expiring_soon: number
  }
  job_cards: {
    total_open: number
    critical: number
  }
  depot_distribution: Record<string, number>
  sensor_health: {
    average_health_score: number
    sensors_online: number
    sensors_critical: number
  }
  last_updated: string
}

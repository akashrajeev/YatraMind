// API Configuration
export const API_BASE_URL = 'http://localhost:8000/api';

// API Key for authentication
export const API_KEY = 'kmrl_api_key_2024';

// API endpoints
export const API_ENDPOINTS = {
  // Dashboard
  DASHBOARD_OVERVIEW: '/dashboard/overview',
  DASHBOARD_ALERTS: '/dashboard/alerts',
  DASHBOARD_PERFORMANCE: '/dashboard/performance',
  
  // Assignments
  ASSIGNMENTS: '/v1/assignments/',
  ASSIGNMENTS_SUMMARY: '/v1/assignments/summary',
  ASSIGNMENTS_APPROVE: '/v1/assignments/approve',
  ASSIGNMENTS_OVERRIDE: '/v1/assignments/override',
  
  // Reports
  REPORTS_DAILY_BRIEFING: '/v1/reports/daily-briefing',
  REPORTS_ASSIGNMENTS: '/v1/reports/assignments',
  REPORTS_AUDIT_LOGS: '/v1/reports/audit-logs',
  REPORTS_FLEET_STATUS: '/v1/reports/fleet-status',
  REPORTS_PERFORMANCE_ANALYSIS: '/v1/reports/performance-analysis',
  REPORTS_COMPLIANCE: '/v1/reports/compliance-report',
  
  // Optimization
  OPTIMIZATION_RUN: '/optimization/run',
  OPTIMIZATION_HISTORY: '/optimization/history',
  OPTIMIZATION_LATEST: '/optimization/latest',
  OPTIMIZATION_CONSTRAINTS: '/optimization/constraints/check',
  OPTIMIZATION_EXPLAIN: '/optimization/explain',
  OPTIMIZATION_SIMULATE: '/optimization/simulate',
  OPTIMIZATION_STABLING: '/optimization/stabling-geometry',
  OPTIMIZATION_SHUNTING: '/optimization/shunting-schedule',
  
  // Trainsets
  TRAINSETS: '/trainsets/',
  TRAINSETS_FITNESS: '/trainsets/{id}/fitness',
  
  // Data Ingestion
  INGESTION_STATUS: '/ingestion/status',
  INGESTION_ALL: '/ingestion/ingest/all',
  INGESTION_MAXIMO: '/ingestion/ingest/maximo',
  INGESTION_IOT: '/ingestion/ingest/iot',
  INGESTION_TIMESERIES: '/ingestion/ingest/timeseries/upload',
  INGESTION_FITNESS: '/ingestion/fitness/upload',
  INGESTION_BRANDING: '/ingestion/branding/upload',
  INGESTION_DEPOT: '/ingestion/depot/upload',
  INGESTION_MQTT_STATUS: '/ingestion/mqtt/status',
  INGESTION_MQTT_START: '/ingestion/mqtt/start',
  INGESTION_MQTT_STOP: '/ingestion/mqtt/stop',
  
  // Notifications
  NOTIFICATIONS: '/notifications',
  NOTIFICATIONS_READ: '/notifications/{id}/read',
  NOTIFICATIONS_READ_ALL: '/notifications/read-all',
} as const;
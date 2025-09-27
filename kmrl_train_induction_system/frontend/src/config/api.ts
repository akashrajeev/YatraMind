// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  API_KEY: import.meta.env.VITE_API_KEY || 'your-api-key-here',
  WS_URL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  TIMEOUT: 10000,
};

// API Endpoints
export const ENDPOINTS = {
  DASHBOARD: {
    OVERVIEW: '/dashboard/overview',
    ALERTS: '/dashboard/alerts',
    PERFORMANCE: '/dashboard/performance',
  },
  ASSIGNMENTS: {
    LIST: '/assignments',
    DETAIL: (id: string) => `/assignments/${id}`,
    CREATE: '/assignments',
    APPROVE: '/assignments/approve',
    OVERRIDE: '/assignments/override',
    SUMMARY: '/assignments/summary',
  },
  REPORTS: {
    DAILY_BRIEFING: '/reports/daily-briefing',
    ASSIGNMENTS: '/reports/assignments',
    AUDIT_LOGS: '/reports/audit-logs',
    FLEET_STATUS: '/reports/fleet-status',
    PERFORMANCE_ANALYSIS: '/reports/performance-analysis',
    COMPLIANCE: '/reports/compliance-report',
  },
  TRAINSETS: {
    LIST: '/trainsets',
    DETAIL: (id: string) => `/trainsets/${id}`,
    UPDATE: (id: string) => `/trainsets/${id}`,
  },
  NOTIFICATIONS: {
    LIST: '/notifications',
    MARK_READ: (id: string) => `/notifications/${id}/read`,
    MARK_ALL_READ: '/notifications/read-all',
  },
  OPTIMIZATION: {
    RUN: '/optimization/run',
    HISTORY: '/optimization/history',
    STATUS: (id: string) => `/optimization/status/${id}`,
  },
};

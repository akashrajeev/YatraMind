import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for API key
api.interceptors.request.use((config) => {
  // Add API key to all requests
  config.headers['X-API-Key'] = 'your-api-key-here'; // This should come from environment or auth
  return config;
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Dashboard API
export const dashboardApi = {
  getOverview: () => api.get('/dashboard/overview'),
  getAlerts: () => api.get('/dashboard/alerts'),
  getPerformance: () => api.get('/dashboard/performance'),
};

// Assignments API
export const assignmentApi = {
  getAll: (params?: any) => api.get('/assignments', { params }),
  getById: (id: string) => api.get(`/assignments/${id}`),
  create: (data: any) => api.post('/assignments', data),
  approve: (data: any) => api.post('/assignments/approve', data),
  override: (data: any) => api.post('/assignments/override', data),
  getSummary: () => api.get('/assignments/summary'),
};

// Reports API
export const reportsApi = {
  getDailyBriefing: (date?: string) => api.get('/reports/daily-briefing', { 
    params: { date },
    responseType: 'blob' 
  }),
  exportAssignments: (format: string, filters?: any) => api.get('/reports/assignments', {
    params: { format, ...filters },
    responseType: 'blob'
  }),
  exportAuditLogs: (filters?: any) => api.get('/reports/audit-logs', {
    params: filters,
    responseType: 'blob'
  }),
  getFleetStatus: (format: string = 'pdf') => api.get('/reports/fleet-status', {
    params: { format },
    responseType: 'blob'
  }),
  getPerformanceAnalysis: (days: number = 30) => api.get('/reports/performance-analysis', {
    params: { days },
    responseType: 'blob'
  }),
  getComplianceReport: (startDate?: string, endDate?: string) => api.get('/reports/compliance-report', {
    params: { start_date: startDate, end_date: endDate },
    responseType: 'blob'
  }),
};

// Optimization API
export const optimizationApi = {
  runOptimization: (data: any) => api.post('/optimization/run', data),
  getHistory: () => api.get('/optimization/history'),
  getStatus: (id: string) => api.get(`/optimization/status/${id}`),
  checkConstraints: () => api.get('/optimization/constraints/check'),
  explainAssignment: (trainsetId: string, decision?: string, format?: string) => 
    api.get(`/optimization/explain/${trainsetId}`, { 
      params: { decision, format } 
    }),
  explainBatch: (assignments: any[], format?: string) => 
    api.post('/optimization/explain/batch', { assignments, format }),
  simulate: (params: any) => api.get('/optimization/simulate', { params }),
  getLatest: () => api.get('/optimization/latest'),
  getStablingGeometry: () => api.get('/optimization/stabling-geometry'),
  getShuntingSchedule: () => api.get('/optimization/shunting-schedule'),
};

// Trainsets API
export const trainsetsApi = {
  getAll: (params?: any) => api.get('/trainsets', { params }),
  getById: (id: string) => api.get(`/trainsets/${id}`),
  update: (id: string, data: any) => api.put(`/trainsets/${id}`, data),
  getFitness: (id: string) => api.get(`/trainsets/${id}/fitness`),
};

// Data Ingestion API
export const ingestionApi = {
  ingestAll: () => api.post('/ingestion/ingest/all'),
  ingestMaximo: () => api.post('/ingestion/ingest/maximo'),
  ingestIoT: () => api.post('/ingestion/ingest/iot'),
  uploadTimeseries: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/ingestion/ingest/timeseries/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  uploadFitness: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/ingestion/fitness/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  uploadBranding: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/ingestion/branding/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  uploadDepot: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/ingestion/depot/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  getStatus: () => api.get('/ingestion/status'),
  startMQTT: () => api.post('/ingestion/mqtt/start'),
  stopMQTT: () => api.post('/ingestion/mqtt/stop'),
  getMQTTStatus: () => api.get('/ingestion/mqtt/status'),
};

// Notifications API
export const notificationsApi = {
  getAll: () => api.get('/notifications'),
  markAsRead: (id: string) => api.put(`/notifications/${id}/read`),
  markAllAsRead: () => api.put('/notifications/read-all'),
};

export default api;
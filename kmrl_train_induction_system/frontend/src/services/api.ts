import axios from 'axios';
import { API_BASE_URL, API_KEY } from '@/config/api';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for API key
api.interceptors.request.use((config) => {
  // Add API key to all requests
  config.headers['X-API-Key'] = API_KEY;
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
  getOverview: () => api.get('/v1/dashboard/overview'),
  getAlerts: () => api.get('/v1/dashboard/alerts'),
  getPerformance: () => api.get('/v1/dashboard/performance'),
};

// Assignments API
export const assignmentApi = {
  getAll: (params?: any) => api.get('/v1/assignments/', { params }),
  getById: (id: string) => api.get(`/v1/assignments/${id}`),
  create: (data: any) => api.post('/v1/assignments/', data),
  approve: (data: any) => api.post('/v1/assignments/approve', data),
  override: (data: any) => api.post('/v1/assignments/override', data),
  getSummary: () => api.get('/v1/assignments/summary'),
  getConflicts: () => api.get('/v1/assignments/conflicts'),
};

// Reports API
export const reportsApi = {
  getDailyBriefing: (date?: string) => api.get('/v1/reports/daily-briefing', {
    params: { date },
    responseType: 'blob'
  }),
  exportAssignments: (format: string, filters?: any) => api.get('/v1/reports/assignments', {
    params: { format, ...filters },
    responseType: 'blob'
  }),
  exportAuditLogs: (filters?: any) => api.get('/v1/reports/audit-logs', {
    params: filters,
    responseType: 'blob'
  }),
  getFleetStatus: (format: string = 'pdf') => api.get('/v1/reports/fleet-status', {
    params: { format },
    responseType: 'blob'
  }),
  getPerformanceAnalysis: (days: number = 30) => api.get('/v1/reports/performance-analysis', {
    params: { days },
    responseType: 'blob'
  }),
  getComplianceReport: (startDate?: string, endDate?: string) => api.get('/v1/reports/compliance-report', {
    params: { start_date: startDate, end_date: endDate },
    responseType: 'blob'
  }),
};

// Optimization API
export const optimizationApi = {
  runOptimization: (data: any) => api.post('/v1/optimization/run', data),
  getHistory: () => api.get('/v1/optimization/history'),
  getStatus: (id: string) => api.get(`/v1/optimization/status/${id}`),
  checkConstraints: () => api.get('/v1/optimization/constraints/check'),
  explainAssignment: (trainsetId: string, decision?: string, format?: string) =>
    api.get(`/v1/optimization/explain/${trainsetId}`, {
      params: { decision, format }
    }),
  explainBatch: (assignments: any[], format?: string) =>
    api.post('/v1/optimization/explain/batch', { assignments, format }),
  simulate: (params: any) => api.get('/v1/optimization/simulate', { params }),
  runSimulation: (scenario: any) => api.post('/v1/simulation/run', scenario),
  getSimulationResult: (id: string) => api.get(`/v1/simulation/result/${id}`),
  getSnapshot: () => api.get('/v1/simulation/snapshot'),
  getLatest: () => api.get('/v1/optimization/latest'),
  getStablingGeometry: () => api.get('/v1/optimization/stabling-geometry'),
  getShuntingSchedule: () => api.get('/v1/optimization/shunting-schedule'),
  reorderRankedList: (data: { trainset_ids: string[]; reason?: string }) =>
    api.post('/v1/optimization/latest/reorder', data),
};

// Trainsets API
export const trainsetsApi = {
  getAll: (params?: any) => api.get('/v1/trainsets/', { params }),
  getById: (id: string) => api.get(`/v1/trainsets/${id}`),
  update: (id: string, data: any) => api.put(`/v1/trainsets/${id}`, data),
  getFitness: (id: string) => api.get(`/v1/trainsets/${id}/fitness`),
  getDetails: (id: string) => api.get(`/v1/trainsets/${id}/details`),
  getReviews: () => api.get('/v1/trainsets/reviews/all'),
};

// Data Ingestion API
export const ingestionApi = {
  ingestAll: () => api.post('/v1/ingestion/ingest/all'),
  ingestMaximo: () => api.post('/v1/ingestion/ingest/maximo'),
  ingestIoT: () => api.post('/v1/ingestion/ingest/iot'),
  uploadTimeseries: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/v1/ingestion/ingest/timeseries/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  uploadFitness: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/v1/ingestion/fitness/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  uploadBranding: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/v1/ingestion/branding/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  uploadDepot: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/v1/ingestion/depot/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  ingestCleaningGoogle: (sheetUrl: string) => {
    const formData = new FormData();
    formData.append('sheet_url', sheetUrl);
    return api.post('/v1/ingestion/cleaning/google', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  uploadN8N: (files: File[]) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    return api.post('/v1/ingestion/ingest/n8n/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  getStatus: () => api.get('/v1/ingestion/status'),
  startMQTT: () => api.post('/v1/ingestion/mqtt/start'),
  stopMQTT: () => api.post('/v1/ingestion/mqtt/stop'),
  getMQTTStatus: () => api.get('/v1/ingestion/mqtt/status'),
};

// Notifications API
export const notificationsApi = {
  getAll: () => api.get('/v1/notifications'),
  markAsRead: (id: string) => api.put(`/v1/notifications/${id}/read`),
  markAllAsRead: () => api.put('/v1/notifications/read-all'),
};

// Auth API
export const authApi = {
  register: (data: {
    username: string;
    password: string;
    name: string;
    email?: string;
    role: string;
  }) => api.post('/v1/auth/register', data),
  login: (credentials: { username: string; password: string }) =>
    api.post('/v1/auth/login', credentials),
  logout: () => api.post('/v1/auth/logout'),
  getProfile: () => api.get('/v1/auth/profile'),
  refreshToken: () => api.post('/v1/auth/refresh-token'),
  changePassword: (data: { current_password: string; new_password: string }) =>
    api.post('/v1/auth/change-password', data),
};

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
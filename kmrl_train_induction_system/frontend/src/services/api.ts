import axios from 'axios'
import { 
  Trainset, 
  InductionDecision, 
  Assignment, 
  OverrideRequest, 
  ApprovalRequest, 
  Alert, 
  DashboardOverview 
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': import.meta.env.VITE_API_KEY || 'dev-key',
  },
})

// Request interceptor for auth
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export const trainsetApi = {
  getAll: (): Promise<Trainset[]> => 
    api.get('/api/trainsets').then(res => res.data),
  
  getById: (id: string): Promise<Trainset> => 
    api.get(`/api/trainsets/${id}`).then(res => res.data),
  
  update: (id: string, updates: Partial<Trainset>): Promise<void> => 
    api.put(`/api/trainsets/${id}`, { updates }).then(res => res.data),
  
  getFitness: (id: string): Promise<any> => 
    api.get(`/api/trainsets/${id}/fitness`).then(res => res.data),
}

export const optimizationApi = {
  runOptimization: (request: any): Promise<InductionDecision[]> => 
    api.post('/api/optimization/run', request).then(res => res.data),
  
  getHistory: (): Promise<any[]> => 
    api.get('/api/optimization/history').then(res => res.data),
}

export const assignmentApi = {
  getAll: (): Promise<Assignment[]> => 
    api.get('/api/v1/assignments').then(res => res.data),
  
  approve: (request: ApprovalRequest): Promise<void> => 
    api.post('/api/v1/assignments/approve', request).then(res => res.data),
  
  override: (request: OverrideRequest): Promise<void> => 
    api.post('/api/v1/assignments/override', request).then(res => res.data),
  
  getById: (id: string): Promise<Assignment> => 
    api.get(`/api/v1/assignments/${id}`).then(res => res.data),
}

export const dashboardApi = {
  getOverview: (): Promise<DashboardOverview> => 
    api.get('/api/dashboard/overview').then(res => res.data),
  
  getAlerts: (): Promise<{ alerts: Alert[]; total_alerts: number }> => 
    api.get('/api/dashboard/alerts').then(res => res.data),
  
  getPerformance: (): Promise<any> => 
    api.get('/api/dashboard/performance').then(res => res.data),
}

export const reportsApi = {
  generateDailyBriefing: (): Promise<Blob> => 
    api.get('/api/v1/reports/daily-briefing', { responseType: 'blob' }).then(res => res.data),
  
  exportAssignments: (format: 'csv' | 'pdf'): Promise<Blob> => 
    api.get(`/api/v1/reports/assignments?format=${format}`, { responseType: 'blob' }).then(res => res.data),
  
  exportAuditLogs: (): Promise<Blob> => 
    api.get('/api/v1/reports/audit-logs', { responseType: 'blob' }).then(res => res.data),
}

export const authApi = {
  login: (credentials: { email: string; password: string }): Promise<{ token: string; user: any }> => 
    api.post('/api/v1/auth/login', credentials).then(res => res.data),
  
  logout: (): Promise<void> => 
    api.post('/api/v1/auth/logout').then(res => res.data),
  
  getProfile: (): Promise<any> => 
    api.get('/api/v1/auth/profile').then(res => res.data),
}

export default api

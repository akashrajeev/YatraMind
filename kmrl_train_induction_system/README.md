# KMRL Operations Decision Support Platform

A comprehensive AI/ML-driven decision support platform for Kochi Metro Rail Ltd (KMRL) train induction planning and operations management.

## 🚀 Features

### I. Workflow & Approvals (UI + API)
- **Ranked Induction List**: Sortable and filterable dashboard showing train induction assignments
- **Detail Panels**: Comprehensive view of each rake with certificates, job cards, telemetry snapshots, predicted risk, and suggested fixes
- **Conflict Alerts**: Real-time alerts for critical issues requiring immediate attention
- **Manual Override Actions**: Supervisor override capabilities with role-based approval requirements
- **Audit Trail**: Complete logging of all override actions and approvals

### J. Reports & Downloads
- **PDF Exports**: Daily briefing reports, assignment summaries, and performance analysis
- **CSV Exports**: Data exports for assignments, audit logs, and fleet status
- **Daily Briefing**: Automated generation of comprehensive daily operational reports
- **Custom Reports**: Pre-configured templates for maintenance, performance, and compliance reports

### K. Real-time Updates & Notifications
- **Socket.IO Integration**: Real-time updates for ingestion status, optimization progress, and system alerts
- **Multi-channel Notifications**: Email, SMS, and in-app notifications for critical alerts
- **Firebase Cloud Messaging**: Push notifications for mobile users
- **MongoDB Persistence**: All notifications and alerts are logged for audit purposes

### L. Authentication & Security
- **JWT Authentication**: Secure token-based authentication with configurable expiration
- **Role-Based Access Control (RBAC)**: 
  - Supervisor: Can approve and override assignments
  - Maintenance Engineer: Can view and edit trainset information
  - Operations Manager: Full system access
  - Read-only Viewer: View-only access
- **Audit Logging**: Complete audit trail of all user actions with IP tracking
- **TLS Security**: Encrypted communication for all endpoints
- **Environment-based Configuration**: Secure secrets management

## 🏗️ Architecture

### Backend (FastAPI + Python)
- **FastAPI**: High-performance async web framework
- **MongoDB**: Primary database for operational data
- **InfluxDB**: Time-series data for sensor metrics
- **Redis**: Caching and Celery task queue
- **Socket.IO**: Real-time communication
- **ReportLab**: PDF generation
- **OR-Tools**: Optimization algorithms
- **PyTorch/TensorFlow**: ML models for risk prediction

### Frontend (React + TypeScript)
- **React 18**: Modern UI framework
- **Material-UI**: Professional component library
- **Formik + Yup**: Form handling and validation
- **Socket.IO Client**: Real-time updates
- **Recharts**: Data visualization
- **jsPDF**: Client-side PDF generation
- **React Query**: Data fetching and caching

## 📁 Project Structure

```
kmrl_train_induction_system/
├── backend/
│   ├── app/
│   │   ├── api/                 # API endpoints
│   │   │   ├── assignments.py   # Assignment management
│   │   │   ├── auth.py         # Authentication
│   │   │   ├── dashboard.py    # Dashboard data
│   │   │   ├── reports.py      # Report generation
│   │   │   └── ...
│   │   ├── models/             # Data models
│   │   │   ├── assignment.py   # Assignment models
│   │   │   ├── audit.py        # Audit logging
│   │   │   ├── notification.py # Notifications
│   │   │   └── user.py         # User management
│   │   ├── services/           # Business logic
│   │   │   ├── auth_service.py # Authentication
│   │   │   ├── notification_service.py # Notifications
│   │   │   └── report_generator.py # Reports
│   │   └── utils/              # Utilities
│   ├── requirements.txt        # Python dependencies
│   └── main.py                # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API services
│   │   ├── contexts/          # React contexts
│   │   └── types/             # TypeScript types
│   ├── package.json           # Node dependencies
│   └── vite.config.ts         # Vite configuration
└── start_development.py       # Development startup script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB Atlas account
- InfluxDB Cloud account (optional)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd kmrl_train_induction_system
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp env.example .env
# Edit .env with your database credentials
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Start Development Environment
```bash
# From project root
python start_development.py
```

This will start:
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:3000
- API Documentation: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables
Copy `backend/env.example` to `backend/.env` and configure:

```env
# Database
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/kmrl_operations
INFLUXDB_URL=https://your-influxdb-instance.com

# Security
SECRET_KEY=your-jwt-secret-key
API_KEY=your-api-key

# Notifications
FIREBASE_PROJECT_ID=your-firebase-project
SMTP_HOST=smtp.gmail.com
```

### Frontend Configuration
Create `frontend/.env.local`:
```env
VITE_API_URL=http://localhost:8000
VITE_SOCKET_URL=http://localhost:8000
VITE_API_KEY=your-api-key
```

## 📊 API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout
- `GET /api/v1/auth/profile` - Get user profile
- `POST /api/v1/auth/refresh-token` - Refresh access token

### Assignments
- `GET /api/v1/assignments` - Get assignments with filtering
- `POST /api/v1/assignments/approve` - Approve assignments
- `POST /api/v1/assignments/override` - Override assignment decisions
- `GET /api/v1/assignments/summary` - Get assignment statistics

### Reports
- `GET /api/v1/reports/daily-briefing` - Generate daily briefing PDF
- `GET /api/v1/reports/assignments` - Export assignments (CSV/PDF)
- `GET /api/v1/reports/audit-logs` - Export audit logs (CSV)
- `GET /api/v1/reports/fleet-status` - Fleet status report

### Real-time Updates
- WebSocket connection for real-time updates
- Events: `optimization_update`, `ingestion_update`, `new_alert`, `assignment_updated`

## 🔐 Security Features

### Authentication & Authorization
- JWT-based authentication with configurable expiration
- Role-based access control (RBAC)
- Password hashing with bcrypt
- Session management

### Audit & Compliance
- Complete audit logging of all user actions
- IP address tracking
- Risk level assessment
- Compliance reporting

### Data Protection
- TLS encryption for all communications
- Environment-based secret management
- Input validation and sanitization
- SQL injection prevention

## 📈 Monitoring & Observability

### Metrics
- Prometheus metrics integration
- Performance monitoring
- Error tracking
- Resource utilization

### Logging
- Structured logging with different levels
- Centralized log aggregation
- Error tracking and alerting
- Audit trail maintenance

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Integration Tests
```bash
cd backend
pytest tests/test_e2e.py -v
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Production Configuration
1. Set `ENVIRONMENT=production` in `.env`
2. Configure production databases
3. Set up SSL certificates
4. Configure load balancing
5. Set up monitoring and alerting

## 📚 Documentation

- **API Documentation**: Available at `/docs` when running the backend
- **Component Documentation**: React Storybook (coming soon)
- **Architecture Guide**: See `docs/architecture.md`
- **Deployment Guide**: See `docs/deployment.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request


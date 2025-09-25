# Vercel Deployment Guide

## Quick Deploy to Vercel

### 1. Prerequisites
- GitHub repository with your code
- Vercel account (free tier available)
- MongoDB Atlas cluster
- InfluxDB Cloud instance
- External worker service (Railway/Render) for background tasks

### 2. Deploy API to Vercel

#### Option A: Using Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from backend directory
cd backend
vercel

# Set environment variables
vercel env add API_KEY
vercel env add MONGODB_URL
vercel env add WORKER_SERVICE_URL
vercel env add WORKER_API_KEY
# ... etc
```

#### Option B: Using Vercel Dashboard
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project" → "Import Git Repository"
3. Select your repository
4. Set root directory to: `backend/`
5. Vercel will auto-detect Python and use `vercel.json`

### 3. Deploy Workers to External Service

#### Option A: Railway (Recommended)
1. Go to [Railway](https://railway.app)
2. Deploy the original `app/main.py` with Celery workers
3. Get the worker service URL

#### Option B: Render
1. Go to [Render](https://render.com)
2. Deploy as background worker service
3. Get the worker service URL

### 4. Set Environment Variables

In Vercel dashboard, add these environment variables:

```bash
# Required
API_KEY=your_secure_api_key_here
MONGODB_URL=mongodb://username:password@host:port/database
DATABASE_NAME=kmrl_db
INFLUXDB_URL=https://your-influxdb-url
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=your_org_id
INFLUXDB_BUCKET=kmrl_sensor_data

# External Worker Service
WORKER_SERVICE_URL=https://your-worker-service.railway.app
WORKER_API_KEY=your_worker_api_key

# Optional
ENVIRONMENT=production
DEBUG=false
CLEANING_SHEET_URL=https://docs.google.com/spreadsheets/d/your_sheet_id
```

### 5. Test Deployment

```bash
# Check API health
curl https://your-app.vercel.app/health

# Check API docs
https://your-app.vercel.app/docs

# Test optimization endpoint
curl -X POST https://your-app.vercel.app/api/optimization/run \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"target_date": "2024-01-01T00:00:00Z", "required_service_hours": 14}'

# Test task management (calls external worker)
curl -X POST https://your-app.vercel.app/tasks/optimization/run \
  -H "X-API-Key: your_api_key"
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Vercel API    │    │  External       │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│  Worker Service │
│                 │    │                 │    │  (Railway)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Databases     │
                       │   (MongoDB,     │
                       │    InfluxDB)    │
                       └─────────────────┘
```

## What Works on Vercel

✅ **API Endpoints** - All FastAPI routes
✅ **Health Checks** - Database connectivity
✅ **Data Retrieval** - MongoDB/InfluxDB queries
✅ **Optimization Requests** - Short optimization runs
✅ **File Uploads** - Data ingestion endpoints
✅ **ML Inference** - Model predictions
✅ **External API Calls** - HTTP requests to worker service

## What Requires External Service

❌ **Background Workers** - Celery tasks
❌ **Scheduled Tasks** - APScheduler
❌ **Long ML Training** - Model training
❌ **Real-time Streaming** - MQTT client

## File Structure

```
backend/
├── api/
│   └── index.py              # Vercel entry point
├── app/
│   ├── vercel_main.py        # Vercel-optimized main
│   └── main.py               # Original (for workers)
├── vercel.json               # Vercel configuration
├── vercel.env.example        # Environment variables template
├── requirements.txt          # Python dependencies
└── README_VERCEL.md          # This guide
```

## Troubleshooting

### Common Issues

1. **Worker service unavailable**: Check WORKER_SERVICE_URL and WORKER_API_KEY
2. **Database connection failed**: Verify MongoDB and InfluxDB URLs
3. **Timeout errors**: Vercel has 30-second limit for functions
4. **Import errors**: Check PYTHONPATH is set correctly

### Logs

View logs in Vercel dashboard:
- Function logs
- Build logs
- Runtime logs

## Cost

- **Free tier**: 100GB bandwidth, 100GB-hours execution
- **Pro tier**: $20/month for unlimited usage
- **External worker**: Additional cost (Railway/Render)

## Scaling

- **Vercel**: Auto-scales based on traffic
- **Worker service**: Manual scaling in external platform
- **Database**: Use managed services (MongoDB Atlas, InfluxDB Cloud)

# Railway Deployment Guide

## Quick Deploy to Railway

### 1. Prerequisites
- GitHub repository with your code
- Railway account (free tier available)
- MongoDB Atlas cluster
- InfluxDB Cloud instance
- Redis instance (for Celery)

### 2. Deploy Web Service

1. Go to [Railway](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Choose the `backend` folder as root directory
5. Railway will auto-detect Python and install dependencies

### 3. Deploy Worker Service

1. In the same project, click "New Service" → "GitHub Repo"
2. Select the same repository and `backend` folder
3. Change the start command to:
   ```
   celery -A app.celery_app.celery_app worker -l info
   ```

### 4. Add Redis Service

1. In Railway dashboard, click "New Service" → "Database" → "Add Redis"
2. Railway will provide a `REDIS_URL` environment variable

### 5. Set Environment Variables

In each service (web + worker), add these environment variables:

```bash
# Required
API_KEY=your_secure_api_key_here
MONGODB_URL=mongodb://username:password@host:port/database
DATABASE_NAME=kmrl_db
INFLUXDB_URL=https://your-influxdb-url
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=your_org_id
INFLUXDB_BUCKET=kmrl_sensor_data
REDIS_URL=redis://username:password@host:port

# Optional
ENVIRONMENT=production
DEBUG=false
CLEANING_SHEET_URL=https://docs.google.com/spreadsheets/d/your_sheet_id
```

### 6. Health Checks

Railway will automatically check:
- Web service: `GET /health`
- Worker service: Celery worker process

### 7. Access Your API

Once deployed, Railway provides:
- Web service URL: `https://your-app.railway.app`
- API docs: `https://your-app.railway.app/docs`
- Health check: `https://your-app.railway.app/health`

## File Structure

```
backend/
├── Procfile                 # Process definitions
├── railway.json            # Railway configuration
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── railway.env.example     # Environment variables template
└── app/                    # Application code
```

## Troubleshooting

### Common Issues

1. **Worker not starting**: Check Redis URL is set correctly
2. **Database connection failed**: Verify MongoDB URL format
3. **Task status 500**: Ensure Redis is accessible from worker
4. **Health check failing**: Check all required env vars are set

### Logs

View logs in Railway dashboard:
- Web service logs
- Worker service logs
- Build logs

### Scaling

- Web service: Railway auto-scales based on traffic
- Worker service: Manually scale in Railway dashboard
- Database: Use Railway's database services or external providers

## Cost

- Free tier: $5 credit monthly
- Web service: ~$0.10/hour
- Worker service: ~$0.10/hour
- Redis: ~$0.05/hour
- Total: ~$15-20/month for small usage

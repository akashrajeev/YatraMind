# 🚀 KMRL Train Induction System - Cloud Services Setup Guide

This guide will help you configure the KMRL system to use real cloud services instead of mock data.

## 📋 Prerequisites

- Python 3.11+ installed
- Active internet connection
- Cloud service accounts (MongoDB Atlas, InfluxDB Cloud, Redis Cloud)

---

## 🔧 Step 1: Set Up Cloud Services

### 1.1 MongoDB Atlas Setup
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free account or sign in
3. Create a new cluster
4. Create a database user with read/write permissions
5. Get your connection string from "Connect" → "Connect your application"
6. Whitelist your IP address (or use 0.0.0.0/0 for development)

### 1.2 InfluxDB Cloud Setup
1. Go to [InfluxDB Cloud](https://cloud2.influxdata.com/)
2. Create a free account or sign in
3. Create a new organization
4. Create a new bucket (e.g., "kmrl_sensor_data")
5. Generate an API token with read/write permissions
6. Note your organization ID and bucket name

### 1.3 Redis Cloud Setup
1. Go to [Redis Cloud](https://redis.com/redis-enterprise-cloud/overview/)
2. Create a free account or sign in
3. Create a new database
4. Get your connection string from the database details
5. Note your host, port, username, and password

### 1.4 MQTT Broker Setup (Optional)
For real-time IoT data streaming:
1. Use a free MQTT broker like [HiveMQ Cloud](https://www.hivemq.com/mqtt-cloud-broker/)
2. Or use [Eclipse Mosquitto](https://mosquitto.org/) for local testing
3. Note your broker host, port, and credentials

---

## 🔑 Step 2: Configure Environment Variables

1. Copy the environment template:
   ```bash
   copy env_template.txt .env
   ```

2. Edit the `.env` file with your actual credentials:

   ```env
   # MongoDB Atlas
   MONGODB_URL=mongodb+srv://your_username:your_password@your_cluster.mongodb.net/kmrl_db?retryWrites=true&w=majority
   DATABASE_NAME=kmrl_db

   # InfluxDB Cloud
   INFLUXDB_URL=https://us-west-2-1.aws.cloud2.influxdata.com
   INFLUXDB_TOKEN=your_actual_token_here
   INFLUXDB_ORG=your_org_id
   INFLUXDB_BUCKET=kmrl_sensor_data

   # Redis Cloud
   REDIS_URL=redis://your_username:your_password@your_redis_host:port

   # MQTT Broker
   MQTT_BROKER=your_mqtt_broker_host
   MQTT_BROKER_HOST=your_mqtt_broker_host
   MQTT_BROKER_PORT=1883
   MQTT_USERNAME=your_mqtt_username
   MQTT_PASSWORD=your_mqtt_password
   ```

---

## 🧪 Step 3: Test Cloud Connections

Run the cloud services setup script:

```bash
py -3.11 setup_cloud_services.py
```

This script will:
- ✅ Test MongoDB Atlas connection
- ✅ Test InfluxDB Cloud connection  
- ✅ Test Redis Cloud connection
- ✅ Test MQTT broker connection
- ✅ Load production data if all connections succeed

---

## 🚀 Step 4: Start Production Server

Once all connections are successful:

```bash
py -3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server will now use real cloud services instead of mock data.

---

## 📊 Step 5: Verify Production Data

Test the API endpoints to verify data is loaded:

```bash
# Test the API
py -3.11 test_api.py

# Or use the Swagger UI
# Open: http://localhost:8000/docs
```

---

## 🔍 Troubleshooting

### Common Issues:

1. **MongoDB Connection Failed**
   - Check your connection string format
   - Verify IP whitelist includes your IP
   - Ensure database user has proper permissions

2. **InfluxDB Connection Failed**
   - Verify your API token is correct
   - Check organization ID and bucket name
   - Ensure token has read/write permissions

3. **Redis Connection Failed**
   - Check your connection string format
   - Verify host, port, username, and password
   - Ensure Redis instance is running

4. **MQTT Connection Failed**
   - Check broker host and port
   - Verify credentials if authentication is required
   - Test with a simple MQTT client first

### Debug Mode:
Set `DEBUG=true` in your `.env` file for detailed error messages.

---

## 🎯 Production Checklist

- [ ] All cloud services connected successfully
- [ ] Production data loaded
- [ ] API endpoints responding correctly
- [ ] Dashboard showing real data
- [ ] Optimization engine working with real data
- [ ] Stabling geometry optimization functional
- [ ] MQTT streaming operational (if configured)

---

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Use strong passwords for all cloud services
- Enable IP whitelisting where possible
- Rotate API tokens regularly
- Use TLS/SSL for production deployments

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your cloud service configurations
3. Check the application logs for detailed error messages
4. Ensure all dependencies are installed correctly

Your KMRL Train Induction System is now ready for production use! 🚇✨

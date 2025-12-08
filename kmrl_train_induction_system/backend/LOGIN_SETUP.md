# Login System Setup Guide

## Overview
A secure login system has been implemented with JWT authentication. The system uses username/password authentication for 6 predefined admin users.

## Predefined Users

The following users have been configured:

1. **adithkp** - Password: `Adith@123`
2. **akashrajeevkv** - Password: `Akash@123`
3. **abindasp** - Password: `Abindas@123`
4. **alanb** - Password: `Alan@123`
5. **pradyodhp** - Password: `Pradyodh@123`
6. **poojacv** - Password: `Pooja@123`

All users have `OPERATIONS_MANAGER` role with full permissions.

## Setup Instructions

### 1. Seed Users into Database

Run the setup script to create the users in MongoDB:

```bash
cd backend
python setup_users.py
```

This script will:
- Connect to MongoDB
- Create all 6 users with hashed passwords
- Skip if users already exist (to prevent duplicates)

### 2. Configure Environment Variables

Make sure your `.env` file has the following:

```env
SECRET_KEY=your-jwt-secret-key-here
MONGODB_URL=your-mongodb-connection-string
```

The `SECRET_KEY` is used for JWT token signing. Use a strong, random secret in production.

### 3. Start the Backend

```bash
cd backend
uvicorn app.main:app --reload
```

### 4. Start the Frontend

```bash
cd frontend
npm install  # If not already done
npm run dev
```

## How It Works

### Backend
- **Login Endpoint**: `POST /api/v1/auth/login`
  - Accepts: `{ "username": "...", "password": "..." }`
  - Returns: JWT token and user information
  - No API key required for login endpoint

- **Protected Endpoints**: All other endpoints require:
  - JWT token in `Authorization: Bearer <token>` header
  - API key in `X-API-Key` header (if configured)

### Frontend
- **Login Page**: First page users see at `/login`
- **Route Protection**: All routes except `/login` require authentication
- **Auto-redirect**: 
  - Unauthenticated users → redirected to `/login`
  - Authenticated users accessing `/login` → redirected to `/`

### Authentication Flow
1. User enters username and password on login page
2. Frontend sends credentials to `/api/v1/auth/login`
3. Backend validates credentials and returns JWT token
4. Frontend stores token in `localStorage`
5. Token is included in all subsequent API requests
6. Token expires after 30 minutes (configurable)

## Security Features

- ✅ Password hashing using bcrypt
- ✅ JWT token-based authentication
- ✅ Secure password storage (never stored in plain text)
- ✅ Token expiration (30 minutes)
- ✅ Route protection on frontend
- ✅ API endpoint protection on backend

## Troubleshooting

### Users not created
- Check MongoDB connection in `.env`
- Ensure `setup_users.py` runs successfully
- Check MongoDB logs for errors

### Login fails
- Verify username and password are correct
- Check backend logs for authentication errors
- Ensure `SECRET_KEY` is set in `.env`
- Check MongoDB connection

### Token not working
- Token may have expired (30 minutes)
- Check `Authorization` header format: `Bearer <token>`
- Verify `SECRET_KEY` matches between token creation and validation

## API Endpoints

- `POST /api/v1/auth/login` - Login (public)
- `POST /api/v1/auth/logout` - Logout (protected)
- `GET /api/v1/auth/profile` - Get user profile (protected)
- `POST /api/v1/auth/refresh-token` - Refresh token (protected)
- `POST /api/v1/auth/change-password` - Change password (protected)


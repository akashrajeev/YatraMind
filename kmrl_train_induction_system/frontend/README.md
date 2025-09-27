# KMRL Operations Dashboard - Frontend

A modern, responsive frontend application for the KMRL Train Induction Management System, built with React, TypeScript, and Tailwind CSS.

## 🚀 Features

### **Modern UI/UX**
- **Shadcn/ui Components** - Professional, accessible UI components
- **Tailwind CSS** - Utility-first styling with industrial theme
- **Responsive Design** - Works seamlessly on all devices
- **Dark/Light Mode Ready** - Built with CSS variables for easy theming

### **Dashboard Features**
- **Real-time Metrics** - Live system monitoring and KPIs
- **Activity Feed** - Real-time updates and user actions
- **System Performance** - Load monitoring, efficiency tracking
- **Quick Actions** - One-click operations for common tasks

### **Assignment Management**
- **Tabbed Interface** - Pending, Approved, and Overridden assignments
- **Status Indicators** - Visual indicators for assignment states
- **Action Buttons** - View, Override, and Approve functionality
- **Priority Management** - High, Medium, Low priority assignments

### **Reports & Analytics**
- **Report Generation** - PDF and CSV export capabilities
- **Performance Analytics** - System performance metrics
- **Compliance Reports** - Safety and regulatory compliance
- **Audit Logs** - Complete audit trail

### **System Settings**
- **Notification Settings** - Email, system, and mobile alerts
- **Security Options** - 2FA, session timeout, audit logging
- **System Configuration** - Auto-backup, monitoring, maintenance mode

## 🛠️ Tech Stack

- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Full type safety throughout the application
- **Vite** - Fast development server and build tool
- **Tailwind CSS** - Utility-first CSS framework
- **Shadcn/ui** - Accessible, customizable component library
- **TanStack Query** - Powerful data synchronization for React
- **Axios** - HTTP client for API requests
- **Lucide React** - Beautiful, customizable icons

## 📦 Installation

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

## 🔧 Configuration

### **API Configuration**
The frontend connects to the backend API. Update the configuration in `src/config/api.ts`:

```typescript
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000/api',
  API_KEY: 'your-api-key-here',
  WS_URL: 'ws://localhost:8000/ws',
  TIMEOUT: 10000,
};
```

### **Environment Variables**
Create a `.env` file in the root directory:

```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_API_KEY=your-api-key-here
VITE_WS_URL=ws://localhost:8000/ws
```

## 🏗️ Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # Base UI components (Shadcn/ui)
│   ├── layout/         # Layout components
│   └── dashboard/      # Dashboard-specific components
├── pages/              # Page components
├── services/           # API services
├── types/              # TypeScript type definitions
├── config/             # Configuration files
├── lib/                # Utility functions
└── hooks/              # Custom React hooks
```

## 🔌 API Integration

### **Dashboard API**
- `GET /dashboard/overview` - Fleet overview and metrics
- `GET /dashboard/alerts` - Real-time alerts and notifications
- `GET /dashboard/performance` - System performance metrics

### **Assignments API**
- `GET /assignments` - List assignments with filtering
- `POST /assignments` - Create new assignment
- `POST /assignments/approve` - Approve assignments
- `POST /assignments/override` - Override assignment decision

### **Reports API**
- `GET /reports/daily-briefing` - Generate daily briefing PDF
- `GET /reports/assignments` - Export assignments (PDF/CSV)
- `GET /reports/fleet-status` - Fleet status report
- `GET /reports/performance-analysis` - Performance analysis

## 🎨 Theming

The application uses a custom industrial theme with CSS variables:

```css
:root {
  --primary: 221.2 83.2% 53.3%;
  --accent: 38 92% 50%;
  --success: 142.1 76.2% 36.3%;
  --warning: 38 92% 50%;
  --destructive: 0 84.2% 60.2%;
}
```

## 📱 Responsive Design

The application is fully responsive with breakpoints:
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

## 🔒 Security

- **API Key Authentication** - All requests include API key
- **Request/Response Interceptors** - Centralized error handling
- **Type Safety** - Full TypeScript coverage
- **Input Validation** - Client-side validation

## 🚀 Deployment

### **Build for Production**
```bash
npm run build
```

### **Preview Production Build**
```bash
npm run preview
```

### **Docker Deployment**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

## 📊 Performance

- **Code Splitting** - Automatic code splitting with Vite
- **Tree Shaking** - Unused code elimination
- **Lazy Loading** - Route-based code splitting
- **Caching** - TanStack Query for intelligent caching
- **Optimized Assets** - Image and asset optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

---

**Built with ❤️ for KMRL Operations**

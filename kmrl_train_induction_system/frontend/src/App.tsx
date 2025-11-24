import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "@/components/theme-provider";
import ErrorBoundary from "@/components/ErrorBoundary";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import Dashboard from "./pages/Dashboard";
import Assignments from "./pages/Assignments";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import Trainsets from "./pages/Trainsets";
import Optimization from "./pages/Optimization";
import DataIngestion from "./pages/DataIngestion";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import { LoadingSpinner } from "@/components/LoadingSpinner";

const queryClient = new QueryClient();

// Protected Route Component
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// Public Route Component (redirects to dashboard if already logged in)
const PublicRoute = ({ children }: { children: React.ReactNode }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (user) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

const AppRoutes = () => (
  <Routes>
    <Route path="/login" element={
      <PublicRoute>
        <Login />
      </PublicRoute>
    } />
    <Route path="/" element={
      <ProtectedRoute>
        <DashboardLayout>
          <Dashboard />
        </DashboardLayout>
      </ProtectedRoute>
    } />
    <Route path="/assignments" element={
      <ProtectedRoute>
        <DashboardLayout>
          <Assignments />
        </DashboardLayout>
      </ProtectedRoute>
    } />
    <Route path="/trainsets" element={
      <ProtectedRoute>
        <DashboardLayout>
          <Trainsets />
        </DashboardLayout>
      </ProtectedRoute>
    } />
    <Route path="/optimization" element={
      <ProtectedRoute>
        <DashboardLayout>
          <Optimization />
        </DashboardLayout>
      </ProtectedRoute>
    } />
    <Route path="/data-ingestion" element={
      <ProtectedRoute>
        <DashboardLayout>
          <DataIngestion />
        </DashboardLayout>
      </ProtectedRoute>
    } />
    <Route path="/reports" element={
      <ProtectedRoute>
        <DashboardLayout>
          <Reports />
        </DashboardLayout>
      </ProtectedRoute>
    } />
    <Route path="/settings" element={
      <ProtectedRoute>
        <DashboardLayout>
          <Settings />
        </DashboardLayout>
      </ProtectedRoute>
    } />
    <Route path="*" element={<NotFound />} />
  </Routes>
);

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="system" storageKey="kmrl-ui-theme">
        <AuthProvider>
          <BrowserRouter>
            <AppRoutes />
          </BrowserRouter>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
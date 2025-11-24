import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "@/components/theme-provider";
import ErrorBoundary from "@/components/ErrorBoundary";
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
import { useAuth } from "@/contexts/AuthContext";
import { Toaster } from "@/components/ui/sonner";

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

const withDashboardLayout = (component: React.ReactNode) => (
  <DashboardLayout>{component}</DashboardLayout>
);

const AppRoutes = () => (
  <Routes>
    <Route
      path="/login"
      element={
        <PublicRoute>
          <Login />
        </PublicRoute>
      }
    />
    <Route
      path="/"
      element={
        <ProtectedRoute>
          {withDashboardLayout(<Dashboard />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/assignments"
      element={
        <ProtectedRoute>
          {withDashboardLayout(<Assignments />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/trainsets"
      element={
        <ProtectedRoute>
          {withDashboardLayout(<Trainsets />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/optimization"
      element={
        <ProtectedRoute>
          {withDashboardLayout(<Optimization />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/data-ingestion"
      element={
        <ProtectedRoute>
          {withDashboardLayout(<DataIngestion />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/reports"
      element={
        <ProtectedRoute>
          {withDashboardLayout(<Reports />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/settings"
      element={
        <ProtectedRoute>
          {withDashboardLayout(<Settings />)}
        </ProtectedRoute>
      }
    />
    <Route path="*" element={<NotFound />} />
  </Routes>
);

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="system" storageKey="kmrl-ui-theme">
        <BrowserRouter>
          <AppRoutes />
          <Toaster />
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
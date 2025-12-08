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
import Signup from "./pages/Signup";
import NotFound from "./pages/NotFound";
import SupervisorDashboard from "./pages/SupervisorDashboard";
import DriverDashboard from "./pages/DriverDashboard";
import PassengerDashboard from "./pages/PassengerDashboard";
import ProtectedRoute from "./components/ProtectedRoute";
import { useAuth } from "@/contexts/AuthContext";
import { Toaster } from "@/components/ui/sonner";
import { UserRole } from "./types/auth";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import AdminUsers from "./pages/AdminUsers";

const queryClient = new QueryClient();

// Public Route Component
const PublicRoute = ({ children }: { children: React.ReactNode }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <LoadingSpinner />
      </div>
    );
  }

  if (user) {
    // Redirect based on role if already logged in
    switch (user.role) {
      case UserRole.ADMIN:
      case UserRole.OPERATIONS_MANAGER:
        return <Navigate to="/admin" replace />;
      case UserRole.STATION_SUPERVISOR:
      case UserRole.SUPERVISOR:
        return <Navigate to="/supervisor" replace />;
      case UserRole.METRO_DRIVER:
        return <Navigate to="/driver" replace />;
      case UserRole.PASSENGER:
        return <Navigate to="/passenger" replace />;
      default:
        return <Navigate to="/admin" replace />;
    }
  }

  return <>{children}</>;
};

const withDashboardLayout = (component: React.ReactNode) => (
  <DashboardLayout>{component}</DashboardLayout>
);

const AppRoutes = () => (
  <Routes>
    {/* Public Routes */}
    <Route
      path="/login"
      element={
        <PublicRoute>
          <Login />
        </PublicRoute>
      }
    />
    <Route
      path="/signup"
      element={
        <PublicRoute>
          <Signup />
        </PublicRoute>
      }
    />

    {/* Admin Routes */}
    <Route
      path="/admin"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER]}>
          {withDashboardLayout(<Dashboard />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/"
      element={<Navigate to="/admin" replace />}
    />
    <Route
      path="/assignments"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER]}>
          {withDashboardLayout(<Assignments />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/trainsets"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER, UserRole.STATION_SUPERVISOR, UserRole.SUPERVISOR]}>
          {withDashboardLayout(<Trainsets />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/optimization"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER]}>
          {withDashboardLayout(<Optimization />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/data-ingestion"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER]}>
          {withDashboardLayout(<DataIngestion />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/reports"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER, UserRole.STATION_SUPERVISOR, UserRole.SUPERVISOR]}>
          {withDashboardLayout(<Reports />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/settings"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER]}>
          {withDashboardLayout(<Settings />)}
        </ProtectedRoute>
      }
    />
    <Route
      path="/users"
      element={
        <ProtectedRoute allowedRoles={[UserRole.ADMIN, UserRole.OPERATIONS_MANAGER]}>
          {withDashboardLayout(<AdminUsers />)}
        </ProtectedRoute>
      }
    />

    {/* Supervisor Routes */}
    <Route
      path="/supervisor"
      element={
        <ProtectedRoute allowedRoles={[UserRole.STATION_SUPERVISOR, UserRole.SUPERVISOR]}>
          {withDashboardLayout(<SupervisorDashboard />)}
        </ProtectedRoute>
      }
    />

    {/* Driver Routes */}
    <Route
      path="/driver"
      element={
        <ProtectedRoute allowedRoles={[UserRole.METRO_DRIVER]}>
          {withDashboardLayout(<DriverDashboard />)}
        </ProtectedRoute>
      }
    />

    {/* Passenger Routes */}
    <Route
      path="/passenger"
      element={
        <ProtectedRoute allowedRoles={[UserRole.PASSENGER]}>
          <PassengerDashboard />
        </ProtectedRoute>
      }
    />

    <Route path="*" element={<NotFound />} />
  </Routes>
);

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="kmrl-ui-theme">
        <BrowserRouter>
          <AppRoutes />
          <Toaster />
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
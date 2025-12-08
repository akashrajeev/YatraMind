import React, { ReactNode } from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { UserRole } from '../types/auth';

interface ProtectedRouteProps {
    allowedRoles?: UserRole[];
    children?: ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ allowedRoles, children }) => {
    const { user, isAuthenticated, loading } = useAuth();

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    if (!isAuthenticated || !user) {
        return <Navigate to="/login" replace />;
    }

    if (allowedRoles && !allowedRoles.includes(user.role)) {
        // Redirect to appropriate dashboard based on role
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
                return <Navigate to="/" replace />;
        }
    }

    // If children are provided, render them; else fallback to Outlet for nested routes
    return <>{children ?? <Outlet />}</>;
};

export default ProtectedRoute;

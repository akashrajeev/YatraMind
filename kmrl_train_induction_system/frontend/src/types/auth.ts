export enum UserRole {
    ADMIN = 'ADMIN',
    STATION_SUPERVISOR = 'STATION_SUPERVISOR',
    METRO_DRIVER = 'METRO_DRIVER',
    PASSENGER = 'PASSENGER',
    // Legacy roles
    SUPERVISOR = 'SUPERVISOR',
    OPERATIONS_MANAGER = 'OPERATIONS_MANAGER',
    MAINTENANCE_ENGINEER = 'MAINTENANCE_ENGINEER',
    MAINTENANCE_HEAD = 'MAINTENANCE_HEAD',
    BRANDING_DEALER = 'BRANDING_DEALER',
    READONLY_VIEWER = 'READONLY_VIEWER'
}

export interface User {
    id: string;
    username: string;
    name: string;
    email?: string;
    role: UserRole;
    permissions: string[];
    is_active: boolean;
    is_approved: boolean;
}

export interface LoginResponse {
    access_token: string;
    token_type: string;
    user: User;
}

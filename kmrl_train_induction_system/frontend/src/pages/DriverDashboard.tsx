import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, Gauge, Power } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useAuth } from '../contexts/AuthContext';
import { ThemeToggle } from "@/components/theme-toggle";

const DriverDashboard = () => {
    const [emergencyActive, setEmergencyActive] = useState(false);
    const { logout } = useAuth();
    const navigate = useNavigate();

    const handleEmergency = async () => {
        setEmergencyActive(!emergencyActive);
        if (!emergencyActive) {
            try {
                // In a real app, this would be an API call
                // await api.post('/notifications/emergency', { type: 'SOS', source: 'DRIVER' });
                toast.error("EMERGENCY SOS SIGNAL SENT! Admins and Supervisors Notified.");
            } catch (error) {
                console.error("Failed to send SOS", error);
                toast.error("Failed to send SOS signal");
            }
        } else {
            toast.info("Emergency signal cancelled.");
        }
    };

    const handleLogout = async () => {
        try {
            await logout();
            navigate('/login');
        } catch (error) {
            console.error('Logout error:', error);
        }
    };

    return (
        <div className="min-h-screen bg-background p-6 transition-colors duration-300">
            <div className="max-w-4xl mx-auto space-y-8">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-foreground">Metro Driver Operations</h1>
                        <div className="flex items-center mt-2">
                            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-900/20 text-green-600 dark:text-green-400 border border-green-900/30">
                                <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
                                System Online
                            </span>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <ThemeToggle />
                        <Button
                            variant="ghost"
                            onClick={handleLogout}
                            className="text-muted-foreground hover:text-foreground"
                        >
                            Sign Out
                        </Button>
                    </div>
                </div>

                <div className="grid grid-cols-1 gap-6">
                    {/* Dashboard Overview Placeholder */}
                    <Card className="bg-card border-border shadow-sm">
                        <CardHeader>
                            <CardTitle className="text-card-foreground flex items-center">
                                <Gauge className="mr-2 h-5 w-5 text-blue-500" />
                                Dashboard Overview
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-center py-12 text-muted-foreground">
                                <p>Main Dashboard View</p>
                                <p className="text-sm mt-2">(Speed, Next Station, and Schedule details would appear here)</p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-destructive/10 border-destructive/20 shadow-sm">
                        <CardHeader>
                            <CardTitle className="text-destructive flex items-center">
                                <AlertTriangle className="mr-2 h-5 w-5" />
                                Emergency Controls
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-muted-foreground mb-6">
                                Use only in case of critical system failure or sudden maintenance.
                                This will immediately notify Admins and Supervisors.
                            </p>
                            <Button
                                variant={emergencyActive ? "destructive" : "default"}
                                className={`w-full h-32 text-2xl font-bold transition-all duration-200 ${emergencyActive
                                    ? "bg-red-600 hover:bg-red-700 animate-pulse shadow-[0_0_20px_rgba(220,38,38,0.5)]"
                                    : "bg-red-900/80 hover:bg-red-800 border-2 border-red-600 shadow-lg text-white"
                                    }`}
                                onClick={handleEmergency}
                            >
                                <Power className="mr-4 h-12 w-12" />
                                {emergencyActive ? "SOS ACTIVE - TAP TO CANCEL" : "EMERGENCY SOS"}
                            </Button>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
};

export default DriverDashboard;

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, Gauge, Power } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useAuth } from '../contexts/AuthContext';

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
        <div className="min-h-screen bg-gray-950 p-6">
            <div className="max-w-4xl mx-auto space-y-8">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-white">Metro Driver Operations</h1>
                        <div className="flex items-center mt-2">
                            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-900 text-green-300">
                                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                System Online
                            </span>
                        </div>
                    </div>
                    <Button
                        variant="ghost"
                        onClick={handleLogout}
                        className="text-gray-400 hover:text-white"
                    >
                        Sign Out
                    </Button>
                </div>

                <div className="grid grid-cols-1 gap-6">
                    {/* Dashboard Overview Placeholder - As requested "There should be the dashboard there" */}
                    <Card className="bg-gray-900 border-gray-800">
                        <CardHeader>
                            <CardTitle className="text-white flex items-center">
                                <Gauge className="mr-2 h-5 w-5 text-blue-400" />
                                Dashboard Overview
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-center py-12 text-gray-500">
                                <p>Main Dashboard View</p>
                                <p className="text-sm mt-2">(Speed, Next Station, and Schedule details would appear here)</p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-red-950/30 border-red-900/50">
                        <CardHeader>
                            <CardTitle className="text-red-400 flex items-center">
                                <AlertTriangle className="mr-2 h-5 w-5" />
                                Emergency Controls
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-gray-400 mb-6">
                                Use only in case of critical system failure or sudden maintenance.
                                This will immediately notify Admins and Supervisors.
                            </p>
                            <Button
                                variant={emergencyActive ? "destructive" : "default"}
                                className={`w-full h-32 text-2xl font-bold transition-all duration-200 ${emergencyActive
                                    ? "bg-red-600 hover:bg-red-700 animate-pulse shadow-[0_0_20px_rgba(220,38,38,0.5)]"
                                    : "bg-red-900/50 hover:bg-red-800 border-2 border-red-600 shadow-lg"
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

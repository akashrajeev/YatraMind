import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, Gauge, Power } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const DriverDashboard = () => {
    const [emergencyActive, setEmergencyActive] = useState(false);

    const handleEmergency = () => {
        setEmergencyActive(!emergencyActive);
        if (!emergencyActive) {
            toast.error("EMERGENCY SIGNAL SENT! Operations Center Notified.");
        } else {
            toast.info("Emergency signal cancelled.");
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-white">Metro Driver Operations</h1>
                <div className="flex items-center space-x-2">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-900 text-green-300">
                        <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                        System Online
                    </span>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                        <CardTitle className="text-gray-400">Train Status</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-center justify-between mb-4">
                            <span className="text-gray-400">Speed</span>
                            <span className="text-2xl font-bold text-white">45 km/h</span>
                        </div>
                        <div className="flex items-center justify-between mb-4">
                            <span className="text-gray-400">Next Station</span>
                            <span className="text-xl font-bold text-blue-400">Aluva</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-gray-400">Schedule Deviation</span>
                            <span className="text-green-500 font-medium">+0:00 (On Time)</span>
                        </div>
                    </CardContent>
                </Card>

                <Card className="bg-red-900/20 border-red-900/50">
                    <CardHeader>
                        <CardTitle className="text-red-400 flex items-center">
                            <AlertTriangle className="mr-2 h-5 w-5" />
                            Emergency Controls
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="text-gray-400 mb-6">
                            Use only in case of critical system failure or immediate danger.
                        </p>
                        <Button
                            variant={emergencyActive ? "destructive" : "default"}
                            className={`w-full h-24 text-xl font-bold ${emergencyActive
                                    ? "bg-red-600 hover:bg-red-700 animate-pulse"
                                    : "bg-red-900/50 hover:bg-red-800 border-2 border-red-600"
                                }`}
                            onClick={handleEmergency}
                        >
                            <Power className="mr-3 h-8 w-8" />
                            {emergencyActive ? "EMERGENCY ACTIVE - TAP TO CANCEL" : "EMERGENCY STOP"}
                        </Button>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default DriverDashboard;

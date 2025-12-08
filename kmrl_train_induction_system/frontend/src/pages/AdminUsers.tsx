import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle, User, Clock } from "lucide-react";
import api from '../services/api';
import { toast } from "sonner";
import { User as UserType } from '../types/auth';

const AdminUsers = () => {
    const [pendingUsers, setPendingUsers] = useState<UserType[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchPendingUsers();
    }, []);

    const fetchPendingUsers = async () => {
        try {
            const response = await api.get('/v1/users/pending');
            setPendingUsers(response.data);
        } catch (error) {
            console.error('Error fetching pending users:', error);
            toast.error("Failed to load pending users");
        } finally {
            setLoading(false);
        }
    };

    const handleApprove = async (userId: string) => {
        try {
            await api.post(`/v1/users/${userId}/approve`);
            toast.success("User approved successfully");
            setPendingUsers(pendingUsers.filter(u => u.id !== userId));
        } catch (error) {
            console.error('Error approving user:', error);
            toast.error("Failed to approve user");
        }
    };

    const handleReject = async (userId: string) => {
        if (!window.confirm("Are you sure you want to reject and delete this user?")) return;

        try {
            await api.post(`/v1/users/${userId}/reject`);
            toast.success("User rejected successfully");
            setPendingUsers(pendingUsers.filter(u => u.id !== userId));
        } catch (error) {
            console.error('Error rejecting user:', error);
            toast.error("Failed to reject user");
        }
    };

    if (loading) {
        return <div className="text-white">Loading...</div>;
    }

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-white">User Management</h1>
            </div>

            <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                    <CardTitle className="text-white flex items-center">
                        <Clock className="mr-2 h-5 w-5 text-yellow-500" />
                        Pending Approvals
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    {pendingUsers.length === 0 ? (
                        <p className="text-gray-400 text-center py-8">No pending user approvals.</p>
                    ) : (
                        <div className="space-y-4">
                            {pendingUsers.map((user) => (
                                <div
                                    key={user.id}
                                    className="flex items-center justify-between bg-gray-900/50 p-4 rounded-lg border border-gray-700"
                                >
                                    <div className="flex items-center space-x-4">
                                        <div className="bg-gray-700 p-2 rounded-full">
                                            <User className="h-6 w-6 text-gray-300" />
                                        </div>
                                        <div>
                                            <h3 className="text-white font-medium">{user.name}</h3>
                                            <p className="text-sm text-gray-400">@{user.username} • {user.role}</p>
                                            {user.email && <p className="text-xs text-gray-500">{user.email}</p>}
                                        </div>
                                    </div>
                                    <div className="flex space-x-2">
                                        <Button
                                            size="sm"
                                            className="bg-green-600 hover:bg-green-700"
                                            onClick={() => handleApprove(user.id)}
                                        >
                                            <CheckCircle className="h-4 w-4 mr-1" />
                                            Approve
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="destructive"
                                            onClick={() => handleReject(user.id)}
                                        >
                                            <XCircle className="h-4 w-4 mr-1" />
                                            Reject
                                        </Button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

export default AdminUsers;

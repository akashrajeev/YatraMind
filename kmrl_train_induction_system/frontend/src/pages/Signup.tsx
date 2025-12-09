import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { UserRole } from '../types/auth';
import { Lock, User, Mail, Briefcase, AlertCircle, CheckCircle } from 'lucide-react';
import { authApi } from '../services/api';

const Signup: React.FC = () => {
    const [formData, setFormData] = useState({
        username: '',
        password: '',
        name: '',
        email: '',
        role: UserRole.PASSENGER
    });
    const [otp, setOtp] = useState('');
    const [userId, setUserId] = useState<string | null>(null);
    const [showOtp, setShowOtp] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState(false);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleVerifyOtp = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            if (!userId) {
                setError("User ID missing. Please register again.");
                return;
            }
            await authApi.verifyEmail({ user_id: userId, otp });
            setSuccess(true);
            setLoading(false);
        } catch (err: any) {
            console.error('OTP Verification error:', err);
            let errorMessage = 'Verification failed. Please check your code.';
            const detail = err.response?.data?.detail;
            if (detail) errorMessage = typeof detail === 'string' ? detail : JSON.stringify(detail);
            setError(errorMessage);
            setLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const response = await authApi.register(formData);
            // Check if response has data property (axios) or is the data itself
            const userData = response.data || response;

            // If needs verification (Supervisor/Driver), show OTP screen
            if (formData.role !== UserRole.PASSENGER) {
                setUserId(userData.id);
                setShowOtp(true);
            } else {
                setSuccess(true);
                setTimeout(() => navigate('/login'), 2000);
            }
        } catch (err: any) {
            console.error('Signup error:', err);
            let errorMessage = 'Registration failed. Please try again.';
            const detail = err.response?.data?.detail;

            if (detail) {
                if (typeof detail === 'string') {
                    errorMessage = detail;
                } else if (Array.isArray(detail)) {
                    // Handle Pydantic validation errors
                    errorMessage = detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ');
                } else if (typeof detail === 'object') {
                    errorMessage = JSON.stringify(detail);
                }
            }
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    if (success) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center px-4">
                <div className="max-w-md w-full bg-gray-800 rounded-lg shadow-xl p-8 border border-gray-700 text-center">
                    <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-white mb-2">Registration Successful!</h2>
                    <p className="text-gray-300 mb-6">
                        {formData.role === UserRole.PASSENGER
                            ? "Your account has been created. Redirecting to login..."
                            : "Your account has been created and is pending admin approval. You will be notified once approved."}
                    </p>
                    <Link
                        to="/login"
                        className="inline-block bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 transition-colors"
                    >
                        Back to Login
                    </Link>
                </div>
            </div>
        );
    }

    if (showOtp) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center px-4 py-12">
                <div className="max-w-md w-full bg-gray-800 rounded-lg shadow-xl p-8 border border-gray-700">
                    <div className="text-center mb-8">
                        <h1 className="text-3xl font-bold text-white mb-2">Verify Email</h1>
                        <p className="text-gray-400">Enter the 6-digit code sent to {formData.email}</p>
                    </div>

                    {error && (
                        <div className="bg-red-900/50 border border-red-500 text-red-200 p-4 rounded-md mb-6 flex items-center">
                            <AlertCircle className="w-5 h-5 mr-2" />
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleVerifyOtp} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">
                                Verification Code
                            </label>
                            <div className="relative">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Lock className="h-5 w-5 text-gray-500" />
                                </div>
                                <input
                                    type="text"
                                    name="otp"
                                    value={otp}
                                    onChange={(e) => setOtp(e.target.value)}
                                    className="block w-full pl-10 bg-gray-700 border border-gray-600 rounded-md py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 tracking-widest text-center text-xl"
                                    placeholder="000000"
                                    required
                                    maxLength={6}
                                    minLength={6}
                                />
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors mt-6"
                        >
                            {loading ? 'Verifying...' : 'Verify Email'}
                        </button>
                    </form>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center px-4 py-12">
            <div className="max-w-md w-full bg-gray-800 rounded-lg shadow-xl p-8 border border-gray-700">
                <div className="text-center mb-8">
                    <h1 className="text-3xl font-bold text-white mb-2">Create Account</h1>
                    <p className="text-gray-400">Join KMRL Induction System</p>
                </div>

                {error && (
                    <div className="bg-red-900/50 border border-red-500 text-red-200 p-4 rounded-md mb-6 flex items-center">
                        <AlertCircle className="w-5 h-5 mr-2" />
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">
                            Full Name
                        </label>
                        <div className="relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <User className="h-5 w-5 text-gray-500" />
                            </div>
                            <input
                                type="text"
                                name="name"
                                value={formData.name}
                                onChange={handleChange}
                                className="block w-full pl-10 bg-gray-700 border border-gray-600 rounded-md py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="John Doe"
                                required
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">
                            Username
                        </label>
                        <div className="relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <User className="h-5 w-5 text-gray-500" />
                            </div>
                            <input
                                type="text"
                                name="username"
                                value={formData.username}
                                onChange={handleChange}
                                className="block w-full pl-10 bg-gray-700 border border-gray-600 rounded-md py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="johndoe"
                                required
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">
                            Email (Optional)
                        </label>
                        <div className="relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <Mail className="h-5 w-5 text-gray-500" />
                            </div>
                            <input
                                type="email"
                                name="email"
                                value={formData.email}
                                onChange={handleChange}
                                className="block w-full pl-10 bg-gray-700 border border-gray-600 rounded-md py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="john@example.com"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">
                            Password
                        </label>
                        <div className="relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <Lock className="h-5 w-5 text-gray-500" />
                            </div>
                            <input
                                type="password"
                                name="password"
                                value={formData.password}
                                onChange={handleChange}
                                className="block w-full pl-10 bg-gray-700 border border-gray-600 rounded-md py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="••••••••"
                                required
                                minLength={8}
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">
                            Role
                        </label>
                        <div className="relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <Briefcase className="h-5 w-5 text-gray-500" />
                            </div>
                            <select
                                name="role"
                                value={formData.role}
                                onChange={handleChange}
                                className="block w-full pl-10 bg-gray-700 border border-gray-600 rounded-md py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value={UserRole.PASSENGER}>Passenger</option>
                                <option value={UserRole.METRO_DRIVER}>Metro Driver</option>
                                <option value={UserRole.STATION_SUPERVISOR}>Station Supervisor</option>
                                <option value={UserRole.MAINTENANCE_HEAD}>Maintenance Head</option>
                                <option value={UserRole.BRANDING_DEALER}>Branding Dealer</option>
                                <option value={UserRole.ADMIN}>Admin</option>
                            </select>
                        </div>
                        {formData.role !== UserRole.PASSENGER && (
                            <p className="text-xs text-yellow-500 mt-1">
                                * Requires admin approval
                            </p>
                        )}
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors mt-6"
                    >
                        {loading ? 'Creating Account...' : 'Sign Up'}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <p className="text-gray-400 text-sm">
                        Already have an account?{' '}
                        <Link to="/login" className="text-blue-400 hover:text-blue-300 font-medium">
                            Sign in
                        </Link>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Signup;

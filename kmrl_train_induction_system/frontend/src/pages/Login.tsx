import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { UserRole } from '../types/auth';
import { Lock, User, AlertCircle, Briefcase } from 'lucide-react';
import { ThemeToggle } from "@/components/theme-toggle";

// Import assets
import slide1 from '../assets/slide1.jpg';
import slide2 from '../assets/slide2.jpg';
import slide3 from '../assets/slide3.jpg';
import slide4 from '../assets/slide4.jpg';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [selectedRole, setSelectedRole] = useState<UserRole>(UserRole.ADMIN);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);

  const { login } = useAuth();
  const navigate = useNavigate();

  const slides = [slide1, slide2, slide3, slide4];

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const user = await login(username, password);

      // Optional: Verify role matches selected role (if strict enforcement needed)
      // For now, we trust the backend user role, but could warn if mismatch
      if (user.role !== selectedRole && selectedRole !== UserRole.ADMIN) {
        // Maybe allow it but warn? Or just ignore the selector and use the real role.
        // The user requirement says "select admin... in the login page".
        // We'll just proceed with the actual user role from DB.
      }

      // Redirect based on role
      switch (user.role) {
        case UserRole.ADMIN:
        case UserRole.OPERATIONS_MANAGER:
          navigate('/admin');
          break;
        case UserRole.STATION_SUPERVISOR:
        case UserRole.SUPERVISOR:
          navigate('/supervisor');
          break;
        case UserRole.METRO_DRIVER:
          navigate('/driver');
          break;
        case UserRole.PASSENGER:
          navigate('/passenger');
          break;
        default:
          navigate('/');
      }
    } catch (err: any) {
      console.error('Login error:', err);
      if (err.response?.status === 403) {
        setError('Account pending approval. Please contact administrator.');
      } else {
        setError('Invalid username or password');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-background transition-colors duration-300">
      {/* Left Side - Slideshow */}
      <div className="hidden lg:block lg:w-1/2 relative overflow-hidden">
        {slides.map((slide, index) => (
          <div
            key={index}
            className={`absolute inset-0 transition-opacity duration-1000 ease-in-out ${index === currentSlide ? 'opacity-100' : 'opacity-0'
              }`}
          >
            <img
              src={slide}
              alt={`Slide ${index + 1}`}
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-black/40" />
          </div>
        ))}
        <div className="absolute bottom-10 left-10 text-white z-10">
          <h2 className="text-4xl font-bold mb-2">KMRL Induction System</h2>
          <p className="text-xl text-gray-200">Advanced Train Management & Operations</p>
        </div>
      </div>

      {/* Right Side - Login Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 bg-background relative">
        <div className="absolute top-4 right-4">
          <ThemeToggle />
        </div>
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-foreground">Welcome Back</h1>
            <p className="mt-2 text-muted-foreground">Sign in to your account</p>
          </div>

          {error && (
            <div className="bg-destructive/10 border border-destructive text-destructive p-4 rounded-md flex items-center">
              <AlertCircle className="w-5 h-5 mr-2" />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-8 space-y-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">
                  I am a...
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Briefcase className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <select
                    value={selectedRole}
                    onChange={(e) => setSelectedRole(e.target.value as UserRole)}
                    className="block w-full pl-10 bg-background border border-input rounded-md py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-input"
                  >
                    <option value={UserRole.ADMIN}>Admin / Operations Manager</option>
                    <option value={UserRole.STATION_SUPERVISOR}>Station Supervisor</option>
                    <option value={UserRole.METRO_DRIVER}>Metro Driver</option>
                    <option value={UserRole.PASSENGER}>Passenger</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">
                  Username
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <User className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="block w-full pl-10 bg-background border border-input rounded-md py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-input"
                    placeholder="Enter your username"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="block w-full pl-10 bg-background border border-input rounded-md py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-input"
                    placeholder="Enter your password"
                    required
                  />
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-primary-foreground bg-primary hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ring disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Signing in...' : 'Sign In'}
            </button>
          </form>

          <div className="text-center mt-4">
            <p className="text-muted-foreground text-sm">
              Don't have an account?{' '}
              <Link to="/signup" className="text-primary hover:text-primary/80 font-medium">
                Sign up
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;

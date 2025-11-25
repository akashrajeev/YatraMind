import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Eye, EyeOff, Loader2, Lock, UserRound, CheckCircle2, AlertCircle } from 'lucide-react';

import bg1 from '@/assets/slide1.jpg'; // Adjust path based on your folder structure
import bg2 from '@/assets/slide2.jpg';
import bg3 from '@/assets/slide3.jpg';
import bg4 from '@/assets/slide4.jpg';

// UPDATE THE ARRAY TO USE THE VARIABLES
const BACKGROUND_SLIDES = [
  bg1,
  bg2,
  bg3,
  bg4,
];

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [toastVariant, setToastVariant] = useState<'success' | 'error'>('success');

  const { login } = useAuth();
  const navigate = useNavigate();
  useEffect(() => {
    BACKGROUND_SLIDES.forEach((slide) => {
      const img = new Image();
      img.src = slide;
    });
  }, []);
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % BACKGROUND_SLIDES.length);
    }, 7000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(username, password);
      setToastVariant('success');
      setToastMessage('Logging you in...');
      setShowToast(true);
      setTimeout(() => {
        setShowToast(false);
        navigate('/');
      }, 1000);
    } catch (err: any) {
      const message = err?.message || 'Invalid username or password';
      setError(message);
      setToastVariant('error');
      setToastMessage(message);
      setShowToast(true);
      setTimeout(() => setShowToast(false), 2500);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="kmrl-login relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950 text-white">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .animate-fade-up {
          animation: fadeUp 0.8s ease-out forwards;
        }

        input:-webkit-autofill,
        input:-webkit-autofill:hover,
        input:-webkit-autofill:focus,
        input:-webkit-autofill:active {
          -webkit-box-shadow: 0 0 0 30px rgba(255,255,255,0.1) inset !important;
          -webkit-text-fill-color: white !important;
          transition: background-color 5000s ease-in-out 0s;
        }

        .kmrl-login {
          font-family: 'Poppins', 'Inter', sans-serif;
        }
      `}</style>

      <div className="absolute inset-0 z-0 overflow-hidden">
      {BACKGROUND_SLIDES.map((slide, index) => (
  <div
    key={slide}
    className="absolute inset-0 h-full w-full"
  >
    <img
      src={slide}
      alt={`Metro slide ${index + 1}`}
      className="absolute inset-0 h-full w-full object-cover"
      style={{
        opacity: currentSlide === index ? 1 : 0,
        transition: 'opacity 3000ms ease-in-out', // 3 second smooth fade
      }}
    />
  </div>
))}
        <div className="absolute inset-0 bg-black/60" />
      </div>

      <div className="relative z-10 w-full max-w-lg px-6 animate-fade-up">
        <div className="rounded-[32px] border border-white/25 bg-white/25 p-8 shadow-2xl backdrop-blur-[22px]">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-semibold tracking-wide text-white">KMRL Operations</h1>
            <p className="text-base text-white/100 mt-2">Enter your credentials to access the system</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-white/100">
                <UserRound className="h-5 w-5" />
              </div>
              <input
                type="text"
                id="username"
                placeholder="Username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={loading}
                className="w-full rounded-xl border border-white/50 bg-transparent py-3.5 pl-11 pr-3 text-white placeholder-white/70 shadow-[0_8px_20px_rgba(0,0,0,0.2)] transition focus:border-white/80 focus:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white/30"
                required
                autoFocus
              />
            </div>

            <div className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-white/100">
                <Lock className="h-5 w-5" />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                placeholder="Password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                className="w-full rounded-xl border border-white/50 bg-transparent py-3.5 pl-11 pr-11 text-white placeholder-white/70 shadow-[0_8px_20px_rgba(0,0,0,0.2)] transition focus:border-white/80 focus:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white/30"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword((prev) => !prev)}
                className="absolute inset-y-0 right-0 flex items-center pr-3 text-white/70 transition hover:text-white"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
              </button>
            </div>

            {error && (
              <div className="flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-100">
                <AlertCircle className="h-4 w-4" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !username || !password}
              className="w-full rounded-2xl bg-white py-3 text-center text-base font-semibold text-slate-900 shadow-[0_15px_40px_rgba(15,23,42,0.35)] transition hover:-translate-y-0.5 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-white/60 disabled:cursor-not-allowed disabled:opacity-100"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-slate-900" />
                  Signing in...
                </span>
              ) : (
                'Sign In'
              )}
            </button>
          </form>
        </div>
      </div>

      <div
        className={`fixed top-6 right-6 z-50 transform transition-all duration-300 ${
          showToast ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
        }`}
      >
        <div className="flex items-center gap-3 rounded-xl border-l-4 border-green-500 bg-white/90 px-5 py-4 text-slate-900 shadow-2xl backdrop-blur">
          {(toastVariant === 'success' ? <CheckCircle2 className="h-5 w-5 text-green-600" /> : <AlertCircle className="h-5 w-5 text-red-500" />)}
          <div>
            <p className="text-sm font-semibold">
              {toastVariant === 'success' ? 'Success' : 'Error'}
            </p>
            <p className="text-xs text-slate-600">{toastMessage}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;


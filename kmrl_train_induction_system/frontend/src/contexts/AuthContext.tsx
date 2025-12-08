import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { User } from '../types'
import { authApi } from '../services/api'

interface AuthContextType {
  user: User | null
  login: (username: string, password: string) => Promise<User>
  logout: () => void
  loading: boolean
  hasPermission: (permission: string) => boolean
  isAuthenticated: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('auth_token')
      if (token) {
        try {
          const response = await authApi.getProfile()
          const safeUser = {
            ...response.data,
            permissions: response.data.permissions ?? [],
          }
          setUser(safeUser)
        } catch (error) {
          localStorage.removeItem('auth_token')
        }
      }
      setLoading(false)
    }
    initAuth()
  }, [])

  const login = async (username: string, password: string): Promise<User> => {
    try {
      const response = await authApi.login({ username, password })
      const data = response.data
      localStorage.setItem('auth_token', data.access_token)
      const safeUser = {
        ...data.user,
        permissions: data.user.permissions ?? [],
      }
      setUser(safeUser)
      return safeUser
    } catch (error: any) {
      const detail = error.response?.data?.detail;
      let message = 'Login failed';
      if (typeof detail === 'string') message = detail;
      else if (Array.isArray(detail)) message = detail.map((e: any) => e.msg).join(', ');
      else if (typeof detail === 'object') message = JSON.stringify(detail);

      const customError: any = new Error(message);
      customError.response = error.response;
      throw customError;
    }
  }

  const logout = async () => {
    try {
      await authApi.logout()
    } catch (error) {
      // Ignore logout errors
    } finally {
      localStorage.removeItem('auth_token')
      setUser(null)
    }
  }

  const hasPermission = (permission: string): boolean => {
    if (!user) return false
    const perms = user.permissions ?? []
    return perms.includes(permission) || user.role === 'OPERATIONS_MANAGER'
  }

  const value = {
    user,
    login,
    logout,
    loading,
    hasPermission,
    isAuthenticated: !!user,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

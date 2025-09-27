import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { User } from '../types'
import { authApi } from '../services/api'

interface AuthContextType {
  user: User | null
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  loading: boolean
  hasPermission: (permission: string) => boolean
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
          const profile = await authApi.getProfile()
          setUser(profile)
        } catch (error) {
          localStorage.removeItem('auth_token')
        }
      }
      setLoading(false)
    }
    initAuth()
  }, [])

  const login = async (email: string, password: string) => {
    try {
      const { token, user: userData } = await authApi.login({ email, password })
      localStorage.setItem('auth_token', token)
      setUser(userData)
    } catch (error) {
      throw new Error('Login failed')
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
    return user.permissions.includes(permission) || user.role === 'OPERATIONS_MANAGER'
  }

  const value = {
    user,
    login,
    logout,
    loading,
    hasPermission,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

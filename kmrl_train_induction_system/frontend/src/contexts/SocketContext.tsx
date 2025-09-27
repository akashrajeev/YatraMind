import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { io, Socket } from 'socket.io-client'
import { Alert } from '../types'

interface SocketContextType {
  socket: Socket | null
  connected: boolean
  alerts: Alert[]
  addAlert: (alert: Alert) => void
  clearAlert: (alertId: string) => void
}

const SocketContext = createContext<SocketContextType | undefined>(undefined)

export const useSocket = () => {
  const context = useContext(SocketContext)
  if (context === undefined) {
    throw new Error('useSocket must be used within a SocketProvider')
  }
  return context
}

interface SocketProviderProps {
  children: ReactNode
}

export const SocketProvider: React.FC<SocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [connected, setConnected] = useState(false)
  const [alerts, setAlerts] = useState<Alert[]>([])

  useEffect(() => {
    const socketUrl = import.meta.env.VITE_SOCKET_URL || 'http://localhost:8000'
    const newSocket = io(socketUrl, {
      transports: ['websocket', 'polling'],
      autoConnect: true,
    })

    newSocket.on('connect', () => {
      console.log('Socket connected')
      setConnected(true)
    })

    newSocket.on('disconnect', () => {
      console.log('Socket disconnected')
      setConnected(false)
    })

    newSocket.on('optimization_update', (data) => {
      console.log('Optimization update:', data)
      // Handle optimization updates
    })

    newSocket.on('ingestion_update', (data) => {
      console.log('Ingestion update:', data)
      // Handle ingestion updates
    })

    newSocket.on('new_alert', (alert: Alert) => {
      console.log('New alert:', alert)
      addAlert(alert)
    })

    newSocket.on('assignment_updated', (data) => {
      console.log('Assignment updated:', data)
      // Handle assignment updates
    })

    setSocket(newSocket)

    return () => {
      newSocket.close()
    }
  }, [])

  const addAlert = (alert: Alert) => {
    setAlerts(prev => [alert, ...prev.slice(0, 49)]) // Keep last 50 alerts
  }

  const clearAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId))
  }

  const value = {
    socket,
    connected,
    alerts,
    addAlert,
    clearAlert,
  }

  return <SocketContext.Provider value={value}>{children}</SocketContext.Provider>
}

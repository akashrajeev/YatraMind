import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Assignments from './pages/Assignments'
import Reports from './pages/Reports'
import Settings from './pages/Settings'
import { SocketProvider } from './contexts/SocketContext'
import { AuthProvider } from './contexts/AuthContext'

function App() {
  return (
    <AuthProvider>
      <SocketProvider>
        <Router>
          <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <Navbar />
            <Box component="main" sx={{ flexGrow: 1, p: 3, ml: '240px' }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/assignments" element={<Assignments />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </SocketProvider>
    </AuthProvider>
  )
}

export default App

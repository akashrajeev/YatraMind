import React from 'react'
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Chip,
  Badge,
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  Assignment as AssignmentIcon,
  Assessment as ReportsIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
} from '@mui/icons-material'
import { useNavigate, useLocation } from 'react-router-dom'
import { useSocket } from '../contexts/SocketContext'
import { useAuth } from '../contexts/AuthContext'

const drawerWidth = 240

const Navbar: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const { alerts } = useSocket()
  const { user } = useAuth()

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Assignments', icon: <AssignmentIcon />, path: '/assignments' },
    { text: 'Reports', icon: <ReportsIcon />, path: '/reports' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ]

  const unreadAlerts = alerts.filter(alert => !alert.acknowledged).length

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: '#1976d2',
          color: 'white',
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
          KMRL Operations
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.8 }}>
          Decision Support Platform
        </Typography>
      </Box>
      
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              onClick={() => navigate(item.path)}
              selected={location.pathname === item.path}
              sx={{
                color: 'white',
                '&.Mui-selected': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                },
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                },
              }}
            >
              <ListItemIcon sx={{ color: 'white', minWidth: 40 }}>
                {item.text === 'Assignments' && unreadAlerts > 0 ? (
                  <Badge badgeContent={unreadAlerts} color="error">
                    {item.icon}
                  </Badge>
                ) : (
                  item.icon
                )}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Box sx={{ mt: 'auto', p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <NotificationsIcon sx={{ mr: 1, fontSize: 20 }} />
          <Typography variant="body2">
            {unreadAlerts} unread alerts
          </Typography>
        </Box>
        
        {user && (
          <Box>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              {user.name}
            </Typography>
            <Chip 
              label={user.role.replace('_', ' ')} 
              size="small" 
              color="secondary"
              sx={{ mt: 0.5 }}
            />
          </Box>
        )}
      </Box>
    </Drawer>
  )
}

export default Navbar

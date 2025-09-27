import React from 'react'
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Box,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Close as CloseIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material'
import { Alert } from '../types'
import { format } from 'date-fns'

interface AlertsPanelProps {
  alerts: Alert[]
}

const AlertsPanel: React.FC<AlertsPanelProps> = ({ alerts }) => {
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'CRITICAL': return <ErrorIcon color="error" />
      case 'HIGH': return <WarningIcon color="warning" />
      case 'WARNING': return <InfoIcon color="info" />
      default: return <InfoIcon />
    }
  }

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'CRITICAL': return 'error'
      case 'HIGH': return 'warning'
      case 'WARNING': return 'info'
      default: return 'default'
    }
  }

  const handleAcknowledge = (alertId: string) => {
    // This would call an API to acknowledge the alert
    console.log('Acknowledging alert:', alertId)
  }

  const unreadAlerts = alerts.filter(alert => !alert.acknowledged)
  const criticalAlerts = unreadAlerts.filter(alert => alert.type === 'CRITICAL')
  const highAlerts = unreadAlerts.filter(alert => alert.type === 'HIGH')
  const warningAlerts = unreadAlerts.filter(alert => alert.type === 'WARNING')

  return (
    <Paper sx={{ p: 2, height: 400, overflow: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Active Alerts
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label={`${criticalAlerts.length} Critical`}
            color="error"
            size="small"
          />
          <Chip
            label={`${highAlerts.length} High`}
            color="warning"
            size="small"
          />
          <Chip
            label={`${warningAlerts.length} Warning`}
            color="info"
            size="small"
          />
        </Box>
      </Box>

      {unreadAlerts.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <CheckCircleIcon color="success" sx={{ fontSize: 48, mb: 2 }} />
          <Typography variant="body1" color="text.secondary">
            No active alerts
          </Typography>
        </Box>
      ) : (
        <List dense>
          {unreadAlerts.slice(0, 10).map((alert) => (
            <ListItem
              key={alert.id}
              sx={{
                borderLeft: `4px solid ${
                  alert.type === 'CRITICAL' ? '#f44336' :
                  alert.type === 'HIGH' ? '#ff9800' : '#2196f3'
                }`,
                mb: 1,
                backgroundColor: 'rgba(0, 0, 0, 0.02)',
                borderRadius: 1,
              }}
            >
              <ListItemIcon>
                {getAlertIcon(alert.type)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography variant="body2" fontWeight="medium">
                      {alert.trainset_id}
                    </Typography>
                    <Chip
                      label={alert.type}
                      color={getAlertColor(alert.type) as any}
                      size="small"
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {alert.message}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {format(new Date(alert.timestamp), 'MMM dd, HH:mm')}
                    </Typography>
                  </Box>
                }
              />
              <Tooltip title="Acknowledge Alert">
                <IconButton
                  size="small"
                  onClick={() => handleAcknowledge(alert.id)}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </ListItem>
          ))}
        </List>
      )}

      {unreadAlerts.length > 10 && (
        <Box sx={{ textAlign: 'center', mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            +{unreadAlerts.length - 10} more alerts
          </Typography>
        </Box>
      )}
    </Paper>
  )
}

export default AlertsPanel

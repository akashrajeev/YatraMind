import React from 'react'
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Alert as MuiAlert,
  Chip,
} from '@mui/material'
import {
  Train as TrainIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Build as BuildIcon,
} from '@mui/icons-material'
import { useQuery } from 'react-query'
import { dashboardApi } from '../services/api'
import { useSocket } from '../contexts/SocketContext'
import FleetOverview from '../components/FleetOverview'
import AlertsPanel from '../components/AlertsPanel'
import PerformanceMetrics from '../components/PerformanceMetrics'

const Dashboard: React.FC = () => {
  const { alerts } = useSocket()
  
  const { data: overview, isLoading: overviewLoading } = useQuery(
    'dashboard-overview',
    dashboardApi.getOverview,
    { refetchInterval: 30000 }
  )

  const { data: performance, isLoading: performanceLoading } = useQuery(
    'dashboard-performance',
    dashboardApi.getPerformance,
    { refetchInterval: 60000 }
  )

  const criticalAlerts = alerts.filter(alert => alert.type === 'CRITICAL')
  const highAlerts = alerts.filter(alert => alert.type === 'HIGH')
  const warningAlerts = alerts.filter(alert => alert.type === 'WARNING')

  if (overviewLoading) {
    return <LinearProgress />
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Operations Dashboard
      </Typography>
      
      {/* Critical Alerts Banner */}
      {criticalAlerts.length > 0 && (
        <MuiAlert severity="error" sx={{ mb: 2 }}>
          <Typography variant="h6">
            {criticalAlerts.length} Critical Alert{criticalAlerts.length > 1 ? 's' : ''} Require Immediate Attention
          </Typography>
        </MuiAlert>
      )}

      <Grid container spacing={3}>
        {/* Fleet Status Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Active Fleet</Typography>
              </Box>
              <Typography variant="h3" color="success.main">
                {overview?.fleet_status.active || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Trainsets in Service
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <BuildIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">Maintenance</Typography>
              </Box>
              <Typography variant="h3" color="warning.main">
                {overview?.fleet_status.maintenance || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Under Maintenance
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrainIcon color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Standby</Typography>
              </Box>
              <Typography variant="h3" color="info.main">
                {overview?.fleet_status.standby || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Available for Induction
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <WarningIcon color="error" sx={{ mr: 1 }} />
                <Typography variant="h6">Alerts</Typography>
              </Box>
              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
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
              <Typography variant="body2" color="text.secondary">
                Total: {alerts.length} alerts
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Fleet Overview Chart */}
        <Grid item xs={12} md={8}>
          <FleetOverview data={overview} />
        </Grid>

        {/* Alerts Panel */}
        <Grid item xs={12} md={4}>
          <AlertsPanel alerts={alerts} />
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12}>
          <PerformanceMetrics data={performance} loading={performanceLoading} />
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard

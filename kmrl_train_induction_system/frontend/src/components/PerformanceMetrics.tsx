import React from 'react'
import {
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  LinearProgress,
  Chip,
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckCircleIcon,
  Build as BuildIcon,
} from '@mui/icons-material'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts'

interface PerformanceMetricsProps {
  data?: any
  loading: boolean
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ data, loading }) => {
  if (loading) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <LinearProgress />
      </Paper>
    )
  }

  if (!data) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <Typography color="text.secondary">No performance data available</Typography>
      </Paper>
    )
  }

  const optimizationHistory = data.optimization_performance?.recent_history || []
  const operationalMetrics = data.operational_metrics || {}
  const systemHealth = data.system_health || {}

  const chartData = optimizationHistory.map((item: any, index: number) => ({
    day: `Day ${index + 1}`,
    confidence: item.average_confidence || 0,
    efficiency: item.efficiency_score || 0,
  }))

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Performance Metrics
      </Typography>

      <Grid container spacing={3}>
        {/* Operational KPIs */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Punctuality</Typography>
              </Box>
              <Typography variant="h3" color="success.main">
                {operationalMetrics.punctuality_rate || 0}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                On-time Performance
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Availability</Typography>
              </Box>
              <Typography variant="h3" color="info.main">
                {operationalMetrics.fleet_availability || 0}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Fleet Availability
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SpeedIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Efficiency</Typography>
              </Box>
              <Typography variant="h3" color="primary.main">
                {operationalMetrics.energy_efficiency || 0}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Energy Efficiency
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <BuildIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">Cost Savings</Typography>
              </Box>
              <Typography variant="h3" color="warning.main">
                {operationalMetrics.maintenance_cost_reduction || 0}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Maintenance Cost Reduction
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Optimization Performance Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Optimization Performance Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="#1976d2"
                    strokeWidth={2}
                    name="Confidence Score"
                  />
                  <Line
                    type="monotone"
                    dataKey="efficiency"
                    stroke="#4caf50"
                    strokeWidth={2}
                    name="Efficiency Score"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* System Health */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    API Response Time
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={100 - (systemHealth.api_response_time_ms || 0)}
                      color="success"
                      sx={{ flexGrow: 1 }}
                    />
                    <Typography variant="body2">
                      {systemHealth.api_response_time_ms || 0}ms
                    </Typography>
                  </Box>
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Optimization Time
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={100 - (systemHealth.optimization_time_seconds || 0) * 2}
                      color="info"
                      sx={{ flexGrow: 1 }}
                    />
                    <Typography variant="body2">
                      {systemHealth.optimization_time_seconds || 0}s
                    </Typography>
                  </Box>
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Database Performance
                  </Typography>
                  <Chip
                    label={systemHealth.database_performance || 'Unknown'}
                    color={
                      systemHealth.database_performance === 'GOOD' ? 'success' :
                      systemHealth.database_performance === 'FAIR' ? 'warning' : 'error'
                    }
                    size="small"
                  />
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    MQTT Connectivity
                  </Typography>
                  <Chip
                    label={systemHealth.mqtt_connectivity || 'Unknown'}
                    color={systemHealth.mqtt_connectivity === 'ONLINE' ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Paper>
  )
}

export default PerformanceMetrics

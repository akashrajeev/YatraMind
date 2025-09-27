import React from 'react'
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Chip,
} from '@mui/material'
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import { DashboardOverview } from '../types'

interface FleetOverviewProps {
  data?: DashboardOverview
}

const FleetOverview: React.FC<FleetOverviewProps> = ({ data }) => {
  if (!data) {
    return (
      <Paper sx={{ p: 3, height: 400 }}>
        <Typography variant="h6" gutterBottom>
          Fleet Overview
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
          <Typography color="text.secondary">Loading fleet data...</Typography>
        </Box>
      </Paper>
    )
  }

  const fleetStatusData = [
    { name: 'Active', value: data.fleet_status.active, color: '#4caf50' },
    { name: 'Maintenance', value: data.fleet_status.maintenance, color: '#ff9800' },
    { name: 'Standby', value: data.fleet_status.standby, color: '#2196f3' },
  ]

  const certificateData = [
    { name: 'Valid', value: data.fitness_certificates.valid, color: '#4caf50' },
    { name: 'Expired', value: data.fitness_certificates.expired, color: '#f44336' },
    { name: 'Expiring Soon', value: data.fitness_certificates.expiring_soon, color: '#ff9800' },
  ]

  const depotData = Object.entries(data.depot_distribution).map(([depot, count]) => ({
    depot,
    count,
  }))

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Fleet Overview
      </Typography>
      
      <Grid container spacing={3}>
        {/* Fleet Status Pie Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Fleet Status Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={fleetStatusData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {fleetStatusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
                {fleetStatusData.map((item) => (
                  <Box key={item.name} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        backgroundColor: item.color,
                        borderRadius: '50%',
                      }}
                    />
                    <Typography variant="body2">{item.name}: {item.value}</Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Certificate Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Certificate Status
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {certificateData.map((item) => (
                  <Box key={item.name} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          backgroundColor: item.color,
                          borderRadius: '50%',
                        }}
                      />
                      <Typography variant="body2">{item.name}</Typography>
                    </Box>
                    <Chip
                      label={item.value}
                      size="small"
                      sx={{ backgroundColor: item.color, color: 'white' }}
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Depot Distribution */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Depot Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={depotData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="depot" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#1976d2" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Paper>
  )
}

export default FleetOverview

import React from 'react'
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  Grid,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
} from '@mui/material'
import {
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  Security as SecurityIcon,
  Person as PersonIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
} from '@mui/icons-material'
import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

const Settings: React.FC = () => {
  const [tabValue, setTabValue] = useState(0)
  const { user } = useAuth()

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Settings
      </Typography>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab
              icon={<PersonIcon />}
              label="Profile"
              id="settings-tab-0"
              aria-controls="settings-tabpanel-0"
            />
            <Tab
              icon={<NotificationsIcon />}
              label="Notifications"
              id="settings-tab-1"
              aria-controls="settings-tabpanel-1"
            />
            <Tab
              icon={<SecurityIcon />}
              label="Security"
              id="settings-tab-2"
              aria-controls="settings-tabpanel-2"
            />
            <Tab
              icon={<SettingsIcon />}
              label="Preferences"
              id="settings-tab-3"
              aria-controls="settings-tabpanel-3"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    User Profile
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                      label="Name"
                      value={user?.name || ''}
                      disabled
                      fullWidth
                    />
                    <TextField
                      label="Email"
                      value={user?.email || ''}
                      disabled
                      fullWidth
                    />
                    <TextField
                      label="Role"
                      value={user?.role?.replace('_', ' ') || ''}
                      disabled
                      fullWidth
                    />
                    <Button variant="outlined" startIcon={<EditIcon />}>
                      Edit Profile
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Permissions
                  </Typography>
                  <List dense>
                    {user?.permissions?.map((permission, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={permission} />
                        <ListItemSecondaryAction>
                          <Chip label="Granted" color="success" size="small" />
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Alert Preferences
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Critical Alerts (Email + SMS)"
                    />
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="High Priority Alerts (Email)"
                    />
                    <FormControlLabel
                      control={<Switch />}
                      label="Warning Alerts (In-app only)"
                    />
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Maintenance Notifications"
                    />
                    <FormControlLabel
                      control={<Switch />}
                      label="Performance Updates"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Notification Channels
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                      label="Email Address"
                      value={user?.email || ''}
                      disabled
                      fullWidth
                    />
                    <TextField
                      label="Phone Number"
                      placeholder="+91 98765 43210"
                      fullWidth
                    />
                    <Button variant="outlined">
                      Update Contact Info
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Password Security
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                      label="Current Password"
                      type="password"
                      fullWidth
                    />
                    <TextField
                      label="New Password"
                      type="password"
                      fullWidth
                    />
                    <TextField
                      label="Confirm New Password"
                      type="password"
                      fullWidth
                    />
                    <Button variant="contained">
                      Change Password
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Two-Factor Authentication
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <FormControlLabel
                      control={<Switch />}
                      label="Enable 2FA"
                    />
                    <Typography variant="body2" color="text.secondary">
                      Add an extra layer of security to your account
                    </Typography>
                    <Button variant="outlined">
                      Setup 2FA
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Active Sessions
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Current Session"
                        secondary="Windows 10 • Chrome • Last active: Now"
                      />
                      <ListItemSecondaryAction>
                        <Chip label="Active" color="success" size="small" />
                      </ListItemSecondaryAction>
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Mobile App"
                        secondary="Android • Chrome • Last active: 2 hours ago"
                      />
                      <ListItemSecondaryAction>
                        <IconButton edge="end" aria-label="delete">
                          <DeleteIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Display Preferences
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Dark Mode"
                    />
                    <FormControlLabel
                      control={<Switch />}
                      label="Compact View"
                    />
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Show Tooltips"
                    />
                    <FormControlLabel
                      control={<Switch />}
                      label="Auto-refresh Data"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Data Preferences
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                      label="Refresh Interval (seconds)"
                      type="number"
                      defaultValue={30}
                      fullWidth
                    />
                    <TextField
                      label="Default Date Range (days)"
                      type="number"
                      defaultValue={7}
                      fullWidth
                    />
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Enable Real-time Updates"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  )
}

export default Settings

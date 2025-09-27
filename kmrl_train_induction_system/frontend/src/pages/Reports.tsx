import React, { useState } from 'react'
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  CircularProgress,
} from '@mui/material'
import {
  Download as DownloadIcon,
  PictureAsPdf as PdfIcon,
  TableChart as CsvIcon,
  Assessment as ReportIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import { useQuery, useMutation } from 'react-query'
import { reportsApi } from '../services/api'
import { format } from 'date-fns'

const Reports: React.FC = () => {
  const [selectedFormat, setSelectedFormat] = useState<'csv' | 'pdf'>('pdf')
  const [dateRange, setDateRange] = useState({
    start: format(new Date(), 'yyyy-MM-dd'),
    end: format(new Date(), 'yyyy-MM-dd'),
  })

  const generateDailyBriefingMutation = useMutation(reportsApi.generateDailyBriefing, {
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `daily-briefing-${format(new Date(), 'yyyy-MM-dd')}.pdf`
      link.click()
      window.URL.revokeObjectURL(url)
    },
  })

  const exportAssignmentsMutation = useMutation(
    () => reportsApi.exportAssignments(selectedFormat),
    {
      onSuccess: (blob) => {
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `assignments-${format(new Date(), 'yyyy-MM-dd')}.${selectedFormat}`
        link.click()
        window.URL.revokeObjectURL(url)
      },
    }
  )

  const exportAuditLogsMutation = useMutation(reportsApi.exportAuditLogs, {
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `audit-logs-${format(new Date(), 'yyyy-MM-dd')}.csv`
      link.click()
      window.URL.revokeObjectURL(url)
    },
  })

  const handleGenerateDailyBriefing = () => {
    generateDailyBriefingMutation.mutate()
  }

  const handleExportAssignments = () => {
    exportAssignmentsMutation.mutate()
  }

  const handleExportAuditLogs = () => {
    exportAuditLogsMutation.mutate()
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Reports & Exports
      </Typography>

      <Grid container spacing={3}>
        {/* Daily Briefing Report */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <ReportIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Daily Briefing Report</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Generate a comprehensive daily briefing PDF with fleet status, 
                assignments, alerts, and performance metrics.
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  label="Date"
                  type="date"
                  value={dateRange.start}
                  onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
                  InputLabelProps={{ shrink: true }}
                />
              </Box>

              <Button
                variant="contained"
                startIcon={generateDailyBriefingMutation.isLoading ? <CircularProgress size={20} /> : <PdfIcon />}
                onClick={handleGenerateDailyBriefing}
                disabled={generateDailyBriefingMutation.isLoading}
                fullWidth
              >
                {generateDailyBriefingMutation.isLoading ? 'Generating...' : 'Generate Daily Briefing'}
              </Button>

              {generateDailyBriefingMutation.isError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  Failed to generate daily briefing. Please try again.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Assignment Exports */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <DownloadIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Assignment Exports</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Export assignment data in various formats for analysis and reporting.
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Export Format</InputLabel>
                <Select
                  value={selectedFormat}
                  onChange={(e) => setSelectedFormat(e.target.value as 'csv' | 'pdf')}
                  label="Export Format"
                >
                  <MenuItem value="pdf">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <PdfIcon fontSize="small" />
                      PDF Report
                    </Box>
                  </MenuItem>
                  <MenuItem value="csv">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CsvIcon fontSize="small" />
                      CSV Data
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>

              <Button
                variant="outlined"
                startIcon={exportAssignmentsMutation.isLoading ? <CircularProgress size={20} /> : <DownloadIcon />}
                onClick={handleExportAssignments}
                disabled={exportAssignmentsMutation.isLoading}
                fullWidth
              >
                {exportAssignmentsMutation.isLoading ? 'Exporting...' : 'Export Assignments'}
              </Button>

              {exportAssignmentsMutation.isError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  Failed to export assignments. Please try again.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Audit Logs Export */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <RefreshIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Audit Logs Export</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Export audit logs for compliance and security analysis.
              </Typography>

              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  label="Start Date"
                  type="date"
                  value={dateRange.start}
                  onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
                  InputLabelProps={{ shrink: true }}
                />
                <TextField
                  label="End Date"
                  type="date"
                  value={dateRange.end}
                  onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
                  InputLabelProps={{ shrink: true }}
                />
              </Box>

              <Button
                variant="outlined"
                startIcon={exportAuditLogsMutation.isLoading ? <CircularProgress size={20} /> : <CsvIcon />}
                onClick={handleExportAuditLogs}
                disabled={exportAuditLogsMutation.isLoading}
                fullWidth
              >
                {exportAuditLogsMutation.isLoading ? 'Exporting...' : 'Export Audit Logs'}
              </Button>

              {exportAuditLogsMutation.isError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  Failed to export audit logs. Please try again.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Report Templates */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AssessmentIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Report Templates</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Pre-configured report templates for common operational needs.
              </Typography>

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  variant="text"
                  startIcon={<PdfIcon />}
                  onClick={() => {/* Generate maintenance report */}}
                >
                  Maintenance Summary Report
                </Button>
                <Button
                  variant="text"
                  startIcon={<PdfIcon />}
                  onClick={() => {/* Generate performance report */}}
                >
                  Performance Analysis Report
                </Button>
                <Button
                  variant="text"
                  startIcon={<CsvIcon />}
                  onClick={() => {/* Generate compliance report */}}
                >
                  Compliance Report
                </Button>
                <Button
                  variant="text"
                  startIcon={<CsvIcon />}
                  onClick={() => {/* Generate cost analysis report */}}
                >
                  Cost Analysis Report
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Reports

import React, { useState } from 'react'
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
} from '@mui/material'
import {
  Add as AddIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { assignmentApi, optimizationApi } from '../services/api'
import { Assignment, OverrideRequest, ApprovalRequest } from '../types'
import AssignmentList from '../components/AssignmentList'
import AssignmentDetail from '../components/AssignmentDetail'
import OverrideDialog from '../components/OverrideDialog'
import ApprovalDialog from '../components/ApprovalDialog'

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
      id={`assignment-tabpanel-${index}`}
      aria-labelledby={`assignment-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

const Assignments: React.FC = () => {
  const [tabValue, setTabValue] = useState(0)
  const [selectedAssignment, setSelectedAssignment] = useState<Assignment | null>(null)
  const [overrideDialogOpen, setOverrideDialogOpen] = useState(false)
  const [approvalDialogOpen, setApprovalDialogOpen] = useState(false)
  const [selectedAssignments, setSelectedAssignments] = useState<string[]>([])

  const queryClient = useQueryClient()

  const { data: assignments, isLoading, refetch } = useQuery(
    'assignments',
    assignmentApi.getAll,
    { refetchInterval: 30000 }
  )

  const runOptimizationMutation = useMutation(optimizationApi.runOptimization, {
    onSuccess: () => {
      queryClient.invalidateQueries('assignments')
    },
  })

  const approveMutation = useMutation(assignmentApi.approve, {
    onSuccess: () => {
      queryClient.invalidateQueries('assignments')
      setApprovalDialogOpen(false)
      setSelectedAssignments([])
    },
  })

  const overrideMutation = useMutation(assignmentApi.override, {
    onSuccess: () => {
      queryClient.invalidateQueries('assignments')
      setOverrideDialogOpen(false)
      setSelectedAssignment(null)
    },
  })

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)
  }

  const handleRunOptimization = () => {
    runOptimizationMutation.mutate({
      target_date: new Date().toISOString(),
      required_service_hours: 14,
    })
  }

  const handleOverride = (assignment: Assignment) => {
    setSelectedAssignment(assignment)
    setOverrideDialogOpen(true)
  }

  const handleApprove = (assignmentIds: string[]) => {
    setSelectedAssignments(assignmentIds)
    setApprovalDialogOpen(true)
  }

  const handleOverrideSubmit = (overrideData: Omit<OverrideRequest, 'assignment_id'>) => {
    if (selectedAssignment) {
      overrideMutation.mutate({
        assignment_id: selectedAssignment.id,
        ...overrideData,
      })
    }
  }

  const handleApprovalSubmit = (approvalData: Omit<ApprovalRequest, 'assignment_ids'>) => {
    approveMutation.mutate({
      assignment_ids: selectedAssignments,
      ...approvalData,
    })
  }

  const pendingAssignments = assignments?.filter(a => a.status === 'PENDING') || []
  const approvedAssignments = assignments?.filter(a => a.status === 'APPROVED') || []
  const overriddenAssignments = assignments?.filter(a => a.status === 'OVERRIDDEN') || []

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Train Induction Assignments
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetch()}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={() => {/* Export functionality */}}
          >
            Export
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleRunOptimization}
            disabled={runOptimizationMutation.isLoading}
          >
            Run Optimization
          </Button>
        </Box>
      </Box>

      {runOptimizationMutation.isError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to run optimization. Please try again.
        </Alert>
      )}

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab 
              label={`Pending (${pendingAssignments.length})`} 
              id="assignment-tab-0"
              aria-controls="assignment-tabpanel-0"
            />
            <Tab 
              label={`Approved (${approvedAssignments.length})`} 
              id="assignment-tab-1"
              aria-controls="assignment-tabpanel-1"
            />
            <Tab 
              label={`Overridden (${overriddenAssignments.length})`} 
              id="assignment-tab-2"
              aria-controls="assignment-tabpanel-2"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <AssignmentList
            assignments={pendingAssignments}
            onOverride={handleOverride}
            onApprove={handleApprove}
            showActions={true}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <AssignmentList
            assignments={approvedAssignments}
            onOverride={handleOverride}
            onApprove={handleApprove}
            showActions={false}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <AssignmentList
            assignments={overriddenAssignments}
            onOverride={handleOverride}
            onApprove={handleApprove}
            showActions={false}
          />
        </TabPanel>
      </Paper>

      {/* Assignment Detail Dialog */}
      {selectedAssignment && (
        <AssignmentDetail
          assignment={selectedAssignment}
          open={!!selectedAssignment}
          onClose={() => setSelectedAssignment(null)}
        />
      )}

      {/* Override Dialog */}
      <OverrideDialog
        open={overrideDialogOpen}
        onClose={() => setOverrideDialogOpen(false)}
        onSubmit={handleOverrideSubmit}
        assignment={selectedAssignment}
        loading={overrideMutation.isLoading}
      />

      {/* Approval Dialog */}
      <ApprovalDialog
        open={approvalDialogOpen}
        onClose={() => setApprovalDialogOpen(false)}
        onSubmit={handleApprovalSubmit}
        assignmentCount={selectedAssignments.length}
        loading={approveMutation.isLoading}
      />
    </Box>
  )
}

export default Assignments

import React, { useState } from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Checkbox,
  Box,
  Typography,
  Tooltip,
  LinearProgress,
} from '@mui/material'
import {
  Visibility as ViewIcon,
  Edit as OverrideIcon,
  CheckCircle as ApproveIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material'
import { Assignment } from '../types'
import { format } from 'date-fns'

interface AssignmentListProps {
  assignments: Assignment[]
  onOverride: (assignment: Assignment) => void
  onApprove: (assignmentIds: string[]) => void
  showActions: boolean
}

const AssignmentList: React.FC<AssignmentListProps> = ({
  assignments,
  onOverride,
  onApprove,
  showActions,
}) => {
  const [selectedIds, setSelectedIds] = useState<string[]>([])

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      setSelectedIds(assignments.map(a => a.id))
    } else {
      setSelectedIds([])
    }
  }

  const handleSelectOne = (assignmentId: string) => {
    setSelectedIds(prev => 
      prev.includes(assignmentId)
        ? prev.filter(id => id !== assignmentId)
        : [...prev, assignmentId]
    )
  }

  const handleApproveSelected = () => {
    onApprove(selectedIds)
    setSelectedIds([])
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'PENDING': return 'warning'
      case 'APPROVED': return 'success'
      case 'REJECTED': return 'error'
      case 'OVERRIDDEN': return 'info'
      default: return 'default'
    }
  }

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'INDUCT': return 'success'
      case 'STANDBY': return 'info'
      case 'MAINTENANCE': return 'warning'
      default: return 'default'
    }
  }

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'success'
    if (score >= 0.6) return 'warning'
    return 'error'
  }

  if (assignments.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="h6" color="text.secondary">
          No assignments found
        </Typography>
      </Box>
    )
  }

  return (
    <Box>
      {showActions && selectedIds.length > 0 && (
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body2">
            {selectedIds.length} assignment{selectedIds.length > 1 ? 's' : ''} selected
          </Typography>
          <IconButton
            color="primary"
            onClick={handleApproveSelected}
            disabled={selectedIds.length === 0}
          >
            <ApproveIcon />
          </IconButton>
        </Box>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              {showActions && (
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedIds.length === assignments.length && assignments.length > 0}
                    indeterminate={selectedIds.length > 0 && selectedIds.length < assignments.length}
                    onChange={handleSelectAll}
                  />
                </TableCell>
              )}
              <TableCell>Trainset ID</TableCell>
              <TableCell>Decision</TableCell>
              <TableCell>Confidence</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Risks</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {assignments.map((assignment) => (
              <TableRow key={assignment.id} hover>
                {showActions && (
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedIds.includes(assignment.id)}
                      onChange={() => handleSelectOne(assignment.id)}
                    />
                  </TableCell>
                )}
                <TableCell>
                  <Typography variant="body2" fontWeight="medium">
                    {assignment.trainset_id}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={assignment.decision.decision}
                    color={getDecisionColor(assignment.decision.decision) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={assignment.decision.confidence_score * 100}
                      color={getConfidenceColor(assignment.decision.confidence_score) as any}
                      sx={{ width: 60, height: 6 }}
                    />
                    <Typography variant="body2">
                      {Math.round(assignment.decision.confidence_score * 100)}%
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip
                    label={assignment.status}
                    color={getStatusColor(assignment.status) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {format(new Date(assignment.created_at), 'MMM dd, HH:mm')}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    {assignment.decision.violations.length > 0 && (
                      <Tooltip title={`${assignment.decision.violations.length} violations`}>
                        <ErrorIcon color="error" fontSize="small" />
                      </Tooltip>
                    )}
                    {assignment.decision.top_risks.length > 0 && (
                      <Tooltip title={`${assignment.decision.top_risks.length} risks identified`}>
                        <WarningIcon color="warning" fontSize="small" />
                      </Tooltip>
                    )}
                    {assignment.decision.top_reasons.length > 0 && (
                      <Tooltip title={`${assignment.decision.top_reasons.length} positive factors`}>
                        <InfoIcon color="info" fontSize="small" />
                      </Tooltip>
                    )}
                  </Box>
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title="View Details">
                      <IconButton
                        size="small"
                        onClick={() => {/* View details */}}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    {showActions && assignment.status === 'PENDING' && (
                      <Tooltip title="Override Decision">
                        <IconButton
                          size="small"
                          onClick={() => onOverride(assignment)}
                        >
                          <OverrideIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  )
}

export default AssignmentList

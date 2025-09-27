import React from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Grid,
  Paper,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  LinearProgress,
  Alert,
} from '@mui/material'
import {
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Train as TrainIcon,
  Build as BuildIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material'
import { Assignment } from '../types'
import { format } from 'date-fns'

interface AssignmentDetailProps {
  assignment: Assignment | null
  open: boolean
  onClose: () => void
}

const AssignmentDetail: React.FC<AssignmentDetailProps> = ({
  assignment,
  open,
  onClose,
}) => {
  if (!assignment) return null

  const { decision } = assignment

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

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <TrainIcon />
          <Typography variant="h6">
            Assignment Details - {assignment.trainset_id}
          </Typography>
          <Chip
            label={assignment.status}
            color={assignment.status === 'PENDING' ? 'warning' : 'success'}
            size="small"
          />
        </Box>
      </DialogTitle>

      <DialogContent>
        <Grid container spacing={3}>
          {/* Decision Summary */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Decision Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Decision
                  </Typography>
                  <Chip
                    label={decision.decision}
                    color={getDecisionColor(decision.decision) as any}
                    sx={{ mt: 1 }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Confidence Score
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={decision.confidence_score * 100}
                      color={getConfidenceColor(decision.confidence_score) as any}
                      sx={{ mb: 1 }}
                    />
                    <Typography variant="body2">
                      {Math.round(decision.confidence_score * 100)}%
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">
                    Composite Score
                  </Typography>
                  <Typography variant="h6" color="primary">
                    {decision.score.toFixed(2)}
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>

          {/* Top Reasons */}
          {decision.top_reasons.length > 0 && (
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom color="success.main">
                  <CheckIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Positive Factors
                </Typography>
                <List dense>
                  {decision.top_reasons.map((reason, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <CheckIcon color="success" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={reason} />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>
          )}

          {/* Top Risks */}
          {decision.top_risks.length > 0 && (
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom color="warning.main">
                  <WarningIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Risk Factors
                </Typography>
                <List dense>
                  {decision.top_risks.map((risk, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <WarningIcon color="warning" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={risk} />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>
          )}

          {/* Violations */}
          {decision.violations.length > 0 && (
            <Grid item xs={12}>
              <Alert severity="error">
                <Typography variant="h6" gutterBottom>
                  Rule Violations
                </Typography>
                <List dense>
                  {decision.violations.map((violation, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <ErrorIcon color="error" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={violation} />
                    </ListItem>
                  ))}
                </List>
              </Alert>
            </Grid>
          )}

          {/* SHAP Values */}
          {decision.shap_values.length > 0 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  <AssessmentIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Feature Impact Analysis
                </Typography>
                <List dense>
                  {decision.shap_values.map((feature, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        {feature.impact === 'positive' ? (
                          <CheckIcon color="success" fontSize="small" />
                        ) : feature.impact === 'negative' ? (
                          <ErrorIcon color="error" fontSize="small" />
                        ) : (
                          <InfoIcon color="info" fontSize="small" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={feature.name}
                        secondary={`Impact: ${feature.value.toFixed(3)}`}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>
          )}

          {/* Assignment Metadata */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Assignment Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Created At
                  </Typography>
                  <Typography variant="body2">
                    {format(new Date(assignment.created_at), 'MMM dd, yyyy HH:mm')}
                  </Typography>
                </Grid>
                {assignment.approved_by && (
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Approved By
                    </Typography>
                    <Typography variant="body2">
                      {assignment.approved_by}
                    </Typography>
                  </Grid>
                )}
                {assignment.override_reason && (
                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary">
                      Override Reason
                    </Typography>
                    <Typography variant="body2">
                      {assignment.override_reason}
                    </Typography>
                  </Grid>
                )}
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  )
}

export default AssignmentDetail

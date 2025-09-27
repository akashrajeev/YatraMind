import React from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Alert,
} from '@mui/material'
import { useFormik } from 'formik'
import * as Yup from 'yup'
import { Assignment } from '../types'

interface OverrideDialogProps {
  open: boolean
  onClose: () => void
  onSubmit: (data: { reason: string; user_id: string; override_decision: string }) => void
  assignment: Assignment | null
  loading: boolean
}

const validationSchema = Yup.object({
  reason: Yup.string()
    .required('Override reason is required')
    .min(10, 'Reason must be at least 10 characters')
    .max(500, 'Reason must be less than 500 characters'),
  override_decision: Yup.string()
    .required('Override decision is required')
    .oneOf(['INDUCT', 'STANDBY', 'MAINTENANCE'], 'Invalid decision'),
})

const OverrideDialog: React.FC<OverrideDialogProps> = ({
  open,
  onClose,
  onSubmit,
  assignment,
  loading,
}) => {
  const formik = useFormik({
    initialValues: {
      reason: '',
      user_id: 'current_user', // This would come from auth context
      override_decision: assignment?.decision.decision || 'STANDBY',
    },
    validationSchema,
    onSubmit: (values) => {
      onSubmit(values)
    },
  })

  const handleClose = () => {
    formik.resetForm()
    onClose()
  }

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        Override Assignment Decision
      </DialogTitle>

      <DialogContent>
        {assignment && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Trainset ID
            </Typography>
            <Typography variant="h6">
              {assignment.trainset_id}
            </Typography>
            
            <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Current Decision
                </Typography>
                <Typography variant="body1">
                  {assignment.decision.decision}
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Confidence Score
                </Typography>
                <Typography variant="body1">
                  {Math.round(assignment.decision.confidence_score * 100)}%
                </Typography>
              </Box>
            </Box>
          </Box>
        )}

        <Alert severity="warning" sx={{ mb: 3 }}>
          Overriding an assignment decision requires supervisor approval and will be logged for audit purposes.
        </Alert>

        <form onSubmit={formik.handleSubmit}>
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel>Override Decision</InputLabel>
            <Select
              name="override_decision"
              value={formik.values.override_decision}
              onChange={formik.handleChange}
              error={formik.touched.override_decision && Boolean(formik.errors.override_decision)}
            >
              <MenuItem value="INDUCT">INDUCT - Deploy for service</MenuItem>
              <MenuItem value="STANDBY">STANDBY - Keep in standby</MenuItem>
              <MenuItem value="MAINTENANCE">MAINTENANCE - Send for maintenance</MenuItem>
            </Select>
          </FormControl>

          <TextField
            fullWidth
            multiline
            rows={4}
            name="reason"
            label="Override Reason"
            placeholder="Please provide a detailed reason for overriding the AI recommendation..."
            value={formik.values.reason}
            onChange={formik.handleChange}
            error={formik.touched.reason && Boolean(formik.errors.reason)}
            helperText={formik.touched.reason && formik.errors.reason}
            sx={{ mb: 3 }}
          />

          <Typography variant="body2" color="text.secondary">
            This override will be recorded in the audit log and may require additional approvals.
          </Typography>
        </form>
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={() => formik.handleSubmit()}
          variant="contained"
          disabled={loading || !formik.isValid}
        >
          {loading ? 'Submitting...' : 'Submit Override'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default OverrideDialog

import React from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Typography,
  Alert,
  List,
  ListItem,
  ListItemText,
  Chip,
} from '@mui/material'
import { useFormik } from 'formik'
import * as Yup from 'yup'

interface ApprovalDialogProps {
  open: boolean
  onClose: () => void
  onSubmit: (data: { user_id: string; comments?: string }) => void
  assignmentCount: number
  loading: boolean
}

const validationSchema = Yup.object({
  comments: Yup.string()
    .max(500, 'Comments must be less than 500 characters'),
})

const ApprovalDialog: React.FC<ApprovalDialogProps> = ({
  open,
  onClose,
  onSubmit,
  assignmentCount,
  loading,
}) => {
  const formik = useFormik({
    initialValues: {
      user_id: 'current_user', // This would come from auth context
      comments: '',
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
        Approve Assignments
      </DialogTitle>

      <DialogContent>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            {assignmentCount} Assignment{assignmentCount > 1 ? 's' : ''} Selected for Approval
          </Typography>
          
          <Alert severity="info" sx={{ mb: 2 }}>
            Approving these assignments will lock them for morning execution. 
            This action cannot be undone.
          </Alert>

          <Typography variant="body2" color="text.secondary" gutterBottom>
            Approval Summary:
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText
                primary="Status Change"
                secondary="PENDING → APPROVED"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Execution"
                secondary="Locked for morning deployment"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Audit Trail"
                secondary="Action will be logged with timestamp and user"
              />
            </ListItem>
          </List>
        </Box>

        <form onSubmit={formik.handleSubmit}>
          <TextField
            fullWidth
            multiline
            rows={3}
            name="comments"
            label="Approval Comments (Optional)"
            placeholder="Add any additional notes about this approval..."
            value={formik.values.comments}
            onChange={formik.handleChange}
            error={formik.touched.comments && Boolean(formik.errors.comments)}
            helperText={formik.touched.comments && formik.errors.comments}
            sx={{ mb: 2 }}
          />

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Chip label="Supervisor Approval Required" color="primary" size="small" />
            <Chip label="Audit Logged" color="secondary" size="small" />
          </Box>

          <Typography variant="body2" color="text.secondary">
            By approving these assignments, you confirm that all AI recommendations 
            have been reviewed and are ready for execution.
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
          color="primary"
          disabled={loading}
        >
          {loading ? 'Approving...' : `Approve ${assignmentCount} Assignment${assignmentCount > 1 ? 's' : ''}`}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default ApprovalDialog

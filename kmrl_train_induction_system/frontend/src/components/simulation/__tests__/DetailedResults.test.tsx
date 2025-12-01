/**
 * Unit tests for DetailedResults component
 * Tests that component never crashes with malformed data
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DetailedResults } from '../DetailedResults';

describe('DetailedResults', () => {
  it('renders empty state when results is null', () => {
    render(<DetailedResults results={null} />);
    expect(screen.getByText(/No detailed results available/i)).toBeInTheDocument();
  });

  it('renders empty state when results is undefined', () => {
    render(<DetailedResults results={undefined} />);
    expect(screen.getByText(/No detailed results available/i)).toBeInTheDocument();
  });

  it('renders empty state when results is empty array', () => {
    render(<DetailedResults results={[]} />);
    expect(screen.getByText(/No detailed results available/i)).toBeInTheDocument();
  });

  it('renders results when results is an object (converts to array)', () => {
    const singleResult = {
      trainset_id: 'T-001',
      decision: 'INDUCT',
      score: 0.85,
      confidence_score: 0.9
    };
    render(<DetailedResults results={singleResult} />);
    expect(screen.getByText('T-001')).toBeInTheDocument();
    expect(screen.getByText('INDUCT')).toBeInTheDocument();
  });

  it('renders results when results is an array', () => {
    const results = [
      { trainset_id: 'T-001', decision: 'INDUCT', score: 0.85, confidence_score: 0.9 },
      { trainset_id: 'T-002', decision: 'STANDBY', score: 0.75, confidence_score: 0.8 }
    ];
    render(<DetailedResults results={results} />);
    expect(screen.getByText('T-001')).toBeInTheDocument();
    expect(screen.getByText('T-002')).toBeInTheDocument();
  });

  it('opens explanation modal when Explain button is clicked', async () => {
    const result = {
      trainset_id: 'T-001',
      decision: 'INDUCT',
      explain: 'This trainset was inducted due to high fitness score and low maintenance needs.'
    };
    render(<DetailedResults results={[result]} />);
    
    const explainButton = screen.getByText('Explain');
    fireEvent.click(explainButton);
    
    await waitFor(() => {
      expect(screen.getByText(/This trainset was inducted/i)).toBeInTheDocument();
    });
  });

  it('extracts explanation from explain_log field', async () => {
    const result = {
      trainset_id: 'T-001',
      decision: 'INDUCT',
      explain_log: ['Reason 1', 'Reason 2']
    };
    render(<DetailedResults results={[result]} />);
    
    const explainButton = screen.getByText('Explain');
    fireEvent.click(explainButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Reason 1/i)).toBeInTheDocument();
    });
  });

  it('extracts explanation from reason field', async () => {
    const result = {
      trainset_id: 'T-001',
      decision: 'INDUCT',
      reason: 'High fitness score'
    };
    render(<DetailedResults results={[result]} />);
    
    const explainButton = screen.getByText('Explain');
    fireEvent.click(explainButton);
    
    await waitFor(() => {
      expect(screen.getByText(/High fitness score/i)).toBeInTheDocument();
    });
  });

  it('shows fallback message when no explanation available', async () => {
    const result = {
      trainset_id: 'T-001',
      decision: 'INDUCT'
    };
    render(<DetailedResults results={[result]} />);
    
    const explainButton = screen.getByText('Explain');
    fireEvent.click(explainButton);
    
    await waitFor(() => {
      expect(screen.getByText(/No explanation available/i)).toBeInTheDocument();
    });
  });

  it('handles missing trainset_id gracefully', () => {
    const result = {
      decision: 'INDUCT',
      score: 0.85
    };
    render(<DetailedResults results={[result]} />);
    // Should render without crashing
    expect(screen.getByText(/Item 1/i)).toBeInTheDocument();
  });

  it('handles missing decision field gracefully', () => {
    const result = {
      trainset_id: 'T-001',
      score: 0.85
    };
    render(<DetailedResults results={[result]} />);
    expect(screen.getByText('T-001')).toBeInTheDocument();
    expect(screen.getByText('UNKNOWN')).toBeInTheDocument();
  });

  it('closes modal when close button is clicked', async () => {
    const result = {
      trainset_id: 'T-001',
      decision: 'INDUCT',
      explain: 'Test explanation'
    };
    render(<DetailedResults results={[result]} />);
    
    const explainButton = screen.getByText('Explain');
    fireEvent.click(explainButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Test explanation/i)).toBeInTheDocument();
    });
    
    const closeButton = screen.getByText('×');
    fireEvent.click(closeButton);
    
    await waitFor(() => {
      expect(screen.queryByText(/Test explanation/i)).not.toBeInTheDocument();
    });
  });

  it('handles malformed result objects without crashing', () => {
    const malformedResults = [
      null,
      undefined,
      {},
      { some: 'random', fields: 123 },
      { trainset_id: null, decision: undefined }
    ];
    render(<DetailedResults results={malformedResults} />);
    // Should render without throwing
    expect(screen.getByText(/Detailed Results/i)).toBeInTheDocument();
  });
});








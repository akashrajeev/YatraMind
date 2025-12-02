/**
 * Defensive component for rendering Detailed Results in What-If Simulation
 * Never crashes even if results are missing, not an array, or malformed
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { asArray, extractExplanation } from "@/utils/simulation";

interface DetailedResultsProps {
  results: any;
  className?: string;
}

export function DetailedResults({ results, className }: DetailedResultsProps) {
  const [selectedExplanation, setSelectedExplanation] = useState<string | null>(null);
  const [showExplanationModal, setShowExplanationModal] = useState(false);

  // Safely convert to array - never crashes
  const safeResults = asArray(results);

  const handleExplainClick = (result: any) => {
    const explanation = extractExplanation(result);
    setSelectedExplanation(explanation);
    setShowExplanationModal(true);
  };

  if (safeResults.length === 0) {
    return (
      <div className={className}>
        <Card>
          <CardHeader>
            <CardTitle>Detailed Results</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-center py-4">
              No detailed results available
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <>
      <div className={className}>
        <Card>
          <CardHeader>
            <CardTitle>Detailed Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {safeResults.map((result: any, index: number) => {
                // Safely extract fields with defaults
                const trainsetId = result?.trainset_id || result?.id || `Item ${index + 1}`;
                const decision = result?.decision || result?.role || "UNKNOWN";
                const score = typeof result?.score === 'number' ? result.score : 0;
                const confidence = typeof result?.confidence_score === 'number' ? result.confidence_score : 0;
                
                return (
                  <div 
                    key={index} 
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    style={{ pointerEvents: 'auto' }} // Ensure clickable
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
                        {index + 1}
                      </div>
                      <span className="font-medium truncate">{trainsetId}</span>
                      <Badge 
                        variant={decision === 'INDUCT' ? 'default' : 'secondary'}
                        className="flex-shrink-0"
                      >
                        {decision}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-4 text-sm flex-shrink-0">
                      <span className="text-muted-foreground">
                        Score: {Math.round(score * 100)}%
                      </span>
                      <span className="text-muted-foreground">
                        Confidence: {Math.round(confidence * 100)}%
                      </span>
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation(); // Prevent event bubbling
                          handleExplainClick(result);
                        }}
                        className="relative z-10" // Ensure button is above other elements
                        style={{ pointerEvents: 'auto' }} // Explicitly enable pointer events
                      >
                        <span className="mr-1">👁️</span>
                        Explain
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Explanation Modal */}
      {showExplanationModal && (
        <div 
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => setShowExplanationModal(false)}
          style={{ pointerEvents: 'auto' }}
        >
          <Card 
            className="max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside
            style={{ pointerEvents: 'auto' }}
          >
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">Explanation</CardTitle>
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowExplanationModal(false)}
                  className="relative z-10"
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none">
                <p className="whitespace-pre-wrap text-sm">
                  {selectedExplanation || "No explanation available"}
                </p>
              </div>
              <div className="flex justify-end mt-4">
                <Button 
                  variant="outline" 
                  onClick={() => setShowExplanationModal(false)}
                  className="relative z-10"
                >
                  Close
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  );
}



/**
 * Utility functions for safely handling What-If simulation results
 * Ensures results is always an array to prevent "filter is not a function" errors
 */

/**
 * Safely convert results to an array
 * Handles cases where results might be an object, null, undefined, or already an array
 */
export function ensureResultsArray(results: any): any[] {
  if (Array.isArray(results)) {
    return results;
  }
  
  if (results && typeof results === 'object') {
    // Convert object to array
    return [results];
  }
  
  // Return empty array for null, undefined, or other falsy values
  return [];
}

/**
 * Alias for ensureResultsArray - simpler name for common use
 */
export function asArray<T = any>(value: any): T[] {
  return ensureResultsArray(value) as T[];
}

/**
 * Safely access results with array methods
 * Wraps results to ensure it's always an array before calling array methods
 */
export function safeResultsAccess<T>(
  results: any,
  callback: (arr: any[]) => T
): T {
  const arrayResults = ensureResultsArray(results);
  return callback(arrayResults);
}

/**
 * Get baseline result from simulation results
 */
export function getBaselineResult(simulationData: any): any | null {
  const results = ensureResultsArray(simulationData?.results);
  return results.find((r: any) => r?.type === 'baseline') || null;
}

/**
 * Get scenario result from simulation results
 */
export function getScenarioResult(simulationData: any): any | null {
  const results = ensureResultsArray(simulationData?.results);
  return results.find((r: any) => r?.type === 'scenario') || null;
}

/**
 * Safely filter results array
 */
export function filterResults(results: any, predicate: (item: any) => boolean): any[] {
  return ensureResultsArray(results).filter(predicate);
}

/**
 * Safely map over results array
 */
export function mapResults<T>(results: any, mapper: (item: any, index: number) => T): T[] {
  return ensureResultsArray(results).map(mapper);
}

/**
 * Get results length safely
 */
export function getResultsLength(results: any): number {
  return ensureResultsArray(results).length;
}

/**
 * Extract explanation text from a result object
 * Checks multiple possible fields in order of preference
 */
export function extractExplanation(result: any): string {
  if (!result || typeof result !== 'object') {
    return "No explanation available";
  }
  
  // Try multiple possible fields
  const possibleFields = [
    result.explain,
    result.explain_log,
    result.reason,
    result.summary?.explain,
    result.summary,
    result.reasons?.join?.('. '),
    Array.isArray(result.reasons) ? result.reasons.join('. ') : null,
    result.explanation,
    result.description
  ];
  
  for (const field of possibleFields) {
    if (field) {
      if (typeof field === 'string' && field.trim()) {
        return field.trim();
      }
      if (Array.isArray(field) && field.length > 0) {
        return field.map((item: any) => typeof item === 'string' ? item : String(item)).join('. ');
      }
    }
  }
  
  return "No explanation available";
}


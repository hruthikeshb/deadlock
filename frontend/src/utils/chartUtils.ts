import type { DisplayVariable } from '../types/weather';
import { getDisplayValue, LAT_COUNT, LON_COUNT } from './weatherUtils';

export interface TimeseriesData {
  days: number[];
  prime: number[];
  multiverse?: number[];
}

/**
 * Computes spatial mean over all 32x36 cells or a subregion for a given variable across rollout steps.
 */
export function computeSpatialMeanTimeseries(
  rollout: number[][][][], // [steps, 6, 32, 36]
  variable: DisplayVariable,
  latRange: [number, number] = [0, LAT_COUNT - 1],
  lonRange: [number, number] = [0, LON_COUNT - 1]
): number[] {
  const steps = rollout.length;
  const values: number[] = [];

  const minLat = Math.min(latRange[0], latRange[1]);
  const maxLat = Math.max(latRange[0], latRange[1]);
  const minLon = Math.min(lonRange[0], lonRange[1]);
  const maxLon = Math.max(lonRange[0], lonRange[1]);

  const cellCount = (maxLat - minLat + 1) * (maxLon - minLon + 1);
  if (cellCount <= 0) return Array(steps).fill(0);

  for (let s = 0; s < steps; s++) {
    const frame = rollout[s];
    let sum = 0;
    for (let i = minLat; i <= maxLat; i++) {
      for (let j = minLon; j <= maxLon; j++) {
        sum += getDisplayValue(frame, i, j, variable);
      }
    }
    values.push(Number((sum / cellCount).toFixed(2)));
  }

  return values;
}

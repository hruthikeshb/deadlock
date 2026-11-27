import { LAT_COUNT, LON_COUNT } from './weatherUtils';

export interface ValidationResult {
  valid: boolean;
  message: string;
}

export function validateRolloutResponse(
  rollout: unknown,
  expectedSteps: number
): ValidationResult {
  if (!Array.isArray(rollout)) {
    return { valid: false, message: 'Rollout is not an array' };
  }

  if (rollout.length !== expectedSteps) {
    return {
      valid: false,
      message: `Rollout step count mismatch: expected ${expectedSteps}, got ${rollout.length}`,
    };
  }

  for (let s = 0; s < rollout.length; s++) {
    const step = rollout[s];
    if (!Array.isArray(step) || step.length !== 6) {
      return { valid: false, message: `Step ${s} channel count mismatch (expected 6)` };
    }

    for (let c = 0; c < 6; c++) {
      const channel = step[c];
      if (!Array.isArray(channel) || channel.length !== LAT_COUNT) {
        return { valid: false, message: `Step ${s} ch ${c} latitude dimension mismatch (expected ${LAT_COUNT})` };
      }

      for (let i = 0; i < LAT_COUNT; i++) {
        const row = channel[i];
        if (!Array.isArray(row) || row.length !== LON_COUNT) {
          return { valid: false, message: `Step ${s} ch ${c} row ${i} lon dimension mismatch (expected ${LON_COUNT})` };
        }

        for (let j = 0; j < LON_COUNT; j++) {
          const val = row[j];
          if (typeof val !== 'number' || isNaN(val) || !isFinite(val)) {
            return { valid: false, message: `Nonfinite value at step ${s}, ch ${c}, [${i},${j}]: ${val}` };
          }
          // Precipitation non-negative check
          if (c === 1 && val < -0.001) {
            return { valid: false, message: `Negative precipitation detected: ${val} mm/day` };
          }
          // Temperature physical bounds check (233.15K to 333.15K)
          if (c === 0 && (val < 200 || val > 360)) {
            return { valid: false, message: `Unphysical temperature detected: ${val} K` };
          }
        }
      }
    }
  }

  return { valid: true, message: `Valid rollout array: [${expectedSteps}, 6, ${LAT_COUNT}, ${LON_COUNT}]` };
}

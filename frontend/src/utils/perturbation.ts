import type { ShockConfiguration } from '../types/weather';
import { CHANNEL_INDICES, LAT_COUNT, LON_COUNT } from './weatherUtils';
import { getAffectedGridCells } from './mapCoordinates';

/**
 * Builds the perturbation array of shape [6, 32, 36] adhering to the backend API contract.
 * Returns null if shock is disabled or no center is selected.
 */
export function buildPerturbationArray(shock: ShockConfiguration): number[][][] | null {
  if (!shock.enabled || !shock.center || shock.magnitude === 0) {
    return null;
  }

  // Shape: [6, 32, 36] filled with zeros
  const perturbation: number[][][] = Array.from({ length: 6 }, () =>
    Array.from({ length: LAT_COUNT }, () => Array(LON_COUNT).fill(0))
  );

  const channelIdx = CHANNEL_INDICES[shock.variable];
  const affectedCells = getAffectedGridCells(
    shock.center.latIndex,
    shock.center.lonIndex,
    shock.radius
  );

  for (const cell of affectedCells) {
    // If radius > 0, apply gaussian or smooth falloff, or direct uniform magnitude
    const dx = cell.lonIndex - shock.center.lonIndex;
    const dy = cell.latIndex - shock.center.latIndex;
    const distSq = dx * dx + dy * dy;
    const maxDistSq = (shock.radius + 0.5) * (shock.radius + 0.5);
    
    // Smooth cosine falloff from center to edge
    const factor = shock.radius === 0 ? 1.0 : Math.cos((Math.PI / 2) * Math.sqrt(distSq / maxDistSq));
    const delta = shock.magnitude * Math.max(0.2, factor);

    perturbation[channelIdx][cell.latIndex][cell.lonIndex] = Number(delta.toFixed(3));
  }

  return perturbation;
}

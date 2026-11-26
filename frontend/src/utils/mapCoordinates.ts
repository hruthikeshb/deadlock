import type { GridCoordinates } from '../types/weather';
import { LAT_COUNT, LON_COUNT, LAT_MIN, LON_MIN } from './weatherUtils';

/**
 * Coordinate grids:
 * Latitudes: 6.5, 7.5, ..., 37.5 (32 points, index 0 is 6.5°N, index 31 is 37.5°N)
 * Longitudes: 65.5, 66.5, ..., 100.5 (36 points, index 0 is 65.5°E, index 35 is 100.5°E)
 */

export function getLatForIndex(latIdx: number): number {
  return LAT_MIN + latIdx * 1.0;
}

export function getLonForIndex(lonIdx: number): number {
  return LON_MIN + lonIdx * 1.0;
}

export function getIndexForLat(lat: number): number {
  const idx = Math.round((lat - LAT_MIN) / 1.0);
  return Math.max(0, Math.min(LAT_COUNT - 1, idx));
}

export function getIndexForLon(lon: number): number {
  const idx = Math.round((lon - LON_MIN) / 1.0);
  return Math.max(0, Math.min(LON_COUNT - 1, idx));
}

/**
 * Maps screen/viewBox coordinates (X: 0..width, Y: 0..height) to grid indices.
 * Note: Y=0 corresponds to Northernmost latitude (LAT_MAX=37.5),
 * Y=height corresponds to Southernmost latitude (LAT_MIN=6.5).
 */
export function viewCoordsToGrid(
  x: number,
  y: number,
  viewWidth: number,
  viewHeight: number
): GridCoordinates | null {
  if (x < 0 || x > viewWidth || y < 0 || y > viewHeight) {
    return null;
  }

  // lon: 0 -> LON_MIN, viewWidth -> LON_MAX
  const lonFraction = x / viewWidth;
  const lonIndex = Math.min(LON_COUNT - 1, Math.max(0, Math.floor(lonFraction * LON_COUNT)));

  // lat: 0 -> LAT_MAX (row 31), viewHeight -> LAT_MIN (row 0)
  const latFraction = y / viewHeight;
  const invertedRow = Math.min(LAT_COUNT - 1, Math.max(0, Math.floor(latFraction * LAT_COUNT)));
  const latIndex = LAT_COUNT - 1 - invertedRow; // Convert screen row to latIndex

  return {
    latIndex,
    lonIndex,
    lat: getLatForIndex(latIndex),
    lon: getLonForIndex(lonIndex),
  };
}

/**
 * Maps grid cell indices to screen/viewBox bounding box [x, y, cellWidth, cellHeight].
 */
export function gridToViewBBox(
  latIndex: number,
  lonIndex: number,
  viewWidth: number,
  viewHeight: number
): { x: number; y: number; width: number; height: number } {
  const cellWidth = viewWidth / LON_COUNT;
  const cellHeight = viewHeight / LAT_COUNT;

  const x = lonIndex * cellWidth;
  const invertedRow = LAT_COUNT - 1 - latIndex;
  const y = invertedRow * cellHeight;

  return {
    x,
    y,
    width: cellWidth,
    height: cellHeight,
  };
}

/**
 * Returns list of grid coordinates within a given radius around a center cell.
 */
export function getAffectedGridCells(
  centerLatIdx: number,
  centerLonIdx: number,
  radius: number
): Array<{ latIndex: number; lonIndex: number }> {
  const cells: Array<{ latIndex: number; lonIndex: number }> = [];

  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy <= radius * radius + 0.5) {
        const latIdx = centerLatIdx + dy;
        const lonIdx = centerLonIdx + dx;
        if (latIdx >= 0 && latIdx < LAT_COUNT && lonIdx >= 0 && lonIdx < LON_COUNT) {
          cells.push({ latIndex: latIdx, lonIndex: lonIdx });
        }
      }
    }
  }

  return cells;
}

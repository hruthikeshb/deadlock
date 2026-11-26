import initialConditionsRaw from '../data/initialConditions.json';
import type { InitialAtmosphericData, SimulationResponse } from '../types/weather';

export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '';
export const PRIMARY_MODEL_PREFIX = 'grast-primary-';

export function isPrimaryGrastModel(modelName: string): boolean {
  return modelName.startsWith(PRIMARY_MODEL_PREFIX);
}

export function isSupportedForecastModel(modelName: string): boolean {
  return isPrimaryGrastModel(modelName) || modelName === 'persistence-p0';
}

export interface HealthResponse {
  status: string;
  model_name: string;
}

export async function checkBackendHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const resp = await fetch(`${BACKEND_URL}/health`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
  });

  if (!resp.ok) {
    throw new Error(`Health check failed with status ${resp.status}`);
  }

  const health: unknown = await resp.json();
  if (
    typeof health !== 'object' ||
    health === null ||
    !('status' in health) ||
    typeof health.status !== 'string' ||
    health.status !== 'ready' ||
    !('model_name' in health) ||
    typeof health.model_name !== 'string'
  ) {
    throw new Error('Backend health response is invalid or not ready');
  }

  return { status: health.status, model_name: health.model_name };
}

export interface SimulationPayload {
  init_states: number[][][][];
  doy_sincos: [number, number];
  steps: number;
  perturbation: number[][][] | null;
}

export async function requestSimulation(
  payload: SimulationPayload,
  signal?: AbortSignal
): Promise<SimulationResponse> {
  const resp = await fetch(`${BACKEND_URL}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`Simulation failed (${resp.status}): ${errorText}`);
  }

  return (await resp.json()) as SimulationResponse;
}

/**
 * Data adapter providing historical observational initial conditions.
 * Reads verified 3-day atmospheric states [3, 6, 32, 36] and day-of-year sincos.
 */
export function getInitialConditions(): InitialAtmosphericData {
  return initialConditionsRaw as unknown as InitialAtmosphericData;
}

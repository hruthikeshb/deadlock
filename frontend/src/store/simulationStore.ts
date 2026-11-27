import { create } from 'zustand';
import type {
  DisplayVariable,
  GridCoordinates,
  ShockConfiguration,
  SimulationDiagnosticsData,
} from '../types/weather';
import {
  checkBackendHealth,
  fetchInitialConditions,
  isSupportedForecastModel,
  requestSimulation,
  type SimulationPayload,
} from '../services/api';
import { buildPerturbationArray } from '../utils/perturbation';
import { validateRolloutResponse } from '../utils/validation';

interface SimulationState {
  // Navigation & mode
  activeTab: 'simulation' | 'docs';

  // Weather display selection
  selectedVariable: DisplayVariable;
  selectedDay: number; // 1 through 10 (0-indexed frame: selectedDay - 1)
  forecastSteps: number;
  isPlaying: boolean;

  // Coordinates & shock state
  hoveredCell: GridCoordinates | null;
  shockSelectionActive: boolean;
  shockConfig: ShockConfiguration;

  // Simulation outputs
  primeRollout: number[][][][] | null; // [steps, 6, 32, 36]
  multiverseRollout: number[][][][] | null; // [steps, 6, 32, 36]

  // Execution states
  isLoadingPrime: boolean;
  isLoadingMultiverse: boolean;
  errorMessage: string | null;

  // Diagnostics
  diagnostics: SimulationDiagnosticsData;

  // Actions
  setActiveTab: (tab: 'simulation' | 'docs') => void;
  setSelectedVariable: (variable: DisplayVariable) => void;
  setSelectedDay: (day: number) => void;
  setForecastSteps: (steps: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setHoveredCell: (cell: GridCoordinates | null) => void;
  setShockSelectionActive: (active: boolean) => void;
  updateShockConfig: (updates: Partial<ShockConfiguration>) => void;
  setShockLocation: (coords: GridCoordinates) => void;

  // Core simulation runners
  checkHealth: () => Promise<boolean>;
  runPrimeSimulation: (force?: boolean) => Promise<void>;
  runMultiverseSimulation: () => Promise<void>;
  testZeroShockConsistency: () => Promise<void>;
}

// In-memory cache to prevent redundant duplicate simulations
let cachedPrime: { steps: number; rollout: number[][][][]; inferenceMs: number } | null = null;
const multiverseCache = new Map<string, { rollout: number[][][][]; inferenceMs: number }>();

export const useSimulationStore = create<SimulationState>((set, get) => ({
  activeTab: 'simulation',

  selectedVariable: 't2m',
  selectedDay: 1,
  forecastSteps: 10,
  isPlaying: false,

  hoveredCell: null,
  shockSelectionActive: false,
  shockConfig: {
    enabled: true,
    variable: 't2m',
    magnitude: 3.0,
    radius: 1, // 3x3 surrounding radius
    center: {
      latIndex: 12, // approx Central India (18.5°N, 77.5°E)
      lonIndex: 12,
      lat: 18.5,
      lon: 77.5,
    },
  },

  primeRollout: null,
  multiverseRollout: null,

  isLoadingPrime: false,
  isLoadingMultiverse: false,
  errorMessage: null,

  diagnostics: {
    backendStatus: 'loading',
    modelName: 'Detecting...',
    primeInferenceMs: null,
    multiverseInferenceMs: null,
    forecastSteps: 10,
    validationStatus: 'untested',
    validationMessage: 'Awaiting simulation output validation',
    zeroShockConsistency: null,
    lastSimulatedAt: null,
    initialDate: null,
  },

  setActiveTab: (tab) => set({ activeTab: tab }),
  setSelectedVariable: (selectedVariable) => set({ selectedVariable }),
  setSelectedDay: (selectedDay) => set({ selectedDay }),
  setForecastSteps: (forecastSteps) => set({ forecastSteps }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setHoveredCell: (hoveredCell) => set({ hoveredCell }),
  setShockSelectionActive: (shockSelectionActive) => set({ shockSelectionActive }),

  updateShockConfig: (updates) =>
    set((state) => ({
      shockConfig: { ...state.shockConfig, ...updates },
    })),

  setShockLocation: (coords) =>
    set((state) => ({
      shockConfig: {
        ...state.shockConfig,
        center: coords,
        enabled: true,
      },
      shockSelectionActive: false, // auto-deactivate after clicking location
    })),

  checkHealth: async () => {
    try {
      const res = await checkBackendHealth();
      if (!isSupportedForecastModel(res.model_name)) {
        const message = `Unsupported simulation engine "${res.model_name}".`;
        cachedPrime = null;
        multiverseCache.clear();
        set((state) => ({
          primeRollout: null,
          multiverseRollout: null,
          errorMessage: message,
          diagnostics: {
            ...state.diagnostics,
            backendStatus: 'error',
            modelName: res.model_name,
            validationStatus: 'untested',
            validationMessage: message,
          },
        }));
        return false;
      }

      if (get().diagnostics.modelName !== res.model_name) {
        cachedPrime = null;
        multiverseCache.clear();
      }
      set((state) => ({
        diagnostics: {
          ...state.diagnostics,
          backendStatus: 'connected',
          modelName: res.model_name,
        },
        errorMessage: null,
      }));
      return true;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      cachedPrime = null;
      multiverseCache.clear();
      set((state) => ({
        primeRollout: null,
        multiverseRollout: null,
        diagnostics: {
          ...state.diagnostics,
          backendStatus: 'disconnected',
          modelName: 'Unavailable',
        },
        errorMessage: `Backend disconnected: ${msg}`,
      }));
      return false;
    }
  },

  runPrimeSimulation: async (force = false) => {
    if (!(await get().checkHealth())) {
      return;
    }

    const { forecastSteps } = get();

    if (!force && cachedPrime && cachedPrime.steps === forecastSteps) {
      set({
        primeRollout: cachedPrime.rollout,
        diagnostics: {
          ...get().diagnostics,
          primeInferenceMs: cachedPrime.inferenceMs,
          forecastSteps,
        },
      });
      return;
    }

    set({ isLoadingPrime: true, errorMessage: null });

    try {
      const initData = await fetchInitialConditions();
      const payload: SimulationPayload = {
        init_states: initData.init_states,
        doy_sincos: initData.doy_sincos,
        steps: forecastSteps,
        perturbation: null,
      };
      const response = await requestSimulation(payload);
      const validation = validateRolloutResponse(response.rollout, forecastSteps);

      if (!validation.valid) {
        throw new Error(`Output validation failed: ${validation.message}`);
      }

      cachedPrime = {
        steps: forecastSteps,
        rollout: response.rollout,
        inferenceMs: response.inference_ms,
      };

      set((state) => ({
        primeRollout: response.rollout,
        isLoadingPrime: false,
        diagnostics: {
          ...state.diagnostics,
          backendStatus: 'connected',
          primeInferenceMs: Math.round(response.inference_ms * 10) / 10,
          forecastSteps,
          validationStatus: 'valid',
          validationMessage: validation.message,
          lastSimulatedAt: new Date().toLocaleTimeString(),
          initialDate: initData.date ?? null,
        },
      }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set((state) => ({
        isLoadingPrime: false,
        errorMessage: `Timeline Prime simulation failed: ${msg}`,
        diagnostics: {
          ...state.diagnostics,
          backendStatus: 'error',
          validationStatus: 'invalid',
          validationMessage: msg,
        },
      }));
    }
  },

  runMultiverseSimulation: async () => {
    if (!get().primeRollout) {
      await get().runPrimeSimulation();
      if (!get().primeRollout) {
        return;
      }
    } else if (!(await get().checkHealth())) {
      return;
    }

    const { shockConfig, forecastSteps } = get();
    const perturbation = buildPerturbationArray(shockConfig);
    if (!perturbation) {
      // If perturbation is null, multiverse equals prime
      set({ multiverseRollout: get().primeRollout });
      return;
    }

    // Cache key
    const cacheKey = JSON.stringify({
      steps: forecastSteps,
      variable: shockConfig.variable,
      mag: shockConfig.magnitude,
      radius: shockConfig.radius,
      center: shockConfig.center,
    });

    if (multiverseCache.has(cacheKey)) {
      const cached = multiverseCache.get(cacheKey)!;
      set((state) => ({
        multiverseRollout: cached.rollout,
        diagnostics: {
          ...state.diagnostics,
          multiverseInferenceMs: cached.inferenceMs,
        },
      }));
      return;
    }

    set({ isLoadingMultiverse: true, errorMessage: null });

    try {
      const initData = await fetchInitialConditions();
      const payload: SimulationPayload = {
        init_states: initData.init_states,
        doy_sincos: initData.doy_sincos,
        steps: forecastSteps,
        perturbation,
      };
      const response = await requestSimulation(payload);
      const validation = validateRolloutResponse(response.rollout, forecastSteps);

      if (!validation.valid) {
        throw new Error(`Multiverse rollout invalid: ${validation.message}`);
      }

      multiverseCache.set(cacheKey, {
        rollout: response.rollout,
        inferenceMs: Math.round(response.inference_ms * 10) / 10,
      });

      set((state) => ({
        multiverseRollout: response.rollout,
        isLoadingMultiverse: false,
        diagnostics: {
          ...state.diagnostics,
          multiverseInferenceMs: Math.round(response.inference_ms * 10) / 10,
          validationStatus: 'valid',
          validationMessage: validation.message,
          lastSimulatedAt: new Date().toLocaleTimeString(),
        },
      }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set({
        isLoadingMultiverse: false,
        errorMessage: `Multiverse simulation failed: ${msg}`,
      });
    }
  },

  testZeroShockConsistency: async () => {
    if (!(await get().checkHealth())) {
      return;
    }

    const { forecastSteps } = get();

    // Perturbation of all zeros
    const zeroPerturbation: number[][][] = Array.from({ length: 6 }, () =>
      Array.from({ length: 32 }, () => Array(36).fill(0))
    );

    try {
      const initData = await fetchInitialConditions();
      const [resNull, resZero] = await Promise.all([
        requestSimulation({
          init_states: initData.init_states,
          doy_sincos: initData.doy_sincos,
          steps: forecastSteps,
          perturbation: null,
        }),
        requestSimulation({
          init_states: initData.init_states,
          doy_sincos: initData.doy_sincos,
          steps: forecastSteps,
          perturbation: zeroPerturbation,
        }),
      ]);

      // Check elementwise equality
      const nullRollout = resNull.rollout;
      const zeroRollout = resZero.rollout;
      let identical = true;

      for (let s = 0; s < forecastSteps && identical; s++) {
        for (let c = 0; c < 6 && identical; c++) {
          for (let i = 0; i < 32 && identical; i++) {
            for (let j = 0; j < 36 && identical; j++) {
              if (Math.abs(nullRollout[s][c][i][j] - zeroRollout[s][c][i][j]) > 1e-4) {
                identical = false;
              }
            }
          }
        }
      }

      set((state) => ({
        diagnostics: {
          ...state.diagnostics,
          zeroShockConsistency: identical,
        },
      }));
    } catch {
      set((state) => ({
        diagnostics: {
          ...state.diagnostics,
          zeroShockConsistency: false,
        },
      }));
    }
  },
}));

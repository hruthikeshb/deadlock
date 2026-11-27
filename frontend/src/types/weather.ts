export type WeatherChannel = 't2m' | 'tp' | 'u10' | 'v10' | 'msl' | 'tcwv';

export type DisplayVariable = 't2m' | 'tp' | 'wind_speed' | 'wind_direction' | 'msl' | 'tcwv';

export interface WeatherVariableMeta {
  id: DisplayVariable;
  channelKey?: WeatherChannel;
  name: string;
  fullName: string;
  unit: string;
  displayUnit: string;
  description: string;
  colorScheme: 'temperature' | 'precipitation' | 'wind' | 'pressure' | 'moisture';
  min: number;
  max: number;
}

export interface GridCoordinates {
  latIndex: number;
  lonIndex: number;
  lat: number;
  lon: number;
}

export interface ShockConfiguration {
  enabled: boolean;
  variable: 't2m' | 'u10' | 'v10';
  magnitude: number;
  radius: number; // radius in grid cells (0 = 1 cell, 1 = 3x3, 2 = 5x5, etc.)
  center: GridCoordinates | null;
}

export interface SimulationDiagnosticsData {
  backendStatus: 'connected' | 'disconnected' | 'loading' | 'error';
  modelName: string;
  primeInferenceMs: number | null;
  multiverseInferenceMs: number | null;
  forecastSteps: number;
  validationStatus: 'valid' | 'invalid' | 'untested';
  validationMessage: string;
  zeroShockConsistency: boolean | null;
  lastSimulatedAt: string | null;
  initialDate: string | null;
}

export interface InitialAtmosphericData {
  date?: string;
  init_states: number[][][][]; // [3, 6, 32, 36]
  doy_sincos: [number, number];
  metadata: {
    lats: number[];
    lons: number[];
    channels: WeatherChannel[];
    units: Record<WeatherChannel, string>;
    description: string;
  };
}

export interface SimulationResponse {
  rollout: number[][][][]; // [steps, 6, 32, 36]
  inference_ms: number;
}

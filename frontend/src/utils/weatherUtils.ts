import type { DisplayVariable, WeatherChannel, WeatherVariableMeta } from '../types/weather';

export const LAT_COUNT = 32;
export const LON_COUNT = 36;
export const LAT_MIN = 6.5;
export const LAT_MAX = 37.5;
export const LON_MIN = 65.5;
export const LON_MAX = 100.5;

export const CHANNEL_INDICES: Record<WeatherChannel, number> = {
  t2m: 0,
  tp: 1,
  u10: 2,
  v10: 3,
  msl: 4,
  tcwv: 5,
};

export const WEATHER_VARIABLES: WeatherVariableMeta[] = [
  {
    id: 't2m',
    channelKey: 't2m',
    name: 'Temperature',
    fullName: '2m Air Temperature',
    unit: 'K',
    displayUnit: '°C',
    description: 'Temperature of air at 2m above ground surface',
    colorScheme: 'temperature',
    min: 0, // In Celsius
    max: 45,
  },
  {
    id: 'tp',
    channelKey: 'tp',
    name: 'Precipitation',
    fullName: 'Total Precipitation Rate',
    unit: 'mm/day',
    displayUnit: 'mm/day',
    description: 'Daily accumulated rainfall water depth',
    colorScheme: 'precipitation',
    min: 0,
    max: 60,
  },
  {
    id: 'wind_speed',
    name: 'Wind Speed',
    fullName: '10m Surface Wind Speed',
    unit: 'm/s',
    displayUnit: 'm/s',
    description: 'Horizontal wind speed derived from vector components √(u²+v²)',
    colorScheme: 'wind',
    min: 0,
    max: 25,
  },
  {
    id: 'wind_direction',
    name: 'Wind Direction',
    fullName: '10m Surface Wind Direction',
    unit: 'deg',
    displayUnit: '°',
    description: 'Meteorological wind direction derived from u10 and v10',
    colorScheme: 'wind',
    min: 0,
    max: 360,
  },
  {
    id: 'msl',
    channelKey: 'msl',
    name: 'Sea Level Pressure',
    fullName: 'Mean Sea Level Pressure',
    unit: 'Pa',
    displayUnit: 'hPa',
    description: 'Atmospheric pressure adjusted to mean sea level',
    colorScheme: 'pressure',
    min: 990,
    max: 1018,
  },
  {
    id: 'tcwv',
    channelKey: 'tcwv',
    name: 'Water Vapour',
    fullName: 'Total Column Water Vapour',
    unit: 'kg/m²',
    displayUnit: 'kg/m²',
    description: 'Total atmospheric precipitable water in vertical column',
    colorScheme: 'moisture',
    min: 10,
    max: 70,
  },
];

export function kelvinToCelsius(k: number): number {
  return k - 273.15;
}

export function paToHpa(pa: number): number {
  return pa / 100.0;
}

export function computeWindSpeed(u: number, v: number): number {
  return Math.sqrt(u * u + v * v);
}

export function computeWindDirection(u: number, v: number): number {
  // Meteorological convention: direction from which wind blows, 0=N, 90=E, 180=S, 270=W
  const rad = Math.atan2(-u, -v);
  let deg = (rad * 180) / Math.PI;
  if (deg < 0) deg += 360;
  return deg;
}

/**
 * Extracts display value for a cell at (latIdx, lonIdx) from a [6, 32, 36] state frame.
 */
export function getDisplayValue(
  state: number[][][], // [6, 32, 36]
  latIdx: number,
  lonIdx: number,
  variable: DisplayVariable
): number {
  if (!state || latIdx < 0 || latIdx >= LAT_COUNT || lonIdx < 0 || lonIdx >= LON_COUNT) {
    return 0;
  }

  const t2m = state[CHANNEL_INDICES.t2m][latIdx][lonIdx];
  const tp = state[CHANNEL_INDICES.tp][latIdx][lonIdx];
  const u10 = state[CHANNEL_INDICES.u10][latIdx][lonIdx];
  const v10 = state[CHANNEL_INDICES.v10][latIdx][lonIdx];
  const msl = state[CHANNEL_INDICES.msl][latIdx][lonIdx];
  const tcwv = state[CHANNEL_INDICES.tcwv][latIdx][lonIdx];

  switch (variable) {
    case 't2m':
      return kelvinToCelsius(t2m);
    case 'tp':
      return Math.max(0, tp);
    case 'wind_speed':
      return computeWindSpeed(u10, v10);
    case 'wind_direction':
      return computeWindDirection(u10, v10);
    case 'msl':
      return paToHpa(msl);
    case 'tcwv':
      return Math.max(0, tcwv);
    default:
      return 0;
  }
}

/**
 * Temperature color scale: blue -> cyan -> yellow -> orange -> red
 */
export function getTemperatureColor(celsius: number, min = 0, max = 45): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, (celsius - min) / (max - min)));
  if (t < 0.25) {
    const s = t / 0.25;
    return [Math.round(30 + s * 40), Math.round(90 + s * 120), Math.round(210 + s * 30), 220]; // Deep blue to Cyan
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return [Math.round(70 + s * 160), Math.round(210 + s * 35), Math.round(240 - s * 180), 220]; // Cyan to Yellow-Green
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return [Math.round(230 + s * 20), Math.round(245 - s * 95), Math.round(60 - s * 50), 225]; // Yellow to Orange
  } else {
    const s = (t - 0.75) / 0.25;
    return [Math.round(250 + s * 5), Math.round(150 - s * 120), Math.round(10 + s * 20), 235]; // Orange to Red
  }
}

/**
 * Precipitation color scale: pale blue -> cyan -> deep blue -> purple
 */
export function getPrecipitationColor(tp: number, _min = 0, max = 50): [number, number, number, number] {
  if (tp <= 0.1) return [240, 245, 250, 40]; // Dry / trace
  const t = Math.max(0, Math.min(1, Math.log1p(tp) / Math.log1p(max)));
  if (t < 0.3) {
    const s = t / 0.3;
    return [Math.round(200 - s * 80), Math.round(230 - s * 30), Math.round(245), 180];
  } else if (t < 0.7) {
    const s = (t - 0.3) / 0.4;
    return [Math.round(120 - s * 90), Math.round(200 - s * 80), Math.round(245 - s * 30), 210];
  } else {
    const s = (t - 0.7) / 0.3;
    return [Math.round(30 + s * 90), Math.round(120 - s * 80), Math.round(215 - s * 40), 235];
  }
}

/**
 * Wind speed color scale: light green -> yellow -> orange -> purple
 */
export function getWindColor(speed: number, min = 0, max = 25): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, (speed - min) / (max - min)));
  if (t < 0.33) {
    const s = t / 0.33;
    return [Math.round(160 + s * 70), Math.round(220 + s * 20), Math.round(200 - s * 100), 190];
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return [Math.round(230 + s * 20), Math.round(240 - s * 110), Math.round(100 - s * 70), 215];
  } else {
    const s = (t - 0.66) / 0.34;
    return [Math.round(250 - s * 90), Math.round(130 - s * 90), Math.round(30 + s * 160), 235];
  }
}

/**
 * Pressure color scale: low (purple/blue) -> high (yellow/orange)
 */
export function getPressureColor(hPa: number, min = 990, max = 1018): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, (hPa - min) / (max - min)));
  const r = Math.round(40 + t * 190);
  const g = Math.round(70 + (1 - Math.abs(t - 0.5) * 2) * 120);
  const b = Math.round(180 - t * 130);
  return [r, g, b, 210];
}

/**
 * Moisture color scale: yellow-brown (dry) -> green -> blue (wet)
 */
export function getMoistureColor(tcwv: number, min = 10, max = 70): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, (tcwv - min) / (max - min)));
  if (t < 0.5) {
    const s = t / 0.5;
    return [Math.round(220 - s * 100), Math.round(210 + s * 10), Math.round(140 - s * 40), 200];
  } else {
    const s = (t - 0.5) / 0.5;
    return [Math.round(120 - s * 100), Math.round(220 - s * 80), Math.round(100 + s * 130), 220];
  }
}

/**
 * Diverging color scale for Multiverse - Prime difference:
 * Blue (negative) -> Neutral off-white (zero) -> Red (positive)
 */
export function getDifferenceColor(diff: number, maxAbsDelta: number): [number, number, number, number] {
  if (maxAbsDelta <= 0 || Math.abs(diff) < 0.001) {
    return [245, 246, 248, 15]; // Nearly transparent neutral
  }
  const t = Math.max(-1, Math.min(1, diff / maxAbsDelta));
  if (t < 0) {
    // Negative delta -> Deep blue
    const s = -t;
    return [Math.round(245 - s * 215), Math.round(246 - s * 156), Math.round(248 - s * 38), Math.round(40 + s * 195)];
  } else {
    // Positive delta -> Deep red/crimson
    const s = t;
    return [Math.round(245 - s * 25), Math.round(246 - s * 206), Math.round(248 - s * 208), Math.round(40 + s * 195)];
  }
}

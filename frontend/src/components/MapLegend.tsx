import React from 'react';
import type { DisplayVariable } from '../types/weather';
import { WEATHER_VARIABLES } from '../utils/weatherUtils';

interface MapLegendProps {
  variable: DisplayVariable;
  isDifference?: boolean;
  maxDiff?: number;
}

export const MapLegend: React.FC<MapLegendProps> = ({
  variable,
  isDifference = false,
  maxDiff = 5,
}) => {
  const meta = WEATHER_VARIABLES.find((v) => v.id === variable) || WEATHER_VARIABLES[0];

  if (isDifference) {
    const formattedMax = Number(maxDiff.toFixed(2));
    return (
      <div className="map-legend-bar">
        <span className="mono" style={{ color: 'var(--diff-neg)', fontWeight: 600 }}>
          -{formattedMax} {meta.displayUnit}
        </span>
        <div
          className="legend-gradient"
          style={{
            background: 'linear-gradient(to right, #0284C7 0%, #F5F6F8 50%, #DC2626 100%)',
          }}
        />
        <span className="mono" style={{ color: 'var(--text-secondary)' }}>
          0.00
        </span>
        <div
          className="legend-gradient"
          style={{
            background: 'linear-gradient(to right, #F5F6F8 0%, #DC2626 100%)',
            display: 'none',
          }}
        />
        <span className="mono" style={{ color: 'var(--diff-pos)', fontWeight: 600 }}>
          +{formattedMax} {meta.displayUnit}
        </span>
      </div>
    );
  }

  let gradientStyle = '';
  switch (meta.colorScheme) {
    case 'temperature':
      gradientStyle = 'linear-gradient(to right, #1E5AB4, #46D2F0, #F0E646, #F08228, #E62814)';
      break;
    case 'precipitation':
      gradientStyle = 'linear-gradient(to right, rgba(240,245,250,0.5), #78C8F5, #1E78D7, #1E46B4)';
      break;
    case 'wind':
      gradientStyle = 'linear-gradient(to right, #A0DCB4, #F0DC50, #E65A28, #7828B4)';
      break;
    case 'pressure':
      gradientStyle = 'linear-gradient(to right, #2846B4, #50B478, #D7A028)';
      break;
    case 'moisture':
      gradientStyle = 'linear-gradient(to right, #DCD28C, #64B496, #285AB4)';
      break;
    default:
      gradientStyle = 'linear-gradient(to right, #38BDF8, #F43F5E)';
  }

  return (
    <div className="map-legend-bar">
      <span className="mono">
        {meta.min} {meta.displayUnit}
      </span>
      <div className="legend-gradient" style={{ background: gradientStyle }} />
      <span className="mono">
        {meta.max} {meta.displayUnit}
      </span>
    </div>
  );
};

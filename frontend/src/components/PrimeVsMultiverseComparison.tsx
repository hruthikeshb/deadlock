import React, { useMemo } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { WeatherMap } from './WeatherMap';
import { MapLegend } from './MapLegend';
import { WeatherVariableSelector } from './WeatherVariableSelector';
import { Layers } from 'lucide-react';

export const PrimeVsMultiverseComparison: React.FC = () => {
  const {
    primeRollout,
    multiverseRollout,
    selectedDay,
    selectedVariable,
    shockConfig,
    setShockLocation,
    shockSelectionActive,
  } = useSimulationStore();

  const dayIndex = selectedDay - 1;

  const primeFrame = useMemo(() => {
    if (!primeRollout || dayIndex >= primeRollout.length) return null;
    return primeRollout[dayIndex];
  }, [primeRollout, dayIndex]);

  const multiverseFrame = useMemo(() => {
    if (!multiverseRollout || dayIndex >= multiverseRollout.length) {
      return primeFrame; // Default to prime before perturbation run
    }
    return multiverseRollout[dayIndex];
  }, [multiverseRollout, primeFrame, dayIndex]);

  // Compute maximum absolute difference for dynamic scaling of difference map
  const maxDiff = useMemo(() => {
    if (shockConfig.variable === 't2m') return Math.max(1, Math.abs(shockConfig.magnitude) * 1.2);
    if (shockConfig.variable === 'u10' || shockConfig.variable === 'v10') return Math.max(1, Math.abs(shockConfig.magnitude) * 1.2);
    return 5;
  }, [shockConfig]);

  return (
    <div className="comparison-section">
      <WeatherVariableSelector />

      {/* Two side-by-side maps: Timeline Prime vs Multiverse */}
      <div className="comparison-maps-grid">
        {/* Left: Timeline Prime */}
        <div className="map-card">
          <div className="map-card-header">
            <span className="map-card-badge prime">TIMELINE PRIME (ORIGINAL)</span>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              Day {selectedDay} Forecast
            </span>
          </div>
          <WeatherMap
            dataFrame={primeFrame}
            variable={selectedVariable}
            interactive={true}
            highlightCenter={shockConfig.center}
            highlightRadius={shockConfig.radius}
            onCellClick={(cell) => {
              if (shockSelectionActive) {
                setShockLocation(cell);
              }
            }}
          />
          <MapLegend variable={selectedVariable} />
        </div>

        {/* Right: Multiverse */}
        <div className="map-card">
          <div className="map-card-header">
            <span className="map-card-badge multiverse">MULTIVERSE (ALTERNATE)</span>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              Day {selectedDay} Perturbed
            </span>
          </div>
          <WeatherMap
            dataFrame={multiverseFrame}
            variable={selectedVariable}
            interactive={true}
            highlightCenter={shockConfig.center}
            highlightRadius={shockConfig.radius}
            onCellClick={(cell) => {
              if (shockSelectionActive) {
                setShockLocation(cell);
              }
            }}
          />
          <MapLegend variable={selectedVariable} />
        </div>
      </div>

      {/* Atmospheric Difference Map below */}
      <div className="map-card" style={{ maxWidth: '820px', margin: '0 auto', width: '100%' }}>
        <div className="map-card-header">
          <span className="map-card-badge diff" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Layers size={13} />
            <span>ATMOSPHERIC DIFFERENCE [ MULTIVERSE − TIMELINE PRIME ]</span>
          </span>
          <span className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
            Day {selectedDay} Net Dynamic Delta
          </span>
        </div>
        <WeatherMap
          dataFrame={multiverseFrame}
          comparisonFrame={primeFrame}
          variable={selectedVariable}
          interactive={true}
          highlightCenter={shockConfig.center}
          highlightRadius={shockConfig.radius}
          maxDiff={maxDiff}
        />
        <MapLegend variable={selectedVariable} isDifference={true} maxDiff={maxDiff} />
      </div>
    </div>
  );
};

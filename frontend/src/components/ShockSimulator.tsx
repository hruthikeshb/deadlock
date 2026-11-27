import React from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { Target, Zap, RotateCcw, AlertTriangle } from 'lucide-react';

export const ShockSimulator: React.FC = () => {
  const {
    shockConfig,
    updateShockConfig,
    shockSelectionActive,
    setShockSelectionActive,
    runMultiverseSimulation,
    isLoadingMultiverse,
    forecastSteps,
    setForecastSteps,
    diagnostics,
  } = useSimulationStore();
  const backendReady = diagnostics.backendStatus === 'connected';

  const getVariableUnit = (v: string) => {
    switch (v) {
      case 't2m':
        return 'K';
      case 'u10':
      case 'v10':
        return 'm/s';
      default:
        return '';
    }
  };

  const minMagnitude = shockConfig.variable === 't2m' ? -10 : -15;
  const maxMagnitude = shockConfig.variable === 't2m' ? 10 : 15;
  const stepMagnitude = shockConfig.variable === 't2m' ? 0.5 : 0.5;

  return (
    <div className="panel shock-panel">
      <div className="section-header">
        <span className="section-title">
          <Zap size={15} color="var(--accent-warning)" />
          Shock Simulator (Multiverse)
        </span>
      </div>

      {/* Location Picker */}
      <div className="control-group">
        <span className="control-label">Perturbation Target</span>
        <button
          className={`btn-secondary ${shockSelectionActive ? 'active' : ''}`}
          onClick={() => setShockSelectionActive(!shockSelectionActive)}
        >
          <Target size={14} />
          <span>{shockSelectionActive ? 'Click on Map...' : 'Enable Shock Selection'}</span>
        </button>

        {shockConfig.center ? (
          <div
            className="mono"
            style={{
              fontSize: '11px',
              padding: '6px 8px',
              background: 'var(--bg-panel-subtle)',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-secondary)',
            }}
          >
            <span>Coordinates: </span>
            <strong style={{ color: 'var(--text-primary)' }}>
              {shockConfig.center.lat.toFixed(1)}°N, {shockConfig.center.lon.toFixed(1)}°E
            </strong>
            <span style={{ marginLeft: '6px', opacity: 0.7 }}>
              [Grid {shockConfig.center.latIndex}, {shockConfig.center.lonIndex}]
            </span>
          </div>
        ) : (
          <span style={{ fontSize: '11px', color: 'var(--accent-warning)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <AlertTriangle size={12} />
            No location selected. Click button above then map.
          </span>
        )}
      </div>

      {/* Perturbation Variable */}
      <div className="control-group">
        <span className="control-label">Atmospheric Channel</span>
        <select
          value={shockConfig.variable}
          onChange={(e) =>
            updateShockConfig({
              variable: e.target.value as 't2m' | 'u10' | 'v10',
            })
          }
          style={{
            padding: '6px 8px',
            border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '12px',
            background: 'var(--bg-panel)',
          }}
        >
          <option value="t2m">Temperature Shock (t2m) [Δ K]</option>
          <option value="u10">Zonal Wind Shock (u10) [Δ m/s]</option>
          <option value="v10">Meridional Wind Shock (v10) [Δ m/s]</option>
        </select>
      </div>

      {/* Perturbation Magnitude */}
      <div className="control-group">
        <div className="control-label">
          <span>Magnitude</span>
          <span className="mono">
            {shockConfig.magnitude > 0 ? `+${shockConfig.magnitude}` : shockConfig.magnitude}{' '}
            {getVariableUnit(shockConfig.variable)}
          </span>
        </div>
        <div className="slider-container">
          <input
            type="range"
            min={minMagnitude}
            max={maxMagnitude}
            step={stepMagnitude}
            value={shockConfig.magnitude}
            onChange={(e) => updateShockConfig({ magnitude: parseFloat(e.target.value) })}
          />
        </div>
      </div>

      {/* Affected Area / Radius */}
      <div className="control-group">
        <div className="control-label">
          <span>Affected Region Radius</span>
          <span className="mono">
            {shockConfig.radius === 0
              ? 'Single Cell (1x1)'
              : `Radius ${shockConfig.radius} (${(2 * shockConfig.radius + 1)}x${(2 * shockConfig.radius + 1)} zone)`}
          </span>
        </div>
        <div className="slider-container">
          <input
            type="range"
            min={0}
            max={4}
            step={1}
            value={shockConfig.radius}
            onChange={(e) => updateShockConfig({ radius: parseInt(e.target.value, 10) })}
          />
        </div>
      </div>

      {/* Forecast Horizon */}
      <div className="control-group">
        <div className="control-label">
          <span>Forecast Horizon</span>
          <span className="mono">{forecastSteps} Days</span>
        </div>
        <div className="slider-container">
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={forecastSteps}
            onChange={(e) => setForecastSteps(parseInt(e.target.value, 10))}
          />
        </div>
      </div>

      {/* Run Action Button */}
      <button
        className="btn-primary"
        onClick={() => runMultiverseSimulation()}
        disabled={isLoadingMultiverse || !shockConfig.center || !backendReady}
        style={{ marginTop: '4px' }}
      >
        <Zap size={14} />
        <span>{isLoadingMultiverse ? 'Computing Rollout...' : 'Run Multiverse Simulation'}</span>
      </button>

      {/* Reset perturbation */}
      <button
        className="btn-secondary"
        onClick={() =>
          updateShockConfig({
            magnitude: 0,
            radius: 0,
          })
        }
        title="Reset shock to 0 (Null perturbation)"
      >
        <RotateCcw size={12} />
        <span>Reset Shock to Null</span>
      </button>
    </div>
  );
};

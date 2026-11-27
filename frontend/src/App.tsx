import React, { useEffect } from 'react';
import { TopNavigation } from './components/TopNavigation';
import { DashboardHeader } from './components/DashboardHeader';
import { ForecastTimeline } from './components/ForecastTimeline';
import { ShockSimulator } from './components/ShockSimulator';
import { PrimeVsMultiverseComparison } from './components/PrimeVsMultiverseComparison';
import { WeatherAnalytics } from './components/WeatherAnalytics';
import { SimulationDiagnostics } from './components/SimulationDiagnostics';
import { useSimulationStore } from './store/simulationStore';
import { WeatherMap } from './components/WeatherMap';
import { MapLegend } from './components/MapLegend';
import { WeatherVariableSelector } from './components/WeatherVariableSelector';
import './styles/dashboard.css';

export const App: React.FC = () => {
  const {
    activeTab,
    runPrimeSimulation,
    primeRollout,
    selectedDay,
    selectedVariable,
    shockConfig,
    setShockLocation,
    shockSelectionActive,
    errorMessage,
    isLoadingPrime,
  } = useSimulationStore();

  useEffect(() => {
    void runPrimeSimulation();
  }, [runPrimeSimulation]);

  const dayIndex = selectedDay - 1;
  const currentPrimeFrame = primeRollout && dayIndex < primeRollout.length ? primeRollout[dayIndex] : null;

  return (
    <div className="app-container">
      <TopNavigation />
      <DashboardHeader />

      <main className="main-content">
        {errorMessage && (
          <div className="error-banner" role="alert">
            <span>{errorMessage}</span>
            <button
              className="btn-secondary"
              onClick={() => void runPrimeSimulation(true)}
              disabled={isLoadingPrime}
            >
              {isLoadingPrime ? 'Checking simulation backend...' : 'Retry backend connection'}
            </button>
          </div>
        )}

        {/* Forecast Timeline Strip */}
        <ForecastTimeline />

        {activeTab === 'simulation' && (
          <>
            <div className="simulation-grid">
              <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div className="section-header">
                  <span className="section-title">
                    Observed & Simulated Atmospheric Field (32 × 36 Grid)
                  </span>
                  <span className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                    Day {selectedDay} Forecast
                  </span>
                </div>

                <WeatherVariableSelector />
                <WeatherMap
                  dataFrame={currentPrimeFrame}
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

              <ShockSimulator />
            </div>

            <div className="panel">
              <PrimeVsMultiverseComparison />
            </div>

            <WeatherAnalytics />
            <SimulationDiagnostics />
          </>
        )}

        {activeTab === 'docs' && (
          <div className="panel" style={{ maxWidth: '900px', margin: '0 auto', width: '100%' }}>
            <h2 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px', color: 'var(--accent-primary)' }}>
              MONSOON / TWIN — Scientific Specifications & Architecture
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px', lineHeight: 1.6 }}>
              <p>
                <strong>MONSOON / TWIN</strong> is an interactive Climate Digital Twin of India designed for
                the Yugaantar DevForge Track 2 challenge. It enables scientists and policy planners to visualize
                India's atmospheric evolution and introduce localized atmospheric disturbances to explore alternate
                weather outcomes (Multiverse scenarios).
              </p>
              <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Simulation Modes</h3>
              <ul style={{ paddingLeft: '20px' }}>
                <li>
                  <strong>Timeline Prime:</strong> The unperturbed forecast generated from initial atmospheric
                  conditions using <code>perturbation = null</code>.
                </li>
                <li>
                  <strong>Multiverse:</strong> The alternate forecast generated after introducing an atmospheric shock
                  tensor [6, 32, 36] at selected coordinates.
                </li>
                <li>
                  <strong>Atmospheric Difference:</strong> Exact pixel-wise delta (<code>Multiverse − Prime</code>)
                  visualizing disturbance propagation and dampening.
                </li>
              </ul>
              <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Grid & Coordinate System</h3>
              <p>
                The model operates on a regular spatial grid of <strong>32 rows × 36 columns (1.0° resolution)</strong>,
                spanning latitudes <strong>6.5°N to 37.5°N</strong> and longitudes <strong>65.5°E to 100.5°E</strong>.
              </p>
              <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Physical Channels</h3>
              <ul style={{ paddingLeft: '20px' }}>
                <li><code>t2m</code>: 2m Temperature (Kelvin, converted to °C for display)</li>
                <li><code>tp</code>: Total Precipitation (mm/day)</li>
                <li><code>u10</code>, <code>v10</code>: 10m Eastward and Northward Wind Velocity (m/s)</li>
                <li><code>msl</code>: Mean Sea Level Pressure (Pascals, converted to hPa for display)</li>
                <li><code>tcwv</code>: Total Column Water Vapour (kg/m²)</li>
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;

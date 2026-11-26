import React from 'react';
import { useSimulationStore } from '../store/simulationStore';

export const DashboardHeader: React.FC = () => {
  const { diagnostics, forecastSteps } = useSimulationStore();

  return (
    <section className="dashboard-header">
      <div className="header-info">
        <h1>India Atmospheric Simulation</h1>
        <p>Explore the evolution of India's atmosphere and simulate alternate climatic conditions.</p>
      </div>

      <div className="metadata-strip">
        <div className="meta-item">
          <span className="meta-label">Spatial Resolution</span>
          <span className="meta-value mono">32 × 36 (1.0°)</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Forecast Horizon</span>
          <span className="meta-value mono">{forecastSteps} Days</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Dynamic Variables</span>
          <span className="meta-value mono">6 Channels</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Simulation Engine</span>
          <span className="meta-value mono">{diagnostics.modelName || 'Detecting...'}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Execution Mode</span>
          <span className="meta-value mono" style={{ color: 'var(--accent-primary)' }}>
            Physical-Unit Copied
          </span>
        </div>
      </div>
    </section>
  );
};

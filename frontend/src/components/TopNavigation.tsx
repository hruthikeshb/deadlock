import React from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { Activity, Cpu } from 'lucide-react';

export const TopNavigation: React.FC = () => {
  const { activeTab, setActiveTab, diagnostics, runPrimeSimulation } = useSimulationStore();

  const getStatusClass = () => {
    switch (diagnostics.backendStatus) {
      case 'connected':
        return 'connected';
      case 'disconnected':
      case 'error':
        return 'disconnected';
      case 'loading':
      default:
        return 'loading';
    }
  };

  const getStatusText = () => {
    switch (diagnostics.backendStatus) {
      case 'connected':
        return 'Simulation Engine Ready';
      case 'disconnected':
        return 'Disconnected';
      case 'error':
        return 'Unsupported Engine';
      case 'loading':
      default:
        return 'Connecting...';
    }
  };

  return (
    <header className="top-nav">
      <div className="brand-section">
        <div className="brand-title">
          <Activity size={18} strokeWidth={2.4} />
          <span>MONSOON / TWIN</span>
        </div>
        <span className="brand-badge">INDIA CLIMATE DIGITAL TWIN</span>
      </div>

      <nav className="nav-tabs">
        <button
          className={`nav-tab-btn ${activeTab === 'simulation' ? 'active' : ''}`}
          onClick={() => setActiveTab('simulation')}
        >
          Simulation
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'docs' ? 'active' : ''}`}
          onClick={() => setActiveTab('docs')}
        >
          Documentation
        </button>
      </nav>

      <div className="status-section">
        <div
          className={`connection-pill ${getStatusClass()}`}
          title="Click to verify the simulation backend and retry the forecast"
          onClick={() => void runPrimeSimulation(true)}
          style={{ cursor: 'pointer' }}
        >
          <span className="connection-dot" />
          <span>{getStatusText()}</span>
        </div>

        <div className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Cpu size={12} />
            <span>v1.0-JUDGE</span>
          </span>
        </div>
      </div>
    </header>
  );
};

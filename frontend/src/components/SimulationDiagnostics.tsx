import React from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { CheckCircle2, XCircle, AlertCircle, RefreshCw } from 'lucide-react';

export const SimulationDiagnostics: React.FC = () => {
  const {
    diagnostics,
    shockConfig,
    testZeroShockConsistency,
  } = useSimulationStore();

  const getValidationIcon = () => {
    switch (diagnostics.validationStatus) {
      case 'valid':
        return <CheckCircle2 size={14} color="var(--accent-success)" />;
      case 'invalid':
        return <XCircle size={14} color="var(--accent-danger)" />;
      default:
        return <AlertCircle size={14} color="var(--accent-warning)" />;
    }
  };

  return (
    <div className="panel">
      <div className="section-header">
        <span className="section-title">Technical Diagnostics & Quality Assurance</span>
        <button
          className="btn-secondary"
          onClick={() => testZeroShockConsistency()}
          style={{ fontSize: '11px', padding: '4px 8px' }}
          title="Verify that null perturbation produces identical rollout as zero perturbation"
        >
          <RefreshCw size={11} />
          <span>Test Zero-Shock Consistency</span>
        </button>
      </div>

      <div className="diagnostics-panel">
        <div className="diagnostic-item">
          <span className="diag-title">Backend Status</span>
          <span className="diag-val" style={{ textTransform: 'uppercase' }}>
            {diagnostics.backendStatus}
          </span>
        </div>

        <div className="diagnostic-item">
          <span className="diag-title">Prime Inference Latency</span>
          <span className="diag-val">
            {diagnostics.primeInferenceMs !== null
              ? `${diagnostics.primeInferenceMs} ms`
              : 'N/A'}
          </span>
        </div>

        <div className="diagnostic-item">
          <span className="diag-title">Multiverse Inference Latency</span>
          <span className="diag-val">
            {diagnostics.multiverseInferenceMs !== null
              ? `${diagnostics.multiverseInferenceMs} ms`
              : 'N/A'}
          </span>
        </div>

        <div className="diagnostic-item">
          <span className="diag-title">Output Array Validation</span>
          <span
            className="diag-val"
            style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px' }}
          >
            {getValidationIcon()}
            <span>{diagnostics.validationStatus.toUpperCase()}</span>
          </span>
        </div>

        <div className="diagnostic-item">
          <span className="diag-title">Zero-Shock Determinism</span>
          <span className="diag-val">
            {diagnostics.zeroShockConsistency === null
              ? 'Untested'
              : diagnostics.zeroShockConsistency
              ? 'Passed (Exact Match)'
              : 'Failed (Divergence)'}
          </span>
        </div>

        <div className="diagnostic-item">
          <span className="diag-title">Active Perturbation Setting</span>
          <span className="diag-val" style={{ fontSize: '12px' }}>
            {shockConfig.center
              ? `${shockConfig.variable} Δ=${shockConfig.magnitude} (R=${shockConfig.radius})`
              : 'None'}
          </span>
        </div>
      </div>
    </div>
  );
};

import React from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { WEATHER_VARIABLES } from '../utils/weatherUtils';
import { Thermometer, CloudRain, Wind, Compass, Gauge, Droplet } from 'lucide-react';

export const WeatherVariableSelector: React.FC = () => {
  const { selectedVariable, setSelectedVariable } = useSimulationStore();

  const getIcon = (id: string) => {
    switch (id) {
      case 't2m':
        return <Thermometer size={14} />;
      case 'tp':
        return <CloudRain size={14} />;
      case 'wind_speed':
        return <Wind size={14} />;
      case 'wind_direction':
        return <Compass size={14} />;
      case 'msl':
        return <Gauge size={14} />;
      case 'tcwv':
        return <Droplet size={14} />;
      default:
        return null;
    }
  };

  return (
    <div className="variable-ribbon">
      <span
        style={{
          fontSize: '11px',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: 'var(--text-secondary)',
          marginRight: '6px',
        }}
      >
        Variable:
      </span>
      {WEATHER_VARIABLES.map((v) => {
        const isActive = selectedVariable === v.id;
        return (
          <button
            key={v.id}
            className={`var-btn ${isActive ? 'active' : ''}`}
            onClick={() => setSelectedVariable(v.id)}
            title={v.description}
          >
            {getIcon(v.id)}
            <span>{v.name}</span>
            <span className="unit-badge">[{v.displayUnit}]</span>
          </button>
        );
      })}
    </div>
  );
};

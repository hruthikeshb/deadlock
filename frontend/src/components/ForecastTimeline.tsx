import React, { useEffect } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { Play, Pause } from 'lucide-react';

export const ForecastTimeline: React.FC = () => {
  const {
    selectedDay,
    setSelectedDay,
    forecastSteps,
    isPlaying,
    setIsPlaying,
  } = useSimulationStore();

  // Animation frame playback loop
  useEffect(() => {
    let timer: number | null = null;
    if (isPlaying) {
      timer = window.setInterval(() => {
        setSelectedDay((selectedDay % forecastSteps) + 1);
      }, 950);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, selectedDay, forecastSteps, setSelectedDay]);

  const daysArray = Array.from({ length: forecastSteps }, (_, i) => i + 1);

  return (
    <div className="timeline-panel">
      <button
        className="btn-secondary"
        onClick={() => setIsPlaying(!isPlaying)}
        title={isPlaying ? 'Pause forecast timeline' : 'Animate forecast timeline'}
        style={{ padding: '6px 10px' }}
      >
        {isPlaying ? <Pause size={14} /> : <Play size={14} />}
        <span>{isPlaying ? 'Pause' : 'Play'}</span>
      </button>

      <div className="timeline-days">
        {daysArray.map((day) => {
          const isSelected = selectedDay === day;
          return (
            <button
              key={day}
              className={`day-btn ${isSelected ? 'active' : ''}`}
              onClick={() => {
                setIsPlaying(false);
                setSelectedDay(day);
              }}
            >
              Day {day}
            </button>
          );
        })}
      </div>

      <div
        className="mono"
        style={{
          fontSize: '11px',
          fontWeight: 600,
          color: 'var(--text-secondary)',
          minWidth: '70px',
          textAlign: 'right',
        }}
      >
        Day {selectedDay} / {forecastSteps}
      </div>
    </div>
  );
};

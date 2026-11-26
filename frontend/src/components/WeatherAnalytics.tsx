import React, { useRef, useEffect } from 'react';
import uPlot from 'uplot';
import { useSimulationStore } from '../store/simulationStore';
import { computeSpatialMeanTimeseries } from '../utils/chartUtils';
import { BarChart3 } from 'lucide-react';

interface SingleChartProps {
  title: string;
  unit: string;
  days: number[];
  primeData: number[];
  multiverseData?: number[];
  primeColor?: string;
  multiverseColor?: string;
}

const SingleUPlotChart: React.FC<SingleChartProps> = ({
  title,
  unit,
  days,
  primeData,
  multiverseData,
  primeColor = '#0F3B66',
  multiverseColor = '#D97706',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartInstance = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current || days.length === 0) return;

    // Data format: [ [day1, day2, ...], [primeVal1, ...], [multiverseVal1, ...] ]
    const data: uPlot.AlignedData = multiverseData
      ? [days, primeData, multiverseData]
      : [days, primeData];

    const series: uPlot.Series[] = [
      {},
      {
        label: 'Timeline Prime',
        stroke: primeColor,
        width: 2,
        points: { show: true, size: 5 },
      },
    ];

    if (multiverseData) {
      series.push({
        label: 'Multiverse',
        stroke: multiverseColor,
        width: 2,
        points: { show: true, size: 5 },
        dash: [4, 4],
      });
    }

    const opts: uPlot.Options = {
      width: containerRef.current.clientWidth || 320,
      height: 180,
      cursor: { drag: { x: false, y: false } },
      scales: {
        x: { time: false },
      },
      axes: [
        {
          label: 'Forecast Day',
          stroke: '#64748B',
          grid: { stroke: 'rgba(100, 116, 139, 0.1)' },
          values: (_u, vals) => vals.map((v) => `D${v}`),
        },
        {
          label: unit,
          stroke: '#64748B',
          grid: { stroke: 'rgba(100, 116, 139, 0.1)' },
        },
      ],
      series,
    };

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    chartInstance.current = new uPlot(opts, data, containerRef.current);

    const handleResize = () => {
      if (chartInstance.current && containerRef.current) {
        chartInstance.current.setSize({
          width: containerRef.current.clientWidth,
          height: 180,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [days, primeData, multiverseData, primeColor, multiverseColor, unit]);

  return (
    <div className="chart-card">
      <div className="chart-header">
        <span>{title}</span>
        <span className="mono" style={{ color: 'var(--text-secondary)' }}>
          Spatial Mean [{unit}]
        </span>
      </div>
      <div ref={containerRef} style={{ width: '100%', minHeight: '180px' }} />
    </div>
  );
};

export const WeatherAnalytics: React.FC = () => {
  const { primeRollout, multiverseRollout, forecastSteps } = useSimulationStore();

  const days = Array.from({ length: forecastSteps }, (_, i) => i + 1);

  // Compute timeseries trends from actual returned rollouts
  const primeTemp = primeRollout
    ? computeSpatialMeanTimeseries(primeRollout, 't2m')
    : Array(forecastSteps).fill(0);
  const multiTemp = multiverseRollout
    ? computeSpatialMeanTimeseries(multiverseRollout, 't2m')
    : undefined;

  const primeRain = primeRollout
    ? computeSpatialMeanTimeseries(primeRollout, 'tp')
    : Array(forecastSteps).fill(0);
  const multiRain = multiverseRollout
    ? computeSpatialMeanTimeseries(multiverseRollout, 'tp')
    : undefined;

  const primeWind = primeRollout
    ? computeSpatialMeanTimeseries(primeRollout, 'wind_speed')
    : Array(forecastSteps).fill(0);
  const multiWind = multiverseRollout
    ? computeSpatialMeanTimeseries(multiverseRollout, 'wind_speed')
    : undefined;

  return (
    <div className="panel">
      <div className="section-header">
        <span className="section-title">
          <BarChart3 size={15} />
          Atmospheric Timeseries Trends & Multiverse Divergence
        </span>
      </div>

      <div className="charts-grid">
        <SingleUPlotChart
          title="2m Temperature Trend"
          unit="°C"
          days={days}
          primeData={primeTemp}
          multiverseData={multiTemp}
          primeColor="#0F3B66"
          multiverseColor="#DC2626"
        />

        <SingleUPlotChart
          title="Precipitation Accumulation"
          unit="mm/day"
          days={days}
          primeData={primeRain}
          multiverseData={multiRain}
          primeColor="#0284C7"
          multiverseColor="#D97706"
        />

        <SingleUPlotChart
          title="10m Wind Speed Trend"
          unit="m/s"
          days={days}
          primeData={primeWind}
          multiverseData={multiWind}
          primeColor="#059669"
          multiverseColor="#7C3AED"
        />
      </div>
    </div>
  );
};

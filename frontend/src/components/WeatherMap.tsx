import React, { useRef, useEffect, useMemo, useCallback } from 'react';
import type { DisplayVariable, GridCoordinates } from '../types/weather';
import {
  LAT_COUNT,
  LON_COUNT,
  getDisplayValue,
  getTemperatureColor,
  getPrecipitationColor,
  getWindColor,
  getPressureColor,
  getMoistureColor,
  getDifferenceColor,
  WEATHER_VARIABLES,
} from '../utils/weatherUtils';
import { viewCoordsToGrid, gridToViewBBox, getAffectedGridCells } from '../utils/mapCoordinates';
import { useSimulationStore } from '../store/simulationStore';

interface WeatherMapProps {
  dataFrame: number[][][] | null; // [6, 32, 36]
  comparisonFrame?: number[][][] | null; // If provided, renders difference (dataFrame - comparisonFrame)
  variable: DisplayVariable;
  interactive?: boolean;
  highlightCenter?: GridCoordinates | null;
  highlightRadius?: number;
  onCellClick?: (coords: GridCoordinates) => void;
  maxDiff?: number;
}

const VIEW_WIDTH = 720;
const VIEW_HEIGHT = 800;

export const WeatherMap: React.FC<WeatherMapProps> = ({
  dataFrame,
  comparisonFrame = null,
  variable,
  interactive = true,
  highlightCenter = null,
  highlightRadius = 0,
  onCellClick,
  maxDiff = 5,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { hoveredCell, setHoveredCell, shockSelectionActive } = useSimulationStore();

  const isDiffMode = Boolean(comparisonFrame);
  const meta = useMemo(
    () => WEATHER_VARIABLES.find((v) => v.id === variable) || WEATHER_VARIABLES[0],
    [variable]
  );

  // Render weather grid on Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, VIEW_WIDTH, VIEW_HEIGHT);

    if (!dataFrame) {
      // Draw placeholder grid
      ctx.fillStyle = '#F1F5F9';
      ctx.fillRect(0, 0, VIEW_WIDTH, VIEW_HEIGHT);
      return;
    }

    const cellWidth = VIEW_WIDTH / LON_COUNT;
    const cellHeight = VIEW_HEIGHT / LAT_COUNT;

    for (let latIdx = 0; latIdx < LAT_COUNT; latIdx++) {
      // Invert row index because row 0 is top (lat 37.5) and row 31 is bottom (lat 6.5)
      const screenRow = LAT_COUNT - 1 - latIdx;
      const y = screenRow * cellHeight;

      for (let lonIdx = 0; lonIdx < LON_COUNT; lonIdx++) {
        const x = lonIdx * cellWidth;
        const val = getDisplayValue(dataFrame, latIdx, lonIdx, variable);

        let rgba: [number, number, number, number];

        if (isDiffMode && comparisonFrame) {
          const compVal = getDisplayValue(comparisonFrame, latIdx, lonIdx, variable);
          const diff = val - compVal;
          rgba = getDifferenceColor(diff, maxDiff);
        } else {
          switch (meta.colorScheme) {
            case 'temperature':
              rgba = getTemperatureColor(val, meta.min, meta.max);
              break;
            case 'precipitation':
              rgba = getPrecipitationColor(val, meta.min, meta.max);
              break;
            case 'wind':
              rgba = getWindColor(val, meta.min, meta.max);
              break;
            case 'pressure':
              rgba = getPressureColor(val, meta.min, meta.max);
              break;
            case 'moisture':
              rgba = getMoistureColor(val, meta.min, meta.max);
              break;
            default:
              rgba = [100, 150, 240, 200];
          }
        }

        ctx.fillStyle = `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${rgba[3] / 255})`;
        ctx.fillRect(x, y, cellWidth + 0.5, cellHeight + 0.5);

        // Crisp 32x36 grid cell outline
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.16)';
        ctx.lineWidth = 0.6;
        ctx.strokeRect(x, y, cellWidth, cellHeight);
      }
    }

    // Subtle 5-degree graticule lines to emphasize the 1.0° (32x36) grid structure
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.28)';
    ctx.lineWidth = 0.9;
    for (let lonIdx = 5; lonIdx < LON_COUNT; lonIdx += 5) {
      const gx = lonIdx * cellWidth;
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, VIEW_HEIGHT);
      ctx.stroke();
    }
    for (let latIdx = 5; latIdx < LAT_COUNT; latIdx += 5) {
      const gy = (LAT_COUNT - latIdx) * cellHeight;
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(VIEW_WIDTH, gy);
      ctx.stroke();
    }
  }, [dataFrame, comparisonFrame, variable, isDiffMode, meta, maxDiff]);

  // Handle pointer interactions on SVG
  const handlePointerMove = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (!interactive || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (VIEW_WIDTH / rect.width);
      const y = (e.clientY - rect.top) * (VIEW_HEIGHT / rect.height);

      const cell = viewCoordsToGrid(x, y, VIEW_WIDTH, VIEW_HEIGHT);
      if (cell) {
        if (!hoveredCell || hoveredCell.latIndex !== cell.latIndex || hoveredCell.lonIndex !== cell.lonIndex) {
          setHoveredCell(cell);
        }
      }
    },
    [interactive, hoveredCell, setHoveredCell]
  );

  const handlePointerLeave = useCallback(() => {
    if (interactive) {
      setHoveredCell(null);
    }
  }, [interactive, setHoveredCell]);

  const handleClick = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (!interactive || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (VIEW_WIDTH / rect.width);
      const y = (e.clientY - rect.top) * (VIEW_HEIGHT / rect.height);

      const cell = viewCoordsToGrid(x, y, VIEW_WIDTH, VIEW_HEIGHT);
      if (cell && onCellClick) {
        onCellClick(cell);
      }
    },
    [interactive, onCellClick]
  );

  // Compute bounding boxes for hover and shock selection
  const hoverBBox = useMemo(() => {
    if (!hoveredCell || !interactive) return null;
    return gridToViewBBox(hoveredCell.latIndex, hoveredCell.lonIndex, VIEW_WIDTH, VIEW_HEIGHT);
  }, [hoveredCell, interactive]);

  const shockCenterBBox = useMemo(() => {
    if (!highlightCenter) return null;
    return gridToViewBBox(highlightCenter.latIndex, highlightCenter.lonIndex, VIEW_WIDTH, VIEW_HEIGHT);
  }, [highlightCenter]);

  const shockAreaCells = useMemo(() => {
    if (!highlightCenter || highlightRadius <= 0) return [];
    return getAffectedGridCells(highlightCenter.latIndex, highlightCenter.lonIndex, highlightRadius);
  }, [highlightCenter, highlightRadius]);

  const hoveredVal = useMemo(() => {
    if (!dataFrame || !hoveredCell) return null;
    const v = getDisplayValue(dataFrame, hoveredCell.latIndex, hoveredCell.lonIndex, variable);
    if (isDiffMode && comparisonFrame) {
      const cv = getDisplayValue(comparisonFrame, hoveredCell.latIndex, hoveredCell.lonIndex, variable);
      return Number((v - cv).toFixed(2));
    }
    return Number(v.toFixed(2));
  }, [dataFrame, comparisonFrame, hoveredCell, variable, isDiffMode]);

  return (
    <div className="map-view-wrapper">
      <canvas
        ref={canvasRef}
        width={VIEW_WIDTH}
        height={VIEW_HEIGHT}
        className="map-canvas-layer"
      />

      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        preserveAspectRatio="none"
        className={`map-svg-layer ${shockSelectionActive ? 'selecting-shock' : ''}`}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onClick={handleClick}
      >
        {/* Bundled India Vector Map Layer */}
        <image
          href="/maps/india.svg"
          x="0"
          y="0"
          width={VIEW_WIDTH}
          height={VIEW_HEIGHT}
          preserveAspectRatio="none"
          style={{ opacity: 0.85, pointerEvents: 'none' }}
        />

        {/* Lat/Lon Degree Axis Labels */}
        {[10, 15, 20, 25, 30, 35].map((lat) => {
          const latIdx = lat - 6.5;
          const y = (LAT_COUNT - 1 - latIdx + 0.5) * (VIEW_HEIGHT / LAT_COUNT);
          return (
            <text
              key={`lat-${lat}`}
              x="6"
              y={y + 3}
              fontSize="10"
              fontFamily="monospace"
              fill="rgba(15, 23, 42, 0.7)"
              style={{ pointerEvents: 'none' }}
            >
              {lat}°N
            </text>
          );
        })}
        {[70, 75, 80, 85, 90, 95].map((lon) => {
          const lonIdx = lon - 65.5;
          const x = (lonIdx + 0.5) * (VIEW_WIDTH / LON_COUNT);
          return (
            <text
              key={`lon-${lon}`}
              x={x - 12}
              y={VIEW_HEIGHT - 6}
              fontSize="10"
              fontFamily="monospace"
              fill="rgba(15, 23, 42, 0.7)"
              style={{ pointerEvents: 'none' }}
            >
              {lon}°E
            </text>
          );
        })}

        {/* Shock Area Highlight */}
        {shockAreaCells.map((cell) => {
          const bbox = gridToViewBBox(cell.latIndex, cell.lonIndex, VIEW_WIDTH, VIEW_HEIGHT);
          return (
            <rect
              key={`shock-area-${cell.latIndex}-${cell.lonIndex}`}
              x={bbox.x}
              y={bbox.y}
              width={bbox.width}
              height={bbox.height}
              className="grid-cell-shock-area"
            />
          );
        })}

        {/* Shock Center Highlight */}
        {shockCenterBBox && (
          <rect
            x={shockCenterBBox.x}
            y={shockCenterBBox.y}
            width={shockCenterBBox.width}
            height={shockCenterBBox.height}
            className="grid-cell-shock-center"
          />
        )}

        {/* Hover Cell Highlight */}
        {hoverBBox && (
          <rect
            x={hoverBBox.x}
            y={hoverBBox.y}
            width={hoverBBox.width}
            height={hoverBBox.height}
            className="grid-cell-hover"
          />
        )}
      </svg>

      {/* Hover Readout Tooltip */}
      {interactive && hoveredCell && hoveredVal !== null && (
        <div className="map-cell-tooltip">
          <span>
            {hoveredCell.lat.toFixed(1)}°N, {hoveredCell.lon.toFixed(1)}°E
          </span>
          <span style={{ opacity: 0.6 }}>
            [Grid {hoveredCell.latIndex}, {hoveredCell.lonIndex}]
          </span>
          <span style={{ fontWeight: 700, color: '#38BDF8' }}>
            {hoveredVal > 0 && isDiffMode ? `+${hoveredVal}` : hoveredVal} {meta.displayUnit}
          </span>
        </div>
      )}
    </div>
  );
};

import { useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, PlotMouseEvent, PlotSelectionEvent } from "plotly.js";
import type { EmbeddingPoint } from "./api";

const Plot = createPlotlyComponent(Plotly);
const labelColors = ["#4976d0", "#e08144", "#38a27c", "#9a6cc3", "#c95567", "#718096"];

function colorMap(points: EmbeddingPoint[], palette: string[]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [label, palette[index % palette.length]]));
}

export default function ScatterPlot2D({
  points,
  selectedIndices,
  onToggleSelection,
  onAddSelection,
  viewKey,
  testIdPrefix = "embedding",
  colorValues,
  labelPalette = labelColors,
  pointSizes,
  scoreValues,
  scoreLabel,
}: {
  points: EmbeddingPoint[];
  selectedIndices: number[];
  onToggleSelection: (index: number) => void;
  onAddSelection: (indices: number[]) => void;
  viewKey: string;
  testIdPrefix?: string;
  colorValues?: number[];
  labelPalette?: string[];
  pointSizes?: number[];
  scoreValues?: number[];
  scoreLabel?: string;
}) {
  const [selectionMode, setSelectionMode] = useState<false | "select" | "lasso">(false);
  const colors = colorMap(points, labelPalette);
  const numericColor = colorValues !== undefined && colorValues.length === points.length;
  const numericScores = scoreValues !== undefined && scoreValues.length === points.length;
  const pointPositions = new Map(points.map((point, position) => [point.index, position]));
  const selectedPositions = selectedIndices
    .map((index) => pointPositions.get(index))
    .filter((position): position is number => position !== undefined);
  const data: Data[] = [{
    type: "scatter",
    mode: "markers",
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    text: points.map((point, position) => `${point.filename}<br>Class: ${point.label}<br>Split: ${point.split}`
      + (numericScores && scoreLabel ? `<br>${scoreLabel}: ${scoreValues[position].toFixed(3)}` : "")),
    customdata: points.map((point) => point.index),
    marker: {
      size: points.map((point, position) => {
        const baseSize = pointSizes?.length === points.length ? pointSizes[position] : 8;
        return selectedIndices.includes(point.index) ? Math.max(11, baseSize) : baseSize;
      }),
      color: numericColor
        ? colorValues
        : points.map((point) => colors.get(point.label) || labelPalette[0]),
      ...(numericColor ? {
        colorscale: "Turbo" as const,
        showscale: true,
        colorbar: { title: { text: scoreLabel || "Score" } },
      } : {}),
      symbol: points.map((point) => point.split.toLowerCase() === "test"
        ? "square" : point.split.toLowerCase() === "valid" ? "diamond" : "circle"),
      line: { color: "#ffffff", width: 1 },
    },
    ...(selectedPositions.length ? { selectedpoints: selectedPositions } : {}),
    selected: { marker: { opacity: 1 } },
    unselected: { marker: { opacity: 0.55 } },
    hovertemplate: "%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
  }];
  const layout: Partial<Layout> = {
    autosize: true,
    height: 440,
    margin: { l: 50, r: 12, t: 12, b: 42 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    showlegend: false,
    dragmode: selectionMode,
    uirevision: viewKey,
    xaxis: { title: { text: "Component 1" }, gridcolor: "#e5eaf1", zerolinecolor: "#d7dde7" },
    yaxis: { title: { text: "Component 2" }, gridcolor: "#e5eaf1", zerolinecolor: "#d7dde7" },
  };

  function selectPoint(event: Readonly<PlotMouseEvent>) {
    const custom = event.points?.[0]?.customdata;
    if (typeof custom === "number") onToggleSelection(custom);
  }

  function selectPoints(event: Readonly<PlotSelectionEvent>) {
    const indices = event.points
      .map((point) => point.customdata)
      .filter((custom): custom is number => typeof custom === "number")
      .filter(Number.isInteger);
    if (indices.length) onAddSelection(indices);
  }

  return (
    <div className="plot-frame plot-frame-2d" data-testid={testIdPrefix === "embedding" ? "embedding-plot" : `${testIdPrefix}-plot`}>
      <div className="selection-mode" role="group" aria-label="2D selection mode">
        <button
          type="button"
          data-testid={`${testIdPrefix}-selection-mode-box`}
          aria-pressed={selectionMode === "select"}
          onClick={() => setSelectionMode("select")}
        >Box select</button>
        <button
          type="button"
          data-testid={`${testIdPrefix}-selection-mode-lasso`}
          aria-pressed={selectionMode === "lasso"}
          onClick={() => setSelectionMode("lasso")}
        >Lasso</button>
      </div>
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false, scrollZoom: true, displayModeBar: true }}
        useResizeHandler
        style={{ width: "100%", height: "440px" }}
        onClick={selectPoint}
        onSelected={selectPoints}
      />
      <div className="screen-reader-points" role="group" aria-label="Accessible 2D points">
        {points.map((point) => (
          <button
            key={point.index}
            type="button"
            data-testid={`${testIdPrefix}-point`}
            aria-label={`Select ${point.filename}, ${point.label}, ${point.split}`}
            aria-pressed={selectedIndices.includes(point.index)}
            onClick={() => onToggleSelection(point.index)}
          />
        ))}
      </div>
    </div>
  );
}

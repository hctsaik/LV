import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, PlotMouseEvent } from "plotly.js";
import type { EmbeddingPoint, KMeansCenter } from "./api";

const Plot = createPlotlyComponent(Plotly);
const labelColors = ["#4976d0", "#e08144", "#38a27c", "#9a6cc3", "#c95567", "#718096"];

function colorMap(points: EmbeddingPoint[], palette: string[]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [label, palette[index % palette.length]]));
}

function objectDescription(point: EmbeddingPoint) {
  return point.obj_index == null ? "" : `, object ${point.obj_index + 1}`;
}

export default function ScatterPlot3D({
  points,
  selectedIndices,
  onToggleSelection,
  viewKey,
  clusterCenters = [],
  colorValues,
  labelPalette = labelColors,
  pointSizes,
  scoreValues,
  scoreLabel,
  testIdPrefix = "embedding",
}: {
  points: EmbeddingPoint[];
  selectedIndices: number[];
  onToggleSelection: (index: number) => void;
  viewKey: string;
  clusterCenters?: KMeansCenter[];
  colorValues?: number[];
  labelPalette?: string[];
  pointSizes?: number[];
  scoreValues?: number[];
  scoreLabel?: string;
  testIdPrefix?: string;
}) {
  const colors = colorMap(points, labelPalette);
  const numericColor = colorValues !== undefined && colorValues.length === points.length;
  const numericScores = scoreValues !== undefined && scoreValues.length === points.length;
  const data: Data[] = [{
    type: "scatter3d",
    mode: "markers",
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    z: points.map((point) => point.z),
    text: points.map((point, index) => point.filename + objectDescription(point)
      + (numericScores && scoreLabel ? `<br>${scoreLabel}: ${(scoreValues?.[index] ?? 0).toFixed(3)}` : "")),
    customdata: points.map((point) => [point.index, point.filename, objectDescription(point), point.label, point.split]),
    marker: {
      size: points.map((point, position) => {
        const baseSize = pointSizes?.length === points.length ? pointSizes[position] : 5;
        return selectedIndices.includes(point.index) ? Math.max(7, baseSize) : baseSize;
      }),
      color: numericColor
        ? colorValues
        : points.map((point) => colors.get(point.label) || labelPalette[0]),
      ...(numericColor ? {
        colorscale: "Turbo" as const,
        showscale: true,
        colorbar: { title: { text: scoreLabel || "Score" } },
      } : {}),
      opacity: 0.9,
      line: { color: "#ffffff", width: 0.5 },
    },
    hovertemplate: "%{customdata[1]}%{customdata[2]}<br>Class: %{customdata[3]}<br>Split: %{customdata[4]}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
  }];
  if (clusterCenters.length) {
    data.push({
      type: "scatter3d",
      mode: "text",
      x: clusterCenters.map((center) => center.x),
      y: clusterCenters.map((center) => center.y),
      z: clusterCenters.map((center) => center.z),
      text: clusterCenters.map((center) => `#${center.block_id}`),
      textposition: "top center",
      textfont: { size: 18, color: "#111827" },
      hoverinfo: "skip",
      showlegend: false,
    });
  }
  const layout: Partial<Layout> = {
    autosize: true,
    height: 440,
    margin: { l: 4, r: 4, t: 5, b: 4 },
    paper_bgcolor: "#ffffff",
    showlegend: false,
    uirevision: viewKey,
    scene: {
      dragmode: "orbit",
      xaxis: { title: { text: "Component 1" }, backgroundcolor: "#ffffff", gridcolor: "#e5eaf1" },
      yaxis: { title: { text: "Component 2" }, backgroundcolor: "#ffffff", gridcolor: "#e5eaf1" },
      zaxis: { title: { text: "Component 3" }, backgroundcolor: "#ffffff", gridcolor: "#e5eaf1" },
      bgcolor: "#ffffff",
    },
  };

  function selectPoint(event: Readonly<PlotMouseEvent>) {
    const custom = event.points?.[0]?.customdata;
    if (Array.isArray(custom)) onToggleSelection(Number(custom[0]));
  }

  return (
    <div className="plot-frame plot-frame-3d" data-testid={`${testIdPrefix}-plot-3d`}>
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false, scrollZoom: true }}
        useResizeHandler
        style={{ width: "100%", height: "440px" }}
        onClick={selectPoint}
      />
      <div className="screen-reader-points" role="group" aria-label="Accessible 3D points">
        {points.map((point) => (
          <button
            key={point.index}
            type="button"
            data-testid={`${testIdPrefix}-point`}
            aria-label={"Select " + point.filename + objectDescription(point) + ", " + point.label + ", " + point.split}
            aria-pressed={selectedIndices.includes(point.index)}
            onClick={() => onToggleSelection(point.index)}
          />
        ))}
      </div>
    </div>
  );
}

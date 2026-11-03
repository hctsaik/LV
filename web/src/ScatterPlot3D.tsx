import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, PlotMouseEvent, PlotSelectionEvent } from "plotly.js";
import type { BlockProjectionPoint, EmbeddingPoint, KMeansBlocks } from "./api";

const Plot = createPlotlyComponent(Plotly);
const labelColors = ["#4976d0", "#e08144", "#38a27c", "#9a6cc3", "#c95567", "#718096"];

function colorMap(points: EmbeddingPoint[]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [label, labelColors[index % labelColors.length]]));
}

export default function ScatterPlot3D({ points, selectedIndex, selectedIndices, clusters, onSelect, viewKey }: {
  points: EmbeddingPoint[];
  selectedIndex?: number | null;
  selectedIndices?: number[];
  clusters?: KMeansBlocks | null;
  onSelect: (index: number) => void;
  viewKey: string;
}) {
  const colors = colorMap(points);
  const selected = new Set(selectedIndices ?? (selectedIndex === null || selectedIndex === undefined ? [] : [selectedIndex]));
  const data: Data[] = [{
    type: "scatter3d",
    mode: "markers",
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    z: points.map((point) => point.z),
    text: points.map((point) => point.filename),
    customdata: points.map((point) => [point.index, point.filename, point.label, point.split]),
    marker: {
      size: points.map((point) => selected.has(point.index) ? 7 : 5),
      color: points.map((point) => colors.get(point.label) || labelColors[0]),
      opacity: 0.9,
      line: {
        color: points.map((point) => selected.has(point.index) ? "#172033" : "#ffffff"),
        width: 1,
      },
    },
    hovertemplate: "%{customdata[1]}<br>Class: %{customdata[2]}<br>Split: %{customdata[3]}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
  }];
  const layout: Partial<Layout> = {
    autosize: true,
    height: 440,
    margin: { l: 4, r: 4, t: 5, b: 4 },
    paper_bgcolor: "#ffffff",
    showlegend: false,
    uirevision: viewKey,
    scene: {
      dragmode: "orbit",
      annotations: clusters?.centers.map((center) => ({
        x: center.x,
        y: center.y,
        z: center.z,
        text: "#" + center.block_id,
        showarrow: false,
        font: { size: 16, color: "#172033" },
        bgcolor: "rgba(255,255,255,0.82)",
        borderpad: 3,
      })),
      xaxis: { title: { text: "Component 1" }, backgroundcolor: "#ffffff", gridcolor: "#e5eaf1" },
      yaxis: { title: { text: "Component 2" }, backgroundcolor: "#ffffff", gridcolor: "#e5eaf1" },
      zaxis: { title: { text: "Component 3" }, backgroundcolor: "#ffffff", gridcolor: "#e5eaf1" },
      bgcolor: "#ffffff",
    },
  };

  function selectPoint(event: Readonly<PlotMouseEvent>) {
    const custom = event.points?.[0]?.customdata;
    if (Array.isArray(custom)) onSelect(Number(custom[0]));
  }

  return (
    <div className="plot-frame plot-frame-3d" data-testid="embedding-plot-3d">
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
            data-testid="embedding-point"
            aria-label={"Select " + point.filename + ", " + point.label + ", " + point.split}
            aria-pressed={selected.has(point.index)}
            onClick={() => onSelect(point.index)}
          />
        ))}
      </div>
    </div>
  );
}

export function BlockSelectionPlot({ points, selectedIndices, onAddSelection }: {
  points: BlockProjectionPoint[];
  selectedIndices: number[];
  onAddSelection: (indices: number[]) => void;
}) {
  const selected = new Set(selectedIndices);
  const data: Data[] = [{
    type: "scatter",
    mode: "markers",
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    customdata: points.map((point) => point.index),
    selectedpoints: points.flatMap((point, index) => selected.has(point.index) ? [index] : []),
    marker: { size: 10, color: "#4976d0", line: { color: "#ffffff", width: 0.75 } },
    selected: { marker: { color: "#df8447", size: 12 } },
    unselected: { marker: { opacity: 0.82 } },
    hovertemplate: "Image #%{customdata}<extra></extra>",
  }];
  const layout: Partial<Layout> = {
    autosize: true,
    height: 360,
    margin: { l: 40, r: 16, t: 10, b: 36 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    dragmode: "select",
    selectdirection: "any",
    xaxis: { title: { text: "Component 1" }, zeroline: false, gridcolor: "#edf0f5" },
    yaxis: { title: { text: "Component 2" }, zeroline: false, gridcolor: "#edf0f5" },
    uirevision: "block-" + points.map((point) => point.index).join("-"),
  };

  function addClickedPoint(event: Readonly<PlotMouseEvent>) {
    const index = Number(event.points?.[0]?.customdata);
    if (Number.isInteger(index)) onAddSelection([index]);
  }

  function addSelectedPoints(event: Readonly<PlotSelectionEvent>) {
    const indices = [...new Set(event.points
      .map((point) => Number(point.customdata))
      .filter((index) => Number.isInteger(index)))];
    if (indices.length) onAddSelection(indices);
  }

  return (
    <div className="plot-frame block-selection-frame" data-testid="block-selection-plot">
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToAdd: ["lasso2d"] }}
        useResizeHandler
        style={{ width: "100%", height: "360px" }}
        onClick={addClickedPoint}
        onSelected={addSelectedPoints}
      />
      <div className="screen-reader-points" role="group" aria-label="Accessible block points">
        {points.map((point) => (
          <button
            key={point.index}
            type="button"
            data-testid="block-selection-point"
            data-index={point.index}
            aria-label={"Add image " + (point.index + 1) + " to selection"}
            aria-pressed={selected.has(point.index)}
            onClick={() => onAddSelection([point.index])}
          />
        ))}
      </div>
    </div>
  );
}

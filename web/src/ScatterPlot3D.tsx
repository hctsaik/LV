import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, PlotMouseEvent } from "plotly.js";
import type { EmbeddingPoint } from "./api";

const Plot = createPlotlyComponent(Plotly);
const labelColors = ["#4976d0", "#e08144", "#38a27c", "#9a6cc3", "#c95567", "#718096"];

function colorMap(points: EmbeddingPoint[]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [label, labelColors[index % labelColors.length]]));
}

export default function ScatterPlot3D({ points, selectedIndex, onSelect, viewKey }: {
  points: EmbeddingPoint[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  viewKey: string;
}) {
  const colors = colorMap(points);
  const data: Data[] = [{
    type: "scatter3d",
    mode: "markers",
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    z: points.map((point) => point.z),
    text: points.map((point) => point.filename),
    customdata: points.map((point) => [point.index, point.filename, point.label, point.split]),
    marker: {
      size: points.map((point) => point.index === selectedIndex ? 7 : 5),
      color: points.map((point) => colors.get(point.label) || labelColors[0]),
      opacity: 0.9,
      line: { color: "#ffffff", width: 0.5 },
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
            aria-pressed={selectedIndex === point.index}
            onClick={() => onSelect(point.index)}
          />
        ))}
      </div>
    </div>
  );
}

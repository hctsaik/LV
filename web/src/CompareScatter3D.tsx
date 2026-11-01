import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, PlotMouseEvent } from "plotly.js";
import type { ComparisonPoint } from "./api";

const Plot = createPlotlyComponent(Plotly);

export default function CompareScatter3D({
  points,
  selected,
  onSelect,
  viewKey,
}: {
  points: ComparisonPoint[];
  selected: string[];
  onSelect: (point: ComparisonPoint) => void;
  viewKey: string;
}) {
  const data: Data[] = (["A", "B"] as const).map((group) => {
    const values = points.filter((point) => point.group === group);
    return {
      type: "scatter3d",
      mode: "markers",
      name: group === "A" ? "Folder A" : "Folder B",
      x: values.map((point) => point.x),
      y: values.map((point) => point.y),
      z: values.map((point) => point.z),
      text: values.map((point) => point.filename),
      customdata: values.map((point) => [point.group, point.index]),
      marker: {
        color: group === "A" ? "#3498db" : "#e74c3c",
        size: values.map((point) => selected.includes(point.group + ":" + point.index) ? 7 : 5),
        opacity: 0.86,
        line: { color: "#ffffff", width: 0.5 },
      },
      hovertemplate: "%{text}<br>Group: " + group + "<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
    } as Data;
  });
  const layout: Partial<Layout> = {
    autosize: true,
    height: 440,
    margin: { l: 4, r: 4, t: 5, b: 4 },
    paper_bgcolor: "#ffffff",
    legend: { orientation: "h" },
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
    if (Array.isArray(custom)) {
      const found = points.find((point) => point.group === custom[0] && point.index === Number(custom[1]));
      if (found) onSelect(found);
    }
  }

  return (
    <div className="plot-frame plot-frame-3d" data-testid="comparison-plot-3d">
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false, scrollZoom: true }}
        useResizeHandler
        style={{ width: "100%", height: "440px" }}
        onClick={selectPoint}
      />
      <div className="screen-reader-points" role="group" aria-label="Accessible comparison points">
        {points.map((point) => (
          <button
            key={point.group + ":" + point.index}
            type="button"
            data-testid="comparison-point"
            aria-label={"Select " + point.filename + " from group " + point.group}
            aria-pressed={selected.includes(point.group + ":" + point.index)}
            onClick={() => onSelect(point)}
          />
        ))}
      </div>
    </div>
  );
}

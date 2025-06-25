# %%
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from bokeh.io import output_file, output_notebook
from bokeh.layouts import column, row
from bokeh.models import Circle, CustomJS, Div, HoverTool, LabelSet, Range1d, TapTool
from bokeh.plotting import ColumnDataSource, figure, from_networkx, show

from alfred_evolve.database import Database, model_to_program
from alfred_evolve.models.database_models import ProgramModel

# %%
root_dir = Path(__file__).parent.parent.parent.parent
data_dir = root_dir / "data"
docs_dir = root_dir / "docs"

# %%
db = Database(str(data_dir / "programs.db"))
with db.get_session() as session:
    query = session.query(ProgramModel)
    programs = [model_to_program(pm) for pm in query.all()]
s = programs[-1].artifacts["packing_0"]
s = "[[0.1, 0.1, 0.1], [0.30000000000000004, 0.1, 0.1], [0.5, 0.1, 0.1], [0.7000000000000001, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.30000000000000004, 0.1], [0.30000000000000004, 0.30000000000000004, 0.1], [0.5, 0.30000000000000004, 0.1], [0.7000000000000001, 0.30000000000000004, 0.1], [0.9, 0.30000000000000004, 0.1], [0.1, 0.5, 0.1], [0.30000000000000004, 0.5, 0.1], [0.5, 0.5, 0.1], [0.7000000000000001, 0.5, 0.1], [0.9, 0.5, 0.1], [0.1, 0.7000000000000001, 0.1], [0.30000000000000004, 0.7000000000000001, 0.1], [0.5, 0.7000000000000001, 0.1], [0.7000000000000001, 0.7000000000000001, 0.1], [0.9, 0.7000000000000001, 0.1], [0.1, 0.9, 0.1], [0.30000000000000004, 0.9, 0.1], [0.5, 0.9, 0.1], [0.7000000000000001, 0.9, 0.1], [0.9, 0.9, 0.1], [1e-09, 1e-09, 1e-09]]"


def parse_packing(s: str) -> list[list[float]]:
    a = [[float(x) for x in l.split(", ")] for l in s[2:-2].split("], [")]
    return a


packing = parse_packing(s)
packing


# %%
def get_evolutionary_tree(db_url: str, n=500) -> nx.DiGraph:
    """
    Load the evolutionary tree from the database and return it as a directed graph.
    """
    db = Database(db_url)
    with db.get_session() as session:
        query = session.query(ProgramModel)
        if n is not None:
            query = query.limit(n)
        programs = [model_to_program(pm) for pm in query.all()]

        graph = nx.DiGraph()
        for program in programs:
            artifacts = program.artifacts
            if artifacts is None:
                packing = None
            else:
                packing = artifacts.get("packing_0", None)
                if packing is None:
                    packing = artifacts.get("packing", None)
                if packing is not None:
                    packing = parse_packing(packing)
            diff = program.diff
            if diff is not None:
                diff_lines = []
                is_red_line = True
                for line in diff.splitlines():
                    if line == "<<<<<<<< SEARCH":
                        is_red_line = True
                    if line == "========":
                        is_red_line = False
                    color_xml = (
                        "<font color='red'>" if is_red_line else "<font color='green'>"
                    )
                    diff_lines.append(f"{color_xml}{line}</font>")
                diff = "\n".join(diff_lines)
            diff = diff.strip() if diff else None
            reasoning = program.reasoning.strip() if program.reasoning else None
            prompt = program.prompt.strip() if program.prompt else None
            graph.add_node(
                program.id,
                parent_id=program.parent_id,
                island=program.island_id,
                generation=program.generation,
                score=program.scores.get("SCORE", 0.0),
                reasoning=reasoning,
                diff=diff,
                prompt=prompt,
                content=program.content,
                packing=packing,
            )
            if program.parent_id:
                graph.add_edge(program.parent_id, program.id, weight=1.0)
            if program.inspired_by_ids:
                for inspired_id in program.inspired_by_ids:
                    graph.add_edge(inspired_id, program.id, weight=0.5)
        return graph


def get_radial_pos(G: nx.DiGraph) -> dict:
    # Compute positions: x by island, y by generation (higher generation = lower y)
    islands = sorted({d["island"] for n, d in G.nodes(data=True)})
    island_to_x = {island: i for i, island in enumerate(islands)}
    generations = {d["generation"] for n, d in G.nodes(data=True)}
    max_gen = max(generations) if generations else 1

    # Use polar coordinates: angle by island, radius by generation
    pos = {}
    num_islands = len(islands)
    angle_per_island = 2 * np.pi / max(1, num_islands)
    for n, d in G.nodes(data=True):
        island_idx = island_to_x.get(d["island"], 0)
        angle = island_idx * angle_per_island
        # Spread out nodes with the same island and generation in the angle
        same_group = [
            nn
            for nn, dd in G.nodes(data=True)
            if dd["island"] == d["island"] and dd["generation"] == d["generation"]
        ]
        if len(same_group) > 1:
            idx = same_group.index(n)
            spread = angle_per_island * 0.8  # spread in radians
            offset = (idx - (len(same_group) - 1) / 2) * (
                spread / max(1, len(same_group) - 1)
            )
            angle = angle + offset
        # root is at center, higher generations further out
        radius = 0.05 + 0.95 * (d["generation"] / max_gen) ** 3
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[n] = (x, y)
    return pos


def display_info(
    source,
    info_div,
    diff_div,
    prompt_div,
    reasoning_div,
    packing_ds,
    default_color,
    hover_color,
    ancestor_color,
    default_idx,
) -> CustomJS:
    return CustomJS(
        args=dict(
            source=source,
            info_div=info_div,
            diff_div=diff_div,
            prompt_div=prompt_div,
            reasoning_div=reasoning_div,
            packing_ds=packing_ds,
        ),
        code=f"""
const node_renderer = cb_obj.renderers[0];
const node_data = node_renderer.data_source;

const {{x, y}} = cb_data.geometries;
let idx = -1;
let min_dist = Infinity;
for (let i = 0; i < node_data.data.x.length; i++) {{
    const dx = node_data.data.x[i] - x;
    const dy = node_data.data.y[i] - y;
    const dist = dx * dx + dy * dy;
    const r = node_data.data.size[i];
    if (dist < r * r) {{
        // Point is inside the circle
        if (dist < min_dist) {{
            min_dist = dist;
            idx = i;
        }}
    }}
}}
// If not inside any circle, fall back to closest node
if (idx === -1) {{
    let idx = {default_idx};
}}

// Reset all nodes to default color
for (let i = 0; i < node_data.data.fill_color.length; i++) {{
    node_data.data.fill_color = node_data.data['fill_color'] || [];
    node_data.data.fill_color[i] = "{default_color}";
}}
// Highlight all ancestors of the hovered node
const id = node_data.data.index[idx];
let current = id;
while (current !== null) {{
    const current_idx = node_data.data.index.indexOf(current);
    if (current === id) {{
        node_data.data.fill_color[current_idx] = "{hover_color}";
    }} else {{
        node_data.data.fill_color[current_idx] = "{ancestor_color}";
    }}
    current = node_data.data.parent_id[current_idx] || null;
}}
node_data.change.emit();

// info div
const score = node_data.data.score[idx];
const content = node_data.data.content[idx];
info_div.text = "<b>Program # " + id + ":</b><br>";
info_div.text += "[Score: " + score + "]<br>";
info_div.text += "<pre style='white-space: pre-wrap;'><code class='language-python'>";
info_div.text += content;
info_div.text += "</code></pre>";

// reasoning div
const reasoning = node_data.data.reasoning[idx];
let reasoning_text = "<b>Reasoning:</b><br>";
reasoning_text += "<pre style='white-space: pre-wrap;'><code class='language-text'>";
reasoning_text += reasoning ? reasoning : "No reasoning available";
reasoning_text += "</code></pre>";
reasoning_div.text = reasoning_text;

// diff div
const diff = node_data.data.diff[idx];
let diff_text = "<b>Diff:</b><br>";
diff_text += "<pre style='white-space: pre-wrap;'><code class='language-diff'>";
diff_text += diff ? diff : "No diff available";
diff_text += "</code></pre>";
diff_div.text = diff_text;

// prompt div
const prompt = node_data.data.prompt[idx];
let prompt_text = "<b>Prompt:</b><br>";
prompt_text += "<pre style='white-space: pre-wrap;'><code class='language-text'>";
if (prompt) {{
    // Escape HTML special characters to show raw XML tags
    prompt_text += prompt.replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;');
}} else {{
    prompt_text += "No prompt available";
}}
prompt_text += "</code></pre>";
prompt_div.text = prompt_text;

// Update the packing circles
const packing = node_data.data.packing[idx];
if (packing) {{
    const packing_x = [];
    const packing_y = [];
    const packing_r = [];
    const packing_idx = [];
    for (let i = 0; i < packing.length; i++) {{
        packing_x.push(packing[i][0]);
        packing_y.push(packing[i][1]);
        packing_r.push(packing[i][2]);
        packing_idx.push(i);
    }}
    packing_ds.data = {{
        x: packing_x,
        y: packing_y,
        r: packing_r,
        idx: packing_idx
    }};
}} else {{
    packing_ds.data = {{x: [], y: [], r: [], idx: []}};
}}
packing_ds.change.emit();
""",
    )


def _make_packing_plot(height, width):
    packing_plot = figure(
        title="Circle Packing Visualization",
        width=width,
        height=height,
        toolbar_location=None,
        match_aspect=True,
    )
    packing_plot.xgrid.visible = False
    packing_plot.ygrid.visible = False
    packing_plot.axis.visible = False
    packing_ds = ColumnDataSource(data=dict(x=[], y=[], r=[], idx=[]))
    packing_plot.quad(
        left=0,
        right=1,
        bottom=0,
        top=1,
        fill_color=None,
        line_color="black",
        line_width=1,
        alpha=1,
    )
    packing_plot.circle(
        x="x",
        y="y",
        radius="r",
        source=packing_ds,
        radius_units="data",
        fill_color="#ffcc00",
        line_color="black",
        line_width=1,
        alpha=0.7,
    )
    return packing_plot, packing_ds


def _make_bokeh_graph(G, pos, height, info_div, diff_div, prompt_div, reasoning_div, packing_ds, default_idx):
    plot = figure(
        title="Evolutionary Tree of Programs",
        width=height,
        height=height,
        toolbar_location=None,
    )
    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.axis.visible = False

    # Create a Bokeh graph from the NetworkX graph
    hover_color = "#ff7f0e"
    default_color = "#1f77b4"
    ancestor_color = "#ffcc00"

    # Normalize scores to a reasonable range for node_size
    scores = np.array([d.get("score", 0.0) for n, d in G.nodes(data=True)])
    min_size, max_size = 0.01, 0.04
    exponent = 2
    sizes = (scores / scores.max()) ** exponent * (max_size - min_size) + min_size
    source = ColumnDataSource(
        data={
            "index": list(G.nodes()),
            "parent_id": [d.get("parent_id", None) for n, d in G.nodes(data=True)],
            "score": scores,
            "size": sizes,
            "content": [d.get("content", "") for n, d in G.nodes(data=True)],
            "reasoning": [d.get("reasoning", "") for n, d in G.nodes(data=True)],
            "diff": [d.get("diff", "") for n, d in G.nodes(data=True)],
            "prompt": [d.get("prompt", "") for n, d in G.nodes(data=True)],
            "fill_color": [default_color] * len(G.nodes()),
            "packing": [d.get("packing", None) for n, d in G.nodes(data=True)],
            "x": [pos[n][0] for n in G.nodes()],
            "y": [pos[n][1] for n in G.nodes()],
        }
    )

    # Draw edges: solid for weight 1.0, dashed for weight 0.5
    edge_attrs = {}
    for u, v, d in G.edges(data=True):
        if d.get("weight", 1.0) == 0.5:
            edge_attrs[(u, v)] = {
                "line_dash": "dashed",
                "line_color": "gray",
                "line_alpha": 0.5,
            }
        else:
            edge_attrs[(u, v)] = {
                "line_dash": "solid",
                "line_color": "black",
                "line_alpha": 1.0,
            }

        u_island = G.nodes[u].get("island")
        v_island = G.nodes[v].get("island")
        if u_island != v_island:
            edge_attrs[(u, v)]["line_color"] = "red"
            edge_attrs[(u, v)]["line_alpha"] = 0.5
    nx.set_edge_attributes(G, edge_attrs)

    nodes = from_networkx(G, pos, scale=1, center=(0, 0))
    nodes.node_renderer.data_source = source
    nodes.node_renderer.glyph = Circle(
        radius="size", fill_color="fill_color", line_color="black"
    )
    nodes.edge_renderer.glyph.line_dash = {"field": "line_dash"}
    nodes.edge_renderer.glyph.line_color = {"field": "line_color"}
    nodes.edge_renderer.glyph.line_alpha = {"field": "line_alpha"}

    hover_tool = HoverTool(
        tooltips=[
            ("ID", "@index"),
            ("Score", "@score"),
        ],
        # renderers=[nodes.node_renderer],
        # callback=display_info(
        #     source,
        #     info_div,
        #     diff_div,
        #     packing_ds,
        #     default_color,
        #     hover_color,
        #     ancestor_color,
        #     default_idx,
        # ),
    )
    tap_tool = TapTool(
        renderers=[nodes.node_renderer],
        callback=display_info(
            source,
            info_div,
            diff_div,
            prompt_div,
            reasoning_div,
            packing_ds,
            default_color,
            hover_color,
            ancestor_color,
            default_idx,
        ),
        behavior="inspect",
    )
    plot.add_tools(hover_tool, tap_tool)
    plot.renderers.append(nodes)
    return plot


def draw_graph_bokeh(G, pos, default_idx):
    height = 1200
    div_width = 420
    packing_plot, packing_ds = _make_packing_plot(int(height * 0.4), div_width)
    div_styles = {
        "overflow-x": "auto",
        "overflow-y": "auto",
        "margin-top": "0px",
        "background": "#f9f9f9",
        "padding": "2px",
        "font-size": "10px",
    }
    info_div = Div(
        text="Hover over nodes to see details",
        width=div_width,
        height=height,
        styles=div_styles,
    )
    diff_div = Div(
        text="Hover over nodes to see diffs",
        width=div_width,
        height=int(height * 0.6),
        styles=div_styles,
    )
    prompt_div = Div(
        text="Click on nodes to see prompts",
        width=div_width,
        height=int(height * 0.7),
        styles=div_styles,
    )
    reasoning_div = Div(
        text="Click on nodes to see reasoning",
        width=div_width,
        height=int(height * 0.3),
        styles=div_styles,
    )
    plot = _make_bokeh_graph(
        G,
        pos,
        height,
        info_div,
        diff_div,
        prompt_div,
        reasoning_div,
        packing_ds,
        default_idx,
    )
    layout = row(
        plot,
        info_div,
        column(packing_plot, diff_div),
        column(prompt_div, reasoning_div),
    )
    return layout


# db_url = data_dir / "sota.db"
db_url = data_dir / "sota.db"
G = get_evolutionary_tree(str(db_url), n=None)
pos = get_radial_pos(G)
best_program = max(G.nodes(data=True), key=lambda x: x[1]["score"])
best_id = best_program[0]
layout = draw_graph_bokeh(G, pos, default_idx=best_id - 1)
output_notebook()
output_file(docs_dir / f"{db_url.stem}.html")
show(layout)

# %%
best_score = best_program[1]["score"]
best_content = best_program[1]["content"]
print(f"Best program ID: {best_id}")
print(f"Score: {best_score}")
print(f"Content:\n{best_content}")

# %%

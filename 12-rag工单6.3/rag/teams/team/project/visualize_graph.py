"""
知识图谱可视化 - 3D WebGL版
比pyvis流畅10倍，支持1000+节点无卡顿
"""

import networkx as nx
import json
from pathlib import Path


def graph_to_json(graphml_path: str, max_nodes: int = 500) -> dict:
    """将graphml转为3d-force-graph需要的JSON格式"""
    G = nx.read_graphml(graphml_path)
    print(f"原始图谱: {len(G.nodes)} 节点, {len(G.edges)} 边")

    # 节点太多时只保留度数最高的
    if len(G.nodes) > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
        print(f"裁剪后: {len(G.nodes)} 节点, {len(G.edges)} 边")

    # 实体类型颜色
    type_colors = {
        "company": "#ff6b6b",
        "person": "#4ecdc4",
        "product": "#45b7d1",
        "method": "#96ceb4",
        "concept": "#ffeaa7",
        "location": "#dda0dd",
        "organization": "#ff9ff3",
        "content": "#a29bfe",
    }

    nodes = []
    for node_id in G.nodes:
        nd = G.nodes[node_id]
        entity_type = nd.get("entity_type", "default")
        degree = G.degree(node_id)
        desc = nd.get("description", "")
        if len(desc) > 300:
            desc = desc[:297] + "..."
        nodes.append({
            "id": str(node_id),
            "label": str(node_id),
            "type": entity_type,
            "color": type_colors.get(entity_type, "#6c5ce7"),
            "size": max(3, min(15, 2 + degree)),
            "degree": degree,
            "description": desc,
        })

    links = []
    for src, tgt in G.edges:
        ed = G.edges[src, tgt]
        desc = ed.get("description", "")
        if len(desc) > 200:
            desc = desc[:197] + "..."
        links.append({
            "source": str(src),
            "target": str(tgt),
            "weight": ed.get("weight", 1),
            "description": desc,
        })

    return {"nodes": nodes, "links": links}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>知识图谱 3D</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a1a; overflow: hidden; font-family: 'Microsoft YaHei', sans-serif; }
#graph { width: 100vw; height: 100vh; }
#search-box {
    position: fixed; top: 16px; left: 16px; z-index: 10;
    display: flex; gap: 8px;
}
#search {
    width: 280px; padding: 8px 14px;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px; color: #fff; font-size: 14px; outline: none;
}
#search:focus { border-color: #4ecdc4; }
#search::placeholder { color: rgba(255,255,255,0.4); }
#stats {
    position: fixed; top: 16px; right: 16px; z-index: 10;
    color: rgba(255,255,255,0.6); font-size: 13px; text-align: right;
}
#tooltip {
    position: fixed; display: none; padding: 12px 16px;
    background: rgba(10,10,30,0.95); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px; color: #fff; font-size: 13px; max-width: 360px;
    pointer-events: none; z-index: 20; backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
#tooltip .title { font-size: 15px; font-weight: bold; margin-bottom: 6px; }
#tooltip .type { color: #4ecdc4; font-size: 12px; }
#tooltip .desc { color: rgba(255,255,255,0.7); margin-top: 6px; line-height: 1.5; }
#legend {
    position: fixed; bottom: 16px; left: 16px; z-index: 10;
    display: flex; flex-wrap: wrap; gap: 10px;
}
.legend-item {
    display: flex; align-items: center; gap: 5px;
    color: rgba(255,255,255,0.6); font-size: 12px;
}
.legend-dot {
    width: 10px; height: 10px; border-radius: 50%;
}
</style>
</head>
<body>
<div id="graph"></div>
<div id="search-box">
    <input id="search" type="text" placeholder="搜索节点...">
</div>
<div id="stats"></div>
<div id="tooltip">
    <div class="title"></div>
    <div class="type"></div>
    <div class="desc"></div>
</div>
<div id="legend"></div>

<script src="https://unpkg.com/3d-force-graph@1.73.4/dist/3d-force-graph.min.js"></script>
<script>
const data = __GRAPH_DATA__;

// 统计
const typeCounts = {};
data.nodes.forEach(n => {
    typeCounts[n.type] = (typeCounts[n.type] || 0) + 1;
});
document.getElementById('stats').innerHTML =
    `<div>${data.nodes.length} 节点 · ${data.links.length} 边</div>`;

// 图例
const legend = document.getElementById('legend');
Object.entries(typeCounts).sort((a,b) => b[1]-a[1]).forEach(([type, count]) => {
    const color = data.nodes.find(n => n.type === type)?.color || '#6c5ce7';
    legend.innerHTML += `<div class="legend-item"><span class="legend-dot" style="background:${color}"></span>${type} (${count})</div>`;
});

const Graph = ForceGraph3D()(document.getElementById('graph'))
    .graphData(data)
    .nodeVal('size')
    .nodeColor('color')
    .nodeLabel(n => `<b>${n.label}</b><br>Type: ${n.type}<br>Degree: ${n.degree}`)
    .nodeOpacity(0.9)
    .linkColor(() => 'rgba(255,255,255,0.12)')
    .linkWidth(l => Math.max(0.3, l.weight * 0.3))
    .linkDirectionalParticles(0)
    .backgroundColor('#0a0a1a')
    .showNavInfo(false)
    .onNodeHover(node => {
        document.getElementById('graph').style.cursor = node ? 'pointer' : 'default';
    })
    .onNodeClick(node => {
        // 聚焦到节点
        const distance = 80;
        const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
        Graph.cameraPosition(
            { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
            node, 2000
        );
        showTooltip(node);
    })
    .onBackgroundClick(() => hideTooltip());

// Tooltip
const tooltip = document.getElementById('tooltip');
function showTooltip(node) {
    tooltip.querySelector('.title').textContent = node.label;
    tooltip.querySelector('.type').textContent = `${node.type} · degree ${node.degree}`;
    tooltip.querySelector('.desc').textContent = node.description || '';
    tooltip.style.display = 'block';
    tooltip.style.left = '50%';
    tooltip.style.top = '20px';
    tooltip.style.transform = 'translateX(-50%)';
}
function hideTooltip() { tooltip.style.display = 'none'; }

// 搜索
const searchInput = document.getElementById('search');
let highlightNodes = new Set();
searchInput.addEventListener('input', e => {
    const q = e.target.value.toLowerCase().trim();
    highlightNodes.clear();
    if (q) {
        data.nodes.forEach(n => {
            if (n.label.toLowerCase().includes(q) || n.type.toLowerCase().includes(q)) {
                highlightNodes.add(n.id);
            }
        });
    }
    Graph.nodeColor(n => highlightNodes.size === 0 ? n.color
        : highlightNodes.has(n.id) ? '#ff0' : 'rgba(255,255,255,0.05)');
    Graph.nodeOpacity(n => highlightNodes.size === 0 ? 0.9
        : highlightNodes.has(n.id) ? 1 : 0.1);
    Graph.linkOpacity(highlightNodes.size === 0 ? 0.12 : 0.02);

    if (highlightNodes.size === 1) {
        const target = data.nodes.find(n => highlightNodes.has(n.id));
        if (target) {
            const dist = 80;
            const ratio = 1 + dist / Math.hypot(target.x||0, target.y||0, target.z||0);
            Graph.cameraPosition(
                { x: (target.x||0)*ratio, y: (target.y||0)*ratio, z: (target.z||0)*ratio },
                target, 1500
            );
        }
    }
});

// 键盘快捷键
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input'));
        hideTooltip();
    }
    if (e.key === '/' && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
    }
});

// 力导向参数调优
Graph.d3Force('charge').strength(-120);
Graph.d3Force('link').distance(60);
</script>
</body>
</html>
"""


def generate_html(graphml_path: str, output_path: str = "knowledge_graph_3d.html", max_nodes: int = 500):
    data = graph_to_json(graphml_path, max_nodes)
    html = HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(data, ensure_ascii=False))
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\n3D可视化已保存: {output_path}")
    print(f"浏览器打开即可，支持:")
    print(f"  - 鼠标拖拽旋转/缩放")
    print(f"  - 点击节点聚焦+查看详情")
    print(f"  - / 键快速搜索")
    print(f"  - ESC 清除搜索")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="知识图谱3D可视化")
    parser.add_argument("--input", default="data/lightrag_storage/graph_chunk_entity_relation.graphml")
    parser.add_argument("--output", default="knowledge_graph_3d.html")
    parser.add_argument("--max-nodes", type=int, default=500)
    args = parser.parse_args()
    generate_html(args.input, args.output, args.max_nodes)

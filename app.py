import streamlit as st
import time
import heapq
import math
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from collections import deque
import pandas as pd

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(page_title="Maze Solver", layout="wide")
st.title("Maze Solver — Búsqueda y Heurísticas")

# ── Movimientos: Arriba → Derecha → Abajo → Izquierda ───────────────────────
MOVIMIENTOS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# ── Funciones del laberinto ──────────────────────────────────────────────────

def cargar_laberinto(texto: str) -> list[list[str]]:
    laberinto = []
    for linea in texto.splitlines():
        fila = list(linea.rstrip())
        if fila:
            laberinto.append(fila)
    return laberinto


def encontrar_celdas(laberinto, valor):
    posiciones = []
    for r, fila in enumerate(laberinto):
        for c, celda in enumerate(fila):
            if celda == valor:
                posiciones.append((r, c))
    return posiciones


def es_transitable(laberinto, r, c):
    filas = len(laberinto)
    cols = len(laberinto[0]) if filas else 0
    return 0 <= r < filas and 0 <= c < cols and laberinto[r][c] != '1'


def reconstruir_camino(padres, inicio, fin):
    camino = []
    nodo = fin
    while nodo is not None:
        camino.append(nodo)
        nodo = padres.get(nodo)
    camino.reverse()
    return camino


# ── Heurísticas ──────────────────────────────────────────────────────────────

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidiana(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ── Algoritmos (retornan resultado + frames para animación) ─────────────────

def bfs(laberinto, inicio, fin):
    t0 = time.perf_counter()
    frontera = deque([inicio])
    padres = {inicio: None}
    visitados_orden = []
    nodos_explorados = 0

    while frontera:
        actual = frontera.popleft()
        nodos_explorados += 1
        visitados_orden.append(actual)

        if actual == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            return camino, nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden

        for dr, dc in MOVIMIENTOS:
            vecino = (actual[0] + dr, actual[1] + dc)
            if vecino not in padres and es_transitable(laberinto, *vecino):
                padres[vecino] = actual
                frontera.append(vecino)

    return [], nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden


def dfs(laberinto, inicio, fin):
    t0 = time.perf_counter()
    pila = [inicio]
    padres = {inicio: None}
    visitados_orden = []
    nodos_explorados = 0

    while pila:
        actual = pila.pop()
        nodos_explorados += 1
        visitados_orden.append(actual)

        if actual == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            return camino, nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden

        for dr, dc in reversed(MOVIMIENTOS):
            vecino = (actual[0] + dr, actual[1] + dc)
            if vecino not in padres and es_transitable(laberinto, *vecino):
                padres[vecino] = actual
                pila.append(vecino)

    return [], nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden


def greedy(laberinto, inicio, fin, heuristica=manhattan):
    t0 = time.perf_counter()
    contador = 0
    frontera = [(heuristica(inicio, fin), contador, inicio)]
    padres = {inicio: None}
    visitados_orden = []
    nodos_explorados = 0

    while frontera:
        _, _, actual = heapq.heappop(frontera)
        nodos_explorados += 1
        visitados_orden.append(actual)

        if actual == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            return camino, nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden

        for dr, dc in MOVIMIENTOS:
            vecino = (actual[0] + dr, actual[1] + dc)
            if vecino not in padres and es_transitable(laberinto, *vecino):
                padres[vecino] = actual
                contador += 1
                heapq.heappush(frontera, (heuristica(vecino, fin), contador, vecino))

    return [], nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden


def a_star(laberinto, inicio, fin, heuristica=manhattan):
    t0 = time.perf_counter()
    contador = 0
    frontera = [(heuristica(inicio, fin), contador, inicio)]
    padres = {inicio: None}
    g = {inicio: 0}
    visitados_orden = []
    nodos_explorados = 0

    while frontera:
        _, _, actual = heapq.heappop(frontera)
        nodos_explorados += 1
        visitados_orden.append(actual)

        if actual == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            return camino, nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden

        for dr, dc in MOVIMIENTOS:
            vecino = (actual[0] + dr, actual[1] + dc)
            if not es_transitable(laberinto, *vecino):
                continue
            g_nuevo = g[actual] + 1
            if vecino not in g or g_nuevo < g[vecino]:
                g[vecino] = g_nuevo
                padres[vecino] = actual
                f_nuevo = g_nuevo + heuristica(vecino, fin)
                contador += 1
                heapq.heappush(frontera, (f_nuevo, contador, vecino))

    return [], nodos_explorados, (time.perf_counter() - t0) * 1000, visitados_orden


# ── Branching factor ─────────────────────────────────────────────────────────

def calcular_branching_factor(nodos, profundidad, tolerancia=1e-4, max_iter=100):
    if profundidad <= 0 or nodos <= 1:
        return 1.0

    def suma_geometrica(b, d):
        if abs(b - 1.0) < 1e-9:
            return float(d + 1)
        try:
            return (b ** (d + 1) - 1) / (b - 1)
        except OverflowError:
            return float("inf")

    lo, hi = 1.0, float(nodos)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = suma_geometrica(mid, profundidad)
        if abs(val - nodos) < tolerancia:
            return mid
        if val < nodos:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── Animación HTML/JS con Canvas ─────────────────────────────────────────────

def generar_animacion_html(laberinto, visitados_orden, camino, titulo, uid):
    """Genera un componente HTML con canvas que anima la exploración del laberinto."""
    filas = len(laberinto)
    cols = len(laberinto[0])

    # Construir grid base: 0=libre, 1=pared, 2=inicio, 3=salida
    grid_base = []
    for r in range(filas):
        row = []
        for c in range(cols):
            v = laberinto[r][c]
            if v == '1':
                row.append(1)
            elif v == '2':
                row.append(2)
            elif v == '3':
                row.append(3)
            else:
                row.append(0)
        grid_base.append(row)

    # Todos los visitados, sin skip — el JS controla la velocidad
    vis_list = [[r, c] for r, c in visitados_orden]
    camino_list = [[r, c] for r, c in camino] if camino else []

    html = f"""
    <div style="text-align:center; margin-bottom: 20px;" id="container_{uid}">
        <h4 style="margin:0 0 8px 0; font-family:sans-serif;">{titulo}</h4>
        <canvas id="canvas_{uid}" style="border:1px solid #ccc; max-width:100%; image-rendering: pixelated;"></canvas>

        <div style="margin-top:10px; font-family:sans-serif; font-size:13px; display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap;">
            <button id="play_{uid}" style="padding:5px 18px; cursor:pointer; font-size:13px; border-radius:4px; border:1px solid #999;">▶ Play</button>
            <button id="reset_{uid}" style="padding:5px 18px; cursor:pointer; font-size:13px; border-radius:4px; border:1px solid #999;">↺ Reset</button>
        </div>

        <div style="margin-top:8px; font-family:sans-serif; font-size:12px; display:flex; align-items:center; justify-content:center; gap:16px; flex-wrap:wrap;">
            <label style="display:flex; align-items:center; gap:4px;">
                Delay (ms):
                <input id="delay_{uid}" type="range" min="0" max="500" value="50" step="10" style="width:120px;">
                <span id="delayval_{uid}" style="min-width:35px;">50</span>
            </label>
            <label style="display:flex; align-items:center; gap:4px;">
                Nodos/paso:
                <input id="batch_{uid}" type="range" min="1" max="50" value="1" step="1" style="width:120px;">
                <span id="batchval_{uid}" style="min-width:20px;">1</span>
            </label>
        </div>

        <div style="margin-top:6px; font-family:sans-serif; font-size:12px;">
            <span id="status_{uid}" style="color:#666;">Listo — {len(visitados_orden)} nodos por explorar</span>
        </div>

        <div style="margin-top:4px; display:flex; align-items:center; justify-content:center; gap:2px; width:100%;">
            <span style="font-family:sans-serif; font-size:11px; color:#888; min-width:20px; text-align:right;">0</span>
            <div style="position:relative; flex:1; max-width:400px; height:8px; background:#e0e0e0; border-radius:4px; overflow:hidden;">
                <div id="bar_{uid}" style="height:100%; width:0%; background:linear-gradient(90deg, #2196F3, #4CAF50); border-radius:4px; transition:width 0.1s;"></div>
            </div>
            <span style="font-family:sans-serif; font-size:11px; color:#888; min-width:35px;" id="bartotal_{uid}">{len(visitados_orden)}</span>
        </div>

        <div style="margin-top:6px; font-family:sans-serif; font-size:11px; color:#888;">
            <span style="display:inline-block;width:12px;height:12px;background:limegreen;border:1px solid #aaa;vertical-align:middle;"></span> Inicio
            <span style="display:inline-block;width:12px;height:12px;background:red;border:1px solid #aaa;vertical-align:middle;margin-left:8px;"></span> Salida
            <span style="display:inline-block;width:12px;height:12px;background:#ADD8E6;border:1px solid #aaa;vertical-align:middle;margin-left:8px;"></span> Visitado
            <span style="display:inline-block;width:12px;height:12px;background:#FF8C00;border:1px solid #aaa;vertical-align:middle;margin-left:8px;"></span> Actual
            <span style="display:inline-block;width:12px;height:12px;background:gold;border:1px solid #aaa;vertical-align:middle;margin-left:8px;"></span> Camino
        </div>
    </div>
    <script>
    (function() {{
        const grid = {json.dumps(grid_base)};
        const visited = {json.dumps(vis_list)};
        const camino = {json.dumps(camino_list)};
        const rows = {filas};
        const cols = {cols};
        const totalVis = visited.length;
        const cellSize = Math.max(4, Math.min(10, Math.floor(600 / Math.max(rows, cols))));

        const canvas = document.getElementById("canvas_{uid}");
        canvas.width = cols * cellSize;
        canvas.height = rows * cellSize;
        const ctx = canvas.getContext("2d");

        const COLORS = {{
            0: "#ffffff",
            1: "#000000",
            2: "#32cd32",
            3: "#ff0000",
            4: "#ADD8E6",
            5: "#ffd700",
            6: "#FF8C00",
        }};

        // Precomputar set de nodos que pertenecen al camino solución
        const caminoSet = new Set();
        for (const [r, c] of camino) {{
            caminoSet.add(cellKey(r, c));
        }}

        let currentIdx = 0;
        let playing = false;
        let timerId = null;
        let finished = false;

        const visitedSet = new Set();
        let lastPainted = null;

        function cellKey(r, c) {{ return r * cols + c; }}

        function paintCell(r, c, colorIdx) {{
            ctx.fillStyle = COLORS[colorIdx];
            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        }}

        // Color correcto para un nodo visitado: dorado si es parte del camino, celeste si no
        function visitColor(r, c) {{
            return caminoSet.has(cellKey(r, c)) ? 5 : 4;
        }}

        function drawFull() {{
            for (let r = 0; r < rows; r++) {{
                for (let c = 0; c < cols; c++) {{
                    let val = grid[r][c];
                    if (visitedSet.has(cellKey(r, c)) && val === 0) val = visitColor(r, c);
                    ctx.fillStyle = COLORS[val];
                    ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                }}
            }}
        }}

        function getDelay() {{
            return parseInt(document.getElementById("delay_{uid}").value);
        }}
        function getBatch() {{
            return parseInt(document.getElementById("batch_{uid}").value);
        }}

        function updateStatus(text, color) {{
            const el = document.getElementById("status_{uid}");
            el.textContent = text;
            el.style.color = color || "#666";
        }}
        function updateBar() {{
            const pct = totalVis > 0 ? (currentIdx / totalVis * 100) : 0;
            document.getElementById("bar_{uid}").style.width = pct.toFixed(1) + "%";
        }}

        // ── Paso de exploración ──
        function stepExplore() {{
            if (!playing) return;

            if (currentIdx < totalVis) {{
                const batch = getBatch();

                // Quitar resaltado naranja del nodo anterior
                if (lastPainted !== null) {{
                    const [pr, pc] = lastPainted;
                    if (grid[pr][pc] === 0) paintCell(pr, pc, visitColor(pr, pc));
                }}

                for (let b = 0; b < batch && currentIdx < totalVis; b++) {{
                    const [r, c] = visited[currentIdx];
                    visitedSet.add(cellKey(r, c));
                    // Pintar dorado si pertenece al camino, celeste si no
                    if (grid[r][c] === 0) paintCell(r, c, visitColor(r, c));
                    lastPainted = [r, c];
                    currentIdx++;
                }}

                // Resaltar el nodo actual en naranja
                if (lastPainted) {{
                    const [lr, lc] = lastPainted;
                    if (grid[lr][lc] === 0) paintCell(lr, lc, 6);
                }}

                updateBar();
                updateStatus("Explorando... " + currentIdx + " / " + totalVis, "#1565C0");

                timerId = setTimeout(stepExplore, getDelay());
            }} else {{
                // Exploración terminada — quitar naranja del último nodo
                if (lastPainted) {{
                    const [pr, pc] = lastPainted;
                    if (grid[pr][pc] === 0) paintCell(pr, pc, visitColor(pr, pc));
                }}

                finished = true;
                playing = false;
                document.getElementById("play_{uid}").textContent = "▶ Play";

                if (camino.length > 0) {{
                    updateStatus("Solución: " + camino.length + " pasos | Explorados: " + totalVis, "#2e7d32");
                }} else {{
                    updateStatus("Sin solución | Explorados: " + totalVis, "#c62828");
                }}
            }}
        }}

        // ── Controles ──
        document.getElementById("play_{uid}").addEventListener("click", function() {{
            if (playing) {{
                playing = false;
                if (timerId) {{ clearTimeout(timerId); timerId = null; }}
                this.textContent = "▶ Play";
            }} else {{
                if (finished) return;
                playing = true;
                this.textContent = "⏸ Pause";
                stepExplore();
            }}
        }});

        document.getElementById("reset_{uid}").addEventListener("click", function() {{
            playing = false;
            if (timerId) {{ clearTimeout(timerId); timerId = null; }}
            currentIdx = 0;
            finished = false;
            lastPainted = null;
            visitedSet.clear();
            drawFull();
            updateBar();
            document.getElementById("play_{uid}").textContent = "▶ Play";
            updateStatus("Listo — " + totalVis + " nodos por explorar", "#666");
        }});

        document.getElementById("delay_{uid}").addEventListener("input", function() {{
            document.getElementById("delayval_{uid}").textContent = this.value;
        }});
        document.getElementById("batch_{uid}").addEventListener("input", function() {{
            document.getElementById("batchval_{uid}").textContent = this.value;
        }});

        // Dibujo inicial
        drawFull();
    }})();
    </script>
    """
    return html


# ── Configuraciones de algoritmos ────────────────────────────────────────────

CONFIGURACIONES = [
    ("BFS", lambda lab, s, e: bfs(lab, s, e)),
    ("DFS", lambda lab, s, e: dfs(lab, s, e)),
    ("Greedy (Manhattan)", lambda lab, s, e: greedy(lab, s, e, manhattan)),
    ("Greedy (Euclidiana)", lambda lab, s, e: greedy(lab, s, e, euclidiana)),
    ("A* (Manhattan)", lambda lab, s, e: a_star(lab, s, e, manhattan)),
    ("A* (Euclidiana)", lambda lab, s, e: a_star(lab, s, e, euclidiana)),
]

# ── Sidebar: carga de archivo ────────────────────────────────────────────────

st.sidebar.header("Cargar laberinto")
archivo = st.sidebar.file_uploader(
    "Sube un archivo .txt con el laberinto",
    type=["txt"],
    help="Formato: 0=camino, 1=pared, 2=inicio, 3=salida"
)

n_simulaciones = st.sidebar.slider(
    "Simulaciones aleatorias", min_value=10, max_value=200, value=50, step=10
)

if archivo is None:
    st.info("Sube un archivo de laberinto (.txt) en la barra lateral para comenzar.")
    st.stop()

# ── Procesar laberinto ───────────────────────────────────────────────────────

texto = archivo.read().decode("utf-8")
laberinto = cargar_laberinto(texto)

inicios = encontrar_celdas(laberinto, '2')
fines = encontrar_celdas(laberinto, '3')

if not inicios or not fines:
    st.error("El laberinto debe contener al menos una celda de inicio (2) y una de salida (3).")
    st.stop()

INICIO = inicios[0]
FIN = fines[0]

st.sidebar.success(f"Laberinto cargado: {len(laberinto)}×{len(laberinto[0])}")
st.sidebar.write(f"**Inicio:** {INICIO}  **Fin:** {FIN}")

# ── Ejecutar algoritmos ──────────────────────────────────────────────────────

st.header("Resultados — Caso base")

resultados = []
for nombre, fn in CONFIGURACIONES:
    camino, nodos, t_ms, visitados_orden = fn(laberinto, INICIO, FIN)
    largo = len(camino) - 1 if camino else 0
    bf = calcular_branching_factor(nodos, largo)
    resultados.append({
        "algoritmo": nombre,
        "encontro": bool(camino),
        "largo": largo,
        "nodos": nodos,
        "tiempo_ms": round(t_ms, 3),
        "bf": round(bf, 4),
        "camino": camino,
        "visitados_orden": visitados_orden,
    })

# ── Tabla de métricas ────────────────────────────────────────────────────────

df = pd.DataFrame([
    {
        "Algoritmo": r["algoritmo"],
        "Solución": "Si" if r["encontro"] else "No",
        "Pasos": r["largo"],
        "Nodos explorados": r["nodos"],
        "Tiempo (ms)": r["tiempo_ms"],
        "Branching Factor": r["bf"],
    }
    for r in resultados
])
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Gráficas comparativas ───────────────────────────────────────────────────

st.subheader("Comparación de métricas")

nombres = [r["algoritmo"] for r in resultados]
x = np.arange(len(nombres))

fig_metricas, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].bar(x, [r["largo"] for r in resultados], color='#2196F3')
axes[0].set_title("Largo del camino (pasos)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(nombres, rotation=35, ha='right', fontsize=7)

axes[1].bar(x, [r["nodos"] for r in resultados], color='#FF9800')
axes[1].set_title("Nodos explorados")
axes[1].set_xticks(x)
axes[1].set_xticklabels(nombres, rotation=35, ha='right', fontsize=7)

axes[2].bar(x, [r["tiempo_ms"] for r in resultados], color='#4CAF50')
axes[2].set_title("Tiempo (ms)")
axes[2].set_xticks(x)
axes[2].set_xticklabels(nombres, rotation=35, ha='right', fontsize=7)

plt.tight_layout()
st.pyplot(fig_metricas)
plt.close(fig_metricas)

# ── Animaciones interactivas por algoritmo ───────────────────────────────────

st.header("Animación interactiva por algoritmo")
st.caption("Presiona **Play** para ver cómo cada algoritmo explora el laberinto paso a paso.")

for i in range(0, len(resultados), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        idx = i + j
        if idx >= len(resultados):
            break
        r = resultados[idx]
        with col:
            st.markdown(
                f"**{r['algoritmo']}** — Pasos: {r['largo']} | "
                f"Nodos: {r['nodos']} | Tiempo: {r['tiempo_ms']} ms"
            )
            html = generar_animacion_html(
                laberinto,
                r["visitados_orden"],
                r["camino"],
                r["algoritmo"],
                uid=f"anim_{idx}",
            )
            st.components.v1.html(html, height=700, scrolling=False)

# ── Simulaciones aleatorias ──────────────────────────────────────────────────

st.header(f"Simulaciones con inicio aleatorio (N={n_simulaciones})")


def celdas_libres(laberinto):
    libres = []
    for r, fila in enumerate(laberinto):
        for c, celda in enumerate(fila):
            if celda != '1':
                libres.append((r, c))
    return libres


def ejecutar_desde_inicio_aleatorio(laberinto, fin, fn_algoritmo, n_simulaciones=50, semilla=42):
    random.seed(semilla)
    candidatos = [c for c in celdas_libres(laberinto) if c != fin]
    largos, nodos_list, tiempos, bfs_list = [], [], [], []
    exitos = 0

    for _ in range(n_simulaciones):
        inicio_rand = random.choice(candidatos)
        camino, nodos, t_ms, _ = fn_algoritmo(laberinto, inicio_rand, fin)
        if camino:
            exitos += 1
            largo = len(camino) - 1
            largos.append(largo)
            nodos_list.append(nodos)
            tiempos.append(t_ms)
            bfs_list.append(calcular_branching_factor(nodos, largo))

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "exito_pct": (exitos / n_simulaciones) * 100,
        "largo_prom": round(avg(largos), 1),
        "nodos_prom": round(avg(nodos_list), 0),
        "tiempo_prom_ms": round(avg(tiempos), 3),
        "bf_prom": round(avg(bfs_list), 4),
    }


with st.spinner("Ejecutando simulaciones aleatorias..."):
    resultados_aleatorios = []
    for nombre, fn in CONFIGURACIONES:
        stats = ejecutar_desde_inicio_aleatorio(laberinto, FIN, fn, n_simulaciones=n_simulaciones)
        stats["algoritmo"] = nombre
        resultados_aleatorios.append(stats)

df_rand = pd.DataFrame([
    {
        "Algoritmo": r["algoritmo"],
        "Éxito (%)": r["exito_pct"],
        "Largo prom.": r["largo_prom"],
        "Nodos prom.": int(r["nodos_prom"]),
        "Tiempo prom. (ms)": r["tiempo_prom_ms"],
        "BF prom.": r["bf_prom"],
    }
    for r in resultados_aleatorios
])
st.dataframe(df_rand, use_container_width=True, hide_index=True)

st.subheader("Comparación — Simulaciones aleatorias")

nombres2 = [r["algoritmo"] for r in resultados_aleatorios]
x2 = np.arange(len(nombres2))

fig_rand, axes2 = plt.subplots(1, 3, figsize=(16, 4))

axes2[0].bar(x2, [r["largo_prom"] for r in resultados_aleatorios], color='#2196F3')
axes2[0].set_title("Largo promedio")
axes2[0].set_xticks(x2)
axes2[0].set_xticklabels(nombres2, rotation=35, ha='right', fontsize=7)

axes2[1].bar(x2, [r["nodos_prom"] for r in resultados_aleatorios], color='#FF9800')
axes2[1].set_title("Nodos promedio")
axes2[1].set_xticks(x2)
axes2[1].set_xticklabels(nombres2, rotation=35, ha='right', fontsize=7)

axes2[2].bar(x2, [r["tiempo_prom_ms"] for r in resultados_aleatorios], color='#4CAF50')
axes2[2].set_title("Tiempo promedio (ms)")
axes2[2].set_xticks(x2)
axes2[2].set_xticklabels(nombres2, rotation=35, ha='right', fontsize=7)

plt.tight_layout()
st.pyplot(fig_rand)
plt.close(fig_rand)

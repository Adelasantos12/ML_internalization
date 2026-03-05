#!/usr/bin/env python3
"""Minimal interactive web app for treaty internalization scenario analysis.

No external dependencies required. Compatible with Railway.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import treaty_model_pipeline as pipeline


class PredictorService:
    def __init__(self) -> None:
        rows, self.features = pipeline.merge_features()
        X, y, _rows_used, means = pipeline.impute_and_matrix(rows, self.features)
        self.means = means
        self.X = X
        self.y = y

        # Fit model on all available observations for interactive scenarios.
        self.model = pipeline.GradientBoostingClassifier(n_estimators=120, learning_rate=0.08)
        self.model.fit(X, y)

        self.feature_ranges = {}
        for i, name in enumerate(self.features):
            vals = [row[i] for row in X]
            self.feature_ranges[name] = {
                "min": min(vals),
                "max": max(vals),
                "mean": means[name],
            }

    def vector_from_payload(self, payload: dict) -> list[float]:
        vec = []
        for f in self.features:
            val = payload.get(f, self.means[f])
            try:
                vec.append(float(val))
            except (TypeError, ValueError):
                vec.append(float(self.means[f]))
        return vec

    def predict(self, payload: dict) -> dict:
        x = self.vector_from_payload(payload)
        p = self.model.predict_proba([x])[0]
        return {
            "probability_improvement": round(p, 4),
            "predicted_class": int(p >= 0.5),
        }

    def sensitivity_curve(self, feature: str, points: int = 30) -> list[dict]:
        if feature not in self.features:
            return []
        points = max(5, min(points, 100))
        info = self.feature_ranges[feature]
        baseline = self.means.copy()
        curve = []
        step = (info["max"] - info["min"]) / (points - 1) if points > 1 else 0
        for i in range(points):
            val = info["min"] + step * i
            baseline[feature] = val
            pred = self.predict(baseline)
            curve.append({"x": round(val, 4), "y": pred["probability_improvement"]})
        return curve


SERVICE = PredictorService()


HTML = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>LATAM Treaty Predictor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background:#fafafa; }
    .wrap { display:grid; grid-template-columns: 320px 1fr; gap:24px; }
    .card { background:white; border-radius:10px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }
    .slider { margin: 14px 0; }
    label { font-size:14px; font-weight:600; }
    input[type=range] { width:100%; }
    .value { font-size:12px; color:#333; }
    .kpi { font-size:28px; font-weight:700; }
  </style>
</head>
<body>
  <h2>Predictor de internalización del tratado pandémico (LATAM)</h2>
  <p>Explora escenarios moviendo variables políticas y de gobernanza.</p>
  <div class="wrap">
    <div class="card" id="controls"></div>
    <div class="card">
      <div>Probabilidad estimada de mejora anual en SPAR</div>
      <div class="kpi" id="kpi">-</div>
      <canvas id="curve" height="120"></canvas>
    </div>
  </div>
<script>
const selected = ['polariz','checks','gov_seat_share','state_capacity_wgi','uhc','health_exp_gdp'];
let meta = null;
let chart = null;

async function getJSON(url, opts={}) { const r = await fetch(url, opts); return await r.json(); }

function currentPayload() {
  const payload = {};
  selected.forEach(f => {
    const el = document.getElementById('s_'+f);
    if (el) payload[f] = Number(el.value);
  });
  return payload;
}

async function refresh() {
  const pred = await getJSON('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(currentPayload())});
  document.getElementById('kpi').innerText = (pred.probability_improvement*100).toFixed(1)+'%';

  const feature = document.getElementById('sensitivity_feature').value;
  const curve = await getJSON('/api/sensitivity?feature='+feature+'&points=40');
  const xs = curve.map(d=>d.x), ys = curve.map(d=>d.y);
  if (!chart) {
    chart = new Chart(document.getElementById('curve'), {type:'line', data:{labels:xs, datasets:[{label:'P(mejora SPAR)', data:ys}]}, options:{responsive:true}});
  } else {
    chart.data.labels = xs;
    chart.data.datasets[0].data = ys;
    chart.update();
  }
}

function addSlider(container, f) {
  const m = meta.feature_ranges[f];
  const div = document.createElement('div');
  div.className='slider';
  div.innerHTML = `<label>${f}</label><input id="s_${f}" type="range" min="${m.min}" max="${m.max}" step="${(m.max-m.min)/200}" value="${m.mean}"><div class="value" id="v_${f}"></div>`;
  container.appendChild(div);
  const inp = div.querySelector('input');
  const vv = div.querySelector('.value');
  const sync = ()=>{ vv.textContent = Number(inp.value).toFixed(3); refresh(); };
  inp.addEventListener('input', sync);
  sync();
}

(async function init(){
  meta = await getJSON('/api/metadata');
  const controls = document.getElementById('controls');
  selected.forEach(f=>addSlider(controls,f));
  const sel = document.createElement('select');
  sel.id='sensitivity_feature';
  selected.forEach(f=>{ const o=document.createElement('option'); o.value=f; o.textContent='Curva de sensibilidad: '+f; sel.appendChild(o); });
  sel.addEventListener('change', refresh);
  controls.appendChild(sel);
  refresh();
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str = "application/json", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/metadata":
            body = json.dumps({"features": SERVICE.features, "feature_ranges": SERVICE.feature_ranges}).encode("utf-8")
            self._send(body)
            return
        if parsed.path == "/api/sensitivity":
            q = parse_qs(parsed.query)
            feature = q.get("feature", [""])[0]
            points = int(q.get("points", [30])[0])
            body = json.dumps(SERVICE.sensitivity_curve(feature, points)).encode("utf-8")
            self._send(body)
            return
        self._send(json.dumps({"error": "not found"}).encode("utf-8"), status=404)

    def do_POST(self):
        if self.path != "/api/predict":
            self._send(json.dumps({"error": "not found"}).encode("utf-8"), status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps(SERVICE.predict(payload)).encode("utf-8")
        self._send(body)


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Server listening on :{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

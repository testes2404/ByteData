# app.py
# ByteData • Backend de Inteligência Geográfica + Filtros em Supabase
# Endpoints:
#   - GET/OPTIONS  /health
#   - GET/OPTIONS  /filters/get         -> lê filtros (nodes) do Supabase
#   - POST/OPTIONS /filters/save        -> salva/upsert filtros no Supabase
#   - POST/OPTIONS /app                 -> processa mapa (prioriza filtros do Supabase; fallback nodes do payload)
#
# Requisitos:
#   pip install -r requirements.txt
#   (flask, flask-cors, python-dotenv, requests, google-generativeai, pytrends, shapely, h3, numpy, rapidfuzz, supabase)
#
# .env:
#   GEMINI_API_KEY=
#   PORT=8080
#   DEBUG=0
#   DEMO_MODE=0
#   HTTP_TIMEOUT=40
#   SUPABASE_URL=https://xxxxx.supabase.co
#   SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...
#   DEFAULT_CONTEXT=pagina_fluxo

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Tuple, Optional

import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from shapely.geometry import shape
from shapely.ops import unary_union

# Gemini (Google Generative AI)
import google.generativeai as genai
# Google Trends
from pytrends.request import TrendReq
# Supabase
from supabase import create_client, Client

# ------------------------------------ Config ------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", "8080"))
DEBUG = os.getenv("DEBUG", "0") == "1"
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "40"))
DEFAULT_CONTEXT = os.getenv("DEFAULT_CONTEXT", "pagina_fluxo")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Use a SERVICE_ROLE_KEY para escrita (passa por RLS em dev). Não exponha no front.
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=[
        "Content-Type",
        "X-User-Id",
        "X-User-Email",
        "X-Company-Email",
        "X-Context",
    ],
    methods=["GET", "POST", "OPTIONS"],
)

os.makedirs("cache", exist_ok=True)

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
log = logging.getLogger("bytedata-app")

# ------------------------------------ Supabase client ------------------------------------

def get_supabase() -> Optional[Client]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY ausentes: persistência OFF")
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        log.error(f"Erro ao criar cliente Supabase: {e}")
        return None

sb: Optional[Client] = get_supabase()

# ------------------------------------ Cache util ------------------------------------

def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return os.path.join("cache", f"{h}.json")

def cache_get(key: str, max_age_sec: int = 3600) -> Optional[Any]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        if (time.time() - os.path.getmtime(path)) > max_age_sec:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def cache_set(key: str, value: Any) -> None:
    path = _cache_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Falha ao salvar cache {path}: {e}")

# ------------------------------------ Helpers numéricos ------------------------------------

def norm_01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    m, M = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
        return np.zeros_like(x)
    return (x - m) / (M - m)

def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)

# ------------------------------------ Fontes ------------------------------------

def _geojson_fallback() -> dict:
    return {"type": "FeatureCollection", "features": []}

def get_ibge_uf_geojson() -> dict:
    key = "ibge_geojson_uf_v1"
    cached = cache_get(key, max_age_sec=86400)
    if cached:
        return cached
    url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados?formato=application/vnd.geo+json"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        gj = r.json()
        cache_set(key, gj)
        return gj
    except Exception as e:
        log.warning(f"Falha ao buscar malhas IBGE: {e}")
        return _geojson_fallback()

def geocode_bounds(query: str) -> Optional[List[List[float]]]:
    try:
        params = {"q": query, "format": "json", "addressdetails": 1, "limit": 1}
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "ByteData/1.0 (+nominatim)"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return None
        b = arr[0].get("boundingbox")  # [south, north, west, east]
        if not b or len(b) != 4:
            return None
        south, north, west, east = map(float, b)
        return [[west, south], [east, north]]
    except Exception as e:
        log.warning(f"Geocode falhou: {e}")
        return None

def overpass_poi_count(bbox: Tuple[float, float, float, float], tags: List[str]) -> int:
    west, south, east, north = bbox
    if west >= east or south >= north or not tags:
        return 0

    tags_filter = "".join([f'node["amenity"="{t}"]({south},{west},{north},{east});' for t in tags])
    query = f"[out:json][timeout:25];({tags_filter});out ids;"

    key = f"overpass_cnt_{hashlib.md5((str(bbox)+str(tags)).encode()).hexdigest()}"
    cached = cache_get(key, max_age_sec=3600)
    if cached is not None:
        return int(cached)

    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "ByteData/1.0 (+overpass)"},
        )
        r.raise_for_status()
        data = r.json()
        count = len(data.get("elements", []))
        cache_set(key, count)
        return count
    except Exception as e:
        log.warning(f"Overpass falhou: {e}")
        return 0

def google_trends_score(keyword: str, geo: str = "BR") -> Dict[str, float]:
    key = f"trends_{keyword}_{geo}"
    cached = cache_get(key, max_age_sec=3600)
    if cached:
        return cached
    try:
        pytrends = TrendReq(hl="pt-BR", tz=0)
        pytrends.build_payload([keyword], timeframe="today 12-m", geo=geo)
        df = pytrends.interest_by_region(resolution="REGION", inc_low_vol=True, inc_geo_code=True)
        scores: Dict[str, float] = {}
        for _, row in df.iterrows():
            uf_code = row.get("geoCode", "")
            if isinstance(uf_code, str) and uf_code.startswith("BR-"):
                uf = uf_code.split("-")[-1]
                try:
                    scores[uf] = float(row[keyword])
                except Exception:
                    pass
        cache_set(key, scores)
        return scores
    except Exception as e:
        log.warning(f"Trends falhou: {e}")
        return {}

# ------------------------------------ Consolidação de filtros ------------------------------------

def flatten_filters(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for n in nodes or []:
        vals = n.get("values", {}) or {}
        for k, v in vals.items():
            kk = (k or "").lower()
            vv = (v or "").strip()
            if not vv:
                continue
            if "região" in kk or "uf" in kk or "cidade" in kk or "local" in kk:
                out.setdefault("localizacao", []).append(vv)
            if "renda" in kk:
                out["faixa_renda"] = vv
            if "idade" in kk:
                out["faixa_idade"] = vv
            if "profissão" in kk or "setor" in kk:
                out.setdefault("setores", []).extend([t.strip() for t in vv.split(",") if t.strip() ])
            if "canal" in kk:
                out.setdefault("canais", []).extend([t.strip() for t in vv.split(",") if t.strip() ])
            if "ticket" in kk or "preço alvo" in kk:
                out["ticket_medio"] = vv
            if "objetivo" in kk or "dor" in kk:
                out.setdefault("palavras", []).extend([t.strip() for t in vv.split(",") if t.strip() ])
            if "concorrente" in kk:
                out.setdefault("concorrentes", []).extend([t.strip() for t in vv.split(",") if t.strip() ])
    return out

# ------------------------------------ Gemini: pesos/labels/keywords ------------------------------------

GEMINI_MODEL = "gemini-1.5-flash"

def gemini_spec_for_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    default = {
        "weights": {"trends_keyword": 0.6, "poi_density": 0.4},
        "penalties": {"poi_competition": 0.0},
        "combine": "score = 0.6*trends + 0.4*poi_density - 0.0*poi_competition",
        "legend_labels": ["Muito baixa", "Baixa", "Média", "Alta", "Muito alta"],
        "rationale": ["Fallback padrão (sem Gemini)."],
        "keywords": ["compras", "loja", "varejo"]
    }
    if not GEMINI_API_KEY:
        return default

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        sys = (
            "Você é um consultor de inteligência de mercado. "
            "Dado um conjunto de filtros (público/mercado/produto), responda APENAS um JSON válido com: "
            "{weights, penalties, combine, legend_labels[5], rationale[3..5], keywords[3..5]}. "
            "Use sinais: trends_keyword, poi_density, poi_competition. "
            "Pesos devem somar ~1 (ignorar penalizadores)."
        )
        user = {"filters": filters}
        prompt = f"<system>{sys}</system>\n<user>{json.dumps(user, ensure_ascii=False)}</user>"

        resp = model.generate_content(prompt)
        txt = (resp.text or "").strip()
        spec = json.loads(txt)

        w = spec.get("weights", {})
        s = sum(abs(float(v)) for v in w.values()) or 1.0
        spec["weights"] = {k: float(v) / s for k, v in w.items()}
        spec.setdefault("penalties", {"poi_competition": 0.0})
        spec.setdefault("legend_labels", ["Muito baixa", "Baixa", "Média", "Alta", "Muito alta"])
        kws = spec.get("keywords") or default["keywords"]
        spec["keywords"] = [str(k) for k in kws][:5]
        return spec
    except Exception as e:
        log.warning(f"Gemini falhou; usando default. Erro: {e}")
        return default

# ------------------------------------ Cálculo por UF ------------------------------------

def uf_bbox(feature: dict) -> Tuple[float, float, float, float]:
    geom = shape(feature["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    return (minx, miny, maxx, maxy)

def _demo_vector_by_uf(features: List[dict], seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = rng.random(len(features))
    vals = norm_01(vals)
    out: Dict[str, float] = {}
    for i, f in enumerate(features):
        uf = f["properties"].get("sigla") or f["properties"].get("UF") or f["properties"].get("uf")
        if uf:
            out[uf] = float(vals[i])
    return out

def compute_scores_by_uf(filters: Dict[str, Any]) -> Tuple[dict, dict]:
    spec = gemini_spec_for_filters(filters)
    weights = spec.get("weights", {"trends_keyword": 0.6, "poi_density": 0.4})
    penalties = spec.get("penalties", {"poi_competition": 0.0})
    legend_labels = spec.get("legend_labels", ["Muito baixa", "Baixa", "Média", "Alta", "Muito alta"])
    keywords = spec.get("keywords", ["compras", "loja", "varejo"])

    gj_uf = get_ibge_uf_geojson()
    feats = gj_uf.get("features", [])
    if not feats:
        legend = {
            "ticks": [0, 0.25, 0.5, 0.75, 1.0],
            "labels": legend_labels,
            "palette": ["#f44336", "#ff9800", "#ffeb3b", "#4caf50", "#2196f3"],
        }
        meta = {
            "weights": {k: float(v) for k, v in weights.items()},
            "penalties": {k: float(v) for k, v in penalties.items()},
            "sources": ["IBGE Malhas (UF)", "Google Trends", "OpenStreetMap/Overpass"],
            "keywords": keywords,
            "generated_at": int(time.time()),
        }
        return {"type": "FeatureCollection", "features": []}, {"legend": legend, "meta": meta}

    # Trends (máximo entre keywords)
    trends_by_uf: Dict[str, float] = {}
    for kw in keywords:
        t = google_trends_score(kw, geo="BR")
        for uf, val in t.items():
            trends_by_uf[uf] = max(trends_by_uf.get(uf, 0.0), float(val))

    if not trends_by_uf and DEMO_MODE:
        trends_by_uf = _demo_vector_by_uf(feats, seed=101)

    if trends_by_uf:
        vals = np.array([trends_by_uf.get(f["properties"]["sigla"], np.nan) for f in feats], dtype=float)
        vals = np.nan_to_num(vals, nan=np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0)
        vals_norm = norm_01(vals)
    else:
        vals_norm = np.zeros(len(feats), dtype=float)

    # POI density
    sectors = [s.lower() for s in filters.get("setores", [])]
    if any("saúd" in s or "health" in s for s in sectors):
        amenity_tags = ["clinic", "hospital", "pharmacy"]
    else:
        amenity_tags = ["mall", "supermarket", "convenience", "fast_food"]

    poi_counts: Dict[str, int] = {}
    for f in feats:
        uf_sigla = f["properties"]["sigla"]
        minx, miny, maxx, maxy = uf_bbox(f)
        cnt = overpass_poi_count((minx, miny, maxx, maxy), amenity_tags)
        poi_counts[uf_sigla] = int(cnt)

    if sum(poi_counts.values()) == 0 and DEMO_MODE:
        demo = _demo_vector_by_uf(feats, seed=202)
        for f in feats:
            uf_sigla = f["properties"]["sigla"]
            poi_counts[uf_sigla] = int(100 + 900 * demo.get(uf_sigla, 0.0))

    areas = np.array([shape(f["geometry"]).area or 1.0 for f in feats], dtype=float)
    counts = np.array([poi_counts.get(f["properties"]["sigla"], 0) for f in feats], dtype=float)
    density = counts / np.maximum(areas, 1e-6)
    density_norm = norm_01(density)

    # Combine
    w_trends = float(weights.get("trends_keyword", 0.6))
    w_poi = float(weights.get("poi_density", 0.4))
    p_comp = float(penalties.get("poi_competition", 0.0))
    pen_comp = density_norm

    sig_trends = vals_norm
    sig_poi = density_norm

    score = w_trends * sig_trends + w_poi * sig_poi - p_comp * pen_comp
    score = clamp01(score)

    out_features: List[dict] = []
    for i, f in enumerate(feats):
        uf_sigla = f["properties"]["sigla"]
        reason = [
            f"Interesse (Trends): {sig_trends[i]:.2f}",
            f"Densidade POI: {sig_poi[i]:.2f}",
        ]
        if p_comp > 0:
            reason.append(f"Penalidade competição: {(p_comp*pen_comp[i]):.2f}")

        props = dict(f["properties"])
        props.update({
            "uf": uf_sigla,
            "score": float(score[i]),
            "reason": "; ".join(reason),
            "signals": [
                {"name": "trends_keyword", "value_norm": float(sig_trends[i]), "value_raw": None},
                {"name": "poi_density", "value_norm": float(sig_poi[i]), "value_raw": int(counts[i])},
            ],
        })
        out_features.append({"type": "Feature", "geometry": f["geometry"], "properties": props})

    out_geojson = {"type": "FeatureCollection", "features": out_features}

    legend = {
        "ticks": [0, 0.25, 0.5, 0.75, 1.0],
        "labels": ["Muito baixa", "Baixa", "Média", "Alta", "Muito alta"],
        "palette": ["#f44336", "#ff9800", "#ffeb3b", "#4caf50", "#2196f3"],
    }
    meta = {
        "weights": {"trends_keyword": w_trends, "poi_density": w_poi},
        "penalties": {"poi_competition": p_comp},
        "sources": ["IBGE Malhas (UF)", "Google Trends", "OpenStreetMap/Overpass"],
        "keywords": keywords,
        "generated_at": int(time.time()),
    }
    return out_geojson, {"legend": legend, "meta": meta}

# ------------------------------------ FitBounds ------------------------------------

def compute_fitbounds_from_filters(filters: Dict[str, Any], gj: dict) -> Optional[List[List[float]]]:
    locs = filters.get("localizacao") or []
    if locs:
        b = geocode_bounds(", ".join(locs[:3]))
        if b:
            return b
    try:
        polys = [shape(f["geometry"]) for f in gj.get("features", [])]
        if not polys:
            raise ValueError("sem features")
        uni = unary_union(polys)
        minx, miny, maxx, maxy = uni.bounds
        return [[float(minx), float(miny)], [float(maxx), float(maxy)]]
    except Exception:
        return [[-73.99, -33.77], [-34.79, 5.39]]

# ------------------------------------ Helpers: identidade & filtros (Supabase) ----

def get_identity_from_request() -> Dict[str, Optional[str]]:
    # Prioridade: headers -> query -> body
    hdr = request.headers
    args = request.args or {}
    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    user_id = hdr.get("X-User-Id") or args.get("user_id") or body.get("user_id")
    user_email = hdr.get("X-User-Email") or args.get("user_email") or body.get("user_email") or "anon@local"
    company_email = hdr.get("X-Company-Email") or args.get("company_email") or body.get("company_email")
    context = hdr.get("X-Context") or args.get("context") or body.get("context") or DEFAULT_CONTEXT

    return {
        "user_id": user_id,
        "user_email": user_email,
        "company_email": company_email,
        "context": context,
    }

def sb_fetch_filters(identity: Dict[str, Optional[str]]) -> Tuple[List[dict], Optional[str]]:
    """
    Busca o registro em 'filters' por prioridade:
      1) (user_id, context)
      2) (user_email, context)
      3) (company_email, context)
    Retorna (nodes, selected_key).
    """
    if not sb:
        return [], None

    ctx = identity.get("context") or DEFAULT_CONTEXT
    uid = identity.get("user_id")
    uem = identity.get("user_email")
    cem = identity.get("company_email")

    try:
        if uid:
            q = sb.table("filters").select("*").eq("user_id", uid).eq("context", ctx).limit(1).execute()
            if q.data:
                row = q.data[0]
                return (row.get("nodes") or []), row.get("selected_key")

        if uem:
            q = sb.table("filters").select("*").eq("user_email", uem).eq("context", ctx).limit(1).execute()
            if q.data:
                row = q.data[0]
                return (row.get("nodes") or []), row.get("selected_key")

        if cem:
            q = sb.table("filters").select("*").eq("company_email", cem).eq("context", ctx).limit(1).execute()
            if q.data:
                row = q.data[0]
                return (row.get("nodes") or []), row.get("selected_key")
    except Exception as e:
        log.warning(f"Supabase GET filters falhou: {e}")

    return [], None

def sb_upsert_filters(identity: Dict[str, Optional[str]], nodes: List[dict], selected_key: Optional[str]) -> tuple[bool, Optional[str]]:
    if not sb:
        return False, "Supabase client não inicializado"
    ctx = identity.get("context") or DEFAULT_CONTEXT
    uid = identity.get("user_id")
    uem = identity.get("user_email")
    cem = identity.get("company_email")

    payload = {
        "context": ctx,
        "nodes": nodes or [],
        "selected_key": selected_key,
        "user_id": uid,
        "user_email": uem,
        "company_email": cem,
    }

    # on_conflict conforme índice único (user_id, context). Se uid vazio, cai em user_email/company_email.
    try:
        if uid:
            res = sb.table("filters").upsert(payload, on_conflict="user_id,context").execute()
        elif uem:
            sel = sb.table("filters").select("id").eq("user_email", uem).eq("context", ctx).limit(1).execute()
            if sel.data:
                row_id = sel.data[0]["id"]
                res = sb.table("filters").update({"nodes": nodes or [], "selected_key": selected_key}).eq("id", row_id).execute()
            else:
                res = sb.table("filters").insert(payload).execute()
        elif cem:
            sel = sb.table("filters").select("id").eq("company_email", cem).eq("context", ctx).limit(1).execute()
            if sel.data:
                row_id = sel.data[0]["id"]
                res = sb.table("filters").update({"nodes": nodes or [], "selected_key": selected_key}).eq("id", row_id).execute()
            else:
                res = sb.table("filters").insert(payload).execute()
        else:
            # fallback anônimo por context
            sel = sb.table("filters").select("id").eq("user_email", "anon@local").eq("context", ctx).limit(1).execute()
            if sel.data:
                row_id = sel.data[0]["id"]
                res = sb.table("filters").update({"nodes": nodes or [], "selected_key": selected_key}).eq("id", row_id).execute()
            else:
                payload["user_email"] = "anon@local"
                res = sb.table("filters").insert(payload).execute()

        return True, None
    except Exception as e:
        log.warning(f"Supabase UPSERT filters falhou: {e}")
        return False, str(e)

# ------------------------------------ Endpoints ------------------------------------

@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"ok": True, "status": "healthy"}), 200

@app.route("/filters/get", methods=["GET", "OPTIONS"])
def filters_get():
    if request.method == "OPTIONS":
        return ("", 204)
    ident = get_identity_from_request()
    nodes, selected = sb_fetch_filters(ident)
    return jsonify({"ok": True, "nodes": nodes, "selectedKey": selected, "context": ident.get("context")}), 200

@app.route("/filters/save", methods=["POST", "OPTIONS"])
def filters_save():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        return jsonify({"ok": False, "message": f"JSON inválido: {e}"}), 400

    ident = get_identity_from_request()
    nodes = payload.get("nodes") or []
    selected = payload.get("selectedKey")

    ok, err = sb_upsert_filters(ident, nodes, selected)
    if not ok:
        return jsonify({"ok": False, "message": f"Falha ao salvar filtros no Supabase: {err}"}), 500

    return jsonify({"ok": True, "message": "Filtros salvos com sucesso.", "context": ident.get("context")}), 200

@app.route("/app", methods=["POST", "OPTIONS"])
def run_app():
    """
    Processa o mapa. Prioriza filtros do Supabase; se vazio, usa nodes do payload.
    Responde no formato que o front espera (message, map, redirect).
    """
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        log.exception("JSON inválido")
        return jsonify({"ok": False, "message": f"JSON inválido: {e}"}), 400

    try:
        action = payload.get("action", "")
        if action != "gerar_fluxo":
            return jsonify({"ok": False, "message": "Ação não suportada."}), 400

        # 0) Identidade
        ident = get_identity_from_request()

        # 1) Obtém nodes do Supabase (preferência)
        nodes_db, selected_db = sb_fetch_filters(ident)

        # 2) Se não houver no DB, usa do payload (retrocompatível com front antigo)
        nodes_payload = payload.get("nodes", []) or []
        nodes = nodes_db if nodes_db else nodes_payload
        selected_key = selected_db or payload.get("selectedKey")

        # 3) Consolida filtros
        filters = flatten_filters(nodes)

        # 4) Cálculo principal
        geojson, extras = compute_scores_by_uf(filters)

        # 5) fitBounds
        fitbounds = compute_fitbounds_from_filters(filters, geojson)

        resp = {
            "ok": True,
            "message": "Mapa gerado com sucesso.",
            "map": {
                "geojson": geojson,
                "legend": extras["legend"],
                "fitBounds": fitbounds,
                "meta": extras["meta"],
            },
            "redirect": "compilador-de-dados.html",
        }
        return jsonify(resp), 200

    except Exception as e:
        log.exception("Falha no processamento; retornando fallback seguro")
        fallback_geojson = {"type": "FeatureCollection", "features": []}
        fallback_legend = {
            "ticks": [0, 0.25, 0.5, 0.75, 1.0],
            "labels": ["Muito baixa", "Baixa", "Média", "Alta", "Muito alta"],
            "palette": ["#f44336", "#ff9800", "#ffeb3b", "#4caf50", "#2196f3"],
        }
        resp = {
            "ok": False,
            "message": f"Gerado em modo parcial (fallback): {e}",
            "map": {
                "geojson": fallback_geojson,
                "legend": fallback_legend,
                "fitBounds": [[-73.99, -33.77], [-34.79, 5.39]],
                "meta": {
                    "weights": {"trends_keyword": 0.6, "poi_density": 0.4},
                    "penalties": {"poi_competition": 0.0},
                    "sources": ["IBGE Malhas (UF)", "Google Trends", "OpenStreetMap/Overpass"],
                    "keywords": [],
                    "generated_at": int(time.time()),
                    "warnings": ["fallback_ativo"],
                },
            },
            "redirect": "compilador-de-dados.html",
        }
        return jsonify(resp), 200

# ------------------------------------ Bootstrap ------------------------------------

if __name__ == "__main__":
    log.info(
        f"ByteData • app.py iniciado • PORT={PORT} • DEBUG={DEBUG} • DEMO_MODE={DEMO_MODE} • "
        f"SB={'on' if sb else 'off'} • CONTEXT={DEFAULT_CONTEXT}"
    )
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)

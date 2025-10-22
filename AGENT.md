# AGENT.md — Clasificador de Valoración Inmobiliaria (RandomForest + Electron GUI)

**Objetivo:** Entrenar y usar un modelo de **RandomForest** para estimar el **precio justo** de una propiedad a partir de un CSV con las columnas:

```
Price, Location, Expensas, surface_total, rooms, bedrooms, garage, type_building, type_operation
```

Luego, con una banda **±10%** (configurable), clasificar cada registro como **Infravalorada**, **Regular** o **Sobrevalorada**, y exponer resultados en un **formato JSON/JSONL listo para una GUI Electron.js** (badges, colores, textos, IPC).
El flujo por defecto es **Regresión (precio justo) + Regla (±banda)**. Se incluye alternativa **Clasificación directa**.

---

## TL;DR

* **Input**: CSV con 9 columnas (ver esquema).
* **ML**: `RandomForestRegressor` + `OneHot`/`Frequency Encoding`; `RandomizedSearchCV`; métricas `R²`, `MAE`, `RMSE`.
* **Etiquetas**: a partir de `deviation_pct = (Price - pred_fair_price)/pred_fair_price` y banda `BAND_PCT` (default `0.10`).
* **Output**: JSON/JSONL con `pred_fair_price`, `deviation_pct`, `class_label`, `confidence`, `ui` (color, badge, summary).
* **CLI**: `python -m src.predict --input data/dataset_alquileres.csv --jsonl --band_pct 0.10`.
* **GUI**: ejemplos de integración con **Electron** (Node + preload + renderer).
* **Gates**: Falla la entrega si `R²_cv < 0.60` o `MAE_cv > 0.25 * median(Price_train)`.

---

## 1) Esquema de Datos

### Entrada (CSV)

Tipos sugeridos:

* `Price` (**float**, moneda unificada)
* `Location` (**string**, **barrio de la propiedad**). *Este dataset viene limpio con barrios de Buenos Aires (ej.: "Belgrano, Capital Federal"; "Palermo Hollywood, Palermo").*
* `Expensas` (**float**)
* `surface_total` (**float**)
* `rooms` (**int**)
* `bedrooms` (**int**)
* `garage` (**int**)
* `type_building` (**categorical**: p.ej., *Departamento*, *Casa*, *PH*, ...)
* `type_operation` (**categorical**: *Venta*, *Alquiler*)

> **Notas**: Si el dataset trae monedas mixtas, habilitar conversión vía `CURRENCY` y `USD_ARS_RATE` antes del entrenamiento.

### Salida (JSON/JSONL apto para Electron)

Cada fila de predicción se emite como un objeto con las claves `id`, `input`, `pred`, `explain`, `ui`, `inference_timestamp`:

```json
{
  "id": "uuid",
  "input": {
    "Price": 120000.0,
    "Location": "Belgrano, Capital Federal",
    "Expensas": 80000.0,
    "surface_total": 56.0,
    "rooms": 2,
    "bedrooms": 1,
    "garage": 0,
    "type_building": "departamentos",
    "type_operation": "alquiler"
  },
  "pred": {
    "pred_fair_price": 108000.0,
    "deviation_pct": 0.1111,
    "class_label": "Sobrevalorada",
    "confidence": 0.1111,
    "band_pct": 0.10,
    "currency": "ARS",
    "locale": "es-AR"
  },
  "explain": {
    "top_features": [
      {"name": "surface_total", "direction": "up", "abs_contrib": 0.27},
      {"name": "Location_FE", "direction": "up", "abs_contrib": 0.18},
      {"name": "Expensas", "direction": "down", "abs_contrib": 0.12}
    ],
    "notes": "Contribuciones relativas a la predicción de precio justo (normalizadas)."
  },
  "ui": {
    "badge": {"text": "Sobrevalorada", "variant": "danger"},
    "color": "#ef4444",
    "icon": "trending-up",
    "summary": "Precio publicado $120.000 vs justo $108.000 (+11,1%)",
    "actions": [
      {"type": "openLink", "label": "Ver comps", "href": "#"},
      {"type": "details", "label": "Ver detalle", "payloadId": "uuid"}
    ]
  },
  "inference_timestamp": "2025-10-22T03:10:00Z"
}
```

**Mapeo de estilo por clase** (sugerido):

* **Infravalorada** → `ui.badge.variant = "success"`, `ui.color = "#16a34a"`, `ui.icon = "trending-down"`
* **Regular** → `ui.badge.variant = "neutral"`, `ui.color = "#64748b"`, `ui.icon = "activity"`
* **Sobrevalorada** → `ui.badge.variant = "danger"`, `ui.color = "#ef4444"`, `ui.icon = "trending-up"`

> **CSV de salida**: puede incluir columnas `pred_fair_price, deviation_pct, class_label, confidence, summary`.

---

## 2) Construcción de Etiquetas (Regla ±Banda)

1. Entrenar **RandomForestRegressor** para estimar `pred_fair_price`.
2. Calcular `deviation_pct = (Price - pred_fair_price) / pred_fair_price`.
3. Con banda `BAND_PCT` (default **0.10**):

   * `deviation_pct <= -BAND_PCT` → **Infravalorada**
   * `|deviation_pct| < BAND_PCT` → **Regular**
   * `deviation_pct >= BAND_PCT` → **Sobrevalorada**
4. `confidence = |deviation_pct|` (o probas si se usa clasificador alternativo).

> **Configurable** vía CLI `--band_pct 0.12` o ENV `BAND_PCT=0.12`.

---

## 3) Validación y Limpieza

* Verificar presencia y tipos de columnas; abortar con mensaje claro si faltan o son inválidas.
* Reglas de calidad:

  * Descartar filas con `Price <= 0` o `surface_total <= 0`.
  * `Expensas <= 1` → tratar como **faltante**; imputar mediana **por Location**.
  * Winsorización (p01–p99) en `Price`, `surface_total`, `Expensas` (configurable).
* Normalizaciones:

  * `Location`: `trim` + `casefold`; consolidar variantes evidentes.
  * `garage`: cast a entero (mapear booleano/string a 0/1 cuando aplique).
  * Categóricas (`type_building`, `type_operation`): **OneHotEncoder(handle_unknown='ignore')**.
* (Opcional) Moneda: si vienen USD y ARS, estandarizar usando `CURRENCY` y `USD_ARS_RATE`.

---

## 4) Features

* **Numéricas**: `surface_total`, `rooms`, `bedrooms`, `garage`, `Expensas`.
* **Categóricas OHE**: `type_building`, `type_operation`.
* **Location**: por defecto **frequency encoding** (conteo relativo por categoría). Alternativa: hashing trick.
* **Derivadas** (sugeridas):

  * `rooms_per_m2 = rooms / surface_total` (con *clipping*),
  * `bedrooms_per_room = bedrooms / max(rooms, 1)`,
  * `log_expensas = log1p(Expensas)`.

> **Importante**: En el flujo de regresión, **Price** es **target** y no se usa como feature.

---

## 5) Entrenamiento y Búsqueda

* **Split**: `StratifiedKFold` por cuantiles de `Price` (k=5 por defecto). Alternativa: estratificar por `Location` si hay fuerte heterogeneidad espacial.
* **Modelo**: `RandomForestRegressor` con hiperparámetros clave: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`.
* **Búsqueda**: `RandomizedSearchCV` (p.ej., `n_iter=40`) con `scoring="neg_mean_absolute_error"`; `random_state=42`.
* **Métricas**: `R²`, `MAE`, `MedAE`, `RMSE`. Log de métricas por *fold* y *hold-out*.
* **Artefactos**: guardar en `artifacts/` → `preprocessor.pkl`, `rf_regressor.pkl`, `feature_importance.csv`, `train_report.json`.

### Alternativa: Clasificación Directa

* Etiquetas desde la regla de banda (sobre *train*), luego entrenar `RandomForestClassifier`.
* Métricas: `macro F1`, `balanced accuracy`, `confusion matrix`.

---

## 6) Evaluación y Gates (Criterios Mínimos)

* **Gate de desempeño** (editable en `artifacts/config.yaml`):

  * `R²_cv >= 0.60`
  * `MAE_cv <= 0.25 * median(Price_train)`
* Reportes obligatorios:

  * Distribución de clases inducida por la banda (train/val/test).
  * Top‑15 importancias (`feature_importance.csv`).
  * Sensibilidad a `BAND_PCT ∈ {0.08, 0.10, 0.12}`.

---

## 7) Predicción (CLI + modos de salida)

### CSV → CSV

```bash
python -m src.predict \
  --input data/dataset_alquileres.csv \
  --output out/predictions.csv \
  --band_pct 0.10
```

### CSV → JSONL (streaming para Electron)

```bash
python -m src.predict \
  --input data/dataset_alquileres.csv \
  --jsonl \
  --band_pct 0.10
```

**Campos mínimos** (ambos modos): `pred_fair_price, deviation_pct, class_label, confidence`.
En JSON/JSONL se agregan `ui` y `summary` listos para render.

---

## 8) Integración con GUI (Electron.js)

### `main.js` (Proceso principal)

```js
const { app, BrowserWindow } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');

function createWindow () {
  const win = new BrowserWindow({
    width: 1200, height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true
    }
  });
  win.loadFile('index.html');

  const py = spawn('python', ['-m','src.predict',
    '--input','data/dataset_alquileres.csv',
    '--jsonl',
    '--band_pct','0.10'
  ]);

  py.stdout.on('data', chunk => {
    const lines = chunk.toString().split('\n').filter(Boolean);
    for (const line of lines) {
      try { win.webContents.send('pred-row', JSON.parse(line)); } catch {}
    }
  });

  py.stderr.on('data', err => win.webContents.send('pred-error', err.toString()));
}

app.whenReady().then(createWindow);
```

### `preload.js`

```js
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('pred', {
  onRow: (cb) => ipcRenderer.on('pred-row', (_e, row) => cb(row)),
  onError: (cb) => ipcRenderer.on('pred-error', (_e, msg) => cb(msg))
});
```

### `index.html` (Renderer)

```html
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Valuación de Propiedades</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: .5rem .75rem; text-align: left; }
    .badge { padding: .25rem .5rem; border-radius: .5rem; color: white; font-weight: 600; }
  </style>
</head>
<body>
  <h1>Clasificación de Propiedades</h1>
  <table>
    <thead>
      <tr>
        <th>Ubicación</th>
        <th>Precio publicado</th>
        <th>Precio justo</th>
        <th>Desvío</th>
        <th>Clasificación</th>
        <th>Resumen</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>

  <script>
    const rows = document.getElementById('rows');
    const badge = (label, color) => `<span class="badge" style="background:${color}">${label}</span>`;

    window.pred.onRow((row) => {
      const loc = row.input?.Location ?? '-';
      const price = row.input?.Price ?? 0;
      const fair = row.pred?.pred_fair_price ?? 0;
      const dev = row.pred?.deviation_pct ?? 0;
      const lab = row.pred?.class_label ?? '-';
      const col = row.ui?.color ?? '#64748b';
      const sum = row.ui?.summary ?? '';
      const fmt = new Intl.NumberFormat(row.pred?.locale || 'es-AR', { style: 'currency', currency: row.pred?.currency || 'ARS' });
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${loc}</td>
        <td>${fmt.format(price)}</td>
        <td>${fmt.format(fair)}</td>
        <td>${(dev*100).toFixed(1)}%</td>
        <td>${badge(lab, col)}</td>
        <td>${sum}</td>
      `;
      rows.appendChild(tr);
    });
  </script>
</body>
</html>
```

**Estados e I18N**: manejar `loading/empty/error`, respetar `pred.locale` y `pred.currency` usando `Intl.NumberFormat`. Garantizar contraste mínimo AA en colores.

---

## 9) Estructura del Repositorio

```
.
├─ data/
│  ├─ raw/                    # CSV de origen
│  └─ processed/              # Post-limpieza
├─ artifacts/
│  ├─ preprocessor.pkl
│  ├─ rf_regressor.pkl
│  ├─ rf_classifier.pkl       # (opcional)
│  ├─ config.yaml
│  └─ train_report.json
├─ src/
│  ├─ data_prep.py            # validación, limpieza, imputación, splits
│  ├─ features.py             # encoding Location, OHE, derivadas
│  ├─ train.py                # entrenamiento + tuning + guardado de artefactos
│  ├─ evaluate.py             # métricas CV/hold‑out, sensibilidad de banda
│  └─ predict.py              # inferencia CSV→CSV/JSONL con UI-helpers
├─ ui/
│  ├─ main.js                 # ejemplo Electron
│  ├─ preload.js
│  └─ index.html
├─ tests/
│  └─ test_schema_and_io.py   # esquemas y smoke tests
├─ AGENT.md
├─ requirements.txt (o pyproject.toml)
└─ Makefile
```

---

## 10) Comandos y Makefile

### Requisitos

* Python 3.10+
* `pip install -r requirements.txt` (o `poetry install`)

### Ejemplos de uso

```bash
# Entrenamiento
make train DATA=./data/raw/dataset_alquileres.csv BAND_PCT=0.10

# Predicción para GUI (stream JSONL por stdout)
python -m src.predict --input data/dataset_alquileres.csv --jsonl --band_pct 0.10

# Predicción a archivo JSONL
python -m src.predict --input data/dataset_alquileres.csv --output out/predictions.jsonl --band_pct 0.10

# Predicción a CSV
python -m src.predict --input data/dataset_alquileres.csv --output out/predictions.csv --band_pct 0.10
```

### Makefile (referencia)

```Makefile
.PHONY: setup train predict test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	python -m src.train --data $(DATA) --band_pct $(BAND_PCT)

predict:
	python -m src.predict --input $(IN) --output $(OUT) --band_pct $(BAND_PCT)

test:
	pytest -q
```

---

## 11) Configuración (`artifacts/config.yaml`)

```yaml
random_state: 42
band_pct: 0.10
cv_folds: 5
winsor: {low: 0.01, high: 0.99}
model:
  n_estimators: 600
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: "sqrt"
  bootstrap: true
search:
  enabled: true
  n_iter: 40
  scoring: "neg_mean_absolute_error"
  n_jobs: -1
```

---

## 12) Artefactos y Reportes

* `preprocessor.pkl` — pipeline de limpieza/encoding/imputación.
* `rf_regressor.pkl` — modelo final.
* `feature_importance.csv` — importancias normalizadas y ordenadas.
* `train_report.json` — métricas CV, hold‑out, hashes de datos, fecha y config.

---

## 13) Ejemplos

### CSV de entrada (mínimo)

```csv
Price,Location,Expensas,surface_total,rooms,bedrooms,garage,type_building,type_operation
650000.0,"Belgrano, Capital Federal",235000.0,58,1,0,0,departamentos,alquiler
1235000.0,"Palermo Hollywood, Palermo",230000.0,55,2,1,1,departamentos,alquiler
```

### Línea JSONL de salida (una por propiedad)

```json
{"id":"a3f...","input":{"Price":650000.0,"Location":"Belgrano, Capital Federal","Expensas":235000.0,"surface_total":58.0,"rooms":1,"bedrooms":0,"garage":0,"type_building":"departamentos","type_operation":"alquiler"},"pred":{"pred_fair_price":585000.0,"deviation_pct":0.1111,"class_label":"Sobrevalorada","confidence":0.1111,"band_pct":0.1,"currency":"ARS","locale":"es-AR"},"ui":{"badge":{"text":"Sobrevalorada","variant":"danger"},"color":"#ef4444","icon":"trending-up","summary":"Precio publicado $650.000 vs justo $585.000 (+11,1%)"},"inference_timestamp":"2025-10-22T03:10:00Z"}

```

---

## 14) Reproducibilidad

* Fijar `random_state=42` en todos los componentes.
* Versionar `artifacts/config.yaml` y registrar hash/fecha de datos en `train_report.json`.
* Mantener `requirements.txt`/`pyproject.toml` con versiones mínimas estables.

---

## 15) Errores y Edge Cases

* `Location` desconocida: el encoding por frecuencia debe tener *fallback* para categorías raras (frecuencia mínima).
* Alto faltante en `Expensas`: registrar tasa de imputación y evaluar impacto.
* Clases muy desbalanceadas: ajustar `band_pct` o evaluar clasificador directo con *class weights*.
* Validación estricta de esquema: si faltan columnas o tipos, abortar con mensaje claro.

---

## 16) Tests

* `tests/test_schema_and_io.py`:

  * Verifica columnas requeridas.
  * Smoke test de `src.predict` en modo `--jsonl` y `--output`.
  * Prueba de consistencia de `band_pct` y de `confidence`.

---

## 17) Licencia y Contacto

* Licencia: MIT (sugerida) o la que prefiera el repositorio.
* Issues y soporte: abrir *issue* en GitHub del proyecto, adjuntando `train_report.json` y ejemplo mínimo de input.

// =============================================================================
//  logistik.js – der Algorithmus
// =============================================================================
//  Ein Modell wird als { k, x0 } gespeichert (wie im Skript):
//      p(x) = sig(k · (x − x₀)) = 1 / (1 + e^(−k·(x − x₀)))
//  k  = Steilheit (je größer, desto schärfer der Übergang)
//  x₀ = Wendepunkt (hier ist p = 0,5)
//
//  Bewertung
//    Log-Loss    mittlerer Fehler über alle Punkte (wird beim Lernen minimiert)
//    Schwellwert ab welcher Wahrscheinlichkeit "Klasse 1" vorhergesagt wird
//    Konfusionsmatrix, Sensitivität, Spezifität, ROC-Kurve, AUC (Skript 4.2)
// =============================================================================

function sig(z) {
  return 1 / (1 + Math.exp(-z));
}

// Wahrscheinlichkeit für Klasse 1
function wahrsch(m, x) {
  return sig(m.k * (x - m.x0));
}

// Logit: Umkehrung der Sigmoid-Funktion
function logit(p) {
  return Math.log(p / (1 - p));
}

// x-Wert, an dem das Modell die Wahrscheinlichkeit p erreicht
function xBeiP(m, p) {
  if (m.k === 0) {
    return NaN;
  }
  return m.x0 + logit(p) / m.k;
}

// Fehler eines Punktes: −ln(p) für Klasse 1, −ln(1 − p) für Klasse 0
function punktFehler(m, q) {
  const p = Math.min(1 - 1e-12, Math.max(1e-12, wahrsch(m, q.x)));
  return q.y === 1 ? -Math.log(p) : -Math.log(1 - p);
}

// Log-Loss (Kreuzentropie) = Mittelwert der Punktfehler
function logLoss(daten, m) {
  let s = 0;
  for (const q of daten) {
    s += punktFehler(m, q);
  }
  return s / daten.length;
}

// -----------------------------------------------------------------------------
// Konfusionsmatrix bei einem Schwellwert
// -----------------------------------------------------------------------------
//  positiv = Klasse 1. Vorhersage "positiv", wenn p(x) ≥ Schwellwert.
function konfusion(daten, m, schwelle) {
  const r = { wp: 0, fn: 0, fp: 0, wn: 0 };
  for (const q of daten) {
    const vorhersage = wahrsch(m, q.x) >= schwelle ? 1 : 0;
    if (q.y === 1 && vorhersage === 1) {
      r.wp++;
    } else if (q.y === 1) {
      r.fn++;
    } else if (vorhersage === 1) {
      r.fp++;
    } else {
      r.wn++;
    }
  }
  r.n = daten.length;
  r.sens = r.wp + r.fn > 0 ? r.wp / (r.wp + r.fn) : NaN; // Wahr-Positiv-Rate
  r.spez = r.wn + r.fp > 0 ? r.wn / (r.wn + r.fp) : NaN;
  r.fpr = r.fp + r.wn > 0 ? r.fp / (r.fp + r.wn) : NaN; // 1 − Spezifität
  r.genau = r.n > 0 ? (r.wp + r.wn) / r.n : NaN;
  return r;
}

// -----------------------------------------------------------------------------
// ROC-Kurve: Sensitivität gegen Falsch-Positiv-Rate für alle Schwellwerte
// -----------------------------------------------------------------------------
function rocKurve(daten, m) {
  // jede vorkommende Wahrscheinlichkeit ist ein sinnvoller Schwellwert
  const ps = daten.map((q) => wahrsch(m, q.x));
  const schwellen = Array.from(new Set(ps)).sort((a, b) => b - a);
  const punkte = [{ fpr: 0, sens: 0, s: Infinity }];
  for (const s of schwellen) {
    const r = konfusion(daten, m, s);
    punkte.push({ fpr: r.fpr, sens: r.sens, s });
  }
  punkte.push({ fpr: 1, sens: 1, s: -Infinity });
  // Fläche unter der Kurve (Trapezregel)
  let auc = 0;
  for (let i = 1; i < punkte.length; i++) {
    const a = punkte[i - 1];
    const b = punkte[i];
    auc += (b.fpr - a.fpr) * (a.sens + b.sens) / 2;
  }
  return { punkte, auc };
}

// -----------------------------------------------------------------------------
// Gradientenabstieg
// -----------------------------------------------------------------------------
//  Damit eine Lernrate für alle Beispiele passt, wird x normiert:
//      x′ = (x − xmin) / (xmax − xmin)
//  Modell:    p = sig(w · x′ + c)
//  Verlust:   L = −(1/n) · Σ [ y·ln p + (1 − y)·ln(1 − p) ]
//  Gradient:  ∂L/∂w = (1/n) · Σ (p − y) · x′
//             ∂L/∂c = (1/n) · Σ (p − y)
//  Schritt:   w ← w − η · ∂L/∂w      c ← c − η · ∂L/∂c
//  Umrechnen: k = w / (xmax − xmin)   x₀ = xmin − c / w · (xmax − xmin)
// -----------------------------------------------------------------------------
function normX(x) {
  return (x - MERKMAL.min) / (MERKMAL.max - MERKMAL.min);
}

function ausNormiert(w, c) {
  const r = MERKMAL.max - MERKMAL.min;
  if (Math.abs(w) < 1e-9) {
    w = w < 0 ? -1e-9 : 1e-9;
  }
  return { k: w / r, x0: MERKMAL.min - (c / w) * r };
}

function inNormiert(m) {
  const r = MERKMAL.max - MERKMAL.min;
  return { w: m.k * r, c: m.k * (MERKMAL.min - m.x0) };
}

function gdSchritt(daten, w, c, eta) {
  const n = daten.length;
  let gw = 0;
  let gc = 0;
  for (const q of daten) {
    const xn = normX(q.x);
    const e = sig(w * xn + c) - q.y;
    gw += e * xn;
    gc += e;
  }
  gw /= n;
  gc /= n;
  return { w: w - eta * gw, c: c - eta * gc, gw, gc };
}

// bestes Modell (für den Vergleich): sehr viele kleine Schritte bergab
function bestesModell(daten) {
  let w = 0;
  let c = 0;
  for (let i = 0; i < 20000; i++) {
    const r = gdSchritt(daten, w, c, 4);
    w = r.w;
    c = r.c;
    if (Math.hypot(r.gw, r.gc) < 1e-7) {
      break;
    }
  }
  return ausNormiert(w, c);
}

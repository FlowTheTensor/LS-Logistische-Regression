// =============================================================================
//  KI-Klassifikation · Logistische Regression
//  Aus Daten mit bekannter Klasse (0 oder 1) lernt das Programm eine
//  Sigmoid-Kurve p(x) = 1 / (1 + e^(−k·(x − x₀))) und klassifiziert damit
//  neue Werte – vier Beispiele zur Auswahl:
//    - Skript 4.2: Schuhgröße → Geschlecht (feste Daten aus dem Skript)
//    - Sortierstation: Reflexion → Metall oder Kunststoff?
//    - Antrieb: Schwinggeschwindigkeit → Lagerschaden?
//    - Servomotor: Wicklungstemperatur → Überlast-Abschaltung?
// =============================================================================
//
//  Dateien
//    basis.js        gemeinsame Grundlage (Layout, Farben, Kopfzeile, UI-Helfer)
//    daten.js        Daten der vier Beispiele
//    logistik.js     der Algorithmus (Sigmoid, Log-Loss, Gradient, ROC)
//    ui.js           Knöpfe, Auswahllisten, Regler
//    sketch.js       Ablauf, Zeichnen, Erklärungen (diese Datei)
//
//  Zwei Verfahren
//    ① per Hand      Wendepunkt x₀ und Steilheit k mit der Maus ziehen
//    ② Gradient      Gradientenabstieg auf dem Log-Loss – so lernt eine KI
//
//  Bewertung (Skript 4.2)
//    Schwellwert → Konfusionsmatrix, Sensitivität, Spezifität, ROC-Kurve, AUC
//
//  Bedienung
//    →         nächster Lernschritt
//    Leertaste automatisch ablaufen lassen / anhalten
//    R         neu starten
//    Klick     ins Diagramm: Prognose für diesen x-Wert (ziehbar) · Esc: weg
//    Ziehen    an der gestrichelten Schwellwert-Linie: Schwellwert ändern
//    Maus      auf ein Feld der Konfusionsmatrix: zugehörige Punkte leuchten
// =============================================================================

const APP_TITEL = "KI-Klassifikation";

const P_DATEN = { x: 24, y: 84, w: 820, h: 800, titel: "" };
const P_ERKL = { x: 860, y: 84, w: 716, h: 244, titel: "" };
const P_EINST = { x: 860, y: 344, w: 716, h: 132, titel: "Einstellungen" };
const P_MODELL = { x: 860, y: 492, w: 716, h: 184, titel: "Modell & Güte" };
const P_KONF = { x: 860, y: 692, w: 350, h: 192, titel: "" };
const P_ROC = { x: 1226, y: 692, w: 350, h: 192, titel: "" };

const PLOT = { x: 104, y: 150, w: 700, h: 580 };
const P_UNTEN = -0.12; // y-Achse des Diagramms reicht von −0,12 …
const P_OBEN = 1.12; //   … bis 1,12, damit die Punkte bei 0 und 1 Platz haben

const TEMPO = {
  langsam: { anim: 380, pause: 520, proFrame: 1 },
  mittel: { anim: 0, pause: 0, proFrame: 4 },
  schnell: { anim: 0, pause: 0, proFrame: 30 },
};
const MAX_SCHRITTE = 6000;
const P_STEIL = sig(2); // Höhe des Steilheits-Griffs: p = sig(2) ≈ 0,88

// -----------------------------------------------------------------------------
// Zustand
// -----------------------------------------------------------------------------
let szenario = "skript";
let startwert = 0;
let daten = [];
let bestes = null; // bestes Modell (Gradientenabstieg bis zum Ende)

let verfahren = "hand"; // hand | gd
let lernrate = 8;
let tempo = "mittel";
let schwelle = 0.5;

let modell = null; // {k, x0}
let phase = "hand"; // hand · gd: leer | lernen | fertig
let gd = { w: 0, c: 0, gw: 0, gc: 0 };
let iteration = 0;
let verlauf = []; // {i, loss}

let anim = null; // {von:{w,c}, nach:{w,c}, t0, dauer}
let laufAuto = false;
let letzterSchritt = 0;

let abstaendeAn = false;
let falscheAn = true;
let vergleichAn = false;

let ziehen = -1; // 0 = Wendepunkt, 1 = Steilheit, 2 = Prognose, 3 = Schwellwert
let prognoseX = null;

// =============================================================================
// Start
// =============================================================================
function setup() {
  ladeSchriften();
  const [cw, ch] = zeichenflaeche();
  cnv = createCanvas(cw, ch);
  S = width / W;

  setzeSzenario(szenario);
  startwert = SZ.startwert || 0;
  daten = erzeugeDaten(startwert);
  bestes = bestesModell(daten);

  erstelleUi();
  cnv.parent(huelle);
  neuStarten();
  layoutUi();
}

function nachFehler() {
  laufAuto = false;
  anim = null;
  ziehen = -1;
}

// =============================================================================
// Hauptschleife – Fehler werden abgefangen, damit nie alles einfriert
// =============================================================================
function draw() {
  try {
    logik();
    zeichneRahmen(() => {
      zeichneStatusZeile(statusText());
      zeichneDatenPanel();
      zeichneErklaerung();
      zeichneEinstellungen();
      zeichneModellPanel();
      zeichneKonfusion();
      zeichneRoc();
      zeichneTooltip();
    });
    aktualisiereUi();
  } catch (err) {
    behandleFehler(err);
  }
}

function logik() {
  if (anim && millis() - anim.t0 > anim.dauer) {
    anim = null;
  }
  if (!laufAuto) {
    return;
  }
  const t = TEMPO[tempo];
  if (!anim && millis() - letzterSchritt > t.pause) {
    for (let i = 0; i < t.proFrame && laufAuto; i++) {
      schritt();
    }
  }
  if (phase === "fertig") {
    laufAuto = false;
  }
}

// =============================================================================
// Daten und Ablauf
// =============================================================================
function szenarioWechseln(neu) {
  if (laufAuto || neu === szenario) {
    return;
  }
  szenario = neu;
  setzeSzenario(neu);
  startwert = SZ.startwert || 0;
  daten = erzeugeDaten(startwert);
  bestes = bestesModell(daten);
  prognoseX = null;
  schwelle = 0.5;
  neuStarten();
}

function neueMessdaten() {
  if (SZ.fest) {
    zeigeHinweis("Das Skript-Beispiel hat feste Daten. Für zufällige Messdaten ein anderes Beispiel wählen.");
    return;
  }
  startwert = Math.floor(Math.random() * 1e9);
  daten = erzeugeDaten(startwert);
  bestes = bestesModell(daten);
  neuStarten();
}

function verfahrenWechseln(neu) {
  if (laufAuto) {
    return;
  }
  const alt = zeigeModell();
  verfahren = neu;
  neuStarten();
  // per Hand: mit der zuletzt gezeigten Kurve weitermachen
  if (neu === "hand" && alt) {
    modell = { k: alt.k, x0: alt.x0 };
  }
}

function startModellHand() {
  const m = MERKMAL;
  const r = m.max - m.min;
  // absichtlich schlecht: Wendepunkt zu weit links, Kurve sehr flach
  return { k: 5 / r, x0: m.min + 0.3 * r };
}

function neuStarten() {
  laufAuto = false;
  anim = null;
  ziehen = -1;
  iteration = 0;
  verlauf = [];
  if (verfahren === "hand") {
    phase = "hand";
    modell = startModellHand();
  } else {
    phase = "leer";
    modell = null;
  }
}

function schritt() {
  if (ziehen >= 0 || verfahren !== "gd") {
    return;
  }
  anim = null;
  letzterSchritt = millis();
  if (phase === "leer") {
    // zufällige, flache Startkurve
    gd = { w: random(-1.5, 1.5), c: random(-1, 1), gw: 0, gc: 0 };
    modell = ausNormiert(gd.w, gd.c);
    iteration = 0;
    verlauf = [{ i: 0, loss: logLoss(daten, modell) }];
    phase = "lernen";
    return;
  }
  if (phase !== "lernen") {
    zeigeHinweis("Die KI ist fertig. „Neu starten“ probiert eine andere Startkurve.");
    laufAuto = false;
    return;
  }
  const von = { w: gd.w, c: gd.c };
  const r = gdSchritt(daten, gd.w, gd.c, lernrate);
  gd = r;
  modell = ausNormiert(gd.w, gd.c);
  iteration++;
  verlauf.push({ i: iteration, loss: logLoss(daten, modell) });
  if (verlauf.length > 600) {
    verlauf.splice(1, 1); // Startwert behalten, Mitte ausdünnen
  }
  const dauer = TEMPO[tempo].anim;
  if (dauer > 0) {
    anim = { von, nach: { w: gd.w, c: gd.c }, t0: millis(), dauer };
  }
  if (Math.hypot(gd.gw, gd.gc) < 4e-4 || iteration >= MAX_SCHRITTE) {
    phase = "fertig";
  }
}

function autoUmschalten() {
  if (laufAuto) {
    laufAuto = false;
    return;
  }
  if (verfahren === "hand" || phase === "fertig") {
    return;
  }
  laufAuto = true;
  letzterSchritt = 0;
}

// angezeigte Kurve (gleitet während einer Animation)
function zeigeModell() {
  if (!modell) {
    return null;
  }
  if (anim) {
    const t = weich(constrain((millis() - anim.t0) / anim.dauer, 0, 1));
    return ausNormiert(lerp(anim.von.w, anim.nach.w, t), lerp(anim.von.c, anim.nach.c, t));
  }
  return modell;
}

function vergleichsModell() {
  return SZ.skriptKurve || bestes;
}

// =============================================================================
// Zahlen
// =============================================================================
function fK(k) {
  return fSig(k, 3);
}

function fX0(x0) {
  return fFix(x0, MERKMAL.stellen + 1);
}

function gleichung(m) {
  const x0 = m.x0 >= 0 ? `x − ${fX0(m.x0)}` : `x + ${fX0(-m.x0)}`;
  const kTxt = m.k >= 0 ? `−${fK(m.k)}` : `+${fK(-m.k)}`;
  return `p(x) = 1 / (1 + e^(${kTxt}·(${x0})))`;
}

function statusText() {
  const name = verfahren === "hand" ? "per Hand" : `Gradientenabstieg (η = ${lernrate})`;
  let s = `${daten.length} ${SZ.wort.mehrzahl} mit bekannter Klasse · Verfahren: ${name}`;
  if (verfahren === "gd" && phase !== "leer") {
    s += ` · Lernschritt ${iteration}`;
  }
  const m = zeigeModell();
  if (m) {
    const r = konfusion(daten, m, schwelle);
    s += ` · Schwellwert ${fKomma(schwelle, 2)} · Genauigkeit ${prozent(r.genau, 0)} · Log-Loss ${fFix(logLoss(daten, m), 3)}`;
  }
  if (phase === "fertig") {
    s += " · fertig ✓";
  }
  return s;
}

// =============================================================================
// Diagramm – Umrechnung
// =============================================================================
function zuPlot(x, p) {
  const m = MERKMAL;
  return [PLOT.x + ((x - m.min) / (m.max - m.min)) * PLOT.w, PLOT.y + PLOT.h - ((p - P_UNTEN) / (P_OBEN - P_UNTEN)) * PLOT.h];
}

function ausPlot(px, py) {
  const m = MERKMAL;
  const x = m.min + ((px - PLOT.x) / PLOT.w) * (m.max - m.min);
  const p = P_UNTEN + ((PLOT.y + PLOT.h - py) / PLOT.h) * (P_OBEN - P_UNTEN);
  return { x: constrain(x, m.min, m.max), p };
}

function imPlot(px, py) {
  return px >= PLOT.x && px <= PLOT.x + PLOT.w && py >= PLOT.y && py <= PLOT.y + PLOT.h;
}

// Punkte mit gleichem x und gleicher Klasse werden leicht übereinander gestapelt
function punktLagen() {
  const zaehler = {};
  return daten.map((q) => {
    const key = q.y + "|" + q.x;
    const i = zaehler[key] || 0;
    zaehler[key] = i + 1;
    const versatz = i * 0.035;
    return { q, p: q.y === 1 ? 1 - versatz : versatz };
  });
}

// Griffe für die Hand-Steuerung: Wendepunkt (x₀ | 0,5) und Steilheit (x₀ + 2/k | 0,88)
function griffe(m) {
  return [
    { x: m.x0, p: 0.5 },
    { x: m.x0 + 2 / m.k, p: P_STEIL },
  ];
}

function griffUnterMaus() {
  if (verfahren !== "hand" || !modell) {
    return -1;
  }
  const g = griffe(modell);
  for (let i = 1; i >= 0; i--) {
    const [px, py] = zuPlot(g[i].x, g[i].p);
    if (dist(mausX(), mausY(), px, py) < 16) {
      return i;
    }
  }
  return -1;
}

function schwelleUnterMaus() {
  if (!modell || !imPlot(mausX(), mausY())) {
    return false;
  }
  const [, sy] = zuPlot(0, schwelle);
  return Math.abs(mausY() - sy) < 6;
}

function prognoseUnterMaus() {
  if (prognoseX === null || !modell) {
    return false;
  }
  const [px] = zuPlot(prognoseX, 0);
  return Math.abs(mausX() - px) < 8 && imPlot(mausX(), mausY());
}

function punktUnterMaus() {
  const mx = mausX();
  const my = mausY();
  if (!imPlot(mx, my)) {
    return null;
  }
  let bester = null;
  let besterAbstand = 10;
  for (const l of punktLagen()) {
    const [px, py] = zuPlot(l.q.x, l.p);
    const d = dist(mx, my, px, py);
    if (d < besterAbstand) {
      besterAbstand = d;
      bester = l.q;
    }
  }
  return bester;
}

// Kategorie eines Punktes in der Konfusionsmatrix
function kategorie(m, q) {
  const v = wahrsch(m, q.x) >= schwelle ? 1 : 0;
  if (q.y === 1) {
    return v === 1 ? "wp" : "fn";
  }
  return v === 1 ? "fp" : "wn";
}

// =============================================================================
// Zeichnen – Datenfeld
// =============================================================================
function zeichneKurve(m, farbe, dicke, strich) {
  const mm = MERKMAL;
  noFill();
  stroke(...farbe);
  strokeWeight(dicke);
  gestrichelt(strich);
  beginShape();
  for (let i = 0; i <= 200; i++) {
    const x = mm.min + (i / 200) * (mm.max - mm.min);
    const [px, py] = zuPlot(x, wahrsch(m, x));
    vertex(px, py);
  }
  endShape();
  gestrichelt(false);
}

function zeichneKlassenPunkt(px, py, klasse, gross) {
  const f = KLASSEN_FARBE[klasse];
  stroke(...FARBE.flaeche);
  strokeWeight(1.4);
  fill(f[0], f[1], f[2], 235);
  const r = gross ? 1.25 : 1;
  if (klasse === 1) {
    triangle(px, py - 8 * r, px + 7.5 * r, py + 5 * r, px - 7.5 * r, py + 5 * r);
  } else {
    circle(px, py, 12 * r);
  }
}

function zeichneDatenPanel() {
  P_DATEN.titel = SZ.titel;
  panel(P_DATEN);
  zeichneLegende();

  const mm = MERKMAL;
  const m = zeigeModell();
  const hoverZelle = zelleUnterMaus();

  // Diagrammfläche
  stroke(...FARBE.linie);
  strokeWeight(1);
  fill(...FARBE.papier);
  rect(PLOT.x, PLOT.y, PLOT.w, PLOT.h);

  clipRechteck(PLOT.x, PLOT.y, PLOT.w, PLOT.h);

  // Vorhersage-Bereiche: links/rechts der Grenze getönt
  const xs = m ? xBeiP(m, schwelle) : NaN;
  if (m && isFinite(xs)) {
    const [gx] = zuPlot(constrain(xs, mm.min - 1e3, mm.max + 1e3), 0);
    const links = m.k > 0 ? 0 : 1; // Klasse links der Grenze
    noStroke();
    const fl = KLASSEN_FARBE[links];
    const fr = KLASSEN_FARBE[1 - links];
    fill(fl[0], fl[1], fl[2], 18);
    rect(PLOT.x, PLOT.y, constrain(gx - PLOT.x, 0, PLOT.w), PLOT.h);
    fill(fr[0], fr[1], fr[2], 18);
    rect(constrain(gx, PLOT.x, PLOT.x + PLOT.w), PLOT.y, PLOT.x + PLOT.w - constrain(gx, PLOT.x, PLOT.x + PLOT.w), PLOT.h);
  }

  // Gitter
  textFont(SCHRIFT);
  textSize(12);
  const anzX = Math.round((mm.max - mm.min) / mm.gitter);
  for (let i = 0; i <= anzX; i++) {
    const v = mm.min + i * mm.gitter;
    const [gx] = zuPlot(v, 0);
    stroke(...FARBE.text, 22);
    strokeWeight(1);
    line(gx, PLOT.y, gx, PLOT.y + PLOT.h);
  }
  for (let i = 0; i <= 10; i++) {
    const [, gy] = zuPlot(mm.min, i / 10);
    stroke(...FARBE.text, i === 0 || i === 10 ? 45 : 18);
    line(PLOT.x, gy, PLOT.x + PLOT.w, gy);
  }

  // Vergleichskurve
  if (vergleichAn) {
    zeichneKurve(vergleichsModell(), FARBE.blass, 2, true);
  }

  if (m) {
    // Fehler je Punkt: Abstand zwischen Punkt und Kurve (Strichstärke ~ −ln p)
    if (abstaendeAn) {
      for (const l of punktLagen()) {
        const [px, py] = zuPlot(l.q.x, l.p);
        const [, ky] = zuPlot(l.q.x, wahrsch(m, l.q.x));
        const f = KLASSEN_FARBE[l.q.y];
        stroke(f[0], f[1], f[2], 150);
        strokeWeight(constrain(1 + punktFehler(m, l.q) * 2.2, 1, 7));
        line(px, py, px, ky);
      }
    }

    // Schwellwert-Linie (waagrecht) und Entscheidungsgrenze (senkrecht)
    const [, sy] = zuPlot(0, schwelle);
    const aktiv = ziehen === 3 || schwelleUnterMaus();
    stroke(...FARBE.senf);
    strokeWeight(aktiv ? 2.8 : 1.8);
    gestrichelt(true);
    line(PLOT.x, sy, PLOT.x + PLOT.w, sy);
    if (isFinite(xs) && xs > mm.min && xs < mm.max) {
      const [gx] = zuPlot(xs, 0);
      line(gx, PLOT.y, gx, PLOT.y + PLOT.h);
    }
    gestrichelt(false);

    zeichneKurve(m, FARBE.gruen, 3.4, false);
  }

  // Punkte
  for (const l of punktLagen()) {
    const [px, py] = zuPlot(l.q.x, l.p);
    const kat = m ? kategorie(m, l.q) : null;
    const leuchtet = hoverZelle && kat === hoverZelle;
    if (leuchtet) {
      noStroke();
      fill(...FARBE.senf, 90);
      circle(px, py, 28);
    }
    zeichneKlassenPunkt(px, py, l.q.y, leuchtet);
    if (m && falscheAn && (kat === "fn" || kat === "fp")) {
      noFill();
      stroke(...FARBE.text);
      strokeWeight(1.8);
      circle(px, py, 22);
    }
    if (l.q.name) {
      noStroke();
      fill(...FARBE.grau);
      textFont(SCHRIFT);
      textSize(11);
      textAlign(CENTER, l.q.y === 1 ? BOTTOM : TOP);
      text(l.q.name, px, l.q.y === 1 ? py - 13 : py + 12);
    }
  }

  if (m) {
    // Wendepunkt markieren bzw. Griffe
    if (verfahren === "hand") {
      const g = griffe(m);
      const aktiv = ziehen >= 0 && ziehen < 2 ? ziehen : griffUnterMaus();
      for (let i = 0; i < 2; i++) {
        const [px, py] = zuPlot(g[i].x, g[i].p);
        noStroke();
        fill(...FARBE.schatten, 40);
        circle(px + 1, py + 2, 28);
        fill(...FARBE.flaeche);
        stroke(...FARBE.gruen);
        strokeWeight(i === aktiv ? 4.5 : 3.5);
        if (i === 0) {
          circle(px, py, i === aktiv ? 28 : 24);
        } else {
          rectMode(CENTER);
          rect(px, py, i === aktiv ? 24 : 20, i === aktiv ? 24 : 20, 5);
          rectMode(CORNER);
        }
        noStroke();
        fill(...FARBE.gruen);
        triangle(px - 3, py - 4, px - 3, py + 4, px - 8, py);
        triangle(px + 3, py - 4, px + 3, py + 4, px + 8, py);
      }
      const [wx, wy] = zuPlot(g[0].x, 0.5);
      marke(`x₀ = ${fX0(m.x0)}`, wx + 18, wy + 24, FARBE.gruen, LEFT);
      const [sx, sy] = zuPlot(g[1].x, g[1].p);
      marke(`k = ${fK(m.k)}`, sx + (m.k > 0 ? 16 : -16), sy - 20, FARBE.gruen, m.k > 0 ? LEFT : RIGHT);
    } else {
      const [wx, wy] = zuPlot(m.x0, 0.5);
      noStroke();
      fill(...FARBE.flaeche);
      stroke(...FARBE.gruen);
      strokeWeight(3);
      circle(wx, wy, 14);
    }

    // Grenze beschriften
    if (isFinite(xs) && xs > mm.min && xs < mm.max) {
      const [gx] = zuPlot(xs, 0);
      marke(`Grenze ${mm.kurz} = ${fFix(xs, mm.stellen + 1)}`, gx, PLOT.y + PLOT.h - 14, FARBE.senf, CENTER);
    }
    const [, sy] = zuPlot(0, schwelle);
    marke(`Schwelle ${fKomma(schwelle, 2)}`, PLOT.x + PLOT.w - 8, sy - 14, FARBE.senf, RIGHT);

    // Prognose
    if (prognoseX !== null) {
      const p = wahrsch(m, prognoseX);
      const [px, py] = zuPlot(prognoseX, p);
      stroke(...FARBE.text);
      strokeWeight(1.6);
      gestrichelt(true);
      line(px, PLOT.y + PLOT.h, px, py);
      line(PLOT.x, py, px, py);
      gestrichelt(false);
      stroke(...FARBE.flaeche);
      strokeWeight(2);
      fill(...FARBE.text);
      zeichneStern(px, py, 13);
      const klasse = p >= schwelle ? 1 : 0;
      const rechts = px < PLOT.x + PLOT.w - 260;
      marke(`p = ${prozent(p, 1)} → ${SZ.kurzKlassen[klasse]}`, rechts ? px + 16 : px - 16, py - 18, FARBE.text, rechts ? LEFT : RIGHT);
      marke(`${mm.kurz} = ${wertText(prognoseX)}`, px, PLOT.y + 18, FARBE.text, CENTER);
    }
  }

  clipEnde();

  // Achsen beschriften
  noStroke();
  fill(...FARBE.grau);
  textFont(SCHRIFT);
  textSize(12);
  for (let i = 0; i <= anzX; i++) {
    const v = mm.min + i * mm.gitter;
    const [gx] = zuPlot(v, 0);
    textAlign(CENTER, TOP);
    text(Number(v.toFixed(3)).toString(), gx, PLOT.y + PLOT.h + 6);
  }
  for (let i = 0; i <= 10; i += 2) {
    const [, gy] = zuPlot(mm.min, i / 10);
    textAlign(RIGHT, CENTER);
    text(fKomma(i / 10, 1), PLOT.x - 8, gy);
  }
  // Klassennamen an den Zeilen 0 und 1
  textStyle(BOLD);
  textSize(11);
  for (let k = 0; k < 2; k++) {
    const [, gy] = zuPlot(mm.min, k);
    fill(...KLASSEN_FARBE[k]);
    textAlign(LEFT, k === 1 ? BOTTOM : TOP);
    text(SZ.klassen[k], PLOT.x + 6, k === 1 ? gy - 4 : gy + 4);
  }
  fill(...FARBE.text);
  textSize(13);
  textAlign(CENTER, TOP);
  text(mm.einheit ? `${mm.name} ${mm.kurz} in ${mm.einheit}` : mm.name, PLOT.x + PLOT.w / 2, PLOT.y + PLOT.h + 26);
  push();
  translate(P_DATEN.x + 18, PLOT.y + PLOT.h / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, TOP);
  text(`Wahrscheinlichkeit p für „${SZ.kurzKlassen[1]}“`, 0, 0);
  pop();
  textStyle(NORMAL);

  zeichnePlotBanner();

  // Hinweis unter den Knöpfen
  noStroke();
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(12);
  textAlign(LEFT, CENTER);
  text(verfahren === "hand" ? "Griffe ziehen: ● Wendepunkt x₀ · ■ Steilheit k" : "Maus auf Punkt: Werte, p(x) und Fehler", 560, 838);
  text("Klick: Prognose · gelbe Linie ziehen: Schwellwert", 560, 856);

  // Mauszeiger
  if (ziehen === 0 || ziehen === 1 || griffUnterMaus() >= 0 || ziehen === 2 || prognoseUnterMaus()) {
    cursor("ew-resize");
  } else if (ziehen === 3 || schwelleUnterMaus()) {
    cursor("ns-resize");
  } else if (imPlot(mausX(), mausY()) && !laufAuto && modell) {
    cursor(CROSS);
  } else {
    cursor(ARROW);
  }
}

function zeichneStern(x, y, r) {
  beginShape();
  for (let i = 0; i < 10; i++) {
    const a = -HALF_PI + (i * PI) / 5;
    const rr = i % 2 === 0 ? r : r * 0.45;
    vertex(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
  endShape(CLOSE);
}

// Legende rechts in der Überschrift des Datenfelds
function zeichneLegende() {
  textFont(SCHRIFT);
  textSize(12);
  textAlign(LEFT, CENTER);
  const y = P_DATEN.y + 17;
  const eintraege = [
    { name: SZ.kurzKlassen[0], art: 0 },
    { name: SZ.kurzKlassen[1], art: 1 },
    { name: "Sigmoid p(x)", art: "kurve" },
    { name: "Schwellwert", art: "schwelle" },
  ];
  let breite = 0;
  for (const e of eintraege) {
    breite += textWidth(e.name) + 34;
  }
  let x = P_DATEN.x + P_DATEN.w - 16 - breite;
  for (const e of eintraege) {
    if (e.art === 0 || e.art === 1) {
      zeichneKlassenPunkt(x + 6, y, e.art, false);
    } else if (e.art === "kurve") {
      stroke(...FARBE.gruen);
      strokeWeight(3);
      line(x - 2, y, x + 14, y);
    } else {
      stroke(...FARBE.senf);
      strokeWeight(2);
      gestrichelt(true);
      line(x - 2, y, x + 14, y);
      gestrichelt(false);
    }
    noStroke();
    fill(...FARBE.text);
    textFont(SCHRIFT);
    textSize(12);
    textAlign(LEFT, CENTER);
    text(e.name, x + 18, y + 1);
    x += textWidth(e.name) + 34;
  }
}

function zeichnePlotBanner() {
  if (verfahren === "gd" && phase === "leer") {
    banner("Mit → bekommt die KI eine zufällige Startkurve", PLOT.x + PLOT.w / 2, PLOT.y + 64, false);
  }
}

// =============================================================================
// Erklärfeld
// =============================================================================
function erklaerung() {
  const m = zeigeModell();
  const wt = SZ.wort;
  const kv = SZ.kurzKlassen;

  if (verfahren === "hand") {
    const loss = logLoss(daten, m);
    const minimum = logLoss(daten, bestes);
    return {
      titel: "① Sigmoid-Kurve per Hand anpassen",
      text:
        `Die Sigmoid-Funktion liefert für jedes x eine Wahrscheinlichkeit zwischen 0 und 1 für „${kv[1]}“. ` +
        "Ziehe den runden Griff (Wendepunkt x₀) und den eckigen Griff (Steilheit k), bis die Kurve unten bei den 0-Punkten " +
        "und oben bei den 1-Punkten liegt. Der Log-Loss bestraft vor allem sichere, aber falsche Aussagen.",
      formeln: [
        "p(x) = 1 / (1 + e^(−k·(x − x₀)))",
        `k = ${fK(m.k)}    x₀ = ${fX0(m.x0)}`,
        `Log-Loss = ${fFix(loss, 3)}     (Minimum: ${fFix(minimum, 3)})`,
        loss - minimum < 0.003 ? "→ perfekt getroffen! ✓" : `→ noch ${fFix(loss - minimum, 3)} über dem Minimum`,
      ],
    };
  }

  if (phase === "leer") {
    return {
      titel: "② Gradientenabstieg · so lernt eine KI",
      text:
        "Die KI startet mit einer zufälligen Kurve und verschiebt k und x₀ in vielen kleinen Schritten so, dass der Log-Loss sinkt. " +
        "Für jeden Punkt zählt nur der Unterschied p − y zwischen Vorhersage und echter Klasse. Die Schrittweite ist die Lernrate η.",
      formeln: [
        "Fehler     L = −(1/n)·Σ [ y·ln p + (1 − y)·ln(1 − p) ]",
        "Gradient   ∂L/∂w = (1/n)·Σ (p − y)·x′     ∂L/∂c = (1/n)·Σ (p − y)",
        "Schritt    w ← w − η·∂L/∂w     c ← c − η·∂L/∂c",
        "(w, c in normierten Werten; k = w/Spannweite, x₀ = Lage bei p = 0,5)",
      ],
    };
  }

  const loss = verlauf.length > 0 ? verlauf[verlauf.length - 1].loss : 0;
  const vorher = verlauf.length > 1 ? verlauf[verlauf.length - 2].loss : loss;
  const formeln = [
    `w = ${fSig(gd.w, 4)}   c = ${fSig(gd.c, 4)}   →   k = ${fK(m.k)}   x₀ = ${fX0(m.x0)}`,
    `∂L/∂w = ${fSig(gd.gw, 3)}    ∂L/∂c = ${fSig(gd.gc, 3)}    η = ${lernrate}`,
    `Log-Loss = ${fFix(loss, 4)}     (Minimum: ${fFix(logLoss(daten, bestes), 4)})`,
  ];
  if (phase === "fertig") {
    return {
      titel: iteration >= MAX_SCHRITTE ? `Abgebrochen nach ${iteration} Lernschritten` : `Gelernt nach ${iteration} Lernschritten ✓`,
      text:
        iteration >= MAX_SCHRITTE
          ? loss > logLoss(daten, bestes) + 0.01
            ? "Die Lernrate ist zu groß: Die KI springt immer wieder über das Minimum hinweg und kommt nicht zur Ruhe. Wähle eine kleinere Lernrate."
            : "Die KI kommt kaum noch voran. Mit einer größeren Lernrate geht es schneller."
          : `Der Gradient ist fast null – bergab geht es nicht mehr. Jetzt den Schwellwert verschieben und beobachten, wie sich Konfusionsmatrix und ROC-Punkt ändern.`,
      formeln,
    };
  }
  let t;
  if (iteration === 0) {
    t = `Die Startkurve ist geraten. Für jeden Punkt wird p − y berechnet: Bei ${wt.mehrzahl} der Klasse 1 soll p groß sein, bei Klasse 0 klein. Daraus ergibt sich der Gradient.`;
  } else if (loss > vorher * 1.0001) {
    t = "Achtung: Der Log-Loss ist gestiegen! Die Lernrate ist so groß, dass die KI über das Minimum hinausspringt.";
  } else if (lernrate <= 0.5) {
    t = "Kleine Lernrate: Die Kurve bewegt sich nur langsam. Das ist sicher, dauert aber lange – probiere „schnell“ oder eine größere Lernrate.";
  } else {
    t = `Lernschritt ${iteration}: Die Kurve wird steiler und rutscht zur Grenze zwischen den Klassen. Der Log-Loss ist kleiner geworden.`;
  }
  return { titel: `Lernschritt ${iteration}`, text: t, formeln };
}

function zeichneErklaerung() {
  const e = erklaerung();
  P_ERKL.titel = e.titel;
  panel(P_ERKL);
  zeichneErklaerText(P_ERKL, e);
}

function zeichneEinstellungen() {
  const p = P_EINST;
  panel(p);
  noStroke();
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(12);
  textAlign(LEFT, TOP);
  const t =
    verfahren === "hand"
      ? "Per Hand: Du bist die KI. Schaffst du das Minimum des Log-Loss? Der Schwellwert ändert die Kurve nicht – nur die Entscheidung."
      : "Gradientenabstieg: Probiere η = 0.5 (langsam) und η = 40 (springt). Der Schwellwert wird nicht gelernt, er wird festgelegt.";
  text(t, p.x + 16, p.y + p.h - 22);
}

// =============================================================================
// Modell & Güte
// =============================================================================
function zeichneModellPanel() {
  const p = P_MODELL;
  panel(p);
  const m = zeigeModell();
  if (!m) {
    leerText(p, "Noch keine Kurve.\nHier stehen gleich Modellgleichung, Deutung und Fehler.");
    return;
  }
  const mm = MERKMAL;
  const x0 = p.x + 16;
  let y = p.y + KOPF_H + 18;

  noStroke();
  fill(...FARBE.text);
  textFont(MONO);
  textStyle(BOLD);
  textSize(14);
  textAlign(LEFT, CENTER);
  text(gleichung(m), x0, y);
  textStyle(NORMAL);

  y += 20;
  textFont(SCHRIFT);
  textSize(12.5);
  fill(...FARBE.grau);
  textAlign(LEFT, TOP);
  const xs = xBeiP(m, schwelle);
  const deutungen = [
    "x₀: " + SZ.wort.deuteX0(wertText(m.x0)),
    `k: Je größer k, desto schärfer der Übergang. ${m.k < 0 ? "k < 0: Die Kurve fällt – die Klassen sind vertauscht!" : `Hier steigt p im Bereich x₀ ± ${fSig(2 / Math.abs(m.k), 2)} von 12 % auf 88 %.`}`,
    `Entscheidung: p ≥ ${fKomma(schwelle, 2)}  →  „${SZ.kurzKlassen[1]}“, also ${m.k > 0 ? "ab" : "bis"} ${mm.kurz} ≈ ${isFinite(xs) ? fFix(xs, mm.stellen + 1) : "–"}`,
  ];
  for (const d of deutungen) {
    for (const z of umbrechen(d, 440).slice(0, 2)) {
      text(z, x0, y);
      y += 16;
    }
    y += 2;
  }

  // Prognose bzw. Frage
  y = p.y + p.h - 18;
  textStyle(BOLD);
  textSize(13);
  textAlign(LEFT, CENTER);
  if (prognoseX !== null) {
    const pr = wahrsch(m, prognoseX);
    fill(...FARBE.text);
    text(`★ ${mm.kurz} = ${wertText(prognoseX)}  →  p = ${prozent(pr, 1)}  →  ${SZ.klassen[pr >= schwelle ? 1 : 0]}`, x0, y);
  } else {
    textStyle(NORMAL);
    fill(...FARBE.blass);
    text("Frage: " + SZ.wort.frage, x0, y);
  }
  textStyle(NORMAL);

  // rechts: Kennzahlen und Lernkurve
  const tx = p.x + 478;
  const tw = p.w - 478 - 16;
  stroke(...FARBE.linie);
  strokeWeight(1);
  line(tx - 14, p.y + KOPF_H + 10, tx - 14, p.y + p.h - 34);
  const r = konfusion(daten, m, schwelle);
  const reihen = [
    ["n", String(daten.length)],
    ["Log-Loss", fFix(logLoss(daten, m), 3)],
    ["Genauigkeit", prozent(r.genau, 1)],
    ["falsch", `${r.fn + r.fp} von ${r.n}`],
  ];
  let ry = p.y + KOPF_H + 16;
  for (const [k, v] of reihen) {
    noStroke();
    fill(...FARBE.blass);
    textFont(SCHRIFT);
    textSize(12);
    textAlign(LEFT, CENTER);
    text(k, tx, ry);
    fill(...FARBE.text);
    textFont(MONO);
    textSize(13);
    textStyle(k === "Log-Loss" ? BOLD : NORMAL);
    textAlign(RIGHT, CENTER);
    text(v, tx + tw, ry);
    textStyle(NORMAL);
    ry += 19;
  }
  if (verfahren === "gd" && verlauf.length > 1) {
    zeichneLernkurve(tx, ry, tw, p.y + p.h - 44 - ry);
  }
}

function zeichneLernkurve(gx, gy, gw, gh) {
  const minX = verlauf[0].i;
  const maxX = Math.max(1, verlauf[verlauf.length - 1].i);
  let maxY = 0;
  for (const q of verlauf) {
    maxY = Math.max(maxY, q.loss);
  }
  maxY = maxY > 0 ? maxY * 1.1 : 1;
  const sx = (x) => gx + ((x - minX) / Math.max(1, maxX - minX)) * gw;
  const sy = (v) => gy + gh - (Math.min(v, maxY) / maxY) * gh;
  stroke(...FARBE.blass);
  strokeWeight(1);
  line(gx, gy + gh, gx + gw, gy + gh);
  noFill();
  stroke(...FARBE.blau);
  strokeWeight(2);
  beginShape();
  for (const q of verlauf) {
    vertex(sx(q.i), sy(q.loss));
  }
  endShape();
  noStroke();
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(10.5);
  textAlign(RIGHT, TOP);
  text("Log-Loss je Lernschritt", gx + gw, gy + gh + 2);
}

// =============================================================================
// Konfusionsmatrix (Skript 4.2)
// =============================================================================
const KONF = { x: P_KONF.x + 90, y: P_KONF.y + KOPF_H + 32, zw: 54, zh: 40 };

function zelleUnterMaus() {
  const mx = mausX();
  const my = mausY();
  const felder = [
    ["wp", 0, 0],
    ["fn", 1, 0],
    ["fp", 0, 1],
    ["wn", 1, 1],
  ];
  for (const [id, sp, ze] of felder) {
    const x = KONF.x + sp * KONF.zw;
    const y = KONF.y + ze * KONF.zh;
    if (mx >= x && mx <= x + KONF.zw && my >= y && my <= y + KONF.zh) {
      return id;
    }
  }
  return null;
}

function zeichneKonfusion() {
  const p = P_KONF;
  p.titel = `Konfusionsmatrix · Schwelle ${fKomma(schwelle, 2)}`;
  panel(p);
  const m = zeigeModell();
  if (!m) {
    leerText(p, "Erst eine Kurve, dann wird gezählt:\nWie viele richtig, wie viele falsch?");
    return;
  }
  const r = konfusion(daten, m, schwelle);
  const kv = SZ.kurzKlassen;
  const hover = zelleUnterMaus();

  noStroke();
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(10.5);
  textAlign(CENTER, BOTTOM);
  text("vorhergesagt", KONF.x + KONF.zw, KONF.y - 15);
  fill(...FARBE.grau);
  textSize(11);
  text(kv[1], KONF.x + KONF.zw / 2, KONF.y - 2);
  text(kv[0], KONF.x + KONF.zw * 1.5, KONF.y - 2);
  textAlign(RIGHT, CENTER);
  text(kv[1], KONF.x - 6, KONF.y + KONF.zh / 2);
  text(kv[0], KONF.x - 6, KONF.y + KONF.zh * 1.5);
  push();
  translate(p.x + 18, KONF.y + KONF.zh);
  rotate(-HALF_PI);
  fill(...FARBE.blass);
  textSize(10.5);
  textAlign(CENTER, CENTER);
  text("tatsächlich", 0, 0);
  pop();

  const felder = [
    ["wp", 0, 0, r.wp, true, "WP"],
    ["fn", 1, 0, r.fn, false, "FN"],
    ["fp", 0, 1, r.fp, false, "FP"],
    ["wn", 1, 1, r.wn, true, "WN"],
  ];
  for (const [id, sp, ze, wert, richtig, kurz] of felder) {
    const x = KONF.x + sp * KONF.zw;
    const y = KONF.y + ze * KONF.zh;
    stroke(...(hover === id ? FARBE.senf : FARBE.linie));
    strokeWeight(hover === id ? 2.5 : 1);
    fill(...(richtig ? FARBE.gruenFlaeche : FARBE.rotFlaeche));
    rect(x, y, KONF.zw, KONF.zh);
    noStroke();
    fill(...(richtig ? FARBE.gruen : FARBE.akzent));
    textFont(MONO);
    textStyle(BOLD);
    textSize(17);
    textAlign(CENTER, CENTER);
    text(String(wert), x + KONF.zw / 2, y + KONF.zh / 2 - 3);
    textStyle(NORMAL);
    textFont(SCHRIFT);
    textSize(9.5);
    text(kurz, x + KONF.zw / 2, y + KONF.zh - 7);
  }

  // Kennzahlen rechts
  const tx = KONF.x + 2 * KONF.zw + 16;
  const tw = p.x + p.w - 14 - tx;
  const reihen = [
    ["Sensitivität", prozent(r.sens, 1), "WP/(WP+FN)"],
    ["Spezifität", prozent(r.spez, 1), "WN/(WN+FP)"],
    ["1 − Spezifität", prozent(r.fpr, 1), "FP/(FP+WN)"],
  ];
  let ry = KONF.y + 2;
  for (const [k, v, f] of reihen) {
    noStroke();
    fill(...FARBE.grau);
    textFont(SCHRIFT);
    textSize(11.5);
    textAlign(LEFT, CENTER);
    text(k, tx, ry);
    fill(...FARBE.text);
    textFont(MONO);
    textStyle(BOLD);
    textSize(12);
    textAlign(RIGHT, CENTER);
    text(v, tx + tw, ry);
    textStyle(NORMAL);
    fill(...FARBE.blass);
    textSize(9.5);
    textAlign(LEFT, CENTER);
    text(f, tx, ry + 12);
    ry += 28;
  }

  // Bedeutung der Fehler
  noStroke();
  fill(...FARBE.akzent);
  textFont(SCHRIFT);
  textSize(11);
  textAlign(LEFT, CENTER);
  const yU = p.y + p.h - 30;
  text(`FN ${r.fn}: ${SZ.wort.fn}`, p.x + 16, yU);
  text(`FP ${r.fp}: ${SZ.wort.fp}`, p.x + 16, yU + 15);
}

// =============================================================================
// ROC-Kurve und AUC (Skript 4.2)
// =============================================================================
function zeichneRoc() {
  const p = P_ROC;
  p.titel = "ROC-Kurve & AUC";
  panel(p);
  const m = zeigeModell();
  if (!m) {
    leerText(p, "Für jeden Schwellwert ein Punkt:\nSensitivität gegen\nFalsch-Positiv-Rate.");
    return;
  }
  const roc = rocKurve(daten, m);
  const r = konfusion(daten, m, schwelle);
  const g = 128;
  const gx = p.x + 44;
  const gy = p.y + KOPF_H + 10;
  const sx = (v) => gx + v * g;
  const sy = (v) => gy + g - v * g;

  // Fläche unter der Kurve
  noStroke();
  fill(...FARBE.gruen, 40);
  beginShape();
  vertex(sx(0), sy(0));
  for (const q of roc.punkte) {
    vertex(sx(q.fpr), sy(q.sens));
  }
  vertex(sx(1), sy(0));
  endShape(CLOSE);

  stroke(...FARBE.linie);
  strokeWeight(1);
  noFill();
  rect(gx, gy, g, g);
  stroke(...FARBE.blass);
  gestrichelt(true);
  line(sx(0), sy(0), sx(1), sy(1));
  gestrichelt(false);

  stroke(...FARBE.gruen);
  strokeWeight(2.4);
  noFill();
  beginShape();
  for (const q of roc.punkte) {
    vertex(sx(q.fpr), sy(q.sens));
  }
  endShape();

  // aktueller Schwellwert
  stroke(...FARBE.flaeche);
  strokeWeight(1.5);
  fill(...FARBE.senf);
  circle(sx(r.fpr), sy(r.sens), 11);

  noStroke();
  fill(...FARBE.grau);
  textFont(SCHRIFT);
  textSize(10);
  textAlign(CENTER, TOP);
  text("0", sx(0), gy + g + 3);
  text("1", sx(1), gy + g + 3);
  text("Falsch-Positiv-Rate", gx + g / 2, gy + g + 3);
  textAlign(RIGHT, CENTER);
  text("1", gx - 4, sy(1));
  text("0", gx - 4, sy(0));
  push();
  translate(p.x + 16, gy + g / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text("Sensitivität", 0, 0);
  pop();

  // AUC und Bewertung
  const tx = gx + g + 18;
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(11.5);
  textAlign(LEFT, TOP);
  text("Fläche unter der Kurve", tx, gy + 2);
  fill(...FARBE.text);
  textFont(MONO);
  textStyle(BOLD);
  textSize(22);
  text("AUC = " + fKomma(roc.auc, 2), tx, gy + 20);
  textStyle(NORMAL);
  textFont(SCHRIFT);
  textSize(11.5);
  fill(...FARBE.grau);
  let urteil;
  if (roc.auc < 0.5) {
    urteil = "unter 0,5: Klassen vertauscht!";
  } else if (roc.auc < 0.7) {
    urteil = "kaum besser als Raten (0,5)";
  } else if (roc.auc < 0.85) {
    urteil = "brauchbar";
  } else if (roc.auc < 0.95) {
    urteil = "gut";
  } else {
    urteil = "sehr gut";
  }
  text(urteil, tx, gy + 52);
  fill(...FARBE.senf);
  textStyle(BOLD);
  text(`● Schwelle ${fKomma(schwelle, 2)}`, tx, gy + 82);
  textStyle(NORMAL);
  fill(...FARBE.blass);
  textSize(10.5);
  text(`Sens. ${prozent(r.sens, 0)} · FPR ${prozent(r.fpr, 0)}`, tx, gy + 99);
  text("Die AUC hängt nicht vom\nSchwellwert ab.", tx, gy + 116);
}

// -----------------------------------------------------------------------------
// Tooltip
// -----------------------------------------------------------------------------
function zeichneTooltip() {
  if (ziehen >= 0) {
    return;
  }
  const q = punktUnterMaus();
  if (!q) {
    return;
  }
  const zeilen = [`${q.name || SZ.wort.einzahl + " " + q.id} · ${MERKMAL.name} ${wertText(q.x)} · Klasse: ${SZ.klassen[q.y]}`];
  const m = zeigeModell();
  if (m) {
    const p = wahrsch(m, q.x);
    const kat = { wp: "richtig (WP)", wn: "richtig (WN)", fp: "falsch (FP)", fn: "falsch (FN)" }[kategorie(m, q)];
    zeilen.push(`p(x) = ${prozent(p, 1)} → vorhergesagt: ${SZ.kurzKlassen[p >= schwelle ? 1 : 0]} · ${kat}`);
    zeilen.push(`Fehler dieses Punktes: −ln(${q.y === 1 ? "p" : "1 − p"}) = ${fFix(punktFehler(m, q), 3)}`);
  }
  tooltip(zeilen);
}

// =============================================================================
// Eingaben
// =============================================================================
function mousePressed() {
  if (laufAuto) {
    return;
  }
  const mx = mausX();
  const my = mausY();
  const gr = griffUnterMaus();
  if (gr >= 0) {
    ziehen = gr;
    return;
  }
  if (!imPlot(mx, my)) {
    return;
  }
  if (schwelleUnterMaus()) {
    ziehen = 3;
    return;
  }
  if (modell) {
    prognoseX = aufRaster(ausPlot(mx, my).x, MERKMAL);
    ziehen = 2;
    return;
  }
  zeigeHinweis("Erst lernen lassen (→), dann kann die Kurve Prognosen machen.");
}

function mouseDragged() {
  if (ziehen < 0) {
    return;
  }
  const pkt = ausPlot(mausX(), mausY());
  if (ziehen === 2) {
    prognoseX = aufRaster(pkt.x, MERKMAL);
  } else if (ziehen === 3) {
    schwelle = Math.round(constrain(pkt.p, 0.05, 0.95) * 100) / 100;
  } else if (ziehen === 0) {
    modell = { k: modell.k, x0: pkt.x };
  } else if (ziehen === 1) {
    // Steilheits-Griff liegt bei x₀ + 2/k  →  k = 2 / (x − x₀)
    const r = MERKMAL.max - MERKMAL.min;
    let d = pkt.x - modell.x0;
    if (Math.abs(d) < r * 0.006) {
      d = d < 0 ? -r * 0.006 : r * 0.006;
    }
    modell = { k: 2 / d, x0: modell.x0 };
  }
}

function mouseReleased() {
  ziehen = -1;
}

function keyPressed() {
  if (!tastaturFrei()) {
    return true;
  }
  if (key === "ArrowRight") {
    if (!laufAuto) {
      schritt();
    }
    return false;
  }
  if (key === " ") {
    autoUmschalten();
    return false;
  }
  if ((key === "r" || key === "R") && !laufAuto) {
    neuStarten();
    return false;
  }
  if (key === "Escape") {
    prognoseX = null;
    return false;
  }
  return true;
}

// =============================================================================
// Daten als CSV speichern (z. B. zur Weiterarbeit in Python, Orange oder Excel)
// =============================================================================
function csvSpeichern() {
  const m = zeigeModell();
  const kopf = [SZ.fest ? "name" : "nr", MERKMAL.csv, "klasse"];
  if (m) {
    kopf.push("p_klasse1", "vorhersage");
  }
  const zeilen = [kopf.join(";")];
  const sortiert = daten.slice().sort((a, b) => a.id - b.id);
  for (const q of sortiert) {
    const z = [q.name || q.id, q.x.toFixed(MERKMAL.stellen), q.y];
    if (m) {
      const p = wahrsch(m, q.x);
      z.push(p.toFixed(4), p >= schwelle ? 1 : 0);
    }
    zeilen.push(z.join(";"));
  }
  ladeHerunter(SZ.csvDatei, zeilen.join("\n") + "\n");
}

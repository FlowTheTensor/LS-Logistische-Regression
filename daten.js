// =============================================================================
//  daten.js – Daten für die Beispiele (jeweils MIT Klasse 0 oder 1 = "Label")
// =============================================================================
//  Logistische Regression ist trotz ihres Namens eine KLASSIFIKATION
//  (überwachtes Lernen). Zu jeder Eingabe x ist die Klasse y ∈ {0, 1} bekannt.
//  Gesucht ist eine Sigmoid-Kurve
//        p(x) = 1 / (1 + e^(−k·(x − x₀)))
//  die für jedes x die Wahrscheinlichkeit für Klasse 1 angibt.
//
//  Beispiel "skript" (Standard):
//    Skript 4.2: Schuhgröße → Geschlecht. Feste Daten aus dem Diagramm.
//    Die Kurve im Skript hat x₀ = 39,5 und k = 0,8.
//
//  Beispiel "sortierung":
//    Sortierstation der Modellanlage. Ein optischer Sensor misst die
//    Reflexion in %. → Ist das Werkstück aus Metall (silber)?
//
//  Beispiel "lager":
//    Antrieb mit Wälzlager. Gemessen wird die Schwinggeschwindigkeit in mm/s.
//    → Liegt ein Lagerschaden vor? (Predictive Maintenance)
//
//  Beispiel "temperatur":
//    Servomotor. Gemessen wird die Wicklungstemperatur in °C.
//    → Schaltet der Motor in der nächsten Stunde wegen Überlast ab?
// =============================================================================

const SZENARIEN = {
  skript: {
    name: "Skript · Schuhgröße",
    titel: "Geschlecht vs. Schuhgröße",
    untertitel: "Beispiel aus dem Skript (4.2) · Logistische Regression · binäre Klassifikation",
    merkmal: { name: "Schuhgröße", kurz: "x", einheit: "", min: 34, max: 48, gitter: 1, stellen: 0, raster: 1, csv: "schuhgroesse" },
    klassen: ["Weiblich (0)", "Männlich (1)"],
    kurzKlassen: ["weiblich", "männlich"],
    fest: [
      { name: "Anna", x: 36, y: 0 },
      { name: "Emma", x: 37, y: 0 },
      { name: "Lisa", x: 38, y: 0 },
      { name: "Sarah", x: 39, y: 0 },
      { name: "Julia", x: 40, y: 0 },
      { name: "Nina", x: 41, y: 0 },
      { name: "Ben", x: 39, y: 1 },
      { name: "Paul", x: 40, y: 1 },
      { name: "Leon", x: 41, y: 1 },
      { name: "Max", x: 42, y: 1 },
      { name: "Tim", x: 43, y: 1 },
      { name: "Tom", x: 44, y: 1 },
      { name: "Jan", x: 45, y: 1 },
    ],
    // Kurve aus dem Skript – kann zum Vergleich eingeblendet werden
    skriptKurve: { k: 0.8, x0: 39.5 },
    wort: {
      einzahl: "Person",
      mehrzahl: "Personen",
      fn: "Männer als weiblich eingestuft",
      fp: "Frauen als männlich eingestuft",
      deuteX0: (x0) => `Bei Schuhgröße ${x0} ist die Wahrscheinlichkeit genau 50 % – hier liegt die Grenze zwischen den Klassen.`,
      frage: "Wie wahrscheinlich ist „männlich“ bei Schuhgröße 40?",
    },
    csvDatei: "schuhgroesse_geschlecht_skript.csv",
  },

  sortierung: {
    name: "Sortierstation · Metall?",
    titel: "Sensordaten · Sortierstation",
    untertitel: "Optischer Sensor · Logistische Regression · Werkstücke sortieren",
    startwert: 20260115,
    merkmal: { name: "Reflexion", kurz: "R", einheit: "%", min: 0, max: 100, gitter: 10, stellen: 0, raster: 1, csv: "reflexion_prozent" },
    klassen: ["Kunststoff schwarz (0)", "Metall silber (1)"],
    kurzKlassen: ["Kunststoff", "Metall"],
    // Klasse 0 und 1: Mittelwert, Streuung, Anzahl
    verteilung: [
      { mittel: 34, streuung: 9, n: 22 },
      { mittel: 60, streuung: 10, n: 22 },
    ],
    wort: {
      einzahl: "Werkstück",
      mehrzahl: "Werkstücke",
      fn: "Metallteile im Kunststoff-Behälter",
      fp: "Kunststoffteile im Metall-Behälter",
      deuteX0: (x0) => `Bei ${x0} Reflexion ist das Modell unentschieden (50 %). Darüber wird eher „Metall“ erkannt.`,
      frage: "Ein Werkstück reflektiert 50 % – in welchen Behälter kommt es?",
    },
    csvDatei: "sensordaten_sortierstation.csv",
  },

  lager: {
    name: "Antrieb · Lagerschaden",
    titel: "Schwingungsmessung · Antrieb M2",
    untertitel: "Zustandsüberwachung · Logistische Regression · Predictive Maintenance",
    startwert: 20260219,
    merkmal: { name: "Schwinggeschw.", kurz: "v", einheit: "mm/s", min: 0, max: 12, gitter: 1, stellen: 1, raster: 0.1, csv: "schwinggeschwindigkeit_mm_s" },
    klassen: ["Lager i. O. (0)", "Lagerschaden (1)"],
    kurzKlassen: ["i. O.", "Schaden"],
    verteilung: [
      { mittel: 2.9, streuung: 1.1, n: 32 },
      { mittel: 6.2, streuung: 1.6, n: 14 },
    ],
    wort: {
      einzahl: "Messung",
      mehrzahl: "Messungen",
      fn: "Lagerschäden übersehen",
      fp: "Fehlalarme (Lager war i. O.)",
      deuteX0: (x0) => `Ab ${x0} ist ein Lagerschaden wahrscheinlicher als ein gesundes Lager.`,
      frage: "Wo setzt man den Schwellwert, wenn ein übersehener Schaden sehr teuer ist?",
    },
    csvDatei: "schwingung_antrieb_m2.csv",
  },

  temperatur: {
    name: "Servomotor · Überlast",
    titel: "Temperaturdaten · Servomotor Achse 3",
    untertitel: "Überlast-Vorhersage · Logistische Regression · binäre Klassifikation",
    startwert: 20251103,
    merkmal: { name: "Wicklungstemp.", kurz: "T", einheit: "°C", min: 30, max: 150, gitter: 10, stellen: 0, raster: 1, csv: "wicklungstemperatur_c" },
    klassen: ["läuft weiter (0)", "Abschaltung (1)"],
    kurzKlassen: ["läuft", "Abschaltung"],
    verteilung: [
      { mittel: 76, streuung: 13, n: 26 },
      { mittel: 106, streuung: 11, n: 18 },
    ],
    wort: {
      einzahl: "Messung",
      mehrzahl: "Messungen",
      fn: "Abschaltungen nicht vorhergesagt",
      fp: "unnötige Warnungen",
      deuteX0: (x0) => `Ab ${x0} rechnet das Modell eher mit einer Überlast-Abschaltung in der nächsten Stunde.`,
      frage: "Ab welcher Temperatur soll die Steuerung die Drehzahl drosseln?",
    },
    csvDatei: "temperatur_servomotor_achse3.csv",
  },
};

// aktives Beispiel – wird mit setzeSzenario() umgeschaltet
let SZ = SZENARIEN.skript;
let MERKMAL = SZ.merkmal;

function setzeSzenario(schluessel) {
  SZ = SZENARIEN[schluessel];
  MERKMAL = SZ.merkmal;
}

// auf das Messraster runden (z. B. 0.1 mm/s oder 1 °C)
function aufRaster(v, m) {
  const r = Math.round(v / m.raster) * m.raster;
  return Number(r.toFixed(Math.max(0, m.stellen)));
}

function erzeugeDaten(startwert) {
  if (SZ.fest) {
    return SZ.fest.map((p, i) => ({ id: i + 1, name: p.name, x: p.x, y: p.y }));
  }
  const rnd = zufallsGenerator(startwert);
  const m = MERKMAL;
  const liste = [];
  SZ.verteilung.forEach((v, klasse) => {
    for (let i = 0; i < v.n; i++) {
      const x = v.mittel + gaussZahl(rnd) * v.streuung;
      liste.push({ x: aufRaster(begrenzen(x, m.min + m.raster, m.max - m.raster), m), y: klasse, r: rnd() });
    }
  });
  // in zufälliger Reihenfolge durchnummerieren (wie gemessen)
  liste.sort((a, b) => a.r - b.r);
  for (let i = 0; i < liste.length; i++) {
    liste[i].id = 101 + i;
    delete liste[i].r;
  }
  return liste;
}

// Messwert mit Einheit, z. B. "6.2 mm/s" oder "Schuhgröße 41"
function wertText(v, m) {
  m = m || MERKMAL;
  return m.einheit ? v.toFixed(m.stellen) + " " + m.einheit : v.toFixed(m.stellen);
}

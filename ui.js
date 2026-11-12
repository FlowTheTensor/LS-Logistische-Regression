// =============================================================================
//  ui.js – HTML-Bedienelemente über der Zeichenfläche
// =============================================================================
//  Positionen im logischen 1600×900-Raster (Helfer in basis.js).
// =============================================================================

function erstelleUi() {
  erstelleHuelle();
  erstelleBeispielReiter(szenarioWechseln);

  ui.neueDaten = knopf(1270, 12, 150, "Neue Messdaten", neueMessdaten, "kopf");
  ui.csv = knopf(1428, 12, 148, "CSV speichern", csvSpeichern, "kopf");

  // --- unter dem Diagramm ----------------------------------------------------
  ui.abstaende = knopf(40, 830, 230, "Fehler je Punkt zeigen", () => {
    abstaendeAn = !abstaendeAn;
  });
  ui.falsche = knopf(280, 830, 260, "Fehlklassifikationen zeigen", () => {
    falscheAn = !falscheAn;
  });

  // --- Ablauf (Erklärfeld) ---------------------------------------------------
  ui.schritt = knopf(876, 284, 190, "Schritt ▶ (→)", schritt, "dunkel");
  ui.auto = knopf(1074, 284, 190, "▶ Automatisch", autoUmschalten, "primaer");
  ui.neu = knopf(1272, 284, 190, "↺ Neu starten (R)", neuStarten);

  // --- Einstellungen ---------------------------------------------------------
  platziere(createSpan("Verfahren"), 876, 380, 80);
  ui.verfahren = auswahl(
    956,
    380,
    262,
    [
      ["① Kurve per Hand anpassen", "hand"],
      ["② Gradientenabstieg (KI lernt)", "gd"],
    ],
    verfahren,
    verfahrenWechseln
  );

  platziere(createSpan("Lernrate η"), 1236, 380, 84);
  ui.lernrate = auswahl(
    1320,
    380,
    78,
    ["0.5", "2", "8", "40"].map((v) => [v, v]),
    lernrate,
    (v) => {
      lernrate = parseFloat(v);
      neuStarten();
    }
  );

  platziere(createSpan("Tempo"), 1418, 380, 56);
  ui.tempo = auswahl(
    1474,
    380,
    90,
    ["langsam", "mittel", "schnell"].map((v) => [v, v]),
    tempo,
    (v) => {
      tempo = v;
    }
  );

  platziere(createSpan("Schwellwert"), 876, 418, 96);
  ui.schwelle = platziere(createSlider(0.05, 0.95, schwelle, 0.01), 972, 422, 190);
  ui.schwelle.input(() => {
    schwelle = Number(ui.schwelle.value());
  });
  ui.schwelle.changed(() => ui.schwelle.elt.blur());
  ui.schwelleWert = platziere(createSpan(""), 1170, 418, 60);

  ui.vergleich = kaestchen(1250, 418, 320, "Beste Kurve zum Vergleich", vergleichAn, (an) => {
    vergleichAn = an;
  });
}

// Beschriftung eines Kästchens ändern (p5 legt den Text in ein <span> oder <label>)
function kaestchenText(el, text) {
  const ziel = el.elt.querySelector("span") || el.elt.querySelector("label");
  if (ziel && ziel.textContent !== " " + text) {
    ziel.textContent = " " + text;
  }
}

// wird in jedem Frame aufgerufen
function aktualisiereUi() {
  beschrifte(ui.auto, laufAuto ? "■ Anhalten" : "▶ Automatisch");
  klasse(ui.auto, "primaer", !laufAuto);
  klasse(ui.auto, "dunkel", laufAuto);

  const fertig = phase === "fertig";
  if (verfahren === "hand") {
    beschrifte(ui.schritt, "Kurve ziehen ✋");
  } else {
    beschrifte(ui.schritt, phase === "leer" ? "Startkurve ▶ (→)" : fertig ? "Fertig ✓" : "Lernschritt ▶ (→)");
  }
  sperre(ui.schritt, laufAuto || fertig || verfahren === "hand");
  sperre(ui.auto, fertig || verfahren === "hand");

  beschrifte(ui.abstaende, abstaendeAn ? "Fehler je Punkt ausblenden" : "Fehler je Punkt zeigen");
  klasse(ui.abstaende, "kopf", true);
  klasse(ui.abstaende, "aktiv", abstaendeAn);
  beschrifte(ui.falsche, falscheAn ? "Fehlklassifikationen ausblenden" : "Fehlklassifikationen zeigen");
  klasse(ui.falsche, "kopf", true);
  klasse(ui.falsche, "aktiv", falscheAn);

  beschrifte(ui.schwelleWert, fKomma(schwelle, 2));
  if (Math.abs(Number(ui.schwelle.value()) - schwelle) > 1e-9) {
    ui.schwelle.value(schwelle);
  }
  kaestchenText(ui.vergleich, SZ.skriptKurve ? "Skript-Kurve (x₀ = 39,5 · k = 0,8)" : "Beste Kurve zum Vergleich");

  if (ui.verfahren.value() !== verfahren) {
    ui.verfahren.selected(verfahren);
  }
  aktualisiereBeispielReiter(szenario, laufAuto);
  sperre(ui.lernrate, laufAuto || verfahren !== "gd");
  sperre(ui.neueDaten, laufAuto || !!SZ.fest);
  for (const el of [ui.verfahren, ui.neu]) {
    sperre(el, laufAuto);
  }
}

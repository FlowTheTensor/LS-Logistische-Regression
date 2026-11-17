# KI-Klassifikation · Logistische Regression

Überwachtes Lernen zum Zuschauen und Ausprobieren: Aus Messwerten mit bekannter Klasse (0 oder 1) lernt das Programm eine Sigmoid-Kurve, die für jeden neuen Messwert eine **Wahrscheinlichkeit** angibt. Die Kurve lässt sich erst per Hand anpassen, dann sieht man zu, wie eine KI sie mit dem Gradientenabstieg selbst findet. Mit Schwellwert, Konfusionsmatrix und ROC-Kurve wird anschließend bewertet, wie gut das Modell entscheidet.

Erstellt für den Unterricht an der Jakob-Preh-Schule ([bsnes.de](https://www.bsnes.de)).

**Live-Version:** [Hier direkt anschauen](https://flowthetensor.github.io/LS-Logistische-Regression/)

---

## Worum geht es?

Trotz ihres Namens ist die logistische Regression **keine Regression**, sondern eine **Klassifikation**: Zu jedem Messwert x ist bekannt, ob er zu Klasse 0 oder Klasse 1 gehört (das „Label“). Gesucht ist eine Kurve, die für jedes x sagt, wie wahrscheinlich Klasse 1 ist. Ab einem **Schwellwert** (meist 50 %) entscheidet das Modell dann „Klasse 1“.

In der roten **Beispiel-Leiste** unter der Kopfzeile stehen vier Beispiele zur Verfügung. Reiter 1 ist das Beispiel aus dem Skript (Kap. 4.2), die anderen drei kommen aus der Industrie.

> Die Industrie-Werte sind simuliert und an eine typische 24-V-Modellanlage für Industrie 4.0 angelehnt. Sobald echte Messdaten der ETS-Anlage verfügbar sind, kann man sie mit realen Werten vergleichen.

### 1. Skript · Schuhgröße → Geschlecht (Standard)

Die 13 Personen aus dem Skript, fest und unverändert (Anna, Emma, … Tom, Jan):

| Merkmal | Bereich | Klasse 0 | Klasse 1 |
|---|---|---|---|
| Schuhgröße | 34 … 48 | weiblich | männlich |

Im Bereich 39 … 41 überlappen sich die Klassen – dort kann keine Kurve alles richtig machen. Die Kurve aus dem Skript ($x_0 = 39{,}5$, $k = 0{,}8$) lässt sich zum Vergleich einblenden. Der Gradientenabstieg findet eine etwas bessere: $x_0 \approx 40$, $k \approx 1{,}0$ (Log-Loss 0,387 statt 0,404).

### 2. Sortierstation · Metall oder Kunststoff?

Ein optischer Sensor misst die Reflexion des Werkstücks. Silberne Metallteile reflektieren mehr als schwarze Kunststoffteile – aber nicht immer eindeutig (Oberfläche, Schmutz, Winkel).

| Klasse | Reflexion | Anzahl |
|---|---|---|
| Kunststoff schwarz (0) | ≈ 34 % ± 9 % | 22 |
| Metall silber (1) | ≈ 60 % ± 10 % | 22 |

### 3. Antrieb M2 · Lagerschaden (Predictive Maintenance)

Am Wälzlager eines Antriebs wird die Schwinggeschwindigkeit gemessen. Ein beginnender Lagerschaden zeigt sich als erhöhte Schwingung, lange bevor der Antrieb ausfällt.

| Klasse | Schwinggeschwindigkeit | Anzahl |
|---|---|---|
| Lager i. O. (0) | ≈ 2,9 mm/s ± 1,1 | 32 |
| Lagerschaden (1) | ≈ 6,2 mm/s ± 1,6 | 14 |

Die Klassen sind **unausgewogen**: Schäden sind seltener als gesunde Lager. Und ein übersehener Schaden ist viel teurer als ein Fehlalarm – ein gutes Beispiel dafür, warum man den Schwellwert nicht immer bei 50 % lässt.

### 4. Servomotor Achse 3 · Überlast-Abschaltung

Gemessen wird die Wicklungstemperatur. Vorhergesagt werden soll, ob der Motor in der nächsten Stunde wegen Überlast abschaltet.

| Klasse | Wicklungstemperatur | Anzahl |
|---|---|---|
| läuft weiter (0) | ≈ 76 °C ± 13 | 26 |
| Abschaltung (1) | ≈ 106 °C ± 11 | 18 |

## Die logistische Regression

**Sigmoid-Kurve** mit Wendepunkt $x_0$ und Steilheit $k$ (wie im Skript):

$$p(x) = \frac{1}{1 + e^{-k\,(x - x_0)}}$$

- $x_0$: Hier ist $p = 0{,}5$ – die Grenze zwischen den Klassen.
- $k$: Je größer, desto schärfer der Übergang. Bei $k < 0$ fällt die Kurve, die Klassen sind dann vertauscht.

**Log-Loss** (Kreuzentropie): mittlerer Fehler über alle $n$ Punkte. Er bestraft vor allem Aussagen, die sicher, aber falsch sind.

$$L = -\frac{1}{n} \sum_i \big[\, y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \,\big]$$

### ① Kurve per Hand anpassen

Den runden Griff (Wendepunkt $x_0$) und den eckigen Griff (Steilheit $k$) mit der Maus ziehen, bis die Kurve unten bei den 0-Punkten und oben bei den 1-Punkten liegt. Das Erklärfeld zeigt, wie weit der Log-Loss noch über dem Minimum liegt.

### ② Gradientenabstieg – so lernt eine KI

Die KI startet mit einer geratenen Kurve und verbessert sie in vielen kleinen Schritten. Damit eine Lernrate für alle Beispiele passt, wird x vorher auf 0 … 1 normiert ($x'$), gelernt werden $w$ und $c$ mit $p = \text{sig}(w \cdot x' + c)$:

$$\frac{\partial L}{\partial w} = \frac{1}{n} \sum_i (p_i - y_i)\, x'_i \qquad \frac{\partial L}{\partial c} = \frac{1}{n} \sum_i (p_i - y_i)$$

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w} \qquad c \leftarrow c - \eta \cdot \frac{\partial L}{\partial c}$$

Für jeden Punkt zählt nur der Unterschied $p - y$ zwischen Vorhersage und echter Klasse. Die Schrittweite ist die **Lernrate η** (0,5 · 2 · 8 · 40). Ist sie zu groß, springt die KI über das Minimum hinweg und der Log-Loss steigt wieder; ist sie zu klein, dauert es sehr lange. Nach höchstens 6000 Lernschritten wird abgebrochen.

### Bewertung (Skript 4.2)

Positiv = Klasse 1. Das Modell sagt „positiv“, wenn $p(x) \ge$ Schwellwert.

| | vorhergesagt 1 | vorhergesagt 0 |
|---|---|---|
| **tatsächlich 1** | wahr positiv (WP) | falsch negativ (FN) |
| **tatsächlich 0** | falsch positiv (FP) | wahr negativ (WN) |

$$\text{Sensitivität} = \frac{WP}{WP + FN} \qquad \text{Spezifität} = \frac{WN}{WN + FP}$$

Die **ROC-Kurve** trägt die Sensitivität gegen die Falsch-Positiv-Rate (1 − Spezifität) für alle möglichen Schwellwerte auf. Die Fläche darunter (**AUC**) ist ein Maß für das Modell unabhängig vom Schwellwert: 1 = perfekt, 0,5 = Raten. Der aktuelle Schwellwert ist als Punkt auf der Kurve markiert.

## Bedienung

| Aktion | Wirkung |
|---|---|
| Beispiel-Leiste (Reiter 1–4) | Schuhgröße, Sortierstation, Lagerschaden oder Servomotor |
| Verfahren | ① Kurve per Hand anpassen oder ② Gradientenabstieg |
| **→** oder „Lernschritt“ | nächster Lernschritt (nur beim Gradientenabstieg) |
| **Leertaste** oder „Automatisch“ | bis zum Ende lernen lassen / anhalten |
| **R** oder „Neu starten“ | neue Startkurve, gleiche Daten |
| Griffe ziehen (● und ■) | Wendepunkt $x_0$ und Steilheit $k$ von Hand einstellen |
| gelbe Linie ziehen | Schwellwert ändern (oder Regler unter „Einstellungen“) |
| Klick ins Diagramm | Prognose für diesen x-Wert (ziehbar), **Esc** blendet sie aus |
| Maus über einem Punkt | Messwert, p(x) und Fehler dieses Punkts |
| Maus auf ein Feld der Konfusionsmatrix | die zugehörigen Punkte leuchten auf |
| Mond / Sonne oben rechts | dunkles oder helles Farbschema; die Wahl wird im Browser gespeichert und gilt für alle Lernsituationen |

**Einstellungen:** Verfahren, Lernrate η, Tempo, Schwellwert (0,05 … 0,95), „Beste Kurve zum Vergleich“ (beim Skript-Beispiel: die Kurve aus dem Skript). Unter dem Diagramm lassen sich der Fehler je Punkt und die Fehlklassifikationen einblenden. „Neue Messdaten“ erzeugt einen neuen Datensatz mit denselben Verteilungen (nicht beim Skript-Beispiel), „CSV speichern“ lädt die Daten herunter (z. B. für Excel oder Python).

## Ideen für den Unterricht

- **Warum keine Gerade?** Durch 0/1-Daten eine Gerade legen: Was wäre bei Schuhgröße 48 die „Wahrscheinlichkeit“? Warum braucht man eine Kurve, die zwischen 0 und 1 bleibt?
- **Per Hand anpassen:** Wer kommt dem Minimum des Log-Loss am nächsten? Was passiert mit dem Log-Loss, wenn die Kurve sehr steil wird und ein Punkt auf der falschen Seite liegt?
- **Skript-Kurve vergleichen:** Kästchen einschalten. Warum findet der Gradientenabstieg eine etwas andere Kurve als im Skript? Wie wahrscheinlich ist „männlich“ bei Schuhgröße 40 – nach Skript-Kurve und nach gelernter Kurve? (≈ 60 % bzw. ≈ 50 %)
- **Lernrate:** Mit η = 0,5, 8 und 40 jeweils lernen lassen. Wann steigt der Log-Loss zwischendurch? Wie viele Lernschritte braucht es?
- **Schwellwert beim Lagerschaden:** Wo setzt man ihn, wenn ein übersehener Schaden sehr teuer ist? Was passiert dabei mit den Fehlalarmen? Beobachte Konfusionsmatrix und ROC-Punkt gleichzeitig.
- **Sensitivität und Spezifität:** Für den Servomotor: Was bedeutet eine hohe Sensitivität für die Produktion, was eine hohe Spezifität?
- **AUC vergleichen:** In welchem Beispiel trennt das Modell die Klassen am besten? Woran sieht man das schon im Diagramm?
- **Prognose:** Ein Werkstück reflektiert 50 % – in welchen Behälter kommt es? Ab welcher Temperatur soll die Steuerung die Drehzahl drosseln?
- **Neue Messdaten:** Mehrmals neu würfeln. Wie stark ändern sich $x_0$ und $k$? Was sagt das über die Menge der Trainingsdaten?
- **Weiterarbeit:** Daten als CSV speichern und in Python mit `sklearn.linear_model.LogisticRegression` nachrechnen.

## Dateien

| Datei | Inhalt |
|---|---|
| `index.html` | Seite, bindet Schriften, p5.js und die Skripte ein |
| `style.css` | Gestaltung im Corporate Design der Schule |
| `basis.js` | gemeinsame Grundlage aller KI-Klassifikations-Seiten (Layout, Farben, Kopfzeile, Beispiel-Leiste, UI-Helfer) |
| `daten.js` | alle vier Beispiele (Merkmal, Klassen, Verteilungen, Texte) und der Datengenerator |
| `logistik.js` | der Algorithmus – Sigmoid, Log-Loss, Gradientenabstieg, Konfusionsmatrix, ROC/AUC – reines JavaScript, ohne p5 |
| `ui.js` | Knöpfe, Auswahllisten, Regler |
| `sketch.js` | Ablauf, Zeichnen, Erklärtexte |
| `assets/logo-jps.png` | Logo der Jakob-Preh-Schule |
| `assets/logo-jps-hell.png` | Logo in Weiß für das dunkle Farbschema |
| `libraries/p5.min.js` | p5.js (lokal, damit die Seite auch offline läuft) |

Ein weiteres Beispiel lässt sich ergänzen, indem man in `daten.js` einen Eintrag in `SZENARIEN` hinzufügt (ein Merkmal, zwei Klassen mit Mittelwert, Streuung und Anzahl – oder feste Daten unter `fest` –, Wörter für die Texte). Es erscheint dann automatisch als weiterer Reiter in der Beispiel-Leiste.

## Lokal starten

Am einfachsten in VS Code mit der Erweiterung **Live Server**: Rechtsklick auf `index.html` → „Open with Live Server“.

## Technische Hinweise

- Läuft mit **p5.js 1.x** (lokal mitgeliefert) und **p5.js 2.x**. Dafür verzichtet der Code bewusst auf Funktionen, die sich zwischen den Versionen geändert haben (z. B. `preload`, `loadFont`).
- Die Schriften (Quicksand, Inter, JetBrains Mono) kommen von Google Fonts. Ohne Internet weicht die Seite automatisch auf Systemschriften aus.
- Alle Schleifen haben geschweifte Klammern: Der p5-Web-Editor baut in Schleifen eine Endlosschleifen-Sicherung ein, die bei Schleifen ohne Klammern fehlschlägt.
- Fehler in `draw()` werden abgefangen und als Hinweis angezeigt, statt das Programm einzufrieren.
- Jeder Datensatz wird mit festem Startwert erzeugt und ist deshalb bei jedem Start gleich. Erst „Neue Messdaten“ würfelt neu.
- `basis.js` ist in allen KI-Klassifikations-Seiten (Logistische Regression, kNN, Naiver Bayes, Entscheidungsbaum) identisch.
- **Farbschema hell/dunkel:** Alle Farben stehen in zwei Paletten (`FARBE_HELL`, `FARBE_DUNKEL`); `FARBE` enthält die aktive Palette und wird beim Umschalten in place überschrieben. Die CSS-Variablen in `style.css` wechseln über die Klasse `dunkelmodus` am `<html>`-Element. Die Wahl steht im `localStorage` unter `jps-lernsituationen-farbschema` (ohne Speicher gilt die Systemeinstellung). Im dunklen Schema zeigt die Kopfzeile `assets/logo-jps-hell.png`.

## Lizenz

[![CC BY 4.0](https://img.shields.io/badge/Lizenz-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/deed.de)

**KI-Klassifikation · Logistische Regression** von **Christoph Rothaug, Jakob-Preh-Schule Bad Neustadt**, steht unter der Lizenz [Creative Commons Namensnennung 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.de). Der vollständige Lizenztext steht in der Datei [`LICENSE`](LICENSE).

Du darfst das Material (Quellcode, Texte, Aufgaben und Grafiken) kopieren, weitergeben und verändern, auch kommerziell. Dafür musst du

- den Urheber nennen: *Christoph Rothaug, Jakob-Preh-Schule Bad Neustadt*,
- auf die Lizenz verlinken: <https://creativecommons.org/licenses/by/4.0/deed.de>,
- angeben, ob du etwas verändert hast.

Vorschlag für die Namensnennung: *„KI-Klassifikation · Logistische Regression“ von Christoph Rothaug, Jakob-Preh-Schule Bad Neustadt, CC BY 4.0, https://github.com/FlowTheTensor/LS-Logistische-Regression*

**Ausgenommen** von dieser Lizenz sind:

| Bestandteil | Rechte |
|---|---|
| `libraries/p5.min.js` | [p5.js](https://p5js.org), GNU LGPL 2.1 |
| `libraries/p5.sound.min.js` | [p5.sound](https://p5js.org/reference/p5.sound/), GNU LGPL 2.1 |
| `assets/logo-jps.png`, `assets/logo-jps-hell.png` | Logo der Jakob-Preh-Schule – alle Rechte bei der Schule |
| Schriften Quicksand, Inter, JetBrains Mono | über Google Fonts eingebunden, SIL Open Font License 1.1 |

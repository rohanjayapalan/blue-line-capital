/* Blue Line Capital - the website.
   Plain JavaScript, no build step. It reads the JSON files that GitHub Actions publishes
   (public/today.json, record.json, model.json, health.json) and the tape (tape/<season>.jsonl).
   Colour rule: red = our model, blue = the market, yellow = a lock or a steal. */
(function () {
  "use strict";

  const CFG = window.BLC_CONFIG || {};
  const TZ = "America/Toronto";
  const REDUCE = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const S = { today: null, record: null, model: null, health: null, view: null };
  let main = null;

  // ------------------------------------------------------------------ small helpers
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const gs = () => (!REDUCE && window.gsap ? window.gsap : null);
  const isLocal = () => location.protocol === "file:" || ["localhost", "127.0.0.1", ""].includes(location.hostname);
  const pct = (p, d = 0) => (p == null || isNaN(p) ? "—" : `${(100 * p).toFixed(d)}%`);
  const pts = (x, d = 1) => (x == null || isNaN(x) ? "—" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`);
  const num = (x, d = 3) => (x == null || isNaN(x) ? "—" : Number(x).toFixed(d));
  const money = (x, d = 0) => (x == null || isNaN(x) ? "—" :
    `${x < 0 ? "−" : ""}$${Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`);
  const section = (title, body, id) =>
    `<section class="section"${id ? ` id="${id}"` : ""}><h2 class="section-title">${esc(title)}</h2>${body}</section>`;
  const rinkX = (p) => 5.5 + 89 * p;         // goal line to goal line: 5.5% .. 94.5% of the strip

  function clockET(iso) {
    if (!iso) return "";
    const s = new Intl.DateTimeFormat("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit" }).format(new Date(iso));
    return s.replace(/\s?AM/, " am").replace(/\s?PM/, " pm");
  }
  const dayLong = (ymd) => new Intl.DateTimeFormat("en-US",
    { timeZone: "UTC", weekday: "long", month: "long", day: "numeric" }).format(new Date(`${ymd}T12:00:00Z`));
  const dayShort = (ymd) => new Intl.DateTimeFormat("en-US",
    { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" }).format(new Date(`${ymd}T12:00:00Z`));
  function hms(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
    if (h >= 24) return `${Math.floor(h / 24)}d ${h % 24}h`;
    return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${String(x).padStart(2, "0")}`;
  }
  const tierClass = (t) => String(t || "").toLowerCase().replace(/[^a-z]+/g, "-");
  const lockIcon = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 7V5a4 4 0 1 1 8 0v2h1v8H3V7h1Zm2 0h4V5a2 2 0 1 0-4 0v2Z"/></svg>';

  // ------------------------------------------------------------------ data
  function dataBase() {
    if (CFG.dataBase) return CFG.dataBase;
    if (isLocal()) return "../";
    return `https://raw.githubusercontent.com/${CFG.owner}/${CFG.repo}/${CFG.branch || "main"}/`;
  }
  async function fetchData(path, asText) {
    const v = Math.floor(Date.now() / 60000);          // new URL every minute: skips stale caches
    const res = await fetch(`${dataBase()}${path}?v=${v}`, { cache: "no-store" });
    if (!res.ok) {
      const e = new Error(`${path} returned ${res.status}`);
      e.status = res.status;
      throw e;
    }
    return asText ? res.text() : res.json();
  }

  // ------------------------------------------------------------------ TONIGHT
  const goalieWord = { Confirmed: "confirmed", Likely: "likely", Unconfirmed: "unconfirmed", Projected: "projected from usage" };
  const score = (g) => `${g.away.abbrev} ${g.away.score == null ? 0 : g.away.score}, ${g.home.abbrev} ${g.home.score == null ? 0 : g.home.score}`;
  const lockAt = (g) => new Date(new Date(g.start_utc).getTime() - (S.today.lock_minutes || 10) * 60000).toISOString();

  function statusHTML(g) {
    const p = g.prediction;
    if (g.status === "final") return `<span class="status">Final: ${esc(score(g))}${g.period && g.period !== "REG" ? ` (${esc(g.period)})` : ""}</span>`;
    if (g.status === "live") return `<span class="status live">Live: ${esc(score(g))}</span>`;
    if (g.status === "postponed") return '<span class="status">Postponed</span>';
    if (p && p.locked) return `<span class="status">${lockIcon}Locked at ${esc(clockET(p.locked.at))}</span>`;
    if (p) return `<span class="status">Provisional, locks at ${esc(clockET(lockAt(g)))}</span>`;
    return '<span class="status">Waiting for the next model run</span>';
  }

  function rinkHTML(g, p) {
    const mk = p && p.market ? p.market.p_home : null;
    const label = p ? `Model gives ${g.home.abbrev} ${pct(p.p_home)}${mk != null ? `, market ${pct(mk)}` : ""}` : `${g.away.abbrev} at ${g.home.abbrev}`;
    return `<div class="rink" role="img" aria-label="${esc(label)}">
      <span class="line goal" style="left:5.5%"></span><span class="line blue" style="left:37.5%"></span>
      <span class="line red" style="left:50%"></span><span class="line blue" style="left:62.5%"></span>
      <span class="line goal" style="left:94.5%"></span>
      <span class="end away">${esc(g.away.abbrev)}</span><span class="end home">${esc(g.home.abbrev)}</span>
      ${mk != null ? `<span class="puck ghost" style="left:${rinkX(mk).toFixed(2)}%"></span>` : ""}
      ${p ? `<span class="puck model" data-x="${rinkX(p.p_home).toFixed(2)}" style="left:${rinkX(p.p_home).toFixed(2)}%"></span>` : ""}
    </div>`;
  }

  function factsHTML(g, p) {
    const home = p.pick === g.home.abbrev;
    const rows = [];
    const mk = p.market;
    if (mk) {
      const mp = home ? mk.p_home : 1 - mk.p_home;
      rows.push(["Market", `<span class="mkt">${pct(mp)}</span> for ${esc(p.pick)}, best price ${esc(home ? mk.home_price : mk.away_price)} at ${esc(home ? mk.home_book : mk.away_book)}`]);
      const e = p.edge_pp;
      rows.push(["Model vs market", `<span class="mdl">${pts(e)} pts</span> ${Math.abs(e) < 2 ? "(in line with the market)" : e > 0 ? "(the model likes this side more)" : "(the market likes this side more)"}`]);
    } else {
      rows.push(["Market", "No sportsbook line posted yet"]);
    }
    const gh = p.goalies.home, ga = p.goalies.away;
    rows.push(["Goalies", `${esc(g.home.abbrev)}: ${esc(gh.name)}, ${esc(goalieWord[gh.status] || gh.status)}<br>${esc(g.away.abbrev)}: ${esc(ga.name)}, ${esc(goalieWord[ga.status] || ga.status)}`]);
    if (p.projected) {
      rows.push(["Projected", `${esc(g.away.abbrev)} ${p.projected.away.toFixed(1)}, ${esc(g.home.abbrev)} ${p.projected.home.toFixed(1)}; overtime ${pct(p.projected.p_ot)}`]);
    }
    if (p.bet) {
      const why = p.bet.steal ? "a steal: the risk desk cleared a bigger stake"
        : p.bet.ev <= 0 ? "minimum stake, no edge after blending with the market"
          : `expected value ${pts(100 * p.bet.ev)}%`;
      rows.push(["Always-in book", `${money(p.bet.stake)} on ${esc(p.bet.team)} at ${esc(p.bet.price)} (${why})`]);
      rows.push(["Edge-only book", p.bet.edge_stake ? `${money(p.bet.edge_stake)} on ${esc(p.bet.team)}` : "No bet"]);
    }
    return `<dl class="facts">${rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
  }

  function whyHTML(g, p, id) {
    const home = p.pick === g.home.abbrev;
    const sgn = home ? 1 : -1;
    const base = home ? p.p_base : 1 - p.p_base;
    const items = (p.why || []).map((w) => ({ label: w.label, detail: w.detail, v: w.pp_home * sgn }));
    const other = 100 * (p.p_pick - base) - items.reduce((a, w) => a + w.v, 0);
    if (Math.abs(other) >= 0.05) items.push({ label: "Everything else", detail: "All smaller inputs combined", v: other });
    const max = Math.max(2, ...items.map((w) => Math.abs(w.v)));
    const rows = items.map((w) => `<div class="wf-row">
        <div class="lbl"><strong>${esc(w.label)}</strong><span>${esc(w.detail)}</span></div>
        <div class="wf-bar" aria-hidden="true"><i class="${w.v >= 0 ? "for" : "against"}" style="width:${(50 * Math.abs(w.v) / max).toFixed(1)}%"></i></div>
        <div class="wf-val">${pts(w.v)}</div></div>`).join("");
    let desk = "";
    if (p.desk && p.desk.triggered) {
      desk = `<div class="desk"><strong>Risk desk</strong>
        <p class="sub">The model and the market disagree by 7+ points, so five checks ran. ${p.desk.passed ? "All passed: this is sized as a steal." : "At least one failed: the bet stays small."}</p>
        <ul>${p.desk.checks.map((c) => `<li class="${c.ok ? "ok" : "fail"}"><b>${c.ok ? "Pass" : "Fail"}</b> ${esc(c.name)}: ${esc(c.detail)}</li>`).join("")}</ul></div>`;
    }
    const gh = p.goalies.home, ga = p.goalies.away;
    const miss = ["home", "away"].filter((s) => p.lineups[s].missing.length)
      .map((s) => `${esc(g[s].abbrev)} without ${esc(p.lineups[s].missing.join(", "))}`).join("; ");
    return `<div class="why" id="${id}" hidden>
      <h4>Why ${esc(p.pick)}</h4>
      <p class="sub">Percentage points each input adds to ${esc(p.pick)}'s chance, starting from ${pct(base, 1)}, the league-wide ${home ? "home" : "road"} baseline.</p>
      ${rows}
      <div class="wf-end"><span>Final call</span><span>${pct(p.p_pick, 1)}</span></div>
      ${desk}
      <div class="tech">
        <p>Goalie talent in goals saved above expected per 60: ${esc(gh.name)} ${pts(gh.talent, 2)}, ${esc(ga.name)} ${pts(ga.talent, 2)}. Streak t-stats ${pts(gh.streak_t, 1)} and ${pts(ga.streak_t, 1)}.</p>
        <p>Lineups from ${esc(p.lineups.home.source)} and ${esc(p.lineups.away.source)}${miss ? `; ${miss}` : "; no core players missing"}.</p>
        <p>Computed at ${esc(clockET(p.computed_at))} ET${p.locked ? `; locked at ${esc(clockET(p.locked.at))} ET with hash <code>${esc(p.locked.hash)}</code>` : "; not locked yet"}.</p>
      </div>
    </div>`;
  }

  function gameHTML(g, i) {
    const p = g.prediction;
    const cls = ["game"];
    if (p && p.bet && p.bet.steal) cls.push("is-steal");
    if (g.status === "final") cls.push("is-final", p && p.hit === 0 ? "is-miss" : "is-hit");
    const meta = `<header class="game-meta"><span class="when">${esc(clockET(g.start_utc))} ET</span>
      <span>${esc(g.away.name)} at ${esc(g.home.name)}</span>${statusHTML(g)}</header>`;
    if (!p) {
      const why = ["live", "final"].includes(g.status)
        ? "No call: this game started before a pick could be locked."
        : "The pick appears with the next model run.";
      return `<article class="${cls.join(" ")}">${meta}<p class="call-none">${esc(why)}</p>${rinkHTML(g, null)}</article>`;
    }
    const home = p.pick === g.home.abbrev;
    const pickName = home ? g.home.name : g.away.name, oppName = home ? g.away.name : g.home.name;
    let result = "";
    if (g.status === "final" && p.hit != null) {
      result = p.hit ? '<span class="result hit"><i class="lamp" aria-hidden="true"></i>Called it</span>'
        : '<span class="result miss">Missed</span>';
    }
    const id = `why-${g.id}`;
    return `<article class="${cls.join(" ")}" data-id="${g.id}">${meta}
      <div class="call">
        <span class="call-team">${esc(p.pick)}</span>
        <span class="call-pct" data-p="${(100 * p.p_pick).toFixed(0)}">${(100 * p.p_pick).toFixed(0)}<small>%</small></span>
        <p class="call-line">${esc(pickName)} to beat ${esc(oppName)} ${home ? "at home" : "on the road"}.
          <span class="tier ${tierClass(p.tier)}">${esc(p.tier)}</span> ${result}</p>
      </div>
      ${rinkHTML(g, p)}
      ${factsHTML(g, p)}
      <button class="why-toggle" type="button" aria-expanded="false" aria-controls="${id}" data-closed="Why ${esc(p.pick)}">Why ${esc(p.pick)}</button>
      ${whyHTML(g, p, id)}
    </article>`;
  }

  function renderTonight() {
    const t = S.today;
    const games = t.games || [];
    let lede, extra = "";
    if (!games.length) {
      lede = t.date < t.season_start
        ? `The ${esc(t.season)} season opens ${esc(dayLong(t.season_start))}. Picks post that morning and lock ${t.lock_minutes} minutes before each puck drop.`
        : "No regular-season games on the schedule today.";
      extra = `<p class="sub">While you wait: the <a href="#record">record</a> page shows how the model did on a three-season exam, and the <a href="#model">model</a> page shows how it works.</p>`;
    } else {
      lede = `${games.length} game${games.length > 1 ? "s" : ""}. Each pick stays provisional until ${t.lock_minutes} minutes before puck drop, then it's locked into the tape for good.`;
      extra = `<p class="sub" id="next-lock"></p>
        <p class="legend-inline"><span><i class="m"></i>Model's chance for the home team</span><span><i class="k"></i>Market's chance, vig removed</span></p>`;
    }
    return `<div class="view-head"><h1 class="view-title">${esc(dayLong(t.date))}</h1><p class="lede">${lede}</p>${extra}</div>
      <div class="slate">${games.map(gameHTML).join("")}</div>`;
  }

  function nextLock() {
    const now = Date.now();
    let best = null;
    for (const g of (S.today && S.today.games) || []) {
      if (!g.prediction || g.prediction.locked || g.status !== "upcoming") continue;
      const t = new Date(lockAt(g)).getTime();
      if (t > now && (!best || t < best.t)) best = { t, g };
    }
    return best;
  }

  function tick() {
    const el = document.getElementById("next-lock");
    if (!el) return;
    const n = nextLock();
    const games = (S.today && S.today.games) || [];
    if (n) {
      el.innerHTML = `Next pick locks in <span class="countdown">${hms(n.t - Date.now())}</span>: ${esc(n.g.away.name)} at ${esc(n.g.home.name)}.`;
    } else if (games.some((g) => g.prediction && g.prediction.locked)) {
      el.textContent = "Every pick on today's board is locked.";
    } else {
      el.textContent = "";
    }
  }

  // ------------------------------------------------------------------ RECORD
  function calibrationSVG(bins) {
    if (!bins || !bins.length) return '<p class="sub">Not enough games yet for a calibration chart.</p>';
    const W = 360, H = 300, L = 46, R = 14, T = 12, B = 40;
    const X = (p) => L + (W - L - R) * p, Y = (p) => T + (H - T - B) * (1 - p);
    const maxN = Math.max(...bins.map((b) => b.n));
    const ticks = [0, 0.25, 0.5, 0.75, 1];
    const grid = ticks.map((v) => `<line class="grid" x1="${X(0)}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}"/>
      <text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v * 100}%</text>
      <text x="${X(v)}" y="${H - B + 18}" text-anchor="middle">${v * 100}%</text>`).join("");
    const dots = bins.map((b) => `<circle cx="${X(b.p).toFixed(1)}" cy="${Y(b.won).toFixed(1)}" r="${(3 + 9 * Math.sqrt(b.n / maxN)).toFixed(1)}" fill="#D62839" fill-opacity=".85"><title>Predicted ${pct(b.p)}, won ${pct(b.won)} of ${b.n} games</title></circle>`).join("");
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration: predicted chance against how often those teams won">
      ${grid}<line class="axis" x1="${X(0)}" x2="${X(1)}" y1="${Y(0)}" y2="${Y(0)}"/>
      <line class="diag" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"/>${dots}
      <text x="${(X(0) + X(1)) / 2}" y="${H - 4}" text-anchor="middle">Model's chance</text>
      <text x="12" y="${(Y(0) + Y(1)) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(Y(0) + Y(1)) / 2})">How often they won</text></svg>`;
  }

  function backtestHTML(bt) {
    const folds = bt.folds || [];
    const hasMkt = folds.some((f) => f.market);
    const row = (label, m, base, mk, lead) => `<tr${lead ? ' class="lead"' : ""}><td>${esc(label)}</td>
      <td class="n">${m.games.toLocaleString("en-US")}</td><td class="n">${pct(m.accuracy, 1)}</td><td class="n">${num(m.log_loss)}</td>
      <td class="n">${base ? num(base.log_loss) : "—"}</td>${hasMkt ? `<td class="n">${mk ? num(mk.market.log_loss) : "—"}</td>` : ""}</tr>`;
    const rows = folds.map((f) => row(f.label, f.model, f.baseline_home, f.market)).join("")
      + row("All three seasons", bt.overall, null, bt.market ? { market: bt.market.market } : null, true);
    const s = folds.length ? folds[folds.length - 1].settings : null;
    return `<p class="sub">The model replayed three full seasons one day at a time. Each season was predicted by a model trained only on seasons at least two years older, then graded game by game, so it never saw the answers.</p>
      <div class="table-wrap"><table><thead><tr><th>Season</th><th class="n">Games</th><th class="n">Picks right</th><th class="n">Log loss</th><th class="n">Always-home log loss</th>${hasMkt ? '<th class="n">Market log loss</th>' : ""}</tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">Log loss grades the probabilities, not just the picks: lower is better, and 0.693 is a coin flip. NHL closing betting lines usually land around 0.67.${s ? ` The tuning picked the same settings every season: form half-life ${s.half_life} games, last season weighted ${pct(s.carryover)}, goalie streak weight ${s.goalie_beta}.` : ""}</p>
      ${calibrationSVG(bt.calibration)}`;
  }

  function logHTML(rows) {
    let out = "", day = null;
    for (const r of rows || []) {
      if (r.date !== day) { day = r.date; out += `<h3 class="log-day">${esc(dayShort(r.date))}</h3>`; }
      if (r.void) { out += `<p class="row-void">${esc(r.away)} at ${esc(r.home)}: postponed, not graded</p>`; continue; }
      const home = r.pick === r.home;
      const opp = home ? r.away : r.home;
      const [as, hs] = String(r.score).split("-");
      const sc = home ? `${hs}–${as}` : `${as}–${hs}`;
      const ot = r.ot && r.ot !== "REG" ? ` in ${r.ot === "SO" ? "a shootout" : "overtime"}` : "";
      const mk = r.market_p_pick != null ? `, market had ${pct(r.market_p_pick)}` : "";
      if (r.hit) {
        out += `<div class="row-hit"><i class="lamp" aria-hidden="true"></i>
          <div class="desc"><b class="pick">${esc(r.pick)}</b> beat ${esc(opp)} ${sc}${ot}<span>Called at ${pct(r.p_pick)}${mk}${r.steal ? ", flagged as a steal" : ""}</span></div>
          <div class="wf-val">${r.profit != null ? money(r.profit) : ""}</div></div>`;
      } else {
        out += `<p class="row-miss"><span>Missed: ${esc(r.pick)} lost to ${esc(opp)} ${sc}${ot}</span><span>called at ${pct(r.p_pick)}${mk}</span></p>`;
      }
    }
    return `<div class="log">${out}</div>`;
  }

  function renderRecord() {
    const r = S.record || {};
    const s = r.season || { games: 0 };
    const head = `<div class="view-head"><h1 class="view-title">Record</h1>
      <p class="lede">Every locked pick, graded the morning after. Hits get the goal light. Misses stay on the board, just smaller.</p></div>`;
    if (!s.games) {
      return head + '<p class="sub">No graded games yet. The first picks lock on opening night and show up here the next morning.</p>'
        + (r.backtest && r.backtest.overall ? section("The exam before going live", backtestHTML(r.backtest)) : "");
    }
    const mk = s.market;
    const tiers = Object.entries(r.by_tier || {}).map(([t, v]) =>
      `<tr><td><span class="tier ${tierClass(t)}">${esc(t)}</span></td><td class="n">${v.games}</td><td class="n">${pct(v.hits / v.games, 1)}</td></tr>`).join("");
    return head + `<div class="big-record"><span class="wl">${s.hits}–${s.games - s.hits}</span><span class="acc">${pct(s.accuracy, 1)}</span></div>
      <div class="stats">
        <div class="stat"><b class="mdl">${num(s.log_loss)}</b><span>Model log loss (lower is better)</span></div>
        ${mk ? `<div class="stat"><b class="mkt">${num(mk.log_loss)}</b><span>Market log loss on the same ${mk.games} games</span></div>` : ""}
        <div class="stat"><b>${num(s.brier)}</b><span>Brier score</span></div>
        ${mk && mk.followed_us != null ? `<div class="stat"><b>${pct(mk.followed_us)}</b><span>of line moves went toward our number</span></div>` : ""}
      </div>
      <div class="section two">
        <div><h2 class="section-title">Calibration</h2><p class="sub">When the model says 60%, that team should win about 60% of the time. Dots on the dashed line mean the probabilities are honest.</p>${calibrationSVG(r.calibration)}</div>
        <div><h2 class="section-title">By confidence</h2><div class="table-wrap"><table><thead><tr><th>Tier</th><th class="n">Games</th><th class="n">Right</th></tr></thead><tbody>${tiers}</tbody></table></div></div>
      </div>
      ${section("Game log", logHTML(r.graded))}
      ${r.backtest && r.backtest.overall ? section("The exam before going live", backtestHTML(r.backtest)) : ""}`;
  }

  // ------------------------------------------------------------------ SCOREBOARD
  const BOOKS = [
    ["always", "#D62839", "", "Always-in: bets every game, Kelly-sized"],
    ["edge", "#0D1F33", "", "Edge-only: bets only with an edge"],
    ["fav", "#1B4F9C", "", "Benchmark: always the favourite"],
    ["home", "#8A9BAA", "", "Benchmark: always the home team"],
    ["pick", "#D62839", "5 5", "Benchmark: model pick, flat $100"],
  ];

  function equitySVG(books) {
    const keys = BOOKS.filter(([k]) => books[k] && books[k].curve && books[k].curve.length);
    if (!keys.length) return "";
    const W = 720, H = 300, L = 70, R = 14, T = 14, B = 34;
    const start = 10000;
    const vals = keys.flatMap(([k]) => books[k].curve.map((c) => c.equity)).concat([start]);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const padv = (hi - lo) * 0.08 || 500;
    lo -= padv; hi += padv;
    const dates = Array.from(new Set(keys.flatMap(([k]) => books[k].curve.map((c) => c.date)))).sort();
    const xi = new Map(dates.map((d, i) => [d, i]));
    const X = (d) => L + (W - L - R) * (dates.length > 1 ? xi.get(d) / (dates.length - 1) : 0.5);
    const Y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    const ticks = [0, 1, 2, 3, 4].map((i) => lo + (hi - lo) * i / 4);
    const grid = ticks.map((v) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/><text x="${L - 8}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${money(v)}</text>`).join("");
    const lines = keys.map(([k, color, dash]) => {
      const d = books[k].curve.map((c, i) => `${i ? "L" : "M"}${X(c.date).toFixed(1)},${Y(c.equity).toFixed(1)}`).join("");
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${k === "always" ? 2.6 : 1.6}"${dash ? ` stroke-dasharray="${dash}"` : ""} stroke-linejoin="round"/>`;
    }).join("");
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Bankroll over time for each betting book">
      ${grid}<line class="diag" x1="${L}" x2="${W - R}" y1="${Y(start).toFixed(1)}" y2="${Y(start).toFixed(1)}"/>${lines}
      <text x="${L}" y="${H - 8}">${esc(dayShort(dates[0]))}</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${esc(dayShort(dates[dates.length - 1]))}</text></svg>
      <p class="legend">${keys.map(([k, color, dash, label]) => `<span><i style="background:${dash ? `repeating-linear-gradient(90deg, ${color} 0 5px, transparent 5px 10px)` : color}"></i>${esc(label)}</span>`).join("")}</p>`;
  }

  function booksTable(books) {
    const rows = BOOKS.filter(([k]) => books[k]).map(([k, , , label]) => {
      const b = books[k];
      return `<tr${k === "always" ? ' class="lead"' : ""}><td>${esc(label.split(":")[0])}</td><td class="n">${money(b.bankroll)}</td><td class="n">${money(b.profit)}</td>
        <td class="n">${pct(b.roi, 1)}</td><td class="n">${pct(b.max_drawdown, 1)}</td><td class="n">${num(b.t_stat, 2)}</td><td class="n">${b.bets}</td></tr>`;
    }).join("");
    return `<div class="table-wrap"><table><thead><tr><th>Book</th><th class="n">Bankroll</th><th class="n">Profit</th><th class="n">Return on stakes</th><th class="n">Worst drawdown</th><th class="n">t-stat</th><th class="n">Bets</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">Every book starts at $10,000. The t-stat asks whether the average return per bet is distinguishable from zero; above 2 is the usual bar. Flat benchmarks bet $100 a game.</p>`;
  }

  function versusHTML(m, k, label) {
    const row = (name, a, b) => `<div><dt>${esc(name)}</dt><dd>${a}</dd></div>`;
    return `<div class="versus">
      <div><h3>Model</h3><dl>${row("Log loss", num(m.log_loss))}${row("Picks right", pct(m.accuracy, 1))}${m.brier != null ? row("Brier score", num(m.brier)) : ""}</dl></div>
      <div><h3>Market</h3><dl>${row("Log loss", num(k.log_loss))}${row("Picks right", pct(k.accuracy, 1))}${k.brier != null ? row("Brier score", num(k.brier)) : ""}</dl></div>
    </div><p class="note">${esc(label)}</p>`;
  }

  function renderScoreboard() {
    const r = S.record || {};
    const s = r.season || {};
    const bt = r.backtest || {};
    let html = `<div class="view-head"><h1 class="view-title">Scoreboard</h1>
      <p class="lede">The model against the betting market, on the same games, plus five simulated betting books.</p>
      <p class="sub">Log loss rewards being confident and right and punishes being confident and wrong. The closing line is the benchmark to beat.</p></div>`;
    if (s.market) {
      html += section("This season", versusHTML({ log_loss: s.market.model_log_loss, accuracy: s.market.model_accuracy },
        { log_loss: s.market.log_loss, accuracy: s.market.accuracy }, `Live, ${s.market.games} graded games with a posted line.`));
    } else if (bt.market) {
      html += section("Backtest, 2023-24 to 2025-26", versusHTML(bt.market.model, bt.market.market,
        `Closing lines on ${bt.market.games.toLocaleString("en-US")} games from ESPN's odds history. The live comparison starts with the first graded games.`));
    } else {
      html += section("Model vs market", '<p class="sub">This fills in once live picks are graded. The backtest comparison appears after the Train workflow loads the historical odds.</p>');
    }
    const liveBooks = r.books && Object.values(r.books).some((b) => b.bets > 0) ? r.books : null;
    const books = liveBooks || (bt.market && bt.market.books);
    if (books) {
      html += section(liveBooks ? "Bankroll" : "Bankroll, backtest", equitySVG(books) + booksTable(books));
    } else {
      html += section("Bankroll", '<p class="sub">Five books start at $10,000 on opening night: always-in, edge-only, and three benchmarks (favourite, home team, model pick at a flat $100).</p>');
    }
    if (bt.overall) html += section("The exam", backtestHTML(bt));
    return html;
  }

  // ------------------------------------------------------------------ MODEL
  function weightsHTML(ws) {
    const max = Math.max(0.05, ...ws.map((w) => Math.max(Math.abs(w.offline), Math.abs(w.weight))));
    const x = (v) => 50 + 50 * v / max;
    return `<div class="weights">${ws.map((w) => {
      const a = Math.min(x(0), x(w.offline)), width = Math.abs(x(w.offline) - x(0));
      const moved = Math.abs(w.weight - w.offline) > 1e-4;
      return `<div class="w-row"><span>${esc(w.label)}</span>
        <span class="w-bar" aria-hidden="true"><i class="${w.offline < 0 ? "neg" : ""}" style="left:${a.toFixed(1)}%;width:${width.toFixed(1)}%"></i>${moved ? `<b style="left:calc(${x(w.weight).toFixed(1)}% - 1.5px)"></b>` : ""}</span>
        <span class="w-val">${pts(w.weight, 3)}</span></div>`;
    }).join("")}</div>`;
  }

  function powerHTML(rows) {
    const M = Math.max(0.3, ...rows.map((r) => Math.abs(r.rating) + r.sd));
    const x = (v) => 50 + 50 * v / M;
    return `<div class="power">${rows.map((r, i) => `<div class="p-row"><span class="rk">${i + 1}</span><span class="tm">${esc(r.team)}</span>
      <span class="p-track" title="${esc(r.name)}: ${pts(r.rating, 2)} goals per game, give or take ${r.sd.toFixed(2)}">
        <span style="left:${x(r.rating - r.sd).toFixed(1)}%;width:${(x(r.rating + r.sd) - x(r.rating - r.sd)).toFixed(1)}%"></span>
        <i style="left:${x(r.rating).toFixed(1)}%"></i></span><span class="w-val">${pts(r.rating, 2)}</span></div>`).join("")}</div>`;
  }

  function renderModel() {
    const m = S.model || {};
    const st = m.settings || {};
    const kal = st.kalman ? Number(String(st.kalman).replace("k", "")) : null;
    const steps = [
      ["Collect", "Every night, each finished game's play-by-play and box score come from the NHL. Every 10 minutes on game days, the schedule with sportsbook lines plus Daily Faceoff's starting goalies and line combinations. History goes back to 2017-18.", "NHL API, a GitHub mirror of NHL game files, Daily Faceoff, NHL EDGE"],
      ["Measure every shot", `An expected-goals model scores every shot's chance of going in from distance, angle, shot type, rebounds, rushes, strength and score${m.xg && m.xg.n_shots ? `, trained on ${m.xg.n_shots.toLocaleString("en-US")} shots` : ""}. Shot counts are adjusted for score and home ice; hits, takeaways and giveaways for each arena's scorekeepers.`, "Logistic regression"],
      ["Rate teams and players", "A Kalman filter keeps a power rating and its uncertainty for every team. Goalies get a talent score in goals saved above expected, plus a streak that only counts if a t-test says it is real. Skaters get Game Score ratings, so tonight's lineup can be priced against the usual one.", "Kalman filter, GSAx, Game Score"],
      ["Predict", "Twenty-five inputs, each home team minus away team, go into a logistic regression and an XGBoost model. Their blend is calibrated separately for early, middle and late season, so the first weeks are deliberately less confident.", "Logistic regression, gradient-boosted trees, Platt scaling"],
      ["Check against the market", "Sportsbook prices become fair probabilities with the vig removed. A disagreement of 7+ points triggers a five-check risk desk. Bets are anchored to the market and sized with fractional Kelly.", "Shin de-vig, Kelly criterion"],
      ["Lock", "Ten minutes before puck drop the pick is written to the tape. Each entry carries the SHA-256 hash of the entry before it, so editing any old pick breaks the chain.", "Hash chain, verified in your browser below"],
      ["Grade and learn", "Every morning results are graded, bets settled, and the logistic weights take one small step toward whatever they got wrong, anchored to their trained values so one strange night can't wreck them.", "Online gradient descent"],
    ];
    const settings = [
      [`${st.half_life} games`, "Form half-life: after the last 20 games, older games lose half their weight every this many games"],
      [pct(st.carryover), "Weight on last season's games; anything older is never used as a stat"],
      [kal != null ? `${kal}% goals` : "—", `The power rating learns from ${kal}% actual goals and ${100 - kal}% expected goals`],
      [String(st.goalie_beta), st.goalie_beta === 0 ? "Goalie streak weight. Tested from 0 to 1, streaks added nothing beyond talent, so the data set it to zero" : "Goalie streak weight, for streaks that pass the t-test"],
      [m.platt ? `${m.platt.early[0].toFixed(2)}×` : "—", "Early-season confidence: in the first 10 games, calls are squeezed toward 50% by this factor"],
      [m.blend != null ? `${pct(m.blend)} / ${pct(1 - m.blend)}` : "—", "Logistic regression / XGBoost blend"],
    ];
    const cv = m.cv || {};
    const learn = (m.learning_log || []).slice(0, 12).map((e) => {
      const mv = e.moves && e.moves[0];
      const miss = e.biggest_miss;
      return `<li><span class="d">${esc(dayShort(e.date))}: ${e.hits} of ${e.games} right.</span>
        ${mv ? ` Biggest weight move: ${esc(mv.label)}, ${pts(mv.before, 3)} to ${pts(mv.after, 3)}.` : ""}
        ${miss ? ` Biggest miss: ${esc(miss.game)}, picked ${esc(miss.pick)} at ${pct(miss.p_pick)}, final ${esc(miss.score)}.` : ""}</li>`;
    }).join("");
    const gl = (m.goalies || []).slice(0, 16).map((g) => `<tr><td>${esc(g.name)}</td><td>${esc(g.team)}</td><td class="n">${pts(g.talent, 2)}</td><td class="n">${pts(g.form, 2)}</td><td class="n">${pts(g.t, 1)}</td><td class="n">${pct(g.gate)}</td></tr>`).join("");
    const sources = Object.entries((S.health && S.health.sources) || {}).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td>${v.status === "ok" ? "Working" : `Problem: ${esc(v.error || "")}`}</td><td>${esc(v.last_ok ? `${dayShort(v.last_ok.slice(0, 10))}, ${clockET(v.last_ok)} ET` : "—")}</td></tr>`).join("");
    const xg = m.xg && m.xg.report ? m.xg.report : null;
    return `<div class="view-head"><h1 class="view-title">How it works</h1>
        <p class="lede">Seven steps run every day on GitHub's servers. No number on this site is typed in by hand.</p></div>
      <ol class="steps">${steps.map(([h, p, src]) => `<li><h3>${esc(h)}</h3><p>${esc(p)}</p><p class="src">${esc(src)}</p></li>`).join("")}</ol>
      ${section("Settings the data chose", `<div class="settings">${settings.map(([v, t]) => `<div><b>${esc(v)}</b><span>${esc(t)}</span></div>`).join("")}</div>
        <p class="note">Chosen by leave-one-season-out cross-validation on ${esc((m.trained_on || []).join(", "))} (${(m.n_train_games || 0).toLocaleString("en-US")} games): log loss ${num(cv.calibrated)}, ${pct(cv.accuracy, 1)} of picks right.</p>`)}
      ${m.weights ? section("What the model weighs", `<p class="sub">Logistic-regression weights, in log-odds per one standard deviation of each input. Bars to the right help the home team. A red tick shows where this season's learning has moved a weight.</p>${weightsHTML(m.weights)}`) : ""}
      ${m.power ? section("Power ratings", `<p class="sub">Kalman-filter strength in goals per game against an average team. The shaded band is one standard deviation of uncertainty; it widens over the summer and narrows as games are played.${m.snapshot_through ? ` Through ${esc(dayShort(m.snapshot_through))}.` : ""}</p>${powerHTML(m.power)}`) : ""}
      ${gl ? section("Goalies", `<p class="sub">Talent is goals saved above expected per 60 minutes. Form is the last 8 starts against that talent. A streak only counts in proportion to its weight, which is zero unless the t-stat clears 1.</p>
        <div class="table-wrap"><table><thead><tr><th>Goalie</th><th>Team</th><th class="n">Talent</th><th class="n">Recent form</th><th class="n">Streak t</th><th class="n">Streak weight</th></tr></thead><tbody>${gl}</tbody></table></div>`) : ""}
      ${xg ? section("Expected goals", `<p class="sub">Graded on ${esc(String(xg.holdout_season).slice(0, 4))}-${esc(String(xg.holdout_season).slice(6))}, a season it never trained on: AUC ${num(xg.auc)} (0.5 is a coin flip), log loss ${num(xg.log_loss, 4)}, ${Number(xg.goals).toLocaleString("en-US")} goals against ${Number(xg.xg).toLocaleString("en-US")} expected.</p>`) : ""}
      ${section("What it learned", learn ? `<ul class="learn">${learn}</ul>` : '<p class="sub">The learning log starts the morning after opening night.</p>')}
      ${section("The tape", `<p class="sub">Every locked pick is a line in a public file, chained by SHA-256 hashes. This button downloads the file and re-checks every hash in your browser.</p>
        <button class="verify" type="button" id="verify">Verify the tape</button><p class="verify-out" id="verify-out" aria-live="polite"></p>`, "tape")}
      ${sources ? section("Data sources", `<div class="table-wrap"><table><thead><tr><th>Source</th><th>Status</th><th>Last good</th></tr></thead><tbody>${sources}</tbody></table></div>
        <p class="note">If a source changes its format, the pipeline opens an issue on GitHub and the site keeps the last good data.</p>`) : ""}`;
  }

  // ------------------------------------------------------------------ the tape
  async function sha256(s) {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
    return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
  }
  async function verifyTape() {
    let text = "";
    try {
      text = await fetchData(`tape/${S.today.season}.jsonl`, true);
    } catch (e) {
      if (e.status === 404) return { ok: true, n: 0 };
      throw e;
    }
    const lines = text.split("\n").filter((l) => l.trim());
    let prev = "0".repeat(64);
    for (let i = 0; i < lines.length; i++) {
      const e = JSON.parse(lines[i]);
      if (e.prev !== prev || (await sha256(prev + e.body)) !== e.hash) return { ok: false, n: lines.length, at: i + 1 };
      prev = e.hash;
    }
    return { ok: true, n: lines.length, head: prev };
  }
  async function checkTape() {
    const chip = document.getElementById("tape-chip");
    if (!chip) return;
    const span = chip.querySelector("span");
    try {
      const r = await verifyTape();
      if (!r.ok) { chip.classList.add("bad"); span.textContent = `Tape broken at entry ${r.at}`; }
      else span.textContent = r.n ? `Tape: ${r.n} locked pick${r.n > 1 ? "s" : ""}, chain intact` : "Tape: first lock on opening night";
    } catch (e) {
      span.textContent = "Tape unavailable right now";
    }
  }

  // ------------------------------------------------------------------ motion
  function intro(first) {
    const g = gs();
    if (!g) return;
    const tl = g.timeline();
    const path = document.querySelector("#trail path");
    if (first && path && typeof path.getTotalLength === "function") {
      try {
        const len = path.getTotalLength();
        g.set(path, { strokeDasharray: len, strokeDashoffset: len });
        tl.to(path, { strokeDashoffset: 0, duration: 1.2, ease: "power2.inOut" }, 0);
      } catch (e) { /* older browsers: skip the trail */ }
    }
    tl.from(".view-title", { y: 24, opacity: 0, duration: 0.6, ease: "power3.out" }, 0.2);
    const pucks = $$(".puck.model");
    if (!pucks.length) return;
    tl.fromTo(pucks, { left: "50%" }, { left: (i, el) => `${el.dataset.x}%`, duration: 1.0, ease: "power3.out", stagger: 0.08 }, 0.5);
    tl.from(".puck.ghost", { opacity: 0, scale: 0.3, duration: 0.5, ease: "back.out(2)", stagger: 0.08 }, 0.95);
    $$(".call-pct[data-p]").forEach((el, i) => {
      const o = { v: 50 };
      const target = Number(el.dataset.p);
      tl.to(o, { v: target, duration: 1.0, ease: "power3.out", onUpdate: () => { el.firstChild.nodeValue = String(Math.round(o.v)); } }, 0.5 + i * 0.08);
    });
  }

  function wipe(swap) {
    const g = gs();
    const z = document.getElementById("zamboni");
    if (!g || !z) { swap(); return; }
    g.timeline()
      .set(z, { display: "block", xPercent: -100 })
      .to(z, { xPercent: 0, duration: 0.42, ease: "power2.in" })
      .add(() => { swap(); window.scrollTo(0, 0); })
      .to(z, { xPercent: 100, duration: 0.5, ease: "power2.out" })
      .set(z, { display: "none", xPercent: -100 });
  }

  // ------------------------------------------------------------------ wiring
  function openWhy(btn, animate) {
    const panel = document.getElementById(btn.getAttribute("aria-controls"));
    btn.setAttribute("aria-expanded", "true");
    btn.textContent = "Hide the reasoning";
    panel.hidden = false;
    const g = animate ? gs() : null;
    if (g) {
      g.from(panel, { height: 0, opacity: 0, duration: 0.4, ease: "power2.out", clearProps: "height,opacity" });
      g.from($$(".wf-bar i", panel), {
        scaleX: 0, duration: 0.6, ease: "power3.out", stagger: 0.04, delay: 0.1,
        transformOrigin: (i, el) => (el.classList.contains("for") ? "left center" : "right center"),
      });
    }
  }
  function bind(view) {
    if (view === "tonight") {
      $$(".why-toggle").forEach((btn) => btn.addEventListener("click", () => {
        if (btn.getAttribute("aria-expanded") === "true") {
          btn.setAttribute("aria-expanded", "false");
          btn.textContent = btn.dataset.closed;
          document.getElementById(btn.getAttribute("aria-controls")).hidden = true;
        } else {
          openWhy(btn, true);
        }
      }));
    }
    if (view === "model") {
      const btn = document.getElementById("verify");
      const out = document.getElementById("verify-out");
      if (btn) btn.addEventListener("click", async () => {
        btn.disabled = true;
        out.className = "verify-out";
        out.textContent = "Downloading the tape and re-computing every hash";
        try {
          const r = await verifyTape();
          if (!r.ok) { out.classList.add("bad"); out.textContent = `The chain breaks at entry ${r.at} of ${r.n}. Something was edited.`; }
          else if (!r.n) out.textContent = "The tape is empty until the first pick locks on opening night.";
          else out.innerHTML = `Chain intact: all ${r.n} hashes match. Latest hash <code>${esc(r.head)}</code>`;
        } catch (e) {
          out.classList.add("bad");
          out.textContent = `Could not download the tape: ${e.message}`;
        }
        btn.disabled = false;
      });
    }
  }

  const VIEWS = { tonight: renderTonight, record: renderRecord, scoreboard: renderScoreboard, model: renderModel };
  const viewFromHash = () => { const v = (location.hash || "").replace("#", ""); return VIEWS[v] ? v : "tonight"; };

  function draw(v) {
    try {
      return VIEWS[v]();
    } catch (e) {
      console.error(e);
      return `<div class="view-head"><h1 class="view-title">Something broke</h1><p class="lede">This page could not be drawn (${esc(e.message)}). The data is safe; try again in a minute.</p></div>`;
    }
  }
  function show(v, how) {
    const swap = () => {
      S.view = v;
      $$("nav a").forEach((a) => a.setAttribute("aria-current", a.dataset.view === v ? "page" : "false"));
      main.innerHTML = draw(v);
      bind(v);
      tick();
    };
    if (how === "wipe") wipe(swap); else swap();
  }

  async function refresh() {
    try {
      const t = await fetchData("public/today.json");
      if (S.today && t.generated_at === S.today.generated_at) return;
      S.today = t;
      if (S.view !== "tonight") return;
      const open = $$('.why-toggle[aria-expanded="true"]').map((b) => b.getAttribute("aria-controls"));
      main.innerHTML = draw("tonight");
      bind("tonight");
      open.forEach((id) => { const b = document.querySelector(`[aria-controls="${id}"]`); if (b) openWhy(b, false); });
      tick();
    } catch (e) { /* keep showing the last good board */ }
  }

  function healthLine() {
    const el = document.getElementById("health-line");
    const src = (S.health && S.health.sources) || {};
    const names = Object.keys(src);
    if (!el || !names.length) return;
    const bad = names.filter((k) => src[k].status === "error");
    if (!bad.length) { el.textContent = `Data sources: all ${names.length} working.`; return; }
    el.classList.add("bad");
    el.textContent = `Data problem: ${bad.join(", ")}. The site keeps the last good data until it's fixed.`;
  }

  async function boot() {
    main = document.getElementById("view");
    if (!CFG.dataBase && !isLocal() && (!CFG.owner || CFG.owner.indexOf("YOUR-") === 0)) {
      main.innerHTML = `<div class="view-head"><h1 class="view-title">One setting left</h1><p class="lede">Open site/js/config.js and put in your GitHub username and repository name, so this page can read the model's data.</p></div>`;
      return;
    }
    try {
      const [today, record, model] = await Promise.all([
        fetchData("public/today.json"), fetchData("public/record.json"), fetchData("public/model.json")]);
      Object.assign(S, { today, record, model });
    } catch (e) {
      main.innerHTML = `<div class="view-head"><h1 class="view-title">No data yet</h1><p class="lede">${esc(e.message)}. Check that the repository is public and the pipeline has run once.</p></div>`;
      return;
    }
    try { S.health = await fetchData("public/health.json"); } catch (e) { S.health = null; }
    healthLine();
    show(viewFromHash(), "none");
    intro(true);
    window.addEventListener("hashchange", () => {
      show(viewFromHash(), "wipe");
      main.focus({ preventScroll: true });
    });
    setInterval(tick, 1000);
    setInterval(refresh, 90000);
    checkTape();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();

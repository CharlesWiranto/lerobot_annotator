const SEG_COLORS = ["#3d6fb8", "#c47a2c", "#3d8a52", "#b44a4a", "#4a9aa3", "#8a6bb0"];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  name: null,
  fps: 30,
  cameras: [],
  episodes: [],
  health: {},
  instruction_segments: {},
  high_level_instruction: {},
  progress: { reviewed: {}, template: null, last_episode: 0 },
  broken: new Set(),
  episodeIndex: 0,
  frame: 0,
  jump: 5,
  rate: 1,
  playing: false,
  lastPlayTs: 0,
  activeSeg: 0,
  dirty: false,
  videoFrames: {},
  length: 0,
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

let toastTimer = 0;
function toast(msg, ms = 4200) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), ms);
}

function epKey(i = state.episodeIndex) {
  return String(i);
}

function isBroken(i = state.episodeIndex) {
  return state.broken.has(i);
}

function brokenList() {
  return [...state.broken].sort((a, b) => a - b);
}

function segsOf(i = state.episodeIndex) {
  return state.instruction_segments[epKey(i)] || [];
}

function lastFrame() {
  return Math.max(0, state.length - 1);
}

function markDirty() {
  state.dirty = true;
  $("#save-state").textContent = "unsaved";
}

function markClean() {
  state.dirty = false;
  $("#save-state").textContent = "saved";
}

function templateState() {
  const n = clamp(parseInt($("#tpl-n").value, 10) || 1, 1, 12);
  const instructions = $$("#tpl-instructions textarea").map((t) => t.value);
  while (instructions.length < n) instructions.push("");
  return {
    n,
    instructions: instructions.slice(0, n),
    hl_mode: $("#tpl-hl-mode").value,
    hl_custom: $("#tpl-hl-custom").value,
  };
}

function persistBroken() {
  return api(`/api/datasets/${encodeURIComponent(state.name)}/broken`, {
    method: "POST",
    body: JSON.stringify({ episodes: brokenList() }),
  })
    .then((res) => {
      state.broken = new Set(res.broken || []);
      renderEpisodeList();
      renderHealth();
    })
    .catch((err) => toast("Could not write broken.json: " + err.message));
}

function setBroken(on) {
  if (on) state.broken.add(state.episodeIndex);
  else state.broken.delete(state.episodeIndex);
  $("#broken").checked = on;
  $("#broken-banner").classList.toggle("hidden", !on);
  renderEpisodeList();
  renderHealth();
  persistBroken();
}

function persistProgressFields() {
  state.progress.template = templateState();
  state.progress.last_episode = state.episodeIndex;
  if (!state.progress.reviewed) state.progress.reviewed = {};
}

function hlForEpisode(i, segs) {
  const tpl = templateState();
  if (tpl.hl_mode === "off") return "";
  if (tpl.hl_mode === "custom") return tpl.hl_custom;
  return (segs || segsOf(i)).map((s) => s.instruction).filter(Boolean).join(", ");
}

function setHL(i, text) {
  state.high_level_instruction[epKey(i)] = { high_level_instruction: text || "" };
}

function clampSuccess(segs) {
  for (const s of segs) {
    const hi = s.end_frame_index - 1;
    s.success_frame_index = clamp(s.success_frame_index, s.start_frame_index, Math.max(s.start_frame_index, hi));
  }
}

function buildSegments(length, instructions) {
  const n = Math.max(1, instructions.length);
  const L = Math.max(1, length);
  const segs = [];
  for (let i = 0; i < n; i++) {
    const start = Math.round((i * L) / n);
    const end = i === n - 1 ? L : Math.round(((i + 1) * L) / n);
    const e = Math.max(end, start + 1);
    segs.push({
      instruction: instructions[i] || "",
      instruction_augmentation: {},
      start_frame_index: start,
      success_frame_index: Math.max(start, e - 1),
      end_frame_index: e,
    });
  }
  for (let i = 1; i < n; i++) segs[i].start_frame_index = segs[i - 1].end_frame_index;
  segs[n - 1].end_frame_index = L;
  clampSuccess(segs);
  return segs;
}

function setSegments(next, i = state.episodeIndex) {
  state.instruction_segments[epKey(i)] = next;
  const ep = state.episodes.find((e) => e.episode_index === i);
  if (ep) ep.n_segments = next.length;
  markDirty();
}

function setSegField(index, field, value, opts = {}) {
  const segs = segsOf().map((s) => ({ ...s, instruction_augmentation: s.instruction_augmentation || {} }));
  if (!segs[index]) return;
  const n = segs.length;
  const L = state.length;
  if (field === "instruction") {
    segs[index].instruction = value;
    setSegments(segs);
    if (!opts.silent) renderSegments();
    return;
  }
  value = Math.round(Number(value));
  if (!Number.isFinite(value)) return;
  if (field === "start_frame_index") {
    const minS = index === 0 ? 0 : segs[index - 1].start_frame_index + 1;
    const maxS = segs[index].end_frame_index - 1;
    value = clamp(value, minS, maxS);
    segs[index].start_frame_index = value;
    if (index > 0) segs[index - 1].end_frame_index = value;
  } else if (field === "end_frame_index") {
    const minE = segs[index].start_frame_index + 1;
    const maxE = index === n - 1 ? L : segs[index + 1].end_frame_index - 1;
    value = clamp(value, minE, maxE);
    segs[index].end_frame_index = value;
    if (index < n - 1) segs[index + 1].start_frame_index = value;
  } else if (field === "success_frame_index") {
    segs[index].success_frame_index = value;
  }
  clampSuccess(segs);
  setSegments(segs);
  if (!opts.silent) {
    renderSegments();
    renderTimeline();
    renderEpisodeList();
    updateActiveHint();
  }
}

function containingSegment(frame) {
  const segs = segsOf();
  for (let i = 0; i < segs.length; i++) {
    if (frame >= segs[i].start_frame_index && frame < segs[i].end_frame_index) return i;
  }
  return segs.length ? segs.length - 1 : 0;
}

function frameUrl(cam, frame) {
  const q = new URLSearchParams({
    episode: String(state.episodeIndex),
    camera: cam.key,
    frame: String(frame),
    max_w: "640",
  });
  return `/api/datasets/${encodeURIComponent(state.name)}/frame?${q}`;
}

function ensureCameras() {
  const row = $("#cam-row");
  const n = Math.max(1, state.cameras.length);
  row.style.gridTemplateColumns = `repeat(${n}, minmax(0, 1fr))`;
  if (row.dataset.sig === state.cameras.map((c) => c.key).join("|")) return;
  row.innerHTML = "";
  row.dataset.sig = state.cameras.map((c) => c.key).join("|");
  for (const cam of state.cameras) {
    const card = document.createElement("article");
    card.className = "cam-card";
    card.innerHTML = `<header><span>${cam.label}</span><span class="mono cam-shape">${(cam.shape || []).join("×")}</span></header>
      <div class="frame-wrap"><img alt="${cam.label}" draggable="false"></div>`;
    row.appendChild(card);
  }
}

function loadCameraImages() {
  ensureCameras();
  const imgs = $$("#cam-row img");
  state.cameras.forEach((cam, i) => {
    if (imgs[i]) imgs[i].src = frameUrl(cam, state.frame);
  });
  prefetch(state.frame);
}

function prefetch(from) {
  const max = lastFrame();
  for (let d = 1; d <= 5; d++) {
    const f = from + d;
    if (f > max) break;
    for (const cam of state.cameras) {
      const img = new Image();
      img.src = frameUrl(cam, f);
    }
  }
}

function setFrame(f) {
  const max = lastFrame();
  f = clamp(Math.round(f), 0, max);
  state.frame = f;
  $("#frame-slider").value = String(f);
  $("#frame-input").value = String(f);
  $("#frame-of").textContent = `/ ${max}  (length ${state.length})`;
  const t = state.fps ? f / state.fps : 0;
  $("#frame-time").textContent = `${t.toFixed(3)}s @ ${state.fps} fps`;
  loadCameraImages();
  renderTimelinePlayhead();
  highlightNowSegment();
  updateActiveHint();
}

function updateActiveHint() {
  const segs = segsOf();
  const s = segs[state.activeSeg];
  const now = containingSegment(state.frame);
  if (!s) {
    $("#active-hint").textContent = "No segments yet. Apply the task template first.";
    return;
  }
  $("#active-hint").textContent =
    `Active segment ${state.activeSeg} · current frame ${state.frame} is in segment ${now}. ` +
    `S/X/E write start / success / end on the active segment. End is exclusive (this frame as last ⇒ end = frame + 1).`;
}

function highlightNowSegment() {
  const now = containingSegment(state.frame);
  $$(".seg-card").forEach((el, i) => el.classList.toggle("now-seg", i === now));
}

function renderTimelinePlayhead() {
  const head = $("#timeline .tl-playhead");
  if (!head || !state.length) return;
  head.style.left = `${((state.frame + 0.5) / state.length) * 100}%`;
}

function xToFrame(clientX, exclusive = false) {
  const el = $("#timeline");
  const r = el.getBoundingClientRect();
  const t = clamp((clientX - r.left) / Math.max(1, r.width), 0, 1);
  if (exclusive) return clamp(Math.round(t * state.length), 0, state.length);
  return clamp(Math.floor(t * state.length), 0, lastFrame());
}

function renderTimeline() {
  const el = $("#timeline");
  const L = state.length;
  const segs = segsOf();
  el.innerHTML = "";
  if (!L) return;
  segs.forEach((s, i) => {
    const color = SEG_COLORS[i % SEG_COLORS.length];
    const left = (s.start_frame_index / L) * 100;
    const width = ((s.end_frame_index - s.start_frame_index) / L) * 100;
    const span = Math.max(1, s.end_frame_index - s.start_frame_index);
    const trained = ((s.success_frame_index - s.start_frame_index + 1) / span) * 100;
    const wrap = document.createElement("div");
    wrap.className = "tl-seg";
    wrap.style.left = `${left}%`;
    wrap.style.width = `${width}%`;
    wrap.innerHTML = `
      <div class="tl-trained" style="width:${trained}%;background:${color}"></div>
      <div class="tl-prompt" style="left:${trained}%;right:0;background:${color};filter:brightness(.42)"></div>
      <div class="tl-label">${i}</div>`;
    el.appendChild(wrap);

    const mkHandle = (cls, field, exclusive) => {
      const h = document.createElement("div");
      h.className = cls;
      const pos = exclusive ? s[field] : s[field] + 0.5;
      h.style.left = `${(pos / L) * 100}%`;
      h.addEventListener("pointerdown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        state.activeSeg = i;
        const move = (e2) => {
          setSegField(i, field, xToFrame(e2.clientX, exclusive), { silent: true });
          renderTimeline();
        };
        const up = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
          renderSegments();
          renderTimeline();
          renderEpisodeList();
          updateActiveHint();
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });
      el.appendChild(h);
    };
    mkHandle("tl-handle", "start_frame_index", false);
    mkHandle("tl-handle", "end_frame_index", true);
    mkHandle("tl-success", "success_frame_index", false);
  });
  const head = document.createElement("div");
  head.className = "tl-playhead";
  el.appendChild(head);
  renderTimelinePlayhead();
}

function renderTemplateFields() {
  const n = clamp(parseInt($("#tpl-n").value, 10) || 1, 1, 12);
  const box = $("#tpl-instructions");
  const prev = $$("textarea", box).map((t) => t.value);
  box.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const lab = document.createElement("label");
    lab.textContent = `Segment ${i} instruction`;
    const ta = document.createElement("textarea");
    ta.rows = 2;
    ta.value = prev[i] || "";
    ta.addEventListener("input", () => {
      persistProgressFields();
      markDirty();
    });
    box.appendChild(lab);
    box.appendChild(ta);
  }
  const custom = $("#tpl-hl-mode").value === "custom";
  $("#tpl-hl-custom").classList.toggle("hidden", !custom);
}

function renderSegments() {
  const box = $("#seg-list");
  const segs = segsOf();
  box.innerHTML = "";
  segs.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "seg-card" + (i === state.activeSeg ? " active" : "");
    const color = SEG_COLORS[i % SEG_COLORS.length];
    card.innerHTML = `
      <header>
        <span><i class="seg-dot" style="background:${color}"></i>Segment ${i}</span>
        <span class="muted mono">${s.start_frame_index} → ${s.end_frame_index} (excl)</span>
      </header>
      <div class="seg-grid">
        <span>start</span><input type="number" data-f="start_frame_index" min="0" value="${s.start_frame_index}">
        <span>success</span><input type="number" data-f="success_frame_index" min="0" value="${s.success_frame_index}">
        <span>end</span><input type="number" data-f="end_frame_index" min="1" value="${s.end_frame_index}">
      </div>
      <textarea data-f="instruction" rows="2">${escapeHtml(s.instruction)}</textarea>`;
    card.addEventListener("click", () => {
      state.activeSeg = i;
      renderSegments();
      updateActiveHint();
    });
    $$("input,textarea", card).forEach((inp) => {
      inp.addEventListener("change", () => {
        state.activeSeg = i;
        setSegField(i, inp.dataset.f, inp.dataset.f === "instruction" ? inp.value : Number(inp.value));
      });
      inp.addEventListener("click", (e) => e.stopPropagation());
    });
    box.appendChild(card);
  });
  highlightNowSegment();
  const hl = (state.high_level_instruction[epKey()] || {}).high_level_instruction || "";
  $("#ep-hl").value = hl;
  $("#hl-warn").classList.toggle("hidden", !hl.trim());
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderEpisodeList() {
  const q = ($("#ep-filter").value || "").trim();
  const box = $("#episode-list");
  box.innerHTML = "";
  let reviewed = 0;
  for (const ep of state.episodes) {
    if (state.progress.reviewed?.[epKey(ep.episode_index)]) reviewed += 1;
    if (q && !String(ep.episode_index).includes(q)) continue;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ep-item";
    if (ep.episode_index === state.episodeIndex) btn.classList.add("current");
    if (state.progress.reviewed?.[epKey(ep.episode_index)]) btn.classList.add("reviewed");
    if (isBroken(ep.episode_index)) btn.classList.add("broken");
    const nseg = segsOf(ep.episode_index).length;
    const marks = [];
    if (isBroken(ep.episode_index)) marks.push("brk");
    if (state.progress.reviewed?.[epKey(ep.episode_index)]) marks.push("rev");
    else if (nseg) marks.push("·");
    btn.innerHTML = `<span class="idx">${ep.episode_index}</span>
      <span>${ep.length} f · ${nseg} seg</span>
      <span class="mark">${marks.join(" ")}</span>`;
    btn.addEventListener("click", () => selectEpisode(ep.episode_index).catch((err) => toast(err.message)));
    box.appendChild(btn);
  }
  $("#ep-progress").textContent = `${reviewed}/${state.episodes.length} rev · ${state.broken.size} brk`;
}

function renderHealth() {
  const h = state.health || {};
  const bits = [];
  bits.push(`<div class="${h.fps_ok ? "ok" : "bad"}">fps ${h.fps} ${h.fps_ok ? "(30)" : "(expected 30)"}</div>`);
  bits.push(`<div>codebase ${h.codebase_version || "?"}</div>`);
  if (h.field_descriptions_ok) bits.push(`<div class="ok">field_descriptions present</div>`);
  else {
    bits.push(`<div class="bad">Missing state: ${(h.missing_state_fields || []).join(", ") || "—"}</div>`);
    bits.push(`<div class="bad">Missing action: ${(h.missing_action_fields || []).join(", ") || "—"}</div>`);
  }
  if ((h.missing_cameras || []).length) bits.push(`<div class="bad">Missing cameras: ${h.missing_cameras.join(", ")}</div>`);
  bits.push(`<div>waist fields: ${h.has_waist ? "yes" : "no"}</div>`);
  bits.push(
    `<div class="${h.episodes_missing_segments_count ? "bad" : "ok"}">${h.episodes_missing_segments_count || 0} episodes missing segments</div>`
  );
  bits.push(
    `<div class="${h.hl_overrides_subtasks ? "warn" : ""}">${h.high_level_instruction_count || 0} episodes have HL text (overrides subtask prompts)</div>`
  );
  bits.push(
    `<div class="${state.broken.size ? "bad" : "ok"}">${state.broken.size} in broken.json${
      state.broken.size ? " [" + brokenList().join(", ") + "]" : ""
    }</div>`
  );
  $("#health").innerHTML = bits.join("");
}

function renderWorkspaceChrome() {
  $("#ds-name").textContent = state.name;
  $("#ds-meta").textContent = `${state.cameras.length} cameras · ${state.fps} fps · ${state.episodes.length} episodes`;
}

let selectGen = 0;

async function selectEpisode(idx) {
  const gen = ++selectGen;
  stopPlay();
  state.episodeIndex = idx;
  state.activeSeg = 0;
  const ep = state.episodes.find((e) => e.episode_index === idx);
  state.length = ep ? ep.length : 0;
  $("#frame-slider").max = String(lastFrame());
  $("#frame-input").max = String(lastFrame());
  $("#reviewed").checked = Boolean(state.progress.reviewed?.[epKey(idx)]);
  $("#broken").checked = isBroken(idx);
  $("#broken-banner").classList.toggle("hidden", !isBroken(idx));
  $("#ep-task").textContent = (ep?.tasks || []).join(" / ") || "No task string in episodes.jsonl";
  try {
    const meta = await api(`/api/datasets/${encodeURIComponent(state.name)}/episodes/${idx}`);
    if (gen !== selectGen) return;
    state.videoFrames = meta.video_frames || {};
    const mismatch = [];
    for (const [k, n] of Object.entries(state.videoFrames)) {
      if (n && n !== state.length) mismatch.push(`${k}: ${n} vs length ${state.length}`);
    }
    if ((meta.missing_videos || []).length) mismatch.push("missing " + meta.missing_videos.join(", "));
    $("#frame-banner").classList.toggle("hidden", mismatch.length === 0);
    $("#frame-banner").textContent = mismatch.length
      ? "Video / length mismatch — check frame alignment: " + mismatch.join("; ")
      : "";
  } catch (err) {
    $("#frame-banner").classList.remove("hidden");
    $("#frame-banner").textContent = String(err.message || err);
  }
  if (!segsOf().length) {
    const tpl = templateState();
    setSegments(buildSegments(state.length, tpl.instructions));
    setHL(idx, hlForEpisode(idx));
  }
  renderEpisodeList();
  renderSegments();
  renderTimeline();
  setFrame(0);
  persistProgressFields();
}

function applyToEpisode(i, { instructionsOnly = false } = {}) {
  const tpl = templateState();
  const ep = state.episodes.find((e) => e.episode_index === i);
  if (!ep) return;
  const existing = segsOf(i);
  if (instructionsOnly) {
    if (!existing.length) {
      setSegments(buildSegments(ep.length, tpl.instructions), i);
    } else {
      const next = existing.map((s, j) => ({
        ...s,
        instruction: tpl.instructions[j] != null ? tpl.instructions[j] : s.instruction,
      }));
      setSegments(next, i);
    }
  } else {
    setSegments(buildSegments(ep.length, tpl.instructions), i);
  }
  setHL(i, hlForEpisode(i, segsOf(i)));
}

function addSegment() {
  const segs = segsOf().map((s) => ({ ...s }));
  const L = state.length;
  const tpl = templateState();
  if (!segs.length) {
    setSegments(buildSegments(L, tpl.instructions.length ? tpl.instructions : [""]));
  } else {
    const last = segs[segs.length - 1];
    const span = last.end_frame_index - last.start_frame_index;
    if (span < 2) {
      toast("Last segment is too short to split");
      return;
    }
    const mid = last.start_frame_index + Math.floor(span / 2);
    last.end_frame_index = mid;
    last.success_frame_index = Math.max(last.start_frame_index, mid - 1);
    segs.push({
      instruction: tpl.instructions[segs.length] || "",
      instruction_augmentation: {},
      start_frame_index: mid,
      success_frame_index: L - 1,
      end_frame_index: L,
    });
    clampSuccess(segs);
    setSegments(segs);
    state.activeSeg = segs.length - 1;
  }
  renderSegments();
  renderTimeline();
  renderEpisodeList();
  updateActiveHint();
}

function removeActive() {
  const segs = segsOf();
  if (segs.length <= 1) {
    toast("Keep at least one segment — otherwise the episode is dropped from training.");
    return;
  }
  const i = state.activeSeg;
  const removed = segs[i];
  const next = segs.filter((_, j) => j !== i).map((s) => ({ ...s }));
  if (i > 0) next[i - 1].end_frame_index = removed.end_frame_index;
  else next[0].start_frame_index = 0;
  next[0].start_frame_index = 0;
  next[next.length - 1].end_frame_index = state.length;
  for (let j = 1; j < next.length; j++) next[j].start_frame_index = next[j - 1].end_frame_index;
  clampSuccess(next);
  setSegments(next);
  state.activeSeg = Math.min(i, next.length - 1);
  renderSegments();
  renderTimeline();
  renderEpisodeList();
}

function copyPrev() {
  if (state.episodeIndex <= 0) {
    toast("No previous episode");
    return;
  }
  const prev = segsOf(state.episodeIndex - 1);
  const prevL = state.episodes[state.episodeIndex - 1].length;
  const L = state.length;
  if (!prev.length || !prevL) return;
  const segs = prev.map((s) => ({
    ...s,
    instruction_augmentation: s.instruction_augmentation || {},
    start_frame_index: Math.round((s.start_frame_index / prevL) * L),
    success_frame_index: Math.round((s.success_frame_index / prevL) * L),
    end_frame_index: Math.round((s.end_frame_index / prevL) * L),
  }));
  segs[0].start_frame_index = 0;
  segs[segs.length - 1].end_frame_index = L;
  for (let i = 1; i < segs.length; i++) segs[i].start_frame_index = segs[i - 1].end_frame_index;
  clampSuccess(segs);
  setSegments(segs);
  setHL(state.episodeIndex, hlForEpisode(state.episodeIndex, segs));
  renderSegments();
  renderTimeline();
  renderEpisodeList();
}

function markHere(field) {
  const segs = segsOf();
  if (!segs[state.activeSeg]) {
    toast("Select a segment first");
    return;
  }
  if (field === "end_frame_index") setSegField(state.activeSeg, field, state.frame + 1);
  else setSegField(state.activeSeg, field, state.frame);
}

function stopPlay() {
  state.playing = false;
  $("#btn-play").textContent = "Play";
}

function togglePlay() {
  if (state.playing) {
    stopPlay();
    return;
  }
  if (state.frame >= lastFrame()) setFrame(0);
  state.playing = true;
  $("#btn-play").textContent = "Pause";
  state.lastPlayTs = 0;
  requestAnimationFrame(playTick);
}

function playTick(ts) {
  if (!state.playing) return;
  const interval = 1000 / Math.max(1, state.fps * state.rate);
  if (!state.lastPlayTs) state.lastPlayTs = ts;
  if (ts - state.lastPlayTs >= interval) {
    state.lastPlayTs = ts;
    if (state.frame >= lastFrame()) {
      stopPlay();
      return;
    }
    setFrame(state.frame + 1);
  }
  requestAnimationFrame(playTick);
}

function step(delta) {
  stopPlay();
  setFrame(state.frame + delta);
}

async function save() {
  persistProgressFields();
  try {
    const res = await api(`/api/datasets/${encodeURIComponent(state.name)}/save`, {
      method: "POST",
      body: JSON.stringify({
        instruction_segments: state.instruction_segments,
        high_level_instruction: state.high_level_instruction,
        progress: state.progress,
        broken: brokenList(),
      }),
    });
    markClean();
    const extra = (res.warnings || []).length ? ` ${res.warnings.length} warning(s).` : "";
    toast(`Saved ${res.episodes_written} episodes to info.json.${extra}`, 6000);
    if ((res.warnings || []).length) console.warn(res.warnings);
    state.health.episodes_missing_segments_count = state.episodes.filter((e) => !segsOf(e.episode_index).length).length;
    renderHealth();
  } catch (err) {
    toast("Save failed: " + err.message);
  }
}

function fillTemplateFromDataset(data) {
  const saved = data.progress?.template;
  const first = (data.instruction_segments || {})["0"] || [];
  const unsplit = first.length <= 1;
  const n = saved?.n || (unsplit ? 2 : first.length || 2);
  $("#tpl-n").value = String(n);
  const texts =
    saved?.instructions ||
    (unsplit ? [first[0]?.instruction || "", ""] : first.map((s) => s.instruction));
  $("#tpl-hl-mode").value = saved?.hl_mode || "off";
  $("#tpl-hl-custom").value = saved?.hl_custom || "";
  renderTemplateFields();
  $$("#tpl-instructions textarea").forEach((t, i) => {
    t.value = texts[i] || "";
  });
}

async function openDataset(name) {
  const data = await api(`/api/datasets/${encodeURIComponent(name)}`);
  state.name = data.name;
  state.fps = Number(data.fps) || 30;
  state.cameras = data.cameras || [];
  state.episodes = data.episodes || [];
  state.health = data.health || {};
  state.instruction_segments = data.instruction_segments || {};
  state.high_level_instruction = data.high_level_instruction || {};
  state.progress = Object.assign({ reviewed: {}, last_episode: 0 }, data.progress || {});
  state.broken = new Set(data.broken || []);
  $("#cam-row").dataset.sig = "";
  fillTemplateFromDataset(data);
  renderWorkspaceChrome();
  renderHealth();
  markClean();
  $("#picker").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  const start = clamp(state.progress.last_episode || 0, 0, Math.max(0, state.episodes.length - 1));
  await selectEpisode(start);
}

function confirmLeave() {
  if (!state.dirty) return true;
  return window.confirm("Unsaved changes will be lost. Continue?");
}

function bind() {
  $("#btn-back").addEventListener("click", () => {
    if (!confirmLeave()) return;
    stopPlay();
    $("#workspace").classList.add("hidden");
    $("#picker").classList.remove("hidden");
  });
  $("#btn-save").addEventListener("click", save);
  $("#reviewed").addEventListener("change", () => {
    state.progress.reviewed[epKey()] = $("#reviewed").checked;
    persistProgressFields();
    markDirty();
    renderEpisodeList();
  });
  $("#broken").addEventListener("change", () => setBroken($("#broken").checked));
  $("#ep-filter").addEventListener("input", renderEpisodeList);
  $("#btn-play").addEventListener("click", togglePlay);
  $("#btn-step-back").addEventListener("click", () => step(-1));
  $("#btn-step-fwd").addEventListener("click", () => step(1));
  $("#btn-jump-back").addEventListener("click", () => step(-state.jump));
  $("#btn-jump-fwd").addEventListener("click", () => step(state.jump));
  $("#btn-prev-ep").addEventListener("click", () => {
    if (state.episodeIndex > 0) selectEpisode(state.episodeIndex - 1).catch((err) => toast(err.message));
  });
  $("#btn-next-ep").addEventListener("click", () => {
    if (state.episodeIndex < state.episodes.length - 1)
      selectEpisode(state.episodeIndex + 1).catch((err) => toast(err.message));
  });
  $("#jump-input").addEventListener("change", () => {
    state.jump = clamp(parseInt($("#jump-input").value, 10) || 1, 1, 300);
    $("#jump-input").value = String(state.jump);
  });
  $$(".jump-preset").forEach((b) =>
    b.addEventListener("click", () => {
      state.jump = Number(b.dataset.n);
      $("#jump-input").value = String(state.jump);
      $$(".jump-preset").forEach((x) => x.classList.toggle("active", x === b));
    })
  );
  $("#play-rate").addEventListener("change", () => {
    state.rate = Number($("#play-rate").value) || 1;
  });
  $("#frame-slider").addEventListener("input", () => {
    stopPlay();
    setFrame(Number($("#frame-slider").value));
  });
  $("#frame-slider").addEventListener("mouseup", () => $("#frame-slider").blur());
  $("#frame-input").addEventListener("change", () => {
    stopPlay();
    setFrame(Number($("#frame-input").value));
  });
  $("#timeline").addEventListener("pointerdown", (ev) => {
    if (ev.target.classList.contains("tl-handle") || ev.target.classList.contains("tl-success")) return;
    stopPlay();
    setFrame(xToFrame(ev.clientX, false));
    const move = (e2) => setFrame(xToFrame(e2.clientX, false));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
  $("#timeline").addEventListener("pointermove", (ev) => {
    $("#tl-hover").textContent = `hover ${xToFrame(ev.clientX, false)}`;
  });
  $("#tpl-n").addEventListener("change", renderTemplateFields);
  $("#tpl-hl-mode").addEventListener("change", () => {
    renderTemplateFields();
    persistProgressFields();
    markDirty();
  });
  $("#tpl-hl-custom").addEventListener("input", () => {
    persistProgressFields();
    markDirty();
  });
  $("#btn-apply-ep").addEventListener("click", () => {
    if (segsOf().length && !confirm("Replace this episode's segments with an even split from the template?")) return;
    applyToEpisode(state.episodeIndex);
    renderSegments();
    renderTimeline();
    renderEpisodeList();
    updateActiveHint();
  });
  $("#btn-apply-instr").addEventListener("click", () => {
    applyToEpisode(state.episodeIndex, { instructionsOnly: true });
    renderSegments();
    renderEpisodeList();
  });
  $("#btn-apply-unreviewed").addEventListener("click", () => {
    if (!confirm(
      "Apply template to every unreviewed episode?\n\n" +
      "This replaces their segments with an even split of the episode length.\n" +
      "First start = 0, last end = length.\n" +
      "Broken episodes and already-reviewed ones are skipped.\n" +
      "HL mode is '" + templateState().hl_mode + "' (off clears high-level text so the model sees subtask instructions)."
    )) return;
    for (const ep of state.episodes) {
      if (state.progress.reviewed?.[epKey(ep.episode_index)]) continue;
      if (isBroken(ep.episode_index)) continue;
      applyToEpisode(ep.episode_index);
    }
    renderSegments();
    renderTimeline();
    renderEpisodeList();
    toast("Template applied to unreviewed episodes (not written until Save)");
  });
  $("#btn-copy-prev").addEventListener("click", copyPrev);
  $("#btn-clear-hl").addEventListener("click", () => {
    if (!confirm("Clear high_level_instruction on every episode?\nTraining will then use per-segment instruction text.")) return;
    for (const ep of state.episodes) setHL(ep.episode_index, "");
    $("#ep-hl").value = "";
    $("#hl-warn").classList.add("hidden");
    markDirty();
    toast("Cleared HL in memory — Save to write info.json");
  });
  $("#btn-add-seg").addEventListener("click", addSegment);
  $("#btn-del-seg").addEventListener("click", removeActive);
  $("#btn-set-start").addEventListener("click", () => markHere("start_frame_index"));
  $("#btn-set-success").addEventListener("click", () => markHere("success_frame_index"));
  $("#btn-set-end").addEventListener("click", () => markHere("end_frame_index"));
  $("#ep-hl").addEventListener("input", () => {
    setHL(state.episodeIndex, $("#ep-hl").value);
    $("#hl-warn").classList.toggle("hidden", !$("#ep-hl").value.trim());
    markDirty();
  });

  window.addEventListener("keydown", (e) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName);
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      save();
      return;
    }
    if (typing) return;
    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      step(e.shiftKey ? -state.jump : -1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      step(e.shiftKey ? state.jump : 1);
    } else if (e.key === "[") {
      if (state.episodeIndex > 0) selectEpisode(state.episodeIndex - 1).catch((err) => toast(err.message));
    } else if (e.key === "]") {
      if (state.episodeIndex < state.episodes.length - 1)
        selectEpisode(state.episodeIndex + 1).catch((err) => toast(err.message));
    } else if (e.key === "Home") {
      e.preventDefault();
      setFrame(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setFrame(lastFrame());
    } else if (e.key === "b" || e.key === "B") {
      setBroken(!isBroken());
    } else if (e.key === "s" || e.key === "S") {
      markHere("start_frame_index");
    } else if (e.key === "x" || e.key === "X") {
      markHere("success_frame_index");
    } else if (e.key === "e" || e.key === "E") {
      markHere("end_frame_index");
    } else if (e.key >= "1" && e.key <= "9") {
      const i = Number(e.key) - 1;
      if (segsOf()[i]) {
        state.activeSeg = i;
        renderSegments();
        updateActiveHint();
      }
    }
  });
  window.addEventListener("beforeunload", (e) => {
    if (!state.dirty) return;
    e.preventDefault();
    e.returnValue = "";
  });
}

function renderPicker(data) {
  $("#picker-root").textContent = data.root;
  const box = $("#dataset-list");
  box.innerHTML = "";
  if (!data.datasets.length) {
    box.innerHTML = "<p>No LeRobot v2.1 folders found (need meta/info.json).</p>";
    return;
  }
  for (const ds of data.datasets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ds-card";
    btn.innerHTML = `<h3>${ds.name}</h3>
      <dl>
        <dt>episodes</dt><dd>${ds.total_episodes}</dd>
        <dt>frames</dt><dd>${ds.total_frames}</dd>
        <dt>fps</dt><dd>${ds.fps}</dd>
        <dt>cameras</dt><dd>${(ds.cameras || []).length}</dd>
        <dt>reviewed</dt><dd>${ds.reviewed_episodes}/${ds.total_episodes}</dd>
        <dt>broken</dt><dd>${(ds.broken || []).length ? ds.broken.join(", ") : "none"}</dd>
        <dt>robot</dt><dd>${ds.robot_type || "—"} ${ds.codebase_version || ""}</dd>
      </dl>`;
    btn.addEventListener("click", () => openDataset(ds.name).catch((err) => toast(err.message)));
    box.appendChild(btn);
  }
}

async function main() {
  bind();
  try {
    const data = await api("/api/datasets");
    renderPicker(data);
  } catch (err) {
    $("#dataset-list").innerHTML = `<p>Failed to list datasets: ${err.message}</p>`;
  }
}

main();

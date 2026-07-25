/* Canvas 标注器 —— 本项目核心前端 */
window.YSLabeler = (function () {
  let root = null;
  let ds = null;
  let app = null;
  let images = [];
  let curIdx = 0;
  let boxes = [];
  let selected = -1;
  let mode = "draw"; // draw | pan | sam
  let classIdx = 0;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let imgNatural = { w: 1, h: 1 };
  let imgEl = null;
  let canvas = null;
  let ctx = null;
  let confMin = 0.15;
  let undoStack = [];
  let redoStack = [];
  let dragging = null; // {type:'new'|'move'|'resize', ...}
  let spaceDown = false;
  let dirty = false;
  let saving = false;
  let saveChain = Promise.resolve();
  let uiBound = false;
  let wrapEl = null;

  const HANDLE = 6;

  function $(sel, el) {
    return (el || root).querySelector(sel);
  }

  function onKeyUp(e) {
    if (e.code === "Space") spaceDown = false;
  }

  function destroy() {
    unbindUi();
    if (root) root.innerHTML = "";
    root = null;
    canvas = null;
    ctx = null;
    imgEl = null;
    images = [];
    boxes = [];
    dirty = false;
  }

  function mount(dataset, vueApp) {
    // 再次进入标注页前先卸旧监听，避免 A/D 连跳
    destroy();
    ds = dataset;
    app = vueApp;
    root = document.getElementById("labeler-root");
    if (!root) return;
    root.innerHTML = `
      <aside class="labeler-side">
        <div class="muted" style="margin-bottom:6px">进度 <span id="lb-progress">-</span></div>
        <div id="lb-thumbs"></div>
      </aside>
      <div class="labeler-center">
        <div class="labeler-toolbar">
          <button data-act="prev">A 上一张</button>
          <button data-act="next">D 下一张</button>
          <button data-act="draw" class="primary">W 画框</button>
          <button data-act="sam">E 点选</button>
          <button data-act="confirm" class="primary">Enter 确认</button>
          <button data-act="copy">C 复制上图</button>
          <button data-act="undo">撤销</button>
          <button data-act="redo">重做</button>
          <label class="muted">置信度过滤
            <input id="lb-conf" class="conf-slider" type="range" min="0" max="100" value="15" />
            <span id="lb-conf-v">0.15</span>
          </label>
          <span class="muted" id="lb-fname"></span>
        </div>
        <div class="labeler-canvas-wrap" id="lb-wrap">
          <canvas id="lb-canvas"></canvas>
        </div>
      </div>
      <aside class="labeler-right">
        <div><strong>类别</strong></div>
        <div id="lb-classes"></div>
        <div style="margin-top:12px"><strong>本图框列表</strong></div>
        <div id="lb-boxes"></div>
        <div class="help-keys">
          <div><strong>快捷键</strong></div>
          W 画框 · E SAM点选 · 空格+拖 平移<br/>
          滚轮缩放 · 1-9 改类别 · Del 删框<br/>
          A/D 切图 · Enter 确认 · C 复制上图<br/>
          Ctrl+Z 撤销 · Ctrl+Shift+Z 重做<br/>
          切换图片自动保存
        </div>
      </aside>
    `;
    canvas = $("#lb-canvas");
    ctx = canvas.getContext("2d");
    bindUi();
    loadImages().catch((e) => {
      if (app) app.showToast(e.message || String(e));
    });
  }

  function bindUi() {
    if (uiBound) unbindUi();
    root.querySelectorAll("[data-act]").forEach((btn) => {
      btn.addEventListener("click", onToolbarClick);
    });
    const conf = $("#lb-conf");
    if (conf) conf.addEventListener("input", onConfInput);
    wrapEl = $("#lb-wrap");
    if (wrapEl) {
      wrapEl.addEventListener("wheel", onWheel, { passive: false });
      wrapEl.addEventListener("mousedown", onDown);
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("resize", fitCanvas);
    uiBound = true;
  }

  function unbindUi() {
    if (!uiBound) return;
    if (root) {
      root.querySelectorAll("[data-act]").forEach((btn) => {
        btn.removeEventListener("click", onToolbarClick);
      });
      const conf = $("#lb-conf");
      if (conf) conf.removeEventListener("input", onConfInput);
    }
    if (wrapEl) {
      wrapEl.removeEventListener("wheel", onWheel);
      wrapEl.removeEventListener("mousedown", onDown);
      wrapEl = null;
    }
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("keyup", onKeyUp);
    window.removeEventListener("resize", fitCanvas);
    uiBound = false;
  }

  function onToolbarClick(e) {
    const btn = e.currentTarget;
    const a = btn.getAttribute("data-act");
    if (a === "prev") nav(-1);
    if (a === "next") nav(1);
    if (a === "draw") mode = "draw";
    if (a === "sam") mode = "sam";
    if (a === "confirm") confirmAndNext().catch((err) => app && app.showToast(err.message));
    if (a === "copy") copyPrev().catch((err) => app && app.showToast(err.message));
    if (a === "undo") undo();
    if (a === "redo") redo();
  }

  function onConfInput(e) {
    confMin = Number(e.target.value) / 100;
    const v = $("#lb-conf-v");
    if (v) v.textContent = confMin.toFixed(2);
    draw();
  }

  async function loadImages() {
    const r = await YS.api(`/api/datasets/${ds.id}/images?page_size=200&sort=uncertainty`);
    images = r.items || [];
    // 继续拉全量
    let page = 2;
    while (images.length < r.total) {
      const more = await YS.api(`/api/datasets/${ds.id}/images?page=${page}&page_size=200&sort=uncertainty`);
      if (!more.items.length) break;
      images = images.concat(more.items);
      page++;
    }
    renderThumbs();
    renderClasses();
    if (images.length) await selectIndex(0);
    else {
      $("#lb-progress").textContent = "0 / 0";
      $("#lb-fname").textContent = "暂无图片，请先上传";
    }
  }

  function renderClasses() {
    const el = $("#lb-classes");
    el.innerHTML = (ds.classes || [])
      .map(
        (c, i) =>
          `<button class="class-btn ${i === classIdx ? "active" : ""}" data-ci="${i}">${i + 1}. ${c}</button>`
      )
      .join("");
    el.querySelectorAll("button").forEach((b) => {
      b.onclick = () => {
        classIdx = Number(b.dataset.ci);
        if (selected >= 0 && boxes[selected]) {
          pushUndo();
          boxes[selected].class_idx = classIdx;
          dirty = true;
          renderBoxList();
          draw();
        }
        renderClasses();
      };
    });
  }

  function renderThumbs() {
    const el = $("#lb-thumbs");
    const labeled = images.filter((x) => ["reviewed", "confirmed", "auto"].includes(x.review_status)).length;
    $("#lb-progress").textContent = `已标 ${labeled} / 共 ${images.length}`;
    el.innerHTML = images
      .map((img, i) => {
        const st = img.review_status || "unlabeled";
        return `<div class="thumb-item ${i === curIdx ? "active" : ""}" data-i="${i}">
          <span class="st st-${st}"></span>
          <img src="/api/images/${img.id}/thumb" loading="lazy" />
          <span class="muted" style="font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:90px">${img.filename}</span>
        </div>`;
      })
      .join("");
    el.querySelectorAll(".thumb-item").forEach((n) => {
      n.onclick = () => selectIndex(Number(n.dataset.i));
    });
  }

  function renderBoxList() {
    const el = $("#lb-boxes");
    el.innerHTML = boxes
      .map((b, i) => {
        const name = (ds.classes || [])[b.class_idx] || b.class_idx;
        const conf = b.conf != null ? Number(b.conf).toFixed(2) : "1.00";
        return `<div class="box-list-item ${i === selected ? "active" : ""}" data-i="${i}">
          #${i + 1} ${name} · ${conf} · ${b.source || "manual"}
        </div>`;
      })
      .join("");
    el.querySelectorAll(".box-list-item").forEach((n) => {
      n.onclick = () => {
        selected = Number(n.dataset.i);
        renderBoxList();
        draw();
      };
    });
  }

  async function selectIndex(i) {
    if (i < 0 || i >= images.length) return;
    // 串行等待未完成的保存，避免 dirty 修改被跳过
    if (dirty) await enqueueSave(false);
    else await saveChain;
    curIdx = i;
    selected = -1;
    undoStack = [];
    redoStack = [];
    const img = images[curIdx];
    $("#lb-fname").textContent = `${img.filename} · ${img.review_status}`;
    const ann = await YS.api(`/api/images/${img.id}/annotations`);
    boxes = (ann.boxes || []).map((b) => ({ ...b }));
    // 缓存本图框，便于下一张 C 复制时快速读取
    images[curIdx]._boxesCache = boxes.map((b) => ({ ...b }));
    imgEl = new Image();
    imgEl.onload = () => {
      imgNatural = { w: imgEl.naturalWidth, h: imgEl.naturalHeight };
      fitCanvas();
      draw();
    };
    imgEl.src = `/api/images/${img.id}/file?t=${Date.now()}`;
    renderThumbs();
    renderBoxList();
    // 后台预热下一张（SAM / 标注）
    if (curIdx + 1 < images.length) {
      // fire-and-forget
      YS.api(`/api/images/${images[curIdx + 1].id}/annotations`).then((a) => {
        images[curIdx + 1]._boxesCache = (a.boxes || []).map((b) => ({ ...b }));
      }).catch(() => {});
    }
  }

  function fitCanvas() {
    const wrap = $("#lb-wrap");
    if (!wrap || !imgEl) return;
    const ww = wrap.clientWidth;
    const wh = wrap.clientHeight;
    canvas.width = ww;
    canvas.height = wh;
    const sx = ww / imgNatural.w;
    const sy = wh / imgNatural.h;
    scale = Math.min(sx, sy) * 0.95;
    offsetX = (ww - imgNatural.w * scale) / 2;
    offsetY = (wh - imgNatural.h * scale) / 2;
    draw();
  }

  function toImage(x, y) {
    return { x: (x - offsetX) / scale, y: (y - offsetY) / scale };
  }
  function toScreen(x, y) {
    return { x: x * scale + offsetX, y: y * scale + offsetY };
  }

  function visibleBoxes() {
    return boxes.filter((b) => (b.conf == null ? 1 : b.conf) >= confMin);
  }

  function draw() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imgEl && imgEl.complete) {
      ctx.drawImage(imgEl, offsetX, offsetY, imgNatural.w * scale, imgNatural.h * scale);
    }
    boxes.forEach((b, i) => {
      if ((b.conf == null ? 1 : b.conf) < confMin) return;
      const x1 = (b.cx - b.w / 2) * imgNatural.w;
      const y1 = (b.cy - b.h / 2) * imgNatural.h;
      const x2 = (b.cx + b.w / 2) * imgNatural.w;
      const y2 = (b.cy + b.h / 2) * imgNatural.h;
      const p1 = toScreen(x1, y1);
      const p2 = toScreen(x2, y2);
      const isAuto = b.source && b.source !== "manual";
      ctx.strokeStyle = i === selected ? "#38bdf8" : isAuto ? "#a78bfa" : "#22c55e";
      ctx.lineWidth = i === selected ? 2.5 : 1.5;
      if (isAuto && b.source !== "manual") ctx.setLineDash([6, 4]);
      else ctx.setLineDash([]);
      ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      ctx.setLineDash([]);
      const name = (ds.classes || [])[b.class_idx] || b.class_idx;
      const conf = b.conf != null ? Number(b.conf).toFixed(2) : "";
      ctx.fillStyle = "rgba(15,23,42,.75)";
      const label = `${name} ${conf}`;
      ctx.font = "12px sans-serif";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillRect(p1.x, Math.max(0, p1.y - 18), tw, 18);
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(label, p1.x + 4, Math.max(12, p1.y - 5));
      if (i === selected) {
        const handles = handlePoints(p1, p2);
        ctx.fillStyle = "#38bdf8";
        handles.forEach((h) => ctx.fillRect(h.x - HANDLE / 2, h.y - HANDLE / 2, HANDLE, HANDLE));
      }
    });
  }

  function handlePoints(p1, p2) {
    const mx = (p1.x + p2.x) / 2;
    const my = (p1.y + p2.y) / 2;
    return [
      { x: p1.x, y: p1.y, corner: "nw" },
      { x: mx, y: p1.y, corner: "n" },
      { x: p2.x, y: p1.y, corner: "ne" },
      { x: p2.x, y: my, corner: "e" },
      { x: p2.x, y: p2.y, corner: "se" },
      { x: mx, y: p2.y, corner: "s" },
      { x: p1.x, y: p2.y, corner: "sw" },
      { x: p1.x, y: my, corner: "w" },
    ];
  }

  function hitTest(sx, sy) {
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      if ((b.conf == null ? 1 : b.conf) < confMin) continue;
      const x1 = (b.cx - b.w / 2) * imgNatural.w;
      const y1 = (b.cy - b.h / 2) * imgNatural.h;
      const x2 = (b.cx + b.w / 2) * imgNatural.w;
      const y2 = (b.cy + b.h / 2) * imgNatural.h;
      const p1 = toScreen(x1, y1);
      const p2 = toScreen(x2, y2);
      if (selected === i) {
        for (const h of handlePoints(p1, p2)) {
          if (Math.abs(sx - h.x) <= HANDLE && Math.abs(sy - h.y) <= HANDLE) {
            return { type: "resize", index: i, corner: h.corner };
          }
        }
      }
      if (sx >= Math.min(p1.x, p2.x) && sx <= Math.max(p1.x, p2.x) && sy >= Math.min(p1.y, p2.y) && sy <= Math.max(p1.y, p2.y)) {
        return { type: "move", index: i };
      }
    }
    return null;
  }

  function onWheel(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const before = toImage(mx, my);
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    scale = Math.min(20, Math.max(0.05, scale * factor));
    offsetX = mx - before.x * scale;
    offsetY = my - before.y * scale;
    draw();
  }

  function onDown(e) {
    if (!imgEl) return;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    if (spaceDown || e.button === 1) {
      dragging = { type: "pan", x: sx, y: sy, ox: offsetX, oy: offsetY };
      return;
    }
    if (mode === "sam") {
      const p = toImage(sx, sy);
      samClick(p.x, p.y);
      return;
    }
    const hit = hitTest(sx, sy);
    if (hit) {
      selected = hit.index;
      pushUndo();
      const b = boxes[selected];
      dragging = {
        type: hit.type,
        index: hit.index,
        corner: hit.corner,
        start: toImage(sx, sy),
        orig: { ...b },
      };
      renderBoxList();
      draw();
      return;
    }
    if (mode === "draw") {
      const p = toImage(sx, sy);
      dragging = { type: "new", x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      selected = -1;
    }
  }

  function onMove(e) {
    if (!dragging || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    if (dragging.type === "pan") {
      offsetX = dragging.ox + (sx - dragging.x);
      offsetY = dragging.oy + (sy - dragging.y);
      draw();
      return;
    }
    const p = toImage(sx, sy);
    if (dragging.type === "new") {
      dragging.x1 = p.x;
      dragging.y1 = p.y;
      draw();
      // 临时框
      const x1 = Math.min(dragging.x0, dragging.x1);
      const y1 = Math.min(dragging.y0, dragging.y1);
      const x2 = Math.max(dragging.x0, dragging.x1);
      const y2 = Math.max(dragging.y0, dragging.y1);
      const a = toScreen(x1, y1);
      const b = toScreen(x2, y2);
      ctx.strokeStyle = "#fbbf24";
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.setLineDash([]);
      return;
    }
    const b = boxes[dragging.index];
    const o = dragging.orig;
    if (dragging.type === "move") {
      const dx = (p.x - dragging.start.x) / imgNatural.w;
      const dy = (p.y - dragging.start.y) / imgNatural.h;
      b.cx = clamp(o.cx + dx, 0, 1);
      b.cy = clamp(o.cy + dy, 0, 1);
      dirty = true;
    } else if (dragging.type === "resize") {
      let x1 = (o.cx - o.w / 2) * imgNatural.w;
      let y1 = (o.cy - o.h / 2) * imgNatural.h;
      let x2 = (o.cx + o.w / 2) * imgNatural.w;
      let y2 = (o.cy + o.h / 2) * imgNatural.h;
      const c = dragging.corner;
      if (c.includes("n")) y1 = p.y;
      if (c.includes("s")) y2 = p.y;
      if (c.includes("w")) x1 = p.x;
      if (c.includes("e")) x2 = p.x;
      const nx1 = Math.min(x1, x2);
      const ny1 = Math.min(y1, y2);
      const nx2 = Math.max(x1, x2);
      const ny2 = Math.max(y1, y2);
      b.cx = ((nx1 + nx2) / 2) / imgNatural.w;
      b.cy = ((ny1 + ny2) / 2) / imgNatural.h;
      b.w = Math.max(2, nx2 - nx1) / imgNatural.w;
      b.h = Math.max(2, ny2 - ny1) / imgNatural.h;
      dirty = true;
    }
    draw();
  }

  function onUp() {
    if (!dragging) return;
    if (dragging.type === "new") {
      const x1 = Math.min(dragging.x0, dragging.x1);
      const y1 = Math.min(dragging.y0, dragging.y1);
      const x2 = Math.max(dragging.x0, dragging.x1);
      const y2 = Math.max(dragging.y0, dragging.y1);
      if (x2 - x1 > 3 && y2 - y1 > 3) {
        pushUndo();
        boxes.push({
          class_idx: classIdx,
          cx: ((x1 + x2) / 2) / imgNatural.w,
          cy: ((y1 + y2) / 2) / imgNatural.h,
          w: (x2 - x1) / imgNatural.w,
          h: (y2 - y1) / imgNatural.h,
          conf: 1,
          source: "manual",
        });
        selected = boxes.length - 1;
        dirty = true;
        renderBoxList();
      }
    }
    dragging = null;
    draw();
  }

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function pushUndo() {
    undoStack.push(JSON.stringify(boxes));
    if (undoStack.length > 20) undoStack.shift();
    redoStack = [];
  }
  function undo() {
    if (!undoStack.length) return;
    redoStack.push(JSON.stringify(boxes));
    boxes = JSON.parse(undoStack.pop());
    dirty = true;
    selected = -1;
    renderBoxList();
    draw();
  }
  function redo() {
    if (!redoStack.length) return;
    undoStack.push(JSON.stringify(boxes));
    boxes = JSON.parse(redoStack.pop());
    dirty = true;
    selected = -1;
    renderBoxList();
    draw();
  }

  async function samClick(ix, iy) {
    if (!images[curIdx]) return;
    try {
      const r = await YS.api(`/api/images/${images[curIdx].id}/sam`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: [[ix, iy]], labels: [1] }),
      });
      pushUndo();
      boxes.push({ ...r.box, class_idx: classIdx, source: "sam" });
      selected = boxes.length - 1;
      dirty = true;
      renderBoxList();
      draw();
    } catch (e) {
      if (app) app.showToast(e.message);
    }
  }

  function enqueueSave(setReviewed = true) {
    saveChain = saveChain.then(() => saveCurrent(setReviewed)).catch((e) => {
      if (app) app.showToast(e.message || String(e));
    });
    return saveChain;
  }

  async function saveCurrent(setReviewed = true) {
    if (!images[curIdx]) return;
    // 串行保存：若正在保存则等上一次结束后再写当前快照
    while (saving) {
      await new Promise((r) => setTimeout(r, 30));
    }
    saving = true;
    const idx = curIdx;
    const snapBoxes = boxes.map((b) => ({ ...b }));
    try {
      const status = setReviewed
        ? snapBoxes.length
          ? images[idx].review_status === "confirmed"
            ? "confirmed"
            : "reviewed"
          : "unlabeled"
        : undefined;
      const body = {
        boxes: snapBoxes.map((b) => ({
          class_idx: b.class_idx,
          cx: b.cx,
          cy: b.cy,
          w: b.w,
          h: b.h,
          conf: b.conf == null ? 1 : b.conf,
          source: b.source || "manual",
        })),
      };
      if (status) body.review_status = status;
      const r = await YS.api(`/api/images/${images[idx].id}/annotations`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      images[idx].box_count = r.box_count;
      images[idx].review_status = r.review_status;
      images[idx]._boxesCache = snapBoxes.map((b) => ({ ...b }));
      if (idx === curIdx) dirty = false;
      renderThumbs();
    } catch (e) {
      if (app) app.showToast(e.message);
      throw e;
    } finally {
      saving = false;
    }
  }

  async function confirmAndNext() {
    if (!images[curIdx]) return;
    try {
      const body = {
        boxes: boxes.map((b) => ({
          class_idx: b.class_idx,
          cx: b.cx,
          cy: b.cy,
          w: b.w,
          h: b.h,
          conf: b.conf == null ? 1 : b.conf,
          source: b.source || "manual",
        })),
        review_status: "confirmed",
      };
      await YS.api(`/api/images/${images[curIdx].id}/annotations`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      images[curIdx].review_status = "confirmed";
      images[curIdx].box_count = boxes.length;
      images[curIdx]._boxesCache = boxes.map((b) => ({ ...b }));
      dirty = false;
      if (curIdx < images.length - 1) await selectIndex(curIdx + 1);
      else {
        renderThumbs();
        if (app) app.showToast("已经是最后一张");
      }
    } catch (e) {
      if (app) app.showToast(e.message);
    }
  }

  async function nav(delta) {
    const ni = curIdx + delta;
    if (ni < 0 || ni >= images.length) return;
    await selectIndex(ni);
  }

  async function copyPrev() {
    if (curIdx <= 0) {
      if (app) app.showToast("已经是第一张，没有上一张可复制");
      return;
    }
    const prev = images[curIdx - 1];
    let prevBoxes = prev._boxesCache;
    if (!prevBoxes) {
      try {
        const ann = await YS.api(`/api/images/${prev.id}/annotations`);
        prevBoxes = (ann.boxes || []).map((b) => ({ ...b }));
        prev._boxesCache = prevBoxes;
      } catch (e) {
        if (app) app.showToast(e.message || "读取上一张标注失败");
        return;
      }
    }
    if (!prevBoxes.length) {
      if (app) app.showToast("上一张没有标注可复制");
      return;
    }
    pushUndo();
    boxes = prevBoxes.map((b) => ({ ...b, source: "manual" }));
    dirty = true;
    renderBoxList();
    draw();
  }

  function onKey(e) {
    if (!root || !document.body.contains(root)) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.code === "Space") {
      spaceDown = true;
      e.preventDefault();
      return;
    }
    if (e.key === "w" || e.key === "W") mode = "draw";
    if (e.key === "e" || e.key === "E") mode = "sam";
    if (e.key === "a" || e.key === "A") {
      e.preventDefault();
      nav(-1);
    }
    if (e.key === "d" || e.key === "D") {
      e.preventDefault();
      nav(1);
    }
    if (e.key === "c" || e.key === "C") {
      e.preventDefault();
      copyPrev();
    }
    if (e.key === "Enter") {
      e.preventDefault();
      confirmAndNext();
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      if (selected >= 0) {
        pushUndo();
        boxes.splice(selected, 1);
        selected = -1;
        dirty = true;
        renderBoxList();
        draw();
      }
    }
    if (e.ctrlKey && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    }
    if (e.key >= "1" && e.key <= "9") {
      const ci = Number(e.key) - 1;
      if (ci < (ds.classes || []).length) {
        classIdx = ci;
        if (selected >= 0) {
          pushUndo();
          boxes[selected].class_idx = classIdx;
          dirty = true;
          renderBoxList();
          draw();
        }
        renderClasses();
      }
    }
  }

  return { mount, destroy };
})();

/* global PIXI */
(function () {
  const statusEl = document.getElementById("status");

  function setStatus(text) {
    if (!statusEl) {
      return;
    }
    statusEl.textContent = text || "";
    statusEl.style.display = text ? "block" : "none";
  }

  if (!window.PIXI || typeof window.PIXI.Application !== "function") {
    setStatus("Live2D加载失败: PIXI运行库不可用");
    console.error("[LIVE2D] PIXI runtime is not available.");
    return;
  }

  const app = new PIXI.Application({
    resizeTo: window,
    antialias: true,
    autoDensity: true,
    backgroundAlpha: 0,
  });

  document.getElementById("stage").appendChild(app.view);

  let live2dModel = null;
  let shouldFollowCursor = true;
  let modelScale = 1.0;
  let targetGazeX = 0.5;
  let targetGazeY = 0.5;

  function centerModelBottom() {
    if (!live2dModel) {
      return;
    }
    live2dModel.anchor.set(0.5, 1.0);
    live2dModel.x = app.renderer.width / 2;
    live2dModel.y = app.renderer.height;
  }

  function applyScale() {
    if (!live2dModel) {
      return;
    }
    const targetHeight = app.renderer.height * 0.92;
    const ratio = targetHeight / Math.max(1, live2dModel.height);
    const s = ratio * Math.max(0.2, modelScale);
    live2dModel.scale.set(s);
  }

  function safePlayMotion(groupCandidates) {
    if (!live2dModel || !Array.isArray(groupCandidates)) {
      return;
    }

    for (const group of groupCandidates) {
      try {
        if (typeof live2dModel.motion === "function") {
          live2dModel.motion(group, 0, 2);
          return;
        }
      } catch (_e) {
        // Try next group.
      }
    }
  }

  function getHitAreaCandidates(kind) {
    if (!live2dModel || !live2dModel.internalModel) {
      return [];
    }

    const settings = live2dModel.internalModel.settings;
    const hitAreas = settings && Array.isArray(settings.hitAreas) ? settings.hitAreas : [];
    const wanted = String(kind || "").toLowerCase();
    const candidates = [];

    for (const area of hitAreas) {
      if (!area) {
        continue;
      }
      const name = String(area.name || "");
      const id = String(area.id || "");
      const lname = name.toLowerCase();
      const lid = id.toLowerCase();

      if (wanted === "head") {
        if (lname.includes("head") || lid.includes("head")) {
          if (name) candidates.push(name);
          if (id) candidates.push(id);
        }
      } else if (wanted === "body") {
        if (lname.includes("body") || lid.includes("body")) {
          if (name) candidates.push(name);
          if (id) candidates.push(id);
        }
      }
    }

    if (wanted === "head") {
      candidates.push("Head", "HitAreaHead");
    } else if (wanted === "body") {
      candidates.push("Body", "HitAreaBody");
    }

    return [...new Set(candidates.filter(Boolean))];
  }

  function hitTestAny(candidates, x, y) {
    if (!live2dModel || typeof live2dModel.hitTest !== "function") {
      return false;
    }

    for (const key of candidates) {
      try {
        if (live2dModel.hitTest(key, x, y)) {
          return true;
        }
      } catch (_e) {
        // Try next candidate.
      }
    }
    return false;
  }

  function applyGazeFallback(nx, ny) {
    if (!live2dModel || !live2dModel.internalModel) {
      return;
    }

    const coreModel = live2dModel.internalModel.coreModel;
    if (!coreModel || typeof coreModel.setParameterValueById !== "function") {
      return;
    }

    const fx = nx * 2 - 1;
    const fy = (1 - ny) * 2 - 1;

    const values = {
      ParamEyeBallX: fx,
      ParamEyeBallY: fy,
      ParamAngleX: fx * 25,
      ParamAngleY: fy * 18,
      ParamBodyAngleX: fx * 10,
    };

    for (const [paramId, value] of Object.entries(values)) {
      try {
        coreModel.setParameterValueById(paramId, value);
      } catch (_e) {
        // Missing parameter IDs are expected on some models.
      }
    }
  }

  function chooseRandomExpression() {
    if (!live2dModel || !live2dModel.internalModel) {
      return;
    }

    const settings = live2dModel.internalModel.settings;
    const expressions = settings && Array.isArray(settings.expressions) ? settings.expressions : [];
    if (!expressions.length) {
      return;
    }

    const idx = Math.floor(Math.random() * expressions.length);
    const expression = expressions[idx];
    if (!expression || !expression.name) {
      return;
    }

    try {
      if (typeof live2dModel.expression === "function") {
        live2dModel.expression(expression.name);
      }
    } catch (_e) {
      // No-op.
    }
  }

  async function loadModel(modelUrl, options) {
    const opts = options || {};
    shouldFollowCursor = opts.followCursor !== false;
    modelScale = Number.isFinite(opts.modelScale) ? opts.modelScale : 1.0;

    if (!PIXI.live2d || !PIXI.live2d.Live2DModel) {
      setStatus("Live2D运行库加载失败");
      return false;
    }

    try {
      setStatus("正在加载Live2D模型...");
      if (live2dModel) {
        app.stage.removeChild(live2dModel);
        live2dModel.destroy();
        live2dModel = null;
      }

      live2dModel = await PIXI.live2d.Live2DModel.from(modelUrl);
      app.stage.addChild(live2dModel);

      applyScale();
      centerModelBottom();
      setStatus("");

      safePlayMotion([opts.idleGroup || "Idle", "idle", "Default"]);
      return true;
    } catch (error) {
      setStatus("模型加载失败，请检查路径和资源");
      console.error("Live2D load failed:", error);
      return false;
    }
  }

  function setPointer(normalizedX, normalizedY) {
    if (!live2dModel || !shouldFollowCursor) {
      return;
    }

    const x = Math.max(0, Math.min(1, normalizedX));
    const y = Math.max(0, Math.min(1, normalizedY));
    targetGazeX = x;
    targetGazeY = y;

    try {
      if (typeof live2dModel.focus === "function") {
        const fx = x * 2 - 1;
        const fy = (1 - y) * 2 - 1;
        live2dModel.focus(fx, fy);
      } else {
        applyGazeFallback(x, y);
      }
    } catch (_e) {
      applyGazeFallback(x, y);
    }
  }

  function tapAt(normalizedX, normalizedY) {
    if (!live2dModel) {
      return;
    }

    const x = Math.max(0, Math.min(1, normalizedX));
    const y = Math.max(0, Math.min(1, normalizedY));

    const worldX = app.renderer.width * x;
    const worldY = app.renderer.height * y;

    const headCandidates = getHitAreaCandidates("head");
    const bodyCandidates = getHitAreaCandidates("body");
    const touchedHead = hitTestAny(headCandidates, worldX, worldY);
    const touchedBody = hitTestAny(bodyCandidates, worldX, worldY);

    try {
      if (typeof live2dModel.tap === "function") {
        live2dModel.tap(worldX, worldY);
      }
    } catch (_e) {
      // No-op.
    }

    if (touchedHead) {
      chooseRandomExpression();
      safePlayMotion(["TapHead", "Head", "touch_head", "Idle"]);
      return;
    }

    if (touchedBody) {
      safePlayMotion(["TapBody", "Body", "touch_body", "Idle"]);
      return;
    }

    safePlayMotion(["Body", "TapBody", "touch_body", "Idle"]);
  }

  function onResize() {
    if (!live2dModel) {
      return;
    }
    applyScale();
    centerModelBottom();
  }

  window.addEventListener("resize", onResize);

  app.ticker.add(() => {
    if (!live2dModel || !shouldFollowCursor) {
      return;
    }
    applyGazeFallback(targetGazeX, targetGazeY);
  });

  window.petLive2d = {
    loadModel,
    setPointer,
    tapAt,
    setFollowCursor(enabled) {
      shouldFollowCursor = Boolean(enabled);
    },
    playMotion(group) {
      safePlayMotion([group]);
    },
    setExpression(name) {
      if (!live2dModel || typeof live2dModel.expression !== "function") {
        return;
      }
      try {
        live2dModel.expression(name);
      } catch (_e) {
        // No-op.
      }
    },
  };
})();

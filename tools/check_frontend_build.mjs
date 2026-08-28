/**
 * 前端 build() 的冒烟测试 —— 拿真实的 /api/page 数据在 node 里跑一遍。
 *
 * 为什么需要：templates/index.html 的 JS 只有浏览器会执行，任何 ReferenceError
 * 都表现为界面永远停在「载入中…」，而服务端日志全是 200 —— 从后端完全看不出来。
 * 实际踩过：判据可视化引用了 buildLinetypeIndex() 里的局部变量 lt，build() 一抛
 * 整个视图就卡住。`node --check` 只验语法，抓不到这种。
 *
 * 做法：把 <script> 抽出来，在一个最小 DOM 桩里 eval，然后用真实页面数据调
 * show()/build()/renderList()。桩只实现被用到的那些 API；缺什么就补什么 ——
 * 补的过程本身就是在记录前端到底依赖了哪些浏览器能力。
 *
 *   node tools/check_frontend_build.mjs [pageJsonUrl ...]
 *   默认拉 gladstone_dog_park 的 P2/P4。
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const HTML = path.join(ROOT, "templates", "index.html");
const BASE = process.env.FENCE_LITE_BASE || "http://127.0.0.1:5062";

const DEFAULT_PAGES = [
  "/api/page/gladstone_dog_park/2",
  "/api/page/gladstone_dog_park/4",
];

const makeNode = (tag) => {
  const node = {
    tagName: String(tag || "div").toUpperCase(),
    nodeName: String(tag || "div").toUpperCase(),
    children: [],
    dataset: {},
    style: {},
    attributes: {},
    classList: {
      _set: new Set(),
      add(...v) { v.forEach((x) => this._set.add(x)); },
      remove(...v) { v.forEach((x) => this._set.delete(x)); },
      toggle(v, on) { if (on === undefined) { this._set.has(v) ? this._set.delete(v) : this._set.add(v); } else if (on) { this._set.add(v); } else { this._set.delete(v); } },
      contains(v) { return this._set.has(v); },
    },
    appendChild(child) { this.children.push(child); return child; },
    removeChild(child) { this.children = this.children.filter((c) => c !== child); },
    remove() {},
    setAttribute(k, v) { this.attributes[k] = v; },
    getAttribute(k) { return this.attributes[k]; },
    removeAttribute(k) { delete this.attributes[k]; },
    addEventListener() {},
    removeEventListener() {},
    querySelector(sel) {
      this._qs = this._qs || new Map();
      const key = String(sel || "*");
      // input 要造成 <input>，否则 .checked 语义看着像是能用其实不是同一个东西
      if (!this._qs.has(key)) this._qs.set(key, makeNode(key.includes("input") ? "input" : "div"));
      return this._qs.get(key);
    },
    querySelectorAll() { return []; },
    getBoundingClientRect() { return { x: 0, y: 0, width: 800, height: 600, top: 0, left: 0, right: 800, bottom: 600 }; },
    getBBox() { return { x: 0, y: 0, width: 10, height: 10 }; },
    scrollIntoView() {},
    focus() {},
    click() {},
    insertBefore(child) { this.children.push(child); return child; },
    closest() { return null; },
    contains() { return false; },
    get firstChild() { return this.children[0] || null; },
    get textContent() { return this._text || ""; },
    set textContent(v) { this._text = v; },
    get innerHTML() { return this._html || ""; },
    set innerHTML(v) { this._html = v; this.children = []; },
    get value() { return this._value || ""; },
    set value(v) { this._value = v; },
    // 页面大量用 el.className='a b c' 而不是 classList.add，桩里必须同步，
    // 否则 classList.contains() 恒为假 —— 检查看着在跑，其实什么都没验。
    // <select>.options —— 侧栏的下拉靠它同步选中值，桩里缺了会抛 TypeError
    get options() { return this.children.filter((c) => c.tagName === "OPTION"); },
    get className() { return [...this.classList._set].join(" "); },
    set className(v) {
      this.classList._set = new Set(String(v || "").split(/\s+/).filter(Boolean));
    },
    checked: false,
    scrollLeft: 0,
    scrollTop: 0,
    clientWidth: 800,
    clientHeight: 600,
    offsetWidth: 800,
    offsetHeight: 600,
  };
  return node;
};

const registry = new Map();
const byId = (id) => {
  if (!registry.has(id)) registry.set(id, makeNode("div"));
  return registry.get(id);
};

const document = {
  readyState: "complete",
  documentElement: makeNode("html"),
  body: makeNode("body"),
  createElement: (t) => makeNode(t),
  createElementNS: (_ns, t) => makeNode(t),
  createTextNode: (t) => ({ textContent: t }),
  getElementById: byId,
  querySelector: () => makeNode("div"),
  querySelectorAll: () => [],
  addEventListener() {},
  createDocumentFragment: () => makeNode("fragment"),
};

const errors = [];
const sandbox = {
  document,
  console,
  Math,
  JSON,
  Date,
  Number,
  String,
  Boolean,
  Array,
  Object,
  Set,
  Map,
  WeakMap,
  Promise,
  Error,
  RegExp,
  parseInt,
  parseFloat,
  isNaN,
  isFinite,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 0,
  clearTimeout: () => {},
  setInterval: () => 0,
  clearInterval: () => {},
  requestAnimationFrame: (fn) => { try { fn(0); } catch (e) { errors.push(e); } return 0; },
  cancelAnimationFrame: () => {},
  // 必须是**真** fetch：页面的 api() 是真实取数，桩返回 {} 的话
  // ensureAllLinetypes() 永远走到 error 分支，真实点击路径就测不到了。
  fetch: async (url, opt) => {
    const full = String(url).startsWith("http") ? String(url) : BASE + String(url);
    return globalThis.fetch(full, opt);
  },
  location: { href: BASE + "/", origin: BASE, pathname: "/", search: "" },
  history: { replaceState() {}, pushState() {} },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  matchMedia: () => ({ matches: false, addEventListener() {}, addListener() {} }),
  IntersectionObserver: class { observe() {} unobserve() {} disconnect() {} },
  ResizeObserver: class { observe() {} unobserve() {} disconnect() {} },
  navigator: { userAgent: "node", clipboard: { writeText: async () => {} } },
  alert() {},
  // 页面在模块顶层就挂了 window 级监听（拖拽平移 / 键盘），桩里必须有，
  // 否则加载脚本这一步就抛 —— 那会掩盖真正要测的 build()。
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent: () => true,
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  devicePixelRatio: 1,
  innerWidth: 1280,
  innerHeight: 800,
  scrollTo() {},
  confirm: () => true,
  URL,
  URLSearchParams,
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.self = sandbox;

const html = readFileSync(HTML, "utf8");
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]
  .map((m) => m[1]);
if (!scripts.length) {
  console.error("no <script> found in index.html");
  process.exit(2);
}

const context = vm.createContext(sandbox);
try {
  vm.runInContext(scripts.join("\n"), context, { filename: "index.html" });
} catch (error) {
  console.error("加载脚本时抛出:", error && error.stack ? error.stack.split("\n")[0] : error);
  process.exit(1);
}

// 桩里的 #bar 不是从 HTML 建的，默认没有 class；真 markup 是
// <div id="bar" class="tucked">。不摆成一致的话页面自己在加载期跑的
// loadOverview() 会以 DEV()==true 渲染一遍画廊，后面的模型名检查抓到的
// 是桩的缺陷、不是页面的问题。
byId("bar").classList.add("tucked");

const urls = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULT_PAGES;
let failures = 0;

// ---- 侧栏改版的执行检查 ----------------------------------------------------
// 下拉选 PDF / 点页面收起预览 这套只在浏览器里跑，写错就是「点了没反应」或
// 整页卡在载入中，服务端看不出来。这里真的调 renderProjectSelect / setSideMode。
{
  const problems = [];
  try {
    const overview = await (await fetch(`${BASE}/api/overview`)).json();
    context.__ov = overview;
    vm.runInContext("OV = __ov; CUR = null;", context, { filename: "inject-ov" });
    if (typeof context.renderProjectSelect !== "function") {
      problems.push("没有 renderProjectSelect()");
    } else {
      context.renderProjectSelect();
      const sel = context.document.getElementById("projectSelect");
      const opts = (sel.children || []).filter((c) => c.tagName === "OPTION");
      const parents = overview.filter((p) => !p.variant_of).length;
      // 首项是占位项，所以是 parents + 1
      if (opts.length !== parents + 1)
        problems.push(`下拉 ${opts.length} 项 != 项目数+1 (${parents + 1})`);
      // 首项是占位项，value 本来就该是空；只要求真正的项目项都有 slug
      const empty = opts.slice(1).filter((o) => !o.value);
      if (empty.length) problems.push(`${empty.length} 个项目项没有 value`);
      const slugs = new Set(opts.slice(1).map((o) => o.value));
      const missing = overview.filter((p) => !p.variant_of && !slugs.has(p.slug));
      if (missing.length)
        problems.push(`下拉漏了 ${missing.length} 个项目: ${missing.slice(0, 3).map((p) => p.slug)}`);
    }
    if (typeof context.setSideMode !== "function") {
      problems.push("没有 setSideMode()");
    } else {
      const side = context.document.getElementById("side");
      context.setSideMode("scope");
      if (!side.classList.contains("scope-mode") || side.classList.contains("pages-mode"))
        problems.push("setSideMode('scope') 没切到 scope-mode");
      context.setSideMode("pages");
      if (!side.classList.contains("pages-mode") || side.classList.contains("scope-mode"))
        problems.push("setSideMode('pages') 没切回 pages-mode");
    }
    if (typeof context.openProject !== "function") problems.push("没有 openProject()");
    if (typeof context.toggleProjectDrawer !== "function")
      problems.push("没有 toggleProjectDrawer()");
    else context.toggleProjectDrawer(false);   // 会碰 bProjectMenu，验它没引用已删元素
  } catch (error) {
    const first = error && error.stack ? error.stack.split("\n")[0] : String(error);
    problems.push(`抛出 -> ${first}`);
  }
  if (problems.length) {
    console.log(`  FAIL 侧栏: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   侧栏: 下拉项数正确, setSideMode 两向切换正常, "
      + "openProject/toggleProjectDrawer 可调用");
  }
}

for (const rel of urls) {
  const url = rel.startsWith("http") ? rel : BASE + rel;
  let payload;
  try {
    const response = await fetch(url);
    payload = await response.json();
  } catch (error) {
    console.log(`  SKIP ${rel}: 拉不到 (${error.message})`);
    continue;
  }
  if (payload.error || payload.pending) {
    console.log(`  SKIP ${rel}: ${payload.error || "pending"}`);
    continue;
  }
  // 脚本里是 `let PAGE`，词法绑定不挂在 global 对象上，直接赋 context.PAGE
  // 改不到它。要在同一个 context 里执行赋值语句才能命中那个绑定。
  context.__payload = payload;
  vm.runInContext("PAGE = __payload;", context, { filename: "inject" });
  const steps = ["makeScopeModel", "build", "renderList", "renderSteps"];
  let ok = true;
  for (const fn of steps) {
    if (typeof context[fn] !== "function") {
      console.log(`  ERR  ${rel}: 没有 ${fn}()`);
      ok = false;
      failures += 1;
      break;
    }
    try {
      context[fn]();
    } catch (error) {
      const first = error && error.stack ? error.stack.split("\n")[0] : String(error);
      console.log(`  FAIL ${rel}: ${fn}() 抛出 -> ${first}`);
      ok = false;
      failures += 1;
      break;
    }
  }
  if (ok && !process.env.SKIP_ALL_LT) {
    // ---- 全部线型（调试层）：真的执行一遍，别只验语法 ----
    // rel 可能是相对路径，也可能是完整 URL（Git Bash 会把 /api/... 翻译成
    // 本地路径，所以调用时常传完整 URL）。两种都要认。
    const m = /\/api\/page\/([^/]+)\/(\d+)\/?$/.exec(rel);
    if (m) {
      const slug = m[1], sheet = m[2];
      let all = null;
      try {
        const r = await fetch(`${BASE}/api/linetypes_all/${slug}/${sheet}`);
        all = await r.json();
      } catch (error) {
        console.log(`  SKIP ${rel} 调试层: 拉不到 (${error.message})`);
      }
      if (all && all.state === "ok") {
        context.__all = all;
        context.__slug = slug;
        try {
          // CUR / ALL_* 都是 let 绑定，只能在同一个 context 里赋值。
          vm.runInContext(
            "CUR = {slug: __slug, page: PAGE.page};"
            + "ALL_LT = __all; ALL_LT_KEY = allKey(); ALL_LT_STATE = 'ok';"
            + "ALL_FOCUS = null; ALL_FILTER = '';"
            + "$('tgAll').querySelector('input').checked = true;",
            context, { filename: "inject-all" });
          context.drawAllLinetypes();
          context.renderList();
          const types = all.types || [];
          // ALL_NODES 是 let 绑定，不挂在 global 对象上（同 PAGE），
          // 只能在同一个 context 里求值才能读到。
          const drawn = vm.runInContext("ALL_NODES.size", context);
          const gAll = context.document.getElementById("gAll");
          const gResid = context.document.getElementById("gResid");
          const items = (context.document.getElementById("list").children || [])
            .filter((n) => n.classList && n.classList.contains("al-item"));
          // 聚焦一个真实编号：op 最多的那个（hatch 误判的典型），再聚焦一个
          // 被绑定过的，两条路径都走到。
          const biggest = [...types].sort((a, b) => (b.op_count || 0) - (a.op_count || 0))[0];
          const bound = types.find((t) => (t.bound_by || []).length);
          if (biggest) context.focusAllType(biggest.line_type_number, true);
          if (bound) context.focusAllType(bound.line_type_number, true);
          const focus = vm.runInContext("ALL_FOCUS", context);
          const problems = [];
          if (drawn !== types.length) problems.push(`ALL_NODES ${drawn} != 类型数 ${types.length}`);
          if (!(gAll.children || []).length) problems.push("gAll 没有任何折线");
          if (all.residual && all.residual.op_count && !(gResid.children || []).length)
            problems.push("residual 有 op 但 gResid 空");
          if (items.length !== types.length) problems.push(`列表 ${items.length} 行 != 类型数 ${types.length}`);
          // 聚焦是 toggle 语义：focus===x 时再点会清掉。按点击顺序
          // [biggest, bound] 推最终值 —— 两者同一个编号时第二下把它关掉。
          const wantFocus = bound
            ? (biggest && bound.line_type_number === biggest.line_type_number
               ? null : bound.line_type_number)
            : (biggest ? biggest.line_type_number : undefined);
          if (wantFocus !== undefined && focus !== wantFocus)
            problems.push(`ALL_FOCUS=${focus} 期望 ${wantFocus}`);
          const focused = vm.runInContext(
            "(ALL_NODES.get(ALL_FOCUS)||[]).filter(n=>n.classList.contains('al-focus')).length",
            context);
          if (wantFocus != null && !focused)
            problems.push("聚焦的线型没有一个节点带 al-focus");
          if (problems.length) {
            console.log(`  FAIL ${rel} 调试层: ${problems.join("; ")}`);
            failures += 1;
          } else {
            console.log(`  OK   ${rel} 调试层: ${types.length} 型 / `
              + `${(gAll.children || []).length} 条折线 / residual `
              + `${(all.residual && all.residual.op_count) || 0} op / 列表 ${items.length} 行`);
          }
          // ---- 真实点击路径：清空状态，只勾开关，让页面自己去取 ----
          vm.runInContext(
            "ALL_LT = null; ALL_LT_KEY = ''; ALL_LT_STATE = 'idle';"
            + "ALL_FOCUS = null; ALL_NODES = new Map();"
            + "document.getElementById('gAll').innerHTML = '';"
            + "document.getElementById('gResid').innerHTML = '';"
            + "$('tgAll').querySelector('input').checked = true;",
            context, { filename: "reset-all" });
          await context.ensureAllLinetypes();
          const liveState = vm.runInContext("ALL_LT_STATE", context);
          const liveDrawn = vm.runInContext("ALL_NODES.size", context);
          const liveNodes = (context.document.getElementById("gAll").children || []).length;
          const liveItems = (context.document.getElementById("list").children || [])
            .filter((n) => n.classList && n.classList.contains("al-item")).length;
          const liveProblems = [];
          if (liveState !== "ok") liveProblems.push(`点击后 ALL_LT_STATE=${liveState}`);
          if (liveDrawn !== types.length) liveProblems.push(`点击后 ALL_NODES ${liveDrawn} != ${types.length}`);
          if (!liveNodes) liveProblems.push("点击后 gAll 空");
          if (liveItems !== types.length) liveProblems.push(`点击后列表 ${liveItems} != ${types.length}`);
          if (liveProblems.length) {
            console.log(`  FAIL ${rel} 真实点击路径: ${liveProblems.join("; ")}`);
            failures += 1;
          } else {
            console.log(`  OK   ${rel} 真实点击路径: 勾选后自动取数并画出 `
              + `${liveDrawn} 型 / ${liveNodes} 折线 / 列表 ${liveItems} 行`);
          }
        } catch (error) {
          const first = error && error.stack ? error.stack.split("\n")[0] : String(error);
          console.log(`  FAIL ${rel} 调试层: 抛出 -> ${first}`);
          failures += 1;
        }
      } else if (all) {
        console.log(`  SKIP ${rel} 调试层: state=${all.state}${all.detail ? " " + all.detail : ""}`);
      }
    }
  }
  if (ok) {
    const lt = (payload.record && payload.record.linetypes) || {};
    const shipped = (lt.line_types || []).filter((t) => t.polylines);
    console.log(`  OK   ${rel}: build() 通过  线型可见=${JSON.stringify(lt.visible || [])} `
      + `发出 ${shipped.length} 个 / ${shipped.reduce((n, t) => n + t.polylines.reduce((m, l) => m + Math.max(0, l.length - 1), 0), 0)} 段`);
  }
}

if (errors.length) {
  console.log(`  注意：requestAnimationFrame 回调里有 ${errors.length} 个异常`);
  failures += errors.length;
}

// ---- 列表里一次只能有一行高亮 ----------------------------------------------
// 同一个 callout 只要绑到了线型，就会同时出现在 FENCE TEXT & GATES 和 FENCELINE
// 两个分节里（scopeId 相同）。按 scope 匹配高亮会把两行一起点亮 —— 这里挑一个
// 真的出现在两节的 callout，断言只有它自己那一行是 on。
{
  const problems = [];
  try {
    const ltRows = vm.runInContext(
      "JSON.stringify((LT_INDEX.rows||[]).map(r=>r.scopeId))", context);
    const scopeIds = vm.runInContext(
      "JSON.stringify(SCOPE_GROUPS.map(g=>g.id))", context);
    const inBoth = JSON.parse(ltRows).filter((id) => JSON.parse(scopeIds).includes(id));
    if (!inBoth.length) {
      console.log("  SKIP 单行高亮: 这一页没有同时出现在两个分节的 callout");
    } else {
      const target = inBoth[0];
      // 点 FENCELINE 那一行
      context.__target = target;
      vm.runInContext(
        "selectedScope = __target; selectedRow = rowKey('FENCELINE', __target);",
        context, { filename: "inject-row" });
      context.renderList();
      const rows = (context.document.getElementById("list").children || [])
        .filter((n) => n.classList && n.classList.contains("scope-item"));
      const on = rows.filter((n) => n.classList.contains("on"));
      const dupes = rows.filter((n) => n.dataset && n.dataset.scope === target);
      if (dupes.length < 2)
        problems.push(`挑中的 callout 只在 ${dupes.length} 个分节里，测不到重复高亮`);
      if (on.length !== 1)
        problems.push(`${on.length} 行是 on（应当只有 1 行）`);
      if (on.length === 1 && on[0].dataset.row !== `FENCELINE|${target}`)
        problems.push(`亮的是 ${on[0].dataset.row}，不是点的那一行`);
    }
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 120)}`);
  }
  if (problems.length) {
    console.log(`  FAIL 单行高亮: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   单行高亮: 同一 callout 跨两节时只有被点的那一行是 on");
  }
}

// ---- 客户视图里不许出现模型名 ----------------------------------------------
// 默认 #bar 带 tucked（图层控制条收起），DEV() 为假，模型相关的东西都不该渲染。
// 这里在渲染完成后扫一遍 DOM，找有没有模型显示名漏出来。
{
  const problems = [];
  try {
    const models = await (await fetch(`${BASE}/api/models`)).json();
    const names = (models.models || []).map((m) => m.display).filter(Boolean);
    const bar = context.document.getElementById("bar");
    // 桩里的 #bar 不是从 HTML 建的，手动把默认状态摆上
    bar.classList.add("tucked");
    // 画廊由 loadOverview() 渲染，必须真的再跑一次才能验到「⋯ 关掉后
    // 重渲染不带模型名」这条路径
    if (typeof context.loadOverview === "function") await context.loadOverview();
    if (typeof context.renderModelSwitch === "function") context.renderModelSwitch();
    if (typeof context.renderJobs === "function") context.renderJobs();
    if (typeof context.renderList === "function") context.renderList();
    const seen = [];
    const walk = (node, depth) => {
      if (!node || depth > 40) return;
      for (const key of ["_html", "_text", "_value"]) {
        const v = node[key];
        if (typeof v !== "string") continue;
        for (const name of names) if (v.includes(name)) seen.push(`${name} @ ${key}`);
      }
      for (const child of (node.children || [])) walk(child, depth + 1);
    };
    for (const node of registry.values()) walk(node, 0);
    if (seen.length) problems.push(`${seen.length} 处模型名: ${[...new Set(seen)].slice(0, 4).join(", ")}`);
    if (!names.length) problems.push("/api/models 没返回 display 名，这项检查等于没做");
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 120)}`);
  }
  if (problems.length) {
    console.log(`  FAIL 模型名泄漏: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   模型名: 客户视图（⋯ 收起）里零模型字样");
  }
}

// ---- 界面上不许出现中文 ----------------------------------------------------
// 独立于源码扫描的检查：不看文件里有没有中文，而是把渲染过程**真正写进 DOM**
// 的每一段文本（innerHTML / textContent / value / 属性）收集起来再扫。源码扫描
// 分不清「注释里的中文」和「文案里的中文」，这个分得清 —— 注释永远不进 DOM。
{
  const CJK = /[一-鿿　-〿！-｠]/;
  const seen = [];
  const walk = (node, depth) => {
    if (!node || depth > 40) return;
    // 报中文**附近**的片段，不是整段的开头 —— 卡片 HTML 动辄上千字符，
    // 打头 120 字看不到中文在哪，等于报了个位置不明的失败。
    const near = (v) => {
      const out = [];
      const re = /[一-鿿　-〿！-｠]+/g;
      let m;
      while ((m = re.exec(v)) && out.length < 3) {
        out.push(v.slice(Math.max(0, m.index - 30), m.index + m[0].length + 20));
      }
      return out.join("  ¶  ");
    };
    for (const key of ["_html", "_text", "_value"]) {
      const v = node[key];
      if (typeof v === "string" && CJK.test(v)) seen.push(near(v));
    }
    for (const [k, v] of Object.entries(node.attributes || {})) {
      if (typeof v === "string" && CJK.test(v)) seen.push("@" + k + "=" + v.slice(0, 100));
    }
    for (const child of (node.children || [])) walk(child, depth + 1);
  };
  for (const node of registry.values()) walk(node, 0);
  if (seen.length) {
    console.log(`  FAIL 界面中文: ${seen.length} 处写进了 DOM`);
    for (const t of seen.slice(0, 8)) console.log(`       ${t}`);
    failures += 1;
  } else {
    console.log(`  OK   界面中文: 渲染写进 DOM 的文本零中文（扫了 ${registry.size} 个节点树）`);
  }
}
console.log(failures ? `\nFRONTEND FAILED: ${failures} 处` : "\nFRONTEND OK");
process.exit(failures ? 1 : 0);

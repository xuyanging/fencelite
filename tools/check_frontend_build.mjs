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
    setAttribute(k, v) {
      this.attributes[k] = v;
      if (String(k).toLowerCase() === "id") this.id = String(v || "");
      if (String(k).toLowerCase() === "class") this.className = String(v || "");
    },
    getAttribute(k) { return this.attributes[k]; },
    removeAttribute(k) {
      delete this.attributes[k];
      if (String(k).toLowerCase() === "id") this.id = "";
    },
    addEventListener(type, fn) {
      this._listeners = this._listeners || new Map();
      const key = String(type || "");
      const list = this._listeners.get(key) || [];
      if (typeof fn === "function" && !list.includes(fn)) list.push(fn);
      this._listeners.set(key, list);
    },
    removeEventListener(type, fn) {
      if (!this._listeners) return;
      const key = String(type || "");
      this._listeners.set(key,
        (this._listeners.get(key) || []).filter((item) => item !== fn));
    },
    dispatchEvent(event) {
      const e = event && typeof event === "object" ? event : {type: String(event || "")};
      e.type = String(e.type || "");
      if (!e.target) e.target = this;
      e.currentTarget = this;
      if (!e.preventDefault) e.preventDefault = () => { e.defaultPrevented = true; };
      if (!e.stopPropagation) e.stopPropagation = () => {};
      const propertyHandler = this[`on${e.type}`];
      if (typeof propertyHandler === "function") propertyHandler.call(this, e);
      for (const fn of [...((this._listeners && this._listeners.get(e.type)) || [])])
        fn.call(this, e);
      return !e.defaultPrevented;
    },
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
    set value(v) {
      this._value = v;
      // A real <input type=file> clears FileList when value is reset.  Mirror
      // that for batch-upload execution tests using lightweight fake files.
      if (v === "" && Array.isArray(this.files)) this.files = [];
    },
    // 页面大量用 el.className='a b c' 而不是 classList.add，桩里必须同步，
    // 否则 classList.contains() 恒为假 —— 检查看着在跑，其实什么都没验。
    // <select>.options —— 侧栏的下拉靠它同步选中值，桩里缺了会抛 TypeError
    get options() { return this.children.filter((c) => c.tagName === "OPTION"); },
    get className() { return [...this.classList._set].join(" "); },
    set className(v) {
      this.classList._set = new Set(String(v || "").split(/\s+/).filter(Boolean));
    },
    // Browser DOM keeps the id property and getElementById registry in sync.
    // build() creates SVG groups with `g.id=id`; without this setter the test
    // would inspect a different empty stub node and could report false results.
    get id() { return this._id || ""; },
    set id(v) {
      const next = String(v || "");
      if (this._id && registry.get(this._id) === this) registry.delete(this._id);
      this._id = next;
      if (next) registry.set(next, this);
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
  if (!registry.has(id)) {
    const node = makeNode("div");
    node.id = id;
  }
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
const sessionValues = new Map();
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
  sessionStorage: {
    getItem: (key) => sessionValues.has(String(key)) ? sessionValues.get(String(key)) : null,
    setItem: (key, value) => sessionValues.set(String(key), String(value)),
    removeItem: (key) => sessionValues.delete(String(key)),
  },
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
  // index.html gives every request an AbortController-backed deadline.  VM
  // contexts do not inherit Node's web globals automatically, so expose the
  // real implementations or the smoke test dies before build() is reached.
  AbortController: globalThis.AbortController,
  AbortSignal: globalThis.AbortSignal,
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

// ---- 主画布缩放：鼠标慢步进，触控板按小 delta 平滑缩放 ---------------------
{
  const problems = [];
  try {
    if (typeof context.wheelZoomFactor !== "function") {
      problems.push("没有 wheelZoomFactor()");
    } else {
      const mouseIn = context.wheelZoomFactor({ deltaY: -100, deltaMode: 0 });
      const mouseOut = context.wheelZoomFactor({ deltaY: 100, deltaMode: 0 });
      const trackpadIn = context.wheelZoomFactor({ deltaY: -5, deltaMode: 0 });
      const lineIn = context.wheelZoomFactor({ deltaY: -3, deltaMode: 1 });
      if (!(mouseIn > 1 && mouseIn < 1.10))
        problems.push(`鼠标放大倍率异常 (${mouseIn})`);
      if (!(mouseOut < 1 && mouseOut > 0.90))
        problems.push(`鼠标缩小倍率异常 (${mouseOut})`);
      if (!(trackpadIn > 1 && trackpadIn < 1.01))
        problems.push(`触控板小步进异常 (${trackpadIn})`);
      if (!(lineIn > 1 && lineIn < 1.05))
        problems.push(`行单位滚轮换算异常 (${lineIn})`);
      if (Math.abs(mouseIn * mouseOut - 1) > 1e-9)
        problems.push("同量正反缩放不对称");
    }
  } catch (error) {
    const first = error && error.stack ? error.stack.split("\n")[0] : String(error);
    problems.push(`抛出 -> ${first}`);
  }
  if (problems.length) {
    console.log(`  FAIL 缩放: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   缩放: 鼠标单步小于 10%，触控板按 delta 平滑缩放");
  }
}

// ---- 多 PDF 上传：多选、逐份提交、单份失败隔离 -----------------------------
{
  const problems = [];
  const source = scripts.join("\n");
  const inputTag = (html.match(/<input\b[^>]*\bid=["']upFile["'][^>]*>/i) || [""])[0];
  const folderTag = (html.match(/<input\b[^>]*\bid=["']upFolder["'][^>]*>/i) || [""])[0];
  const folderButtonTag = (html.match(
    /<button\b[^>]*\bid=["']upPickFolder["'][^>]*>[\s\S]*?<\/button>/i) || [""])[0];
  const savedPollState = vm.runInContext(
    "({jobs:JOBS,poll:POLL,timer:POLL_TIMER,inflight:POLL_INFLIGHT,seq:POLL_SEQ})",
    context);
  const oldUploadPdfFile = context.uploadPdfFile;
  const oldLoadOverview = context.loadOverview;
  const upFile = context.document.getElementById("upFile");
  const upFolder = context.document.getElementById("upFolder");
  const oldUpFileClick = upFile.click;
  const oldUpFolderClick = upFolder.click;
  const upRun = context.document.getElementById("upRun");
  const upName = context.document.getElementById("upName");
  const upErr = context.document.getElementById("upErr");
  const modal = context.document.getElementById("modal");
  try {
    if (!/\bmultiple\b/i.test(inputTag)) problems.push("#upFile 没有 multiple");
    if (!/\.pdf|application\/pdf/i.test(inputTag)) problems.push("#upFile 没有限定 PDF");
    for (const attr of ["webkitdirectory", "directory", "multiple"])
      if (!new RegExp(`\\b${attr}\\b`, "i").test(folderTag))
        problems.push(`#upFolder 缺少 ${attr}`);
    if (!/\.pdf|application\/pdf/i.test(folderTag))
      problems.push("#upFolder 没有限定 PDF");
    if (!/Choose Folder/i.test(folderButtonTag))
      problems.push("HTML 缺少 Choose Folder 按钮");
    if (typeof context.enqueuePdfUploads !== "function")
      problems.push("没有 enqueuePdfUploads()");

    let filePickerClicks = 0, folderPickerClicks = 0;
    upFile.click = () => { filePickerClicks += 1; };
    upFolder.click = () => { folderPickerClicks += 1; };
    context.document.getElementById("upPick").onclick();
    context.document.getElementById("upPickFolder").onclick();
    if (filePickerClicks !== 1) problems.push("Choose PDFs 没有触发 #upFile");
    if (folderPickerClicks !== 1) problems.push("Choose Folder 没有触发 #upFolder");

    const calls = [];
    let overviewCalls = 0;
    context.__batchUploadMock = async (file, target) => {
      const relative = String(file.webkitRelativePath || file.relativePath || file.name);
      calls.push({name:file.name,path:relative,target});
      if (file.name === "broken.pdf")
        return {ok:false,status:400,body:{error:"not a PDF file"}};
      const slug = relative.replace(/\.pdf$/i, "").replace(/[^a-z0-9]+/gi, "_")
        .replace(/^_+|_+$/g, "").toLowerCase();
      return {ok:true,status:200,body:{slug,
        job:{slug,stage:"queued",done:false}}};
    };
    let overviewThrows = false;
    context.__batchOverviewMock = async () => {
      overviewCalls += 1;
      if (overviewThrows) throw new Error("gallery render failed");
    };
    vm.runInContext(
      "uploadPdfFile = __batchUploadMock; loadOverview = __batchOverviewMock; "
      + "JOBS = {}; POLL = {}; POLL_TIMER = null; POLL_INFLIGHT = false;",
      context, { filename: "batch-upload-mocks" });

    // A recursive directory FileList can contain arbitrary depths and files
    // ignored by accept=.pdf. Same basenames in different subdirectories are
    // distinct uploads and must also have distinct idempotency fingerprints.
    const folderA = {name:"same.pdf",type:"application/pdf",size:101,lastModified:7,
      webkitRelativePath:"root/area_a/same.pdf"};
    const folderB = {name:"same.pdf",type:"application/pdf",size:101,lastModified:7,
      webkitRelativePath:"root/area_b/deep/same.pdf"};
    const folderUpper = {name:"UPPER.PDF",type:"",size:202,lastModified:8,
      webkitRelativePath:"root/area_b/deep/UPPER.PDF"};
    const folderIgnored = {name:"notes.txt",type:"text/plain",size:303,lastModified:9,
      webkitRelativePath:"root/area_b/deep/notes.txt"};
    upFolder.files = [folderA,folderB,folderUpper,folderIgnored];
    upFolder.onchange();
    const selectedFolder = vm.runInContext(
      "selectedPdfFiles().map(f=>uploadRelativePath(f))", context);
    if (Array.from(selectedFolder).join("|")
        !== "root/area_a/same.pdf|root/area_b/deep/same.pdf|root/area_b/deep/UPPER.PDF")
      problems.push(`目录递归 PDF 过滤/顺序错误: ${Array.from(selectedFolder).join("|")}`);
    if (!upName.textContent.includes("3 PDFs selected")
        || !upName.title.includes("root/area_b/deep/UPPER.PDF")
        || !upErr.textContent.includes("1 non-PDF file was ignored"))
      problems.push("目录选择没有显示层级路径、大小写 PDF 数量或非 PDF 过滤摘要");
    context.__folderTokenA = folderA;
    context.__folderTokenB = folderB;
    const fingerprints = vm.runInContext(
      "[uploadFileFingerprint(__folderTokenA,'folder target'),"
      + "uploadFileFingerprint(__folderTokenB,'folder target')]", context);
    const folderTokenA = context.uploadTokenFor(folderA,"folder target");
    const folderTokenB = context.uploadTokenFor(folderB,"folder target");
    if (fingerprints[0] === fingerprints[1] || folderTokenA === folderTokenB)
      problems.push("同名不同目录 PDF 错误复用 fingerprint/token");

    // Cancelling the native directory picker dispatches an empty change in
    // some browsers. It must be a no-op, preserving both the recursive set and
    // an independent Choose PDFs selection.
    const loosePdf = {name:"loose.pdf",type:"application/pdf",size:404,lastModified:10};
    upFile.files = [loosePdf];
    upFile.onchange();
    upFolder.files = [];
    upFolder.onchange();
    const afterFolderCancel = vm.runInContext(
      "selectedPdfFiles().map(f=>uploadRelativePath(f))", context);
    if (Array.from(afterFolderCancel).join("|")
        !== "loose.pdf|root/area_a/same.pdf|root/area_b/deep/same.pdf|root/area_b/deep/UPPER.PDF"
        || upRun.disabled)
      problems.push("取消 Choose Folder 后改动了文件/目录选择或禁用了 Start");
    upFile.files = [];
    upFile.onchange();
    const folderOnlyAfterFileReset = vm.runInContext(
      "selectedPdfFiles().map(f=>uploadRelativePath(f))", context);
    if (Array.from(folderOnlyAfterFileReset).join("|") !== Array.from(selectedFolder).join("|"))
      problems.push("清空 Choose PDFs 入口时误清空了目录选择");

    modal.style.display = "flex";
    context.document.getElementById("upTarget").value = "folder target";
    await upRun.onclick();
    const folderCalls = calls.map(x=>x.path).join("|");
    if (folderCalls
        !== "root/area_a/same.pdf|root/area_b/deep/same.pdf|root/area_b/deep/UPPER.PDF")
      problems.push(`目录 PDF 没有逐份排队或同名文件被覆盖: ${folderCalls}`);
    const folderQueued = vm.runInContext("Object.keys(JOBS).sort()", context);
    if (Array.from(folderQueued).join("|")
        !== "root_area_a_same|root_area_b_deep_same|root_area_b_deep_upper")
      problems.push(`同名不同目录 PDF 没有形成三个任务: ${Array.from(folderQueued).join("|")}`);
    if (overviewCalls !== 1) problems.push(`目录批次 overview 刷新 ${overviewCalls} 次，不是 1 次`);
    const resetState = vm.runInContext(
      "({selected:selectedPdfFiles().length,file:UPLOAD_FILE_SELECTION.length,"
      + "folder:UPLOAD_FOLDER_SELECTION.length})", context);
    if (resetState.selected || resetState.file || resetState.folder
        || (upFile.files || []).length || (upFolder.files || []).length)
      problems.push("目录批次完成后没有同时 reset 两个上传入口");

    calls.length = 0;
    overviewCalls = 0;
    vm.runInContext("JOBS = {}; POLL = {}; POLL_TIMER = null; POLL_INFLIGHT = false;",
      context, {filename:"batch-upload-after-folder"});

    upFile.files = [{name:"alpha.pdf"},{name:"broken.pdf"},{name:"charlie.pdf"}];
    upFile.onchange();
    if (!upName.textContent.includes("3 PDFs selected")
        || !upName.classList.contains("has"))
      problems.push("多选后没有显示文件数");
    if (upRun.disabled || !upRun.textContent.includes("3 PDFs"))
      problems.push("多选后 Start 按钮状态/数量不对");
    modal.style.display = "flex";
    context.document.getElementById("upTarget").value = "shared target";
    const partialRun = upRun.onclick();
    context.document.getElementById("upCancel").onclick(); // Hide while active
    await partialRun;
    if (calls.map(x=>x.name).join(",") !== "alpha.pdf,broken.pdf,charlie.pdf")
      problems.push("没有按选择顺序逐份提交，或失败后未继续");
    if (calls.some(x=>x.target !== "shared target"))
      problems.push("批内 PDF 没有共用 detection target");
    const queued = vm.runInContext("Object.keys(JOBS).sort()", context);
    if (Array.from(queued).join(",") !== "alpha,charlie")
      problems.push("成功/失败文件进入任务队列的范围不对");
    if (overviewCalls !== 1) problems.push(`overview 刷新 ${overviewCalls} 次，不是 1 次`);
    if (modal.style.display !== "none"
        || !context.document.getElementById("bUp").textContent.includes("1 upload failed"))
      problems.push("隐藏后部分失败没有在顶栏保留可找回的摘要");
    await context.document.getElementById("bUp").onclick();
    if (modal.style.display !== "flex" || !upErr.textContent.includes("broken.pdf")
        || !upErr.textContent.includes("not a PDF file")
        || !upErr.classList.contains("bad"))
      problems.push("部分失败摘要无法重新打开或缺少文件名/原因");
    if ((upFile.files || []).length || !upRun.disabled)
      problems.push("部分失败后没有清空选择，可能重复提交成功文件");

    // 全成功批次即使 overview 渲染异常也必须收口，不能留下可重复提交的 FileList。
    calls.length = 0;
    overviewCalls = 0;
    overviewThrows = true;
    let releaseOverview = null;
    context.__batchOverviewMock = async () => {
      overviewCalls += 1;
      await new Promise((resolve) => { releaseOverview = resolve; });
      if (overviewThrows) throw new Error("gallery render failed");
    };
    vm.runInContext("loadOverview = __batchOverviewMock", context,
      { filename: "batch-overview-deferred" });
    vm.runInContext(
      "JOBS = {}; POLL = {}; POLL_TIMER = null; POLL_INFLIGHT = false;",
      context, { filename: "batch-upload-success" });
    upFile.files = Array.from({length:30}, (_,i) => ({
      name:`bulk_${String(i+1).padStart(2,"0")}.pdf`
    }));
    upFile.onchange();
    if (!upName.textContent.includes("30 PDFs selected")
        || !upRun.textContent.includes("30 PDFs"))
      problems.push("30 份选择没有正确显示批次数量");
    modal.style.display = "flex";
    const successRun = upRun.onclick();
    for (let i = 0; i < 200 && !releaseOverview; i += 1) await Promise.resolve();
    if (!releaseOverview) {
      problems.push("成功批次没有进入 overview settling 阶段");
    } else {
      if (!context.document.getElementById("upTarget").disabled
          || context.document.getElementById("bUp").textContent !== "Uploading…")
        problems.push("overview settling 时过早解除 busy，可交错启动新批次");
      modal.style.display = "none";
      await context.document.getElementById("bUp").onclick();
      if (modal.style.display !== "flex")
        problems.push("overview settling 时顶栏没有只恢复旧批次弹窗");
      releaseOverview();
    }
    await successRun;
    if (modal.style.display !== "none"
        || !context.document.getElementById("side").classList.contains("project-open"))
      problems.push("全成功批次没有关闭弹窗并打开任务队列");
    if (overviewCalls !== 1 || calls.length !== 30)
      problems.push("30 份全成功批次提交/overview 次数不对");
    if ((upFile.files || []).length || !upRun.disabled)
      problems.push("overview 异常后仍可重复提交成功批次");
    if (context.uploadSelectionText([{name:"single.pdf"}]) !== "single.pdf")
      problems.push("单文件选择文案不兼容");
    upFile.files = [];
    upFile.onchange();
    if (!upRun.disabled) problems.push("空选择时 Start 按钮仍可用");
    if (!/30\s*\*\s*60\s*\*\s*1000/.test(source))
      problems.push("单文件上传长超时丢失");
    if (!/fd\.append\(['"]upload_token['"],\s*token\)/.test(source))
      problems.push("上传请求没有携带幂等 token");
    const tokenFile = {name:"stable.pdf",size:123,lastModified:456};
    const token1 = context.uploadTokenFor(tokenFile,"target A");
    const token2 = context.uploadTokenFor({...tokenFile},"target A");
    if (!token1 || token1 !== token2) problems.push("同一文件重选后 token 不稳定");
    const tokenOtherTarget = context.uploadTokenFor(tokenFile,"target B");
    if (tokenOtherTarget === token1) problems.push("不同 target 错误复用同一 token");
    context.forgetUploadToken(tokenFile,"target A");
    const token3 = context.uploadTokenFor(tokenFile,"target A");
    if (token3 === token1) problems.push("成功后 token 没有释放，无法主动重新上传");

    // 上传中可以隐藏并从顶栏重新打开；文件选择和 target 均被锁住。
    upFile.files = [{name:"busy.pdf",size:1,lastModified:2}];
    upFile.onchange();
    context.setUploadBusy(true);
    if (!context.document.getElementById("upPick").disabled
        || !context.document.getElementById("upPickFolder").disabled
        || !context.document.getElementById("upTarget").disabled
        || upRun.disabled !== true
        || context.document.getElementById("upCancel").textContent !== "Hide")
      problems.push("上传忙碌态没有同时锁住文件/目录选择、target 或缺少 Hide");
    modal.style.display = "flex";
    context.document.getElementById("upCancel").onclick();
    if (modal.style.display !== "none") problems.push("上传中无法隐藏弹窗");
    await context.document.getElementById("bUp").onclick();
    if (modal.style.display !== "flex") problems.push("上传中无法从顶栏找回弹窗");
    context.setUploadBusy(false);
    context.clearUploadResultPending();
    upFolder.files = [folderA];
    upFolder.onchange();
    context.resetUploadSelections();
    context.renderUploadSelection();
    const manualReset = vm.runInContext(
      "({selected:selectedPdfFiles().length,file:UPLOAD_FILE_SELECTION.length,"
      + "folder:UPLOAD_FOLDER_SELECTION.length})", context);
    if (manualReset.selected || manualReset.file || manualReset.folder
        || (upFile.files || []).length || (upFolder.files || []).length
        || !upRun.disabled)
      problems.push("resetUploadSelections 没有同时清空文件/目录两个入口");

    // 多任务顶栏优先显示真正运行的任务，不被更新但仍 queued 的 0% 卡盖住。
    context.renderTopJobStatus([
      {slug:"new_queued",stage:"queued",started:2,done:false,progress:0},
      {slug:"old_running",stage:"text",started:1,done:false,progress:.4,
       stage_done:4,stage_total:10,stage_unit:"percent"},
    ]);
    const topHtml = context.document.getElementById("jobStatus").innerHTML;
    if (!topHtml.includes("2 PDFs") || !topHtml.includes("Step 1 · Find text"))
      problems.push("批量顶栏没有优先显示正在运行的任务");
  } catch (error) {
    const first = error && error.stack ? error.stack.split("\n")[0] : String(error);
    problems.push(`抛出 -> ${first}`);
  } finally {
    context.__oldUploadPdfFile = oldUploadPdfFile;
    context.__oldLoadOverview = oldLoadOverview;
    context.__savedPollState = savedPollState;
    vm.runInContext(
      "uploadPdfFile = __oldUploadPdfFile; loadOverview = __oldLoadOverview; "
      + "JOBS = __savedPollState.jobs; POLL = __savedPollState.poll; "
      + "POLL_TIMER = __savedPollState.timer; "
      + "POLL_INFLIGHT = __savedPollState.inflight; POLL_SEQ = __savedPollState.seq;",
      context, { filename: "restore-batch-upload" });
    context.setUploadBusy(false);
    context.clearUploadResultPending();
    context.resetUploadSelections();
    upFile.click = oldUpFileClick;
    upFolder.click = oldUpFolderClick;
    upFile.files = [];
    upFile.value = "";
    upFolder.files = [];
    upFolder.value = "";
    modal.style.display = "none";
    context.toggleProjectDrawer(false);
    context.renderJobs();
  }
  if (problems.length) {
    console.log(`  FAIL 批量上传: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   批量上传: 文件/递归目录、多级路径、过滤、逐份隔离和防重复均正常");
  }
}

// ---- 30 个任务共享一次轮询 / 一次渲染 ------------------------------------
// 这项必须执行定时器回调，源码里出现 /api/jobs 并不能证明 startPoll 调 30 次
// 后只挂了一个 timer，也不能证明同一 tick 完成两份时只刷新一次 overview。
{
  const problems = [];
  const savedState = vm.runInContext(
    "({jobs:JOBS,poll:POLL,timer:POLL_TIMER,inflight:POLL_INFLIGHT,seq:POLL_SEQ,cur:CUR})",
    context);
  const oldApi = context.api;
  const oldRenderJobs = context.renderJobs;
  const oldLoadOverview = context.loadOverview;
  const oldShow = context.show;
  const oldSetTimeout = context.setTimeout;
  const oldClearTimeout = context.clearTimeout;
  const scheduled = [];
  const cleared = [];
  const apiCalls = [];
  const showCalls = [];
  let timerSeq = 0;
  let renderCalls = 0;
  let overviewCalls = 0;
  let mode = "finish-two";
  const initial = Array.from({length:30}, (_,i) => ({
    slug:`poll_${String(i+1).padStart(2,"0")}`,
    done:false, ok:false, stage:i===0?"text":"queued", started:i+1,
    progress:i===0?.2:0, warnings:[],
  }));
  const finished = initial.map((j,i) => i===0
    ? {...j,done:true,ok:false,outcome:"partial",results_available:true,stage:"done"}
    : i===1 ? {...j,done:true,ok:true,outcome:"success",results_available:true,stage:"done"}
    : {...j});
  try {
    context.setTimeout = (fn, delay) => {
      const item = {id:++timerSeq,fn,delay,cancelled:false};
      scheduled.push(item);
      return item.id;
    };
    context.clearTimeout = (id) => {
      cleared.push(id);
      const item = scheduled.find((x) => x.id === id);
      if (item) item.cancelled = true;
    };
    context.__pollApiMock = async (url, opt) => {
      apiCalls.push({url:String(url),opt:opt||{}});
      if (String(url) !== "/api/jobs")
        return {ok:true,status:200,body:{}};
      if (mode === "network")
        return {ok:false,status:0,body:{error:"Network error"}};
      if (mode === "missing")
        return {ok:true,status:200,body:finished.slice(3).map((j) => ({...j,done:false}))};
      return {ok:true,status:200,body:finished};
    };
    context.__pollRenderMock = () => { renderCalls += 1; };
    context.__pollOverviewMock = async () => { overviewCalls += 1; };
    context.__pollShowMock = (...args) => { showCalls.push(args); };
    context.__pollInitial = Object.fromEntries(initial.map((j) => [j.slug,j]));
    vm.runInContext(
      "api = __pollApiMock; renderJobs = __pollRenderMock; "
      + "loadOverview = __pollOverviewMock; show = __pollShowMock; "
      + "JOBS = __pollInitial; POLL = {}; POLL_TIMER = null; "
      + "POLL_INFLIGHT = false; POLL_SEQ = 0; CUR = {slug:'poll_01',page:7};",
      context, { filename: "batch-poll-mocks" });

    for (const j of initial) context.startPoll(j.slug);
    const firstTimers = scheduled.filter((x) => !x.cancelled);
    if (firstTimers.length !== 1 || firstTimers[0].delay !== 0)
      problems.push(`30 次 startPoll 调度了 ${firstTimers.length} 个首轮 timer`);
    if (vm.runInContext("Object.keys(POLL).length", context) !== 30)
      problems.push("没有同时追踪 30 个 slug");

    const first = scheduled.shift();
    if (first) await first.fn();
    if (apiCalls.length !== 1 || apiCalls[0].url !== "/api/jobs")
      problems.push(`首轮请求不是单个 /api/jobs（${apiCalls.map((x)=>x.url).join(",")}）`);
    if (renderCalls !== 1)
      problems.push(`首轮 30 状态触发了 ${renderCalls} 次 renderJobs`);
    if (overviewCalls !== 1)
      problems.push(`同 tick 两任务完成触发了 ${overviewCalls} 次 overview`);
    if (showCalls.length !== 1 || showCalls[0][0] !== "poll_01" || showCalls[0][1] !== 7)
      problems.push("完成的当前项目没有按 results_available 刷新原页");
    if (vm.runInContext("Object.keys(POLL).length", context) !== 28)
      problems.push("完成任务没有退出全局 watcher");
    let next = scheduled.shift();
    if (!next || next.delay !== 2000 || scheduled.length)
      problems.push("成功 tick 后没有且仅有一个 2s timer");

    // 网络失败保留全部 watcher，并且仍然只渲染、重试一次。
    mode = "network";
    apiCalls.length = 0;
    if (next) await next.fn();
    if (apiCalls.length !== 1 || renderCalls !== 2
        || vm.runInContext("Object.keys(POLL).length", context) !== 28)
      problems.push("网络失败没有用单次批量 tick 保留任务并重试");
    if (!vm.runInContext("JOBS.poll_03.detail", context).includes("retrying"))
      problems.push("网络失败没有显示 retrying 状态");
    next = scheduled.shift();
    if (!next || next.delay !== 2000 || scheduled.length)
      problems.push("网络失败后没有且仅有一个 2s retry timer");

    // 权威列表缺 slug 等价于旧单任务 endpoint 的 404：终止该 watcher，
    // 但其他任务继续由同一个 timer 轮询。
    mode = "missing";
    apiCalls.length = 0;
    if (next) await next.fn();
    const missing = vm.runInContext("JOBS.poll_03", context);
    if (!missing || missing.stage !== "error"
        || !String(missing.error||"").includes("no longer exists")
        || vm.runInContext("isPolling('poll_03')", context))
      problems.push("批量列表缺项没有保留原 404 语义");
    if (apiCalls.length !== 1 || renderCalls !== 3)
      problems.push("缺项 tick 不是一次请求/一次 render");
    next = scheduled.shift();
    if (!next || next.delay !== 2000 || scheduled.length)
      problems.push("缺项后其余任务没有共用一个 2s timer");

    // Cancel 仍只做乐观状态并保留 watcher；实际 POST endpoint 不变。
    mode = "action";
    apiCalls.length = 0;
    await context.cancelJob("poll_04");
    if (!vm.runInContext("JOBS.poll_04.cancel_requested", context)
        || !vm.runInContext("isPolling('poll_04')", context)
        || !apiCalls.some((x) => x.url === "/api/cancel/poll_04"))
      problems.push("取消操作改变了原有 POST/watcher 语义");

    // 删除路径调用 stopPoll；移除最后一个 watcher 时共享 timer 也必须清掉。
    vm.runInContext("for(const slug of Object.keys(POLL))stopPoll(slug)", context);
    if (vm.runInContext("Object.keys(POLL).length", context)
        || vm.runInContext("POLL_TIMER", context) !== null
        || !cleared.includes(next&&next.id))
      problems.push("stopPoll 没有在最后一个任务删除时清理共享 timer");
    if (!/for\(const s of \[slug,[\s\S]*?stopPoll\(s\)/.test(scripts.join("\n")))
      problems.push("删除项目没有接入 stopPoll");
  } catch (error) {
    const first = error && error.stack ? error.stack.split("\n")[0] : String(error);
    problems.push(`抛出 -> ${first}`);
  } finally {
    context.api = oldApi;
    context.renderJobs = oldRenderJobs;
    context.loadOverview = oldLoadOverview;
    context.show = oldShow;
    context.setTimeout = oldSetTimeout;
    context.clearTimeout = oldClearTimeout;
    context.__savedBatchPollState = savedState;
    vm.runInContext(
      "JOBS = __savedBatchPollState.jobs; POLL = __savedBatchPollState.poll; "
      + "POLL_TIMER = __savedBatchPollState.timer; "
      + "POLL_INFLIGHT = __savedBatchPollState.inflight; "
      + "POLL_SEQ = __savedBatchPollState.seq; CUR = __savedBatchPollState.cur;",
      context, { filename: "restore-batch-poll" });
    context.renderJobs();
  }
  if (problems.length) {
    console.log(`  FAIL 批量轮询: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   批量轮询: 30 任务共用一次请求/渲染，完成合并刷新，断线/404/取消/删除正常");
  }
}

// ---- 上传任务卡：阶段进度 / partial / 安全重试 -----------------------------
// 这几条不能只扫源码：99% 卡住的根因正是数学没错、展示语义错。用真实
// renderJobs() 验 3/9 线型页是 33%，并让 partial / 历史任务各走一次过滤。
{
  const problems = [];
  const savedJobs = vm.runInContext("JOBS", context);
  const jobs = context.document.getElementById("jobs");
  const top = context.document.getElementById("jobStatus");
  const renderOne = (job) => {
    context.__jobCase = { [job.slug]: job };
    vm.runInContext("JOBS = __jobCase", context, { filename: "job-card-case" });
    context.renderJobs();
    return { html: (jobs.children[0] && jobs.children[0].innerHTML) || jobs.innerHTML,
      top: top.innerHTML, count: jobs.children.length,
      className: jobs.children[0] ? jobs.children[0].className : "" };
  };
  try {
    const now = Date.now() / 1000;
    let card = renderOne({slug:"line_progress",done:false,ok:false,stage:"linetypes",
      progress:.9867,stage_done:3,stage_total:9,stage_unit:"sheets",
      updated_at:now-4,detail:"Line-type engine active",warnings:[]});
    if (!card.html.includes("Extra · Line types · 3/9 sheets"))
      problems.push("线型阶段没有显示 3/9 sheets");
    if (!card.html.includes("width:33%") || !card.html.includes(">33%</span>"))
      problems.push("3/9 sheets 没有画成 33%");
    if (!card.top.includes("3/9 sheets") || !card.top.includes(">33%</b>"))
      problems.push("顶栏没有同步显示当前阶段 33%");
    if (!card.html.includes("Engine active · updated ") || !card.html.includes("s ago"))
      problems.push("active 卡没有 updated_at 心跳");

    card = renderOne({slug:"percent_progress",done:false,stage:"text",progress:.5,
      stage_done:67,stage_total:100,stage_unit:"percent",warnings:[]});
    if (!card.html.includes("Step 1 · Find text · 67%"))
      problems.push("percent 阶段没有显示百分比");
    if (card.html.includes("67/100") || card.html.includes("sheets"))
      problems.push("percent 阶段误写了 sheets/fraction");

    card = renderOne({slug:"not_done_100",done:false,stage:"linetypes",progress:1,
      stage_done:9,stage_total:9,stage_unit:"sheets",warnings:[]});
    if (!card.html.includes("width:99%") || card.html.includes("width:100%"))
      problems.push("未完成任务被四舍五入成 100%");

    card = renderOne({slug:"overall_fallback",done:false,stage:"queued",progress:.42,
      stage_done:0,stage_total:0,warnings:[]});
    if (!card.html.includes("width:42%"))problems.push("无 stage_total 时没有回退 overall progress");

    card = renderOne({slug:"partial_job",done:true,ok:true,outcome:"partial",
      results_available:true,progress:1,warnings:["P17 remains unresolved"]});
    if (card.count !== 1 || !card.className.includes("warn"))
      problems.push("partial 卡被过滤或没有琥珀状态");
    if (!card.html.includes("P17 remains unresolved")
        || !card.html.includes("Retry unresolved sheets"))
      problems.push("partial 卡没有 warning / Retry unresolved sheets");

    card = renderOne({slug:"failed_job",done:true,ok:false,outcome:"failed",
      results_available:false,error:"worker failed",warnings:[]});
    if (!card.html.includes("Retry unresolved sheets") || !card.html.includes("Delete project"))
      problems.push("failed 卡没有安全重试和删除按钮");

    card = renderOne({slug:"legacy_ok",done:true,ok:true,progress:1,
      warnings:["historical warning"]});
    if (card.count || !jobs.innerHTML.includes("No jobs"))
      problems.push("旧 done+ok 历史卡重新冒出");
    card = renderOne({slug:"new_success",done:true,ok:true,outcome:"success",
      results_available:true,progress:1,warnings:[]});
    if (card.count)problems.push("新 success 卡没有过滤");
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 160)}`);
  } finally {
    context.__savedJobs = savedJobs;
    vm.runInContext("JOBS = __savedJobs", context, { filename: "restore-jobs" });
    context.renderJobs();
  }
  if (problems.length) {
    console.log(`  FAIL 任务 UX: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   任务 UX: 当前阶段进度、99% cap、partial/failed/legacy 状态均正确");
  }
}

// Retry 按钮必须复用当前 cache，只修没完成的页；同时静态钉住完成后刷新
// results_available 的分支（失败但已有部分结果也能立刻看到）。
{
  const problems = [];
  const oldFetch = context.fetch;
  const calls = [];
  const savedState = vm.runInContext(
    "({jobs:JOBS,poll:POLL,timer:POLL_TIMER,inflight:POLL_INFLIGHT,seq:POLL_SEQ})",
    context);
  try {
    context.fetch = async (url, opt) => {
      calls.push({url:String(url),opt:opt||{}});
      const isRetry=String(url).includes("/api/rerun/retry_case");
      return {ok:true,status:200,json:async()=>isRetry
        ? {slug:"retry_case",job:{slug:"retry_case",done:false,stage:"queued",progress:0}}
        : (String(url).includes("/api/overview")?[]:{})};
    };
    context.__retryJobs = {retry_case:{slug:"retry_case",done:true,ok:true,
      outcome:"partial",results_available:true,warnings:["one unresolved sheet"]}};
    vm.runInContext(
      "JOBS = __retryJobs; POLL = {}; POLL_TIMER = null; POLL_INFLIGHT = false;",
      context, { filename: "retry-job" });
    await context.retryUnresolved("retry_case");
    const call = calls.find((x) => x.url.includes("/api/rerun/retry_case"));
    if (!call || call.opt.method !== "POST")problems.push("Retry 没有 POST rerun endpoint");
    else {
      let body = null;
      try { body = JSON.parse(call.opt.body); } catch (_error) {}
      if (!body || body.reset !== false)problems.push("Retry 没有发送 reset:false");
    }
    if (!/j&&\(j\.results_available\|\|j\.ok\)&&CUR\)show\(j\.slug,CUR\.page\)/s
      .test(scripts.join("\n")))
      problems.push("批量 poll 完成后没有按 results_available 刷新当前页");
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 160)}`);
  } finally {
    context.fetch = oldFetch;
    context.__savedRetryState = savedState;
    vm.runInContext(
      "JOBS = __savedRetryState.jobs; POLL = __savedRetryState.poll; "
      + "POLL_TIMER = __savedRetryState.timer; "
      + "POLL_INFLIGHT = __savedRetryState.inflight; POLL_SEQ = __savedRetryState.seq;",
      context,
      { filename: "restore-retry-job" });
    context.renderJobs();
  }
  if (problems.length) {
    console.log(`  FAIL 任务重试: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   任务重试: POST reset:false；部分结果完成后会刷新当前页");
  }
}

// ---- fence + gate 同句必须由 fence 语义优先 -------------------------------
// 后端的同名判据决定是否允许绑定线型；这里钉住前端颜色/scope 的镜像规则，
// 防止只改一边后再次出现“文字是蓝色 fence，线型却被 gate 闸掉”的分裂状态。
{
  const samples = [
    ["GATE", true],
    ["DOUBLE SWING GATE", true],
    ["EXISTING GATES", true],
    ["5' ORNAMENTAL STEEL FENCE & GATE", false],
    ["FENCING / GATES", false],
    ["FENCED ACCESS GATE", false],
    ["FENCES AND GATES", false],
    ["FENCELINE GATE", false],
    ["FENCE LINE / GATE", false],
    ["AGGREGATE BASE", false],
  ];
  const problems = [];
  try {
    for (const [value, expected] of samples) {
      context.__gateProbe = value;
      const actual = vm.runInContext("isGateText(__gateProbe)", context,
        { filename: "fence-gate-priority" });
      if (actual !== expected)
        problems.push(`${JSON.stringify(value)} -> ${actual}, expected ${expected}`);
    }
    context.__scopePriorityPage = {
      items: [
        {text:"5' ORNAMENTAL STEEL FENCE & GATE",box_2d:[10,10,20,90]},
        {text:"DOUBLE SWING GATE",box_2d:[30,10,40,90]},
      ],
      marker_codes:[],suppressed_items:[],symbols:{symbols:[]},record:{},
    };
    const scopes = JSON.parse(vm.runInContext(
      "(()=>{const saved=PAGE;PAGE=__scopePriorityPage;makeScopeModel();"
      + "const out=SCOPE_GROUPS.map(g=>({text:g.text,gate:g.gate,cls:scopeClass(g)}));"
      + "PAGE=saved;makeScopeModel();return JSON.stringify(out)})()",
      context, { filename: "fence-gate-scope-priority" }));
    if (scopes.length !== 2
        || scopes[0].gate !== false || scopes[0].cls !== "fence"
        || scopes[1].gate !== true || scopes[1].cls !== "gate")
      problems.push(`scope 分类错误: ${JSON.stringify(scopes)}`);
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 160)}`);
  }
  if (problems.length) {
    console.log(`  FAIL fence/gate 优先级: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   fence/gate 优先级: 同句显式 fence 胜过 gate，纯 gate 保持 gate");
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
            + "$('bar').classList.remove('tucked');"
            + "$('tgLtMeta').querySelector('input').checked = true;"
            + "$('tgAll').querySelector('input').checked = true;",
            context, { filename: "inject-all" });
          context.drawAllLinetypes();
          context.renderList();
          const types = all.types || [];
          // ALL_NODES 是 let 绑定，不挂在 global 对象上（同 PAGE），
          // 只能在同一个 context 里求值才能读到。
          const drawn = vm.runInContext("ALL_NODES.size", context);
          const emptyTypes = JSON.parse(vm.runInContext(
            "JSON.stringify([...ALL_NODES].filter(([,nodes])=>!nodes.length).map(([number])=>number))",
            context));
          const missingGeometry = emptyTypes.filter((number) => {
            const type = types.find((t) => t.line_type_number === number);
            return type && Number(type.segment_count || 0) > 0;
          });
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
          if (missingGeometry.length)
            problems.push(`${missingGeometry.length} 个有线段的类型没有折线节点: ${missingGeometry.slice(0, 8).join(",")}`);
          const recognized = Number((((payload.record || {}).linetypes || {}).page || {}).line_types);
          if (recognized && types.length !== recognized)
            problems.push(`全量接口 ${types.length} 型 != 主结果识别 ${recognized} 型`);
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
          const liveEmpty = JSON.parse(vm.runInContext(
            "JSON.stringify([...ALL_NODES].filter(([,nodes])=>!nodes.length).map(([number])=>number))",
            context));
          const liveMissingGeometry = liveEmpty.filter((number) => {
            const type = types.find((t) => t.line_type_number === number);
            return type && Number(type.segment_count || 0) > 0;
          });
          const liveNodes = (context.document.getElementById("gAll").children || []).length;
          const liveItems = (context.document.getElementById("list").children || [])
            .filter((n) => n.classList && n.classList.contains("al-item")).length;
          const liveProblems = [];
          if (liveState !== "ok") liveProblems.push(`点击后 ALL_LT_STATE=${liveState}`);
          if (liveDrawn !== types.length) liveProblems.push(`点击后 ALL_NODES ${liveDrawn} != ${types.length}`);
          if (liveMissingGeometry.length)
            liveProblems.push(`点击后 ${liveMissingGeometry.length} 个有线段的类型没有折线节点`);
          if (!liveNodes) liveProblems.push("点击后 gAll 空");
          if (liveItems !== types.length) liveProblems.push(`点击后列表 ${liveItems} != ${types.length}`);
          if (liveProblems.length) {
            console.log(`  FAIL ${rel} 真实点击路径: ${liveProblems.join("; ")}`);
            failures += 1;
          } else {
            console.log(`  OK   ${rel} 真实点击路径: 勾选后自动取数并画出 `
              + `${liveDrawn} 型 / ${liveNodes} 折线 / 列表 ${liveItems} 行`);
          }

          // ---- 真实 checkbox change 路径：隐藏不等于丢缓存 -----------
          // 不直接调 syncLayers()/ensureAllLinetypes()；由页面在 LAYERS 循环里
          // 注册的 change listener 决定是纯隐藏还是恢复绘制。
          const toggleProblems = [];
          const allInput = context.document.getElementById("tgAll").querySelector("input");
          const residInput = context.document.getElementById("tgResid").querySelector("input");
          const allGroup = context.document.getElementById("gAll");
          const residGroup = context.document.getElementById("gResid");
          const oldFetch = context.fetch;
          const savedToggle = vm.runInContext(
            "({all:ALL_LT,key:ALL_LT_KEY,state:ALL_LT_STATE,detail:ALL_LT_DETAIL,"
            + "allChecked:$('tgAll').querySelector('input').checked,"
            + "residChecked:$('tgResid').querySelector('input').checked})", context);
          context.__toggleAllCache = savedToggle.all;
          let toggleNetworkCalls = 0;
          try {
            // 已有缓存的开关路径不应碰网络；若碰了，返回显式失败
            // 而不是真正重跑，让断言能稳定指出多余 GET/POST。
            context.fetch = async () => {
              toggleNetworkCalls += 1;
              return {ok: false, status: 599,
                json: async () => ({error: "unexpected toggle network request"})};
            };
            const change = (input, checked) => {
              input.checked = checked;
              input.dispatchEvent({type: "change"});
            };

            // 两个开关都打开：全部线型和 residual 各显示各自的层。
            change(residInput, true);
            change(allInput, true);
            await Promise.resolve();
            if (allGroup.style.display === "none" || !(allGroup.children || []).length)
              toggleProblems.push("勾选 All line types 后 gAll 未显示/无几何");
            if (all.residual && all.residual.op_count
                && (residGroup.style.display === "none" || !(residGroup.children || []).length))
              toggleProblems.push("勾选 residual 后 gResid 未显示/无几何");

            const allNodeCount = (allGroup.children || []).length;
            const residNodeCount = (residGroup.children || []).length;
            change(allInput, false);
            if (allGroup.style.display !== "none")
              toggleProblems.push("取消 All line types 后 gAll 仍显示");
            if (residGroup.style.display === "none")
              toggleProblems.push("取消 All line types 误隐藏了独立 residual 层");
            const hidden = vm.runInContext(
              "({same:ALL_LT===__toggleAllCache,key:ALL_LT_KEY,state:ALL_LT_STATE,"
              + "inflight:!!ALL_LT_INFLIGHT})", context);
            if (!hidden.same || hidden.key !== savedToggle.key || hidden.state !== "ok")
              toggleProblems.push("取消勾选后缓存/key/状态未保留");
            if (hidden.inflight) toggleProblems.push("取消勾选启动了请求");

            // 再次打开必须仅用内存 ALL_LT 重画，节点数不变且无网络。
            change(allInput, true);
            await Promise.resolve();
            if (allGroup.style.display === "none"
                || (allGroup.children || []).length !== allNodeCount)
              toggleProblems.push("再次勾选未从缓存恢复 gAll");
            const restored = vm.runInContext(
              "({same:ALL_LT===__toggleAllCache,state:ALL_LT_STATE,"
              + "inflight:!!ALL_LT_INFLIGHT})", context);
            if (!restored.same || restored.state !== "ok" || restored.inflight)
              toggleProblems.push("再次勾选改变了缓存/状态或启动请求");

            // residual 是独立层：关它不得动 gAll，再开要恢复原节点。
            change(residInput, false);
            if (residGroup.style.display !== "none")
              toggleProblems.push("取消 residual 后 gResid 仍显示");
            if (allGroup.style.display === "none")
              toggleProblems.push("取消 residual 误隐藏了 gAll");
            change(residInput, true);
            if (residGroup.style.display === "none"
                || (residGroup.children || []).length !== residNodeCount)
              toggleProblems.push("再次勾选 residual 未恢复原几何");
            if (toggleNetworkCalls)
              toggleProblems.push(`开关过程额外发出 ${toggleNetworkCalls} 次 GET/POST`);
          } finally {
            context.fetch = oldFetch;
            context.__toggleSaved = savedToggle;
            vm.runInContext(
              "ALL_LT=__toggleSaved.all; ALL_LT_KEY=__toggleSaved.key;"
              + "ALL_LT_STATE=__toggleSaved.state; ALL_LT_DETAIL=__toggleSaved.detail;"
              + "$('tgAll').querySelector('input').checked=__toggleSaved.allChecked;"
              + "$('tgResid').querySelector('input').checked=__toggleSaved.residChecked;"
              + "drawAllLinetypes();syncLayers();", context,
              {filename: "restore-all-toggle"});
          }
          if (toggleProblems.length) {
            console.log(`  FAIL ${rel} All/residual 开关: ${toggleProblems.join("; ")}`);
            failures += 1;
          } else {
            console.log(`  OK   ${rel} All/residual 开关: 隐藏保缓存，再开零请求，residual 独立`);
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

// ---- Method 2 confirmed pattern 框：normal / All / 来源 / 开关缓存 ---------
// 用刚拉到的真实 page payload 做外壳，注入一个已经绑定且会发到前端的线型
// metadata。这样既走真实 build()/makeScopeModel()/change listener，又不依赖盘上
// All cache 是否恰好已经由新版 sidecar 重算。
{
  const problems = [];
  const oldFetch = context.fetch;
  const saved = vm.runInContext(
    "({cur:CUR,page:PAGE,epoch,all:ALL_LT,key:ALL_LT_KEY,state:ALL_LT_STATE,"
    + "detail:ALL_LT_DETAIL,focus:ALL_FOCUS,filter:ALL_FILTER,inflight:ALL_LT_INFLIGHT,"
    + "selectedScope,selectedRow,ltChecked:$('tgLt').querySelector('input').checked,"
    + "allChecked:$('tgAll').querySelector('input').checked,"
    + "provChecked:$('tgLtSrc').querySelector('input').checked,"
    + "metaChecked:$('tgLtMeta').querySelector('input').checked,"
    + "barTucked:$('bar').classList.contains('tucked')})", context);
  let skipped = false;
  try {
    const fixture = saved.page ? JSON.parse(JSON.stringify(saved.page)) : null;
    if (!fixture) {
      skipped = true;
      console.log("  SKIP Method 2 pattern 框: 没有可用页面外壳");
    } else {
      fixture.record = fixture.record || {};
      // 固定放一个 plan 框：普通几何仍应可选中，但不再弹 Type / box_2d
      // 之类的通用调试状态卡。
      fixture.plan_boxes = [[22,20,580,450]];
      fixture.items = Array.isArray(fixture.items) ? fixture.items : [];
      let scopeItem = fixture.items.find((item) => item && String(item.text || "").trim());
      if (!scopeItem) {
        scopeItem = {text:"8' SIDELINE FENCE",label:"callout",source:"probe",
          box_2d:[80,80,95,180]};
        fixture.items = [scopeItem];
      }
      const scopeText = String(scopeItem.text).replace(/\s+/g, " ").trim();
      const scopeToken = scopeText.normalize("NFKC").toLocaleUpperCase();
      const instances = [
        {region_id:"probe-a",group_id:"G-A",op_indices:[11,12],literal_text:"8'",
          pattern_source:"inline_measurement",bbox:[100,120,145,205]},
        {region_id:"probe-b",group_id:"G-A",op_indices:[21,22],literal_text:"8'",
          pattern_source:"inline_measurement",bbox:[100,240,145,325]},
      ];
      const target = {
        line_type_number:701,line_type_id:"probe-method2",
        signature_family:"synthetic_method2",recognition_source:"method2",
        op_count:4,segment_count:1,member_count:2,runs:[{run_id:"r1"}],
        bbox:[100,120,145,325],polylines:[[[110,120],[110,325]]],
        pattern_instance_count:instances.length,
      };
      // 第三个退化 bbox 是防御性负例：不能制造一个 0 面积框。
      target.pattern_instances = [...instances,
        {region_id:"invalid",bbox:[400,400,400,450]}];
      fixture.record.linetypes = {
        visible:[target.line_type_number],
        groups:[{group:"t:"+scopeToken,text:scopeText,
          visible_line_type_number:target.line_type_number,in_plan_count:1,
          votes_in_plan:{}}],
        line_types:[target],bindings:[
          {visible:true,line_type_number:target.line_type_number,key:0,ti:0},
          {visible:true,line_type_number:target.line_type_number,key:"s2:1",ti:0},
          {visible:true,line_type_number:target.line_type_number,key:"s0:0",ti:0,
            source:"legend_template"},
          // 同编号但最终不可见的 binding 不能污染 provenance。
          {visible:false,line_type_number:target.line_type_number,key:"s9:9",ti:0},
        ],page:{line_types:1},
      };

      context.__patternPage = fixture;
      vm.runInContext(
        "PAGE=__patternPage; CUR={slug:'method2_pattern_probe',page:PAGE.page}; epoch+=1;"
        + "selectedScope=null;selectedRow=null;ALL_LT=null;ALL_LT_KEY='';"
        + "ALL_LT_STATE='idle';ALL_LT_DETAIL='';ALL_LT_INFLIGHT=null;ALL_FOCUS=null;"
        + "$('tgLt').querySelector('input').checked=true;"
        + "$('tgAll').querySelector('input').checked=false;"
        + "$('tgLtSrc').querySelector('input').checked=false;"
        + "$('tgLtMeta').querySelector('input').checked=false;"
        + "$('bar').classList.add('tucked');"
        + "makeScopeModel();build();renderList();syncLayers();", context,
        {filename:"method2-pattern-normal"});
      context.__patternNumber = target.line_type_number;
      const planPick = JSON.parse(vm.runInContext(
        "$('sel').classList.add('has');$('sel').innerHTML='stale inspector';"
        + "pick('pl0',false);JSON.stringify({exists:!!LOOK.pl0,picked,"
        + "selected:!!(LOOK.pl0&&LOOK.pl0.rect.classList.contains('pick')),"
        + "has:$('sel').classList.contains('has'),html:$('sel').innerHTML})", context,
        {filename:"hidden-generic-box-status"}));
      if (!planPick.exists || planPick.picked !== "pl0" || !planPick.selected)
        problems.push("plan 框不再能保持选中态");
      if (planPick.has || planPick.html)
        problems.push("plan 框仍会显示 Type / box_2d 通用状态卡");
      const normal = JSON.parse(vm.runInContext(
        "JSON.stringify({"
        + "nodes:(LT_NODES.get(__patternNumber)||[]).map(n=>({tag:n.tagName,"
        + "cls:n.getAttribute('class'),pattern:n.dataset.pattern,lt:n.dataset.lt,"
        + "x:n.getAttribute('x'),y:n.getAttribute('y'),w:n.getAttribute('width'),"
        + "h:n.getAttribute('height'),title:((n.children||[]).find(c=>"
        + "c.tagName==='TITLE')||{}).textContent||''})),"
        + "rows:(LT_INDEX.rows||[]).filter(r=>r.number===__patternNumber),"
        + "method1Leak:method2PatternInstances({recognition_source:'method1',"
        + "pattern_instances:__patternPage.record.linetypes.line_types.find("
        + "t=>t.line_type_number===__patternNumber).pattern_instances}).length})", context));
      const normalBoxes = normal.nodes.filter((node) => node.cls === "lt-pattern");
      if (normalBoxes.length !== 2)
        problems.push(`normal 画了 ${normalBoxes.length} 个框，应为 2`);
      if (normalBoxes.some((node) => node.tag !== "RECT"
          || node.pattern !== "confirmed" || Number(node.lt) !== target.line_type_number
          || !(Number(node.w) > 0) || !(Number(node.h) > 0)
          || node.title !== "Fenceline pattern"))
        problems.push("normal pattern 框的 rect/data/尺寸不完整");
      if (normal.method1Leak !== 0)
        problems.push("Method 1 误带 pattern_instances 时仍会画框");
      if (!normal.rows.length || normal.rows.some((row) =>
          row.source !== "Method 2" || row.patternCount !== 2))
        problems.push("normal FENCELINE 内部索引丢失 Method 2 / pattern 数据");
      const normalListRows = (context.document.getElementById("list").children || [])
        .filter((node) => node.dataset
          && Number(node.dataset.lt) === target.line_type_number);
      const customerLeaks = ["#701", "Line type", "synthetic_method2", "segments",
        "Method 2", "confirmed pattern", "terminals"];
      if (!normalListRows.length || normalListRows.some((node) =>
          !String(node.innerHTML).includes(scopeText)))
        problems.push("默认 FENCELINE 没保留识别出的客户可读文字");
      if (normalListRows.some((node) => customerLeaks.some((token) =>
          String(node.innerHTML).includes(token))))
        problems.push("默认 FENCELINE 泄漏编号/family/算法/几何/pattern/terminal 元数据");
      const currentLtTitles = () => JSON.parse(vm.runInContext(
        "JSON.stringify((((svg&&svg.children)||[]).find(n=>n.id==='gLt')||"
        + "{children:[]}).children.filter(n=>n.tagName==='TITLE').map(n=>n.textContent))",
        context));
      const currentPatternTitles = () => JSON.parse(vm.runInContext(
        "JSON.stringify((((svg&&svg.children)||[]).find(n=>n.id==='gLt')||"
        + "{children:[]}).children.filter(n=>n.getAttribute&&"
        + "n.getAttribute('class')==='lt-pattern').map(n=>"
        + "(((n.children||[]).find(c=>c.tagName==='TITLE')||{}).textContent||'')))",
        context));
      const defaultTitles = currentLtTitles();
      if (!defaultTitles.length || defaultTitles.some((value) => value !== "Fenceline"))
        problems.push(`客户视图 SVG tooltip 仍含算法详情: ${defaultTitles.join(" | ")}`);
      if (currentPatternTitles().some((value) => value !== "Fenceline pattern"))
        problems.push("客户视图 pattern tooltip 仍含 Method/group/source 算法详情");

      // 线型诊断必须同时满足“Debug Console 已展开”和显式 checkbox。关闭
      // Console 后 checkbox 可以保留，但客户视图必须立即恢复为纯文字。
      const metaTag = (html.match(
        /<label\b[^>]*\bid=["']tgLtMeta["'][\s\S]*?<\/label>/i) || [""])[0];
      if (!metaTag || !/line-type diagnostics/i.test(metaTag))
        problems.push("Debug Console 缺少 line-type diagnostics 开关");
      if (/<input\b[^>]*\bchecked\b/i.test(metaTag))
        problems.push("line-type diagnostics 开关在 HTML 中不是默认关闭");
      const metaInput = context.document.getElementById("tgLtMeta").querySelector("input");
      const normalNodeCount = vm.runInContext(
        "(LT_NODES.get(__patternNumber)||[]).length", context);
      const toggleConsole = () => vm.runInContext(
        "__patternOv=OV;OV=[];$('bLayers').onclick();OV=__patternOv;", context,
        {filename:"line-type-diagnostics-console"});
      toggleConsole();
      metaInput.checked = true;
      metaInput.dispatchEvent({type:"change"});
      const debugRows = (context.document.getElementById("list").children || [])
        .filter((node) => node.dataset
          && Number(node.dataset.lt) === target.line_type_number);
      const debugTitles = currentLtTitles();
      if (!debugRows.length || debugRows.some((node) =>
          !String(node.innerHTML).includes("#701")
          || !String(node.innerHTML).includes("synthetic_method2")
          || !String(node.innerHTML).includes("segments")
          || !String(node.innerHTML).includes("Method 2")
          || !String(node.innerHTML).includes("confirmed patterns boxed")))
        problems.push("诊断开关开启后没有恢复 FENCELINE 技术元数据");
      if (!debugTitles.some((value) => value.includes("#701")
          && value.includes("Method 2") && value.includes("segments")))
        problems.push("诊断开关开启后没有恢复 SVG 技术 tooltip");
      if (!currentPatternTitles().some((value) =>
          value.includes("Method 2 confirmed repeated pattern")))
        problems.push("诊断开关开启后没有恢复 pattern 技术 tooltip");
      toggleConsole();
      const collapsedRows = (context.document.getElementById("list").children || [])
        .filter((node) => node.dataset
          && Number(node.dataset.lt) === target.line_type_number);
      const collapsedTitles = currentLtTitles();
      if (collapsedRows.some((node) => customerLeaks.some((token) =>
          String(node.innerHTML).includes(token)))
          || collapsedTitles.some((value) => value !== "Fenceline")
          || currentPatternTitles().some((value) => value !== "Fenceline pattern"))
        problems.push("收起 Debug Console 后 checkbox=true 仍泄漏线型诊断信息");
      toggleConsole();
      const reopenedRows = (context.document.getElementById("list").children || [])
        .filter((node) => node.dataset
          && Number(node.dataset.lt) === target.line_type_number);
      if (!reopenedRows.length
          || !String(reopenedRows[0].innerHTML).includes("synthetic_method2"))
        problems.push("重新展开 Debug Console 后已勾选的诊断信息未恢复");
      metaInput.checked = false;
      metaInput.dispatchEvent({type:"change"});
      toggleConsole();
      const normalNodeCountAfterMeta = vm.runInContext(
        "(LT_NODES.get(__patternNumber)||[]).length", context);
      if (normalNodeCountAfterMeta !== normalNodeCount)
        problems.push("诊断开关改变了普通 FENCELINE 高亮几何");

      const provenanceKinds = JSON.parse(vm.runInContext(
        "JSON.stringify(lineTypeProvenance(__patternNumber))", context));
      if (provenanceKinds.join("|")
          !== "Text callout|Symbol callout|Legend line sample")
        problems.push(`binding provenance 分类/顺序错误: ${provenanceKinds.join("|")}`);
      if (normalListRows.some((node) => String(node.innerHTML).includes("Matched via"))
          || normal.nodes.some((node) => node.cls === "lt-prov-label"))
        problems.push("provenance 默认关闭时仍泄露到普通列表/画布");

      const provenanceInput = context.document.getElementById("tgLtSrc")
        .querySelector("input");
      toggleConsole();
      provenanceInput.checked = true;
      provenanceInput.dispatchEvent({type:"change"});
      const provenanceOn = JSON.parse(vm.runInContext(
        "(()=>{const row=(LT_INDEX.rows||[]).find(r=>r.number===__patternNumber);"
        + "if(row)selectScope(row.scopeId,false);"
        + "const label=(LT_NODES.get(__patternNumber)||[]).find(n=>"
        + "n.getAttribute('class')==='lt-prov-label');"
        + "const item=($('list').children||[]).find(n=>n.dataset&&"
        + "Number(n.dataset.lt)===__patternNumber);"
        + "return JSON.stringify({row:item?item.innerHTML:'',detail:$('sel').innerHTML,"
        + "label:label?label.textContent:'',focus:!!(label&&"
        + "label.classList.contains('lt-focus'))});})()", context));
      const allProvenance = "Matched via Text callout + Symbol callout + Legend line sample";
      if (!provenanceOn.row.includes(allProvenance)
          || !provenanceOn.detail.includes(allProvenance)
          || !provenanceOn.label.includes(allProvenance)
          || !provenanceOn.focus)
        problems.push("provenance 开启后普通列表/选择详情/聚焦标签未同时显示三类来源");
      provenanceInput.checked = false;
      provenanceInput.dispatchEvent({type:"change"});
      toggleConsole();

      const targetRow = normal.rows[0];
      if (targetRow) {
        context.__patternScope = targetRow.scopeId;
        vm.runInContext("selectScope(__patternScope,false);", context,
          {filename:"method2-pattern-normal-focus"});
        const focused = JSON.parse(vm.runInContext(
          "JSON.stringify((LT_NODES.get(__patternNumber)||[])"
          + ".filter(n=>n.getAttribute('class')==='lt-pattern')"
          + ".map(n=>({focus:n.classList.contains('lt-focus'),"
          + "dim:n.classList.contains('lt-dim')})))", context));
        if (focused.length !== 2 || focused.some((node) => !node.focus || node.dim))
          problems.push("normal pattern 框没有跟随 callout focus/dim");
      }
      const ltInput = context.document.getElementById("tgLt").querySelector("input");
      const ltGroup = context.document.getElementById("gLt");
      ltInput.checked = false; ltInput.dispatchEvent({type:"change"});
      if (ltGroup.style.display !== "none")
        problems.push("关闭 Line types 后 normal pattern 框层仍显示");
      ltInput.checked = true; ltInput.dispatchEvent({type:"change"});
      const restoredNormalBoxes = vm.runInContext(
        "(LT_NODES.get(__patternNumber)||[]).filter(n=>"
        + "n.getAttribute('class')==='lt-pattern').length", context);
      if (ltGroup.style.display === "none"
          || restoredNormalBoxes !== 2)
        problems.push("重开 Line types 后 normal pattern 框未保留");

      const allM2 = {
        line_type_number:701,signature_family:"synthetic_method2",
        recognition_source:"method2",op_count:4,segment_count:1,
        member_count:2,runs:[{run_id:"r1"}],
        bound_by:[{key:"s1:0",ti:0,distance:0}],
        bbox:[100,120,145,325],polylines:[[[110,120],[110,325]]],
        pattern_instance_count:2,pattern_instances:instances,
      };
      const allM1 = {
        line_type_number:702,signature_family:"synthetic_method1",
        recognition_source:"method1",op_count:2,segment_count:1,
        member_count:2,runs:[{run_id:"r2"}],bound_by:[],
        bbox:[300,120,345,205],polylines:[[[310,120],[310,205]]],
        // 故意带同名字段，验证 source gate，而不是仅靠后端“通常不会发”。
        pattern_instance_count:1,pattern_instances:[instances[0]],
      };
      context.__patternAll = {state:"ok",page:{path_ops:8,owned_path_ops:6},
        types:[allM2,allM1],residual:{op_count:1,polylines:[[[500,500],[510,510]]]}};
      vm.runInContext(
        "ALL_LT=__patternAll;ALL_LT_KEY=allKey();ALL_LT_STATE='ok';"
        + "ALL_LT_DETAIL='';ALL_LT_INFLIGHT=null;ALL_FOCUS=null;ALL_FILTER='';"
        + "$('tgAll').querySelector('input').checked=true;drawAllLinetypes();"
        + "renderList();syncLayers();", context, {filename:"method2-pattern-all"});
      const allState = JSON.parse(vm.runInContext(
        "JSON.stringify({m2:(ALL_NODES.get(701)||[]).map(n=>({"
        + "cls:n.getAttribute('class'),pattern:n.dataset.pattern})),"
        + "m1:(ALL_NODES.get(702)||[]).map(n=>n.getAttribute('class'))})", context));
      const allBoxes = allState.m2.filter((node) => node.cls === "al-pattern");
      if (allBoxes.length !== 2 || allBoxes.some((node) => node.pattern !== "confirmed"))
        problems.push(`All Method 2 框=${allBoxes.length}，应为两个 confirmed rect`);
      if (allState.m1.includes("al-pattern"))
        problems.push("All 层给 Method 1 画了 pattern 框");

      const allInput = context.document.getElementById("tgAll").querySelector("input");
      // build() replaces the SVG and therefore the actual <g id="gAll">.
      // Always resolve it again; keeping an old node would make layer checks
      // pass or fail against detached geometry instead of the live canvas.
      const allGroupNow = () => context.document.getElementById("gAll");
      const sel = context.document.getElementById("sel");
      const listAllItems = () => (context.document.getElementById("list").children || [])
        .filter((node) => node.classList && node.classList.contains("al-item"));
      const hasAllSection = () => (context.document.getElementById("list").children || [])
        .some((node) => String(node.textContent || "") === "ALL LINE TYPES (DEBUG)");

      // tgAll 只控制画布几何和取数。客户默认视图里即使几何已打开，也不能
      // 把完整聚类清单或点击后的算法卡塞进 Fence scope。
      if (listAllItems().length || hasAllSection())
        problems.push("诊断关闭时 tgAll 仍把 ALL LINE TYPES(DEBUG) 渲染进侧栏");
      sel.classList.add("has"); sel.dataset.detail = "line-type-diagnostic";
      sel.innerHTML = "stale all-line detail";
      context.focusAllType(701, false);
      const allFocused = JSON.parse(vm.runInContext(
        "JSON.stringify({m2:(ALL_NODES.get(701)||[]).filter(n=>"
        + "n.getAttribute('class')==='al-pattern').map(n=>({"
        + "focus:n.classList.contains('al-focus'),dim:n.classList.contains('al-dim')})),"
        + "m1:(ALL_NODES.get(702)||[]).map(n=>({"
        + "focus:n.classList.contains('al-focus'),dim:n.classList.contains('al-dim')}))})",
        context));
      if (allFocused.m2.length !== 2
          || allFocused.m2.some((node) => !node.focus || node.dim)
          || allFocused.m1.some((node) => node.focus || !node.dim))
        problems.push("All pattern 框没有随类型 focus，其余类型没有 dim");
      context.showAllTypeDetail(allM2);
      if (sel.classList.contains("has") || sel.innerHTML)
        problems.push("诊断关闭时 show/focus All type 仍显示算法详情");

      let networkCalls = 0;
      context.fetch = async () => {
        networkCalls += 1;
        return {ok:false,status:599,json:async()=>({error:"unexpected request"})};
      };
      toggleConsole();
      metaInput.checked = true;
      metaInput.dispatchEvent({type:"change"});
      context.showAllTypeDetail(allM2);
      const detail = sel.innerHTML;
      const allItems = listAllItems();
      const m2Item = allItems.find((node) => Number(node.dataset.al) === 701);
      const m1Item = allItems.find((node) => Number(node.dataset.al) === 702);
      if (!hasAllSection() || allItems.length !== 2)
        problems.push("诊断开启后没有恢复 ALL LINE TYPES(DEBUG) 清单");
      if (!sel.classList.contains("has")
          || !detail.includes("source Method 2")
          || !detail.includes("2 confirmed pattern instances boxed")
          || !detail.includes("Bound by: s1:0/0 @0pt"))
        problems.push("诊断开启后没有恢复算法详情/Bound by");
      if (!m2Item || !String(m2Item.innerHTML).includes("Method 2")
          || !String(m2Item.innerHTML).includes("2 confirmed patterns boxed"))
        problems.push("All 列表没有 Method 2 / pattern 数");
      if (!m1Item || !String(m1Item.innerHTML).includes("Method 1"))
        problems.push("All 列表没有 Method 1 来源");

      provenanceInput.checked = true;
      vm.runInContext(
        "drawAllLinetypes();renderList();showAllTypeDetail(__patternAll.types[0]);",
        context, {filename:"linetype-provenance-all"});
      const provenanceAll = JSON.parse(vm.runInContext(
        "(()=>{const label=(ALL_NODES.get(701)||[]).find(n=>"
        + "n.getAttribute('class')==='al-prov-label');"
        + "const item=($('list').children||[]).find(n=>n.dataset&&Number(n.dataset.al)===701);"
        + "return JSON.stringify({row:item?item.innerHTML:'',detail:$('sel').innerHTML,"
        + "label:label?label.textContent:'',focus:!!(label&&"
        + "label.classList.contains('al-focus'))});})()", context));
      if (!provenanceAll.row.includes(allProvenance)
          || !provenanceAll.detail.includes(allProvenance)
          || !provenanceAll.label.includes(allProvenance)
          || !provenanceAll.focus)
        problems.push("provenance 开启后 All 列表/详情/聚焦标签未显示三类来源");
      provenanceInput.checked = false;
      vm.runInContext("drawAllLinetypes();renderList();", context,
        {filename:"linetype-provenance-all-off"});

      const allNodeCount = (allGroupNow().children || []).length;
      toggleConsole();
      if (listAllItems().length || hasAllSection()
          || sel.classList.contains("has") || sel.innerHTML)
        problems.push("收起 Debug Console 后 All 清单/详情仍残留");
      if (allGroupNow().style.display === "none"
          || (allGroupNow().children || []).length !== allNodeCount)
        problems.push("收起 Debug Console 误隐藏或重算 tgAll 画布几何");

      // Console 重新打开会恢复已勾选的诊断清单。关闭 tgAll 则必须立即清除
      // 这张清单和当前 All detail；之后选普通 FENCELINE 不得偷偷重开 tgAll。
      toggleConsole();
      context.showAllTypeDetail(allM2);
      allInput.checked = false; allInput.dispatchEvent({type:"change"});
      if (allGroupNow().style.display !== "none")
        problems.push("关闭 All 后 pattern 框层仍显示");
      if (listAllItems().length || hasAllSection()
          || sel.classList.contains("has") || sel.innerHTML)
        problems.push("关闭 All 后调试侧栏/All detail 仍残留");
      if (targetRow) context.selectScope(targetRow.scopeId, false);
      if (allInput.checked || allGroupNow().style.display !== "none")
        problems.push("选择普通 FENCELINE 因 ALL_LT 缓存而强制重开 tgAll");
      const focusedNormalLines = vm.runInContext(
        "(LT_NODES.get(__patternNumber)||[]).filter(n=>"
        + "n.getAttribute('class')==='lt-line'&&n.classList.contains('lt-focus')).length",
        context);
      if (!focusedNormalLines)
        problems.push("关闭 tgAll 后选择普通 FENCELINE 没有正常高亮");
      allInput.checked = true; allInput.dispatchEvent({type:"change"});
      await Promise.resolve();
      const restoredBoxes = (allGroupNow().children || []).filter((node) =>
        node.getAttribute && node.getAttribute("class") === "al-pattern");
      if (allGroupNow().style.display === "none" || restoredBoxes.length !== 2)
        problems.push("再次打开 All 没从缓存恢复两个 pattern 框");
      if (listAllItems().length !== 2 || !hasAllSection())
        problems.push("再次打开 All 没从缓存恢复调试侧栏");

      metaInput.checked = false;
      metaInput.dispatchEvent({type:"change"});
      if (listAllItems().length || hasAllSection()
          || sel.classList.contains("has") || sel.innerHTML)
        problems.push("关闭 line-type diagnostics 后清单/详情仍残留");
      if (allGroupNow().style.display === "none"
          || (allGroupNow().children || []).length !== allNodeCount)
        problems.push("关闭 line-type diagnostics 误影响 tgAll 画布几何");
      allInput.checked = false; allInput.dispatchEvent({type:"change"});
      allInput.checked = true; allInput.dispatchEvent({type:"change"});
      await Promise.resolve();
      const finalBoxes = (allGroupNow().children || []).filter((node) =>
        node.getAttribute && node.getAttribute("class") === "al-pattern");
      if (allGroupNow().style.display === "none" || finalBoxes.length !== 2)
        problems.push("诊断关闭时 All 开→关→开没有从缓存恢复几何");
      if (listAllItems().length || hasAllSection())
        problems.push("诊断关闭时 All 开→关→开重新泄漏调试侧栏");
      toggleConsole();
      if (networkCalls)
        problems.push(`诊断/All 开关过程额外请求 ${networkCalls} 次`);
    }
  } catch (error) {
    const first = error && error.stack ? error.stack.split("\n")[0] : String(error);
    problems.push(`抛出 -> ${first}`);
  } finally {
    context.fetch = oldFetch;
    context.__patternSaved = saved;
    vm.runInContext(
      "CUR=__patternSaved.cur;PAGE=__patternSaved.page;epoch=__patternSaved.epoch;"
      + "ALL_LT=__patternSaved.all;ALL_LT_KEY=__patternSaved.key;"
      + "ALL_LT_STATE=__patternSaved.state;ALL_LT_DETAIL=__patternSaved.detail;"
      + "ALL_FOCUS=__patternSaved.focus;ALL_FILTER=__patternSaved.filter;"
      + "ALL_LT_INFLIGHT=__patternSaved.inflight;selectedScope=__patternSaved.selectedScope;"
      + "selectedRow=__patternSaved.selectedRow;"
      + "$('tgLt').querySelector('input').checked=__patternSaved.ltChecked;"
      + "$('tgAll').querySelector('input').checked=__patternSaved.allChecked;"
      + "$('tgLtSrc').querySelector('input').checked=__patternSaved.provChecked;"
      + "$('tgLtMeta').querySelector('input').checked=__patternSaved.metaChecked;"
      + "$('bar').classList.toggle('tucked',__patternSaved.barTucked);"
      + "if(PAGE){makeScopeModel();build();renderList();syncLayers();}", context,
      {filename:"method2-pattern-restore"});
  }
  if (problems.length) {
    console.log(`  FAIL Method 2 pattern 框: ${problems.join("; ")}`);
    failures += 1;
  } else if (!skipped) {
    console.log("  OK   客户 FENCELINE 纯文字；线型诊断/All 详情/Method 2 pattern/provenance 开关通过");
  }
}

// ---- All line types 缺失/过期时的自动生成状态机 -------------------------
// 真实服务若已有 .all.json，只能覆盖 GET=ok 的快路径；若没有，冒烟测试
// 又不应真的触发一次数分钟聚类。这里用可控 fetch 真正执行
// GET not-run -> building -> POST ok，并验证重复点击不重入、切页后旧 POST 不回写。
{
  const problems = [];
  const oldFetch = context.fetch;
  const oldSetTimeout = context.setTimeout;
  const oldClearTimeout = context.clearTimeout;
  const saved = vm.runInContext(
    "({cur:CUR,page:PAGE,epoch,all:ALL_LT,key:ALL_LT_KEY,state:ALL_LT_STATE,"
    + "detail:ALL_LT_DETAIL,focus:ALL_FOCUS,filter:ALL_FILTER,"
    + "checked:$('tgAll').querySelector('input').checked})", context);
  const response = (body, ok = true, status = 200) => ({
    ok, status, json: async () => body,
  });
  const full = {
    state: "ok",
    page: {path_ops: 2, owned_path_ops: 1},
    types: [{
      line_type_number: 901, signature_family: "synthetic_periodic",
      recognition_source: "method1", op_count: 1, segment_count: 1,
      member_count: 3, runs: [{run_id: "1"}], bound_by: [],
      bbox: [100, 100, 200, 200], polylines: [[[100, 100], [200, 200]]],
    }],
    residual: {op_count: 1, polylines: [[[250, 250], [300, 300]]]},
  };
  try {
    if (!saved.page) {
      console.log("  SKIP All line types 自动生成: 没有可用页面");
    } else {
      context.__allAutoPage = saved.page;
      vm.runInContext(
        "PAGE=__allAutoPage; CUR={slug:'all_autobuild_probe',page:PAGE.page}; epoch+=1;"
        + "ALL_LT=null; ALL_LT_KEY=''; ALL_LT_STATE='idle'; ALL_LT_DETAIL='';"
        + "ALL_LT_INFLIGHT=null; ALL_FOCUS=null; ALL_FILTER='';"
        + "$('tgAll').querySelector('input').checked=true; build(); renderList();",
        context, {filename: "all-autobuild-setup"});

      const calls = [], deadlines = [];
      let releasePost;
      let sawPost;
      const postSeen = new Promise((resolve) => { sawPost = resolve; });
      context.setTimeout = (_fn, ms) => { deadlines.push(ms); return deadlines.length; };
      context.clearTimeout = () => {};
      context.fetch = async (url, opt = {}) => {
        const method = String(opt.method || "GET").toUpperCase();
        calls.push({url: String(url), method});
        if (method === "GET") return response({state: "not-run"});
        sawPost();
        return await new Promise((resolve) => {
          releasePost = () => resolve(response(full));
        });
      };

      const first = context.ensureAllLinetypes();
      const reachedPost = await Promise.race([
        postSeen.then(() => true),
        new Promise((resolve) => globalThis.setTimeout(() => resolve(false), 2000)),
      ]);
      if (!reachedPost) {
        problems.push("GET not-run 后没有自动 POST");
      } else {
        const during = vm.runInContext(
          "({state:ALL_LT_STATE,inflight:!!ALL_LT_INFLIGHT})", context);
        if (during.state !== "building")
          problems.push(`POST 等待期状态=${during.state}，应为 building`);
        if (!during.inflight) problems.push("POST 等待期没有 inflight token");
        const duplicate = context.ensureAllLinetypes();
        await Promise.resolve();
        if (calls.length !== 2)
          problems.push(`并发勾选发出了 ${calls.length} 次请求，应为 GET+POST`);
        releasePost();
        await Promise.all([first, duplicate]);
      }
      const built = vm.runInContext(
        "({state:ALL_LT_STATE,count:ALL_NODES.size,"
        + "empty:[...ALL_NODES].filter(([,nodes])=>!nodes.length).length,"
        + "inflight:!!ALL_LT_INFLIGHT})", context);
      if (reachedPost && built.state !== "ok")
        problems.push(`POST 完成后状态=${built.state}`);
      if (reachedPost && (built.count !== 1 || built.empty))
        problems.push(`POST 完成后绘制结果 count=${built.count}, empty=${built.empty}`);
      if (built.inflight) problems.push("POST 完成后 inflight 未清理");
      if (calls.map((x) => x.method).join(",") !== "GET,POST")
        problems.push(`请求顺序=${calls.map((x) => x.method).join(",")}`);
      const fetchLimit = vm.runInContext("ALL_LT_FETCH_TIMEOUT_MS", context);
      const buildLimit = vm.runInContext("ALL_LT_BUILD_TIMEOUT_MS", context);
      if (!deadlines.includes(fetchLimit) || !deadlines.includes(buildLimit))
        problems.push(`没有使用 GET/POST 专用超时: ${deadlines.join(",")}`);
      if (buildLimit < 3600 * 1000)
        problems.push(`POST 超时 ${buildLimit}ms 短于密页边车上限`);

      // Legend-only 页的 GET 会带一个可验证但不完整的 supervised 子集。
      // partial 绝不能进入 ok/绘图终态，必须和 missing/stale 一样只触发
      // 一次完整 POST；最终画布只能来自 POST 返回的 full audit。
      vm.runInContext(
        "epoch+=1; ALL_LT=null; ALL_LT_KEY=''; ALL_LT_STATE='idle';"
        + "ALL_LT_DETAIL=''; ALL_LT_INFLIGHT=null; build();",
        context, {filename: "all-partial-setup"});
      const partialCalls = [];
      context.fetch = async (_url, opt = {}) => {
        const method = String(opt.method || "GET").toUpperCase();
        partialCalls.push(method);
        if (method === "GET") return response({
          state: "partial", partial: true, cache_state: "not-run",
          types: [{...full.types[0], line_type_number: 999}], residual: null,
        });
        return response(full);
      };
      await context.ensureAllLinetypes();
      const partialBuilt = vm.runInContext(
        "({state:ALL_LT_STATE,partial:!!(ALL_LT&&ALL_LT.partial),"
        + "hasFull:ALL_NODES.has(901),hasSubset:ALL_NODES.has(999),"
        + "inflight:!!ALL_LT_INFLIGHT})", context);
      if (partialCalls.join(",") !== "GET,POST")
        problems.push(`partial 请求顺序=${partialCalls.join(",")}`);
      if (partialBuilt.state !== "ok" || partialBuilt.partial
          || !partialBuilt.hasFull || partialBuilt.hasSubset)
        problems.push("partial 子集被误当成完整 ok 或写进最终画布");
      if (partialBuilt.inflight)
        problems.push("partial POST 完成后 inflight 未清理");

      // 再走 stale 分支，在 POST 飞行期间使页面代次失效。
      vm.runInContext(
        "epoch+=1; ALL_LT=null; ALL_LT_KEY=''; ALL_LT_STATE='idle';"
        + "ALL_LT_DETAIL=''; ALL_LT_INFLIGHT=null; build();",
        context, {filename: "all-stale-setup"});
      const staleCalls = [];
      let releaseStale;
      let sawStalePost;
      const stalePostSeen = new Promise((resolve) => { sawStalePost = resolve; });
      context.fetch = async (_url, opt = {}) => {
        const method = String(opt.method || "GET").toUpperCase();
        staleCalls.push(method);
        if (method === "GET") return response({state: "stale"});
        sawStalePost();
        return await new Promise((resolve) => {
          releaseStale = () => resolve(response({...full,
            types: [{...full.types[0], line_type_number: 902}]}));
        });
      };
      const staleRun = context.ensureAllLinetypes();
      const reachedStalePost = await Promise.race([
        stalePostSeen.then(() => true),
        new Promise((resolve) => globalThis.setTimeout(() => resolve(false), 2000)),
      ]);
      if (!reachedStalePost) {
        problems.push("GET stale 后没有自动 POST");
      } else {
        vm.runInContext("epoch+=1; build();", context,
          {filename: "all-change-page-generation"});
        releaseStale();
        await staleRun;
        const guarded = vm.runInContext(
          "({all:ALL_LT,state:ALL_LT_STATE,has902:ALL_NODES.has(902),"
          + "inflight:!!ALL_LT_INFLIGHT})", context);
        if (guarded.all !== null || guarded.state !== "idle" || guarded.has902)
          problems.push("切页后旧 POST 响应回写了新页状态/几何");
        if (guarded.inflight) problems.push("切页后旧 inflight 未收尾");
      }
      if (staleCalls.join(",") !== "GET,POST")
        problems.push(`stale 请求顺序=${staleCalls.join(",")}`);
    }
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 180)}`);
  } finally {
    context.fetch = oldFetch;
    context.setTimeout = oldSetTimeout;
    context.clearTimeout = oldClearTimeout;
    context.__allAutoSaved = saved;
    vm.runInContext(
      "CUR=__allAutoSaved.cur; PAGE=__allAutoSaved.page; epoch=__allAutoSaved.epoch;"
      + "ALL_LT=__allAutoSaved.all; ALL_LT_KEY=__allAutoSaved.key;"
      + "ALL_LT_STATE=__allAutoSaved.state; ALL_LT_DETAIL=__allAutoSaved.detail;"
      + "ALL_FOCUS=__allAutoSaved.focus; ALL_FILTER=__allAutoSaved.filter;"
      + "ALL_LT_INFLIGHT=null;"
      + "$('tgAll').querySelector('input').checked=__allAutoSaved.checked;"
      + "if(PAGE){build();renderList();syncLayers();}",
      context, {filename: "all-autobuild-restore"});
  }
  if (problems.length) {
    console.log(`  FAIL All line types 自动生成: ${problems.join("; ")}`);
    failures += 1;
  } else if (saved.page) {
    console.log("  OK   All line types 自动生成: GET→building→POST；去重、长超时和切页防回写通过");
  }
}

if (errors.length) {
  console.log(`  注意：requestAnimationFrame 回调里有 ${errors.length} 个异常`);
  failures += errors.length;
}

// ---- 同一页重载也必须作废旧调试线型 --------------------------------------
// 后台把 stale 主结果原子换成 current 后，show() 会再次打开同一 slug/page。
// 若缓存 key 只有 slug|page，旧的 ALL_LT 几何会绕过后端签名闸继续重画。
{
  const problems = [];
  try {
    const state = JSON.parse(vm.runInContext(
      "(()=>{if(!PAGE)return JSON.stringify({skip:true});"
      + "if(!CUR)CUR={slug:'same_page_reload',page:PAGE.page};"
      + "ALL_LT={types:[],residual:null}; ALL_LT_STATE='ok';"
      + "ALL_LT_KEY=allKey(); const oldKey=ALL_LT_KEY; epoch+=1; build();"
      + "return JSON.stringify({oldKey,newKey:allKey(),cleared:ALL_LT===null,state:ALL_LT_STATE});})()",
      context, { filename: "same-page-linetype-generation" }));
    if (!state.skip) {
      if (state.oldKey === state.newKey) problems.push("同页重载没有更换缓存代次 key");
      if (!state.cleared) problems.push("同页重载后旧 ALL_LT 仍在内存");
      if (state.state !== "idle") problems.push(`清理后状态=${state.state}，应为 idle`);
    }
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 160)}`);
  }
  if (problems.length) {
    console.log(`  FAIL 同页调试线型作废: ${problems.join("; ")}`);
    failures += 1;
  } else {
    console.log("  OK   同页调试线型作废: 页面代次变化会清除旧几何");
  }
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

// ---- 从任意入口选 callout 都必须聚焦它绑定的线型 --------------------------
// 真实故障：普通 FENCE TEXT & GATES 行曾显式传 withLines=false；后端已经把
// callout 箭头末端绑到线型，点击那行却给所有线加 lt-dim。只有再点一遍重复的
// FENCELINE 行才看得到线。这里实际点击普通行，再走一次图上箭头共用的入口；
// 最后用一个无绑定 scope 验 gate / residual 不会继承上一条线的焦点。
{
  const problems = [];
  let skipped = false;
  try {
    const snapshot = JSON.parse(vm.runInContext(
      "JSON.stringify({rows:(LT_INDEX.rows||[]).map(r=>({scopeId:r.scopeId,number:r.number})),"
      + "scopes:SCOPE_GROUPS.map(g=>({id:g.id,gate:!!g.gate}))})", context));
    const boundRows = snapshot.rows.filter(
      (row) => snapshot.scopes.some((g) => g.id === row.scopeId));
    // Civil P4 的实际回归目标是 PROPOSED FENCE -> #24；该型存在时优先点它，
    // 让冒烟测试不只碰巧验证同页排在前面的 #4。
    const hit = boundRows.find((row) => row.number === 24) || boundRows[0];
    if (!hit) {
      skipped = true;
      console.log("  SKIP 绑定线型选择: 这一页没有可点击的 bound callout");
    } else {
      context.__target = hit.scopeId;
      context.__number = hit.number;
      vm.runInContext("ALL_LT=null; clearScopeSelection();", context,
        { filename: "reset-bound-focus" });
      context.renderList();
      const rows = (context.document.getElementById("list").children || [])
        .filter((n) => n.classList && n.classList.contains("scope-item"));
      const ordinary = rows.find((n) => n.dataset && n.dataset.scope === hit.scopeId
        && n.dataset.row !== `FENCELINE|${hit.scopeId}`);
      if (!ordinary || typeof ordinary.onclick !== "function") {
        problems.push("找不到 bound callout 在普通分节里的可点击行");
      } else {
        ordinary.onclick();
        const focusState = () => JSON.parse(vm.runInContext(
          "JSON.stringify({selectedScope,layer:$('tgLt').querySelector('input').checked,"
          + "target:[...(LT_NODES.get(__number)||[])].map(n=>({focus:n.classList.contains('lt-focus'),dim:n.classList.contains('lt-dim')})),"
          + "other:[...LT_NODES].filter(([n])=>n!==__number).flatMap(([,nodes])=>nodes.map(n=>({focus:n.classList.contains('lt-focus'),dim:n.classList.contains('lt-dim')})))})",
          context));
        let state = focusState();
        if (state.selectedScope !== hit.scopeId) problems.push("普通 callout 行没有选中目标 scope");
        if (!state.layer) problems.push("普通 callout 行没有自动打开 Line types 图层");
        if (!state.target.length) problems.push(`目标线型 #${hit.number} 没有 SVG 节点`);
        if (state.target.some((n) => !n.focus || n.dim))
          problems.push("普通 callout 行没有只聚焦目标线型");
        if (state.other.some((n) => n.focus || !n.dim))
          problems.push("普通 callout 行没有隐藏其他线型");

        // 图上的文字 / 引线 / 箭头最终都调用 selectScope(scopeId, false)。
        vm.runInContext("clearScopeSelection(); selectScope(__target,false);", context,
          { filename: "arrow-scope-focus" });
        state = focusState();
        if (!state.target.length || state.target.some((n) => !n.focus || n.dim))
          problems.push("图上 callout / 箭头入口没有聚焦目标线型");
        if (state.other.some((n) => n.focus || !n.dim))
          problems.push("图上 callout / 箭头入口没有隐藏其他线型");

        const unbound = snapshot.scopes.find(
          (scope) => scope.gate
            && !snapshot.rows.some((row) => row.scopeId === scope.id))
          || snapshot.scopes.find(
            (scope) => !snapshot.rows.some((row) => row.scopeId === scope.id));
        if (unbound) {
          context.__unbound = unbound.id;
          const none = JSON.parse(vm.runInContext(
            "selectScope(__unbound,false); JSON.stringify([...LT_NODES.values()].flatMap(nodes=>nodes.map(n=>({focus:n.classList.contains('lt-focus'),dim:n.classList.contains('lt-dim')}))))",
            context, { filename: "unbound-scope-focus" }));
          if (none.some((n) => n.focus || !n.dim))
            problems.push("gate / residual scope 仍继承了上一条线型焦点");
        }
      }
    }
  } catch (error) {
    problems.push(`抛出 -> ${String(error).slice(0, 160)}`);
  }
  if (problems.length) {
    console.log(`  FAIL 绑定线型选择: ${problems.join("; ")}`);
    failures += 1;
  } else if (!skipped) {
    console.log("  OK   绑定线型选择: 普通 callout / 箭头自动聚焦；gate / residual 不继承焦点");
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

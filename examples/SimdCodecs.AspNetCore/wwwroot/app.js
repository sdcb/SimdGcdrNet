const $ = id => document.getElementById(id);
const fileInput = $("file"), backendSelect = $("backend"), formatSelect = $("format");
const rerunButton = $("rerun"), downloadLink = $("download"), statusText = $("status");
const stage = $("stage"), compare = $("compare"), placeholder = $("placeholder");
const before = $("before"), after = $("after"), divider = $("divider"), slider = $("slider");

let current = null; // { blob, name }
let beforeUrl = null, afterUrl = null;

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function setSplit(percent) {
  after.style.clipPath = `inset(0 0 0 ${percent}%)`;
  divider.style.left = `${percent}%`;
}

async function enhance() {
  if (!current) return;
  rerunButton.disabled = true;
  setStatus("增强中…");

  const form = new FormData();
  form.append("file", current.blob, current.name);
  form.append("backend", backendSelect.value);
  form.append("format", formatSelect.value);

  try {
    const started = performance.now();
    const response = await fetch("/api/enhance", { method: "POST", body: form });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(error.error || response.statusText);
    }
    const blob = await response.blob();
    const roundTrip = performance.now() - started;

    if (afterUrl) URL.revokeObjectURL(afterUrl);
    afterUrl = URL.createObjectURL(blob);
    after.src = afterUrl;
    downloadLink.href = afterUrl;
    downloadLink.download = current.name.replace(/\.[^.]*$/, "") + "-enhanced." + formatSelect.value;
    downloadLink.hidden = false;

    const h = name => response.headers.get(name);
    setStatus(`${h("X-Image-Size")} · ${h("X-Backend")} · 增强 ${h("X-Enhance-Ms")} ms` +
      ` · 解码 ${h("X-Decode-Ms")} ms · 编码 ${h("X-Encode-Ms")} ms · 往返 ${roundTrip.toFixed(0)} ms`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    rerunButton.disabled = false;
  }
}

function load(blob, name) {
  current = { blob, name };
  if (beforeUrl) URL.revokeObjectURL(beforeUrl);
  beforeUrl = URL.createObjectURL(blob);
  before.src = beforeUrl;
  after.removeAttribute("src");
  downloadLink.hidden = true;
  compare.hidden = false;
  placeholder.hidden = true;
  slider.value = 50;
  setSplit(50);
  enhance();
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) load(file, file.name);
  fileInput.value = "";
});

$("example").addEventListener("click", async () => {
  setStatus("下载示例图片…");
  const response = await fetch("example.jpg");
  if (!response.ok) return setStatus("示例图片不存在", true);
  load(await response.blob(), "example.jpg");
});

rerunButton.addEventListener("click", enhance);
backendSelect.addEventListener("change", enhance);
formatSelect.addEventListener("change", enhance);
slider.addEventListener("input", () => setSplit(slider.value));

stage.addEventListener("dragover", e => { e.preventDefault(); stage.classList.add("dragover"); });
stage.addEventListener("dragleave", () => stage.classList.remove("dragover"));
stage.addEventListener("drop", e => {
  e.preventDefault();
  stage.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) load(file, file.name);
});

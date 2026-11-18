// Adds an "Open this lesson in JupyterLab" button to every lesson page.
document.addEventListener("DOMContentLoaded", () => {
  const page = location.pathname.split("/").pop().replace(/\.html$/, "") || "00_welcome";
  if (!/^\d\d_/.test(page)) return;
  const url = `${location.protocol}//${location.hostname}:8888/lab/tree/workspace/${page}.md`;
  const box = document.createElement("div");
  box.style.cssText = "margin:0 0 1.2em;padding:.6em 1em;border-radius:6px;background:var(--pst-color-surface,#eef);";
  box.innerHTML = `🧪 <b>Hands-on:</b> <a href="${url}" target="_blank">open this lesson in JupyterLab</a> to run the code and do the exercises.`;
  const article = document.querySelector("article") || document.querySelector("main");
  if (article) article.prepend(box);
});

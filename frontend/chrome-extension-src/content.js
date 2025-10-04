(function() {
  try {
    // Avoid duplicate runs on SPA navigation
    if (window.__dejaview_captured__) return;
    window.__dejaview_captured__ = true;

    // clone document to avoid modifying original DOM used by page scripts
    const docClone = document.cloneNode(true);
    const reader = new Readability(docClone);
    const article = reader.parse();

    if (!article || !article.textContent) {
      // fallback: short body text
      const fallback = (document.querySelector("main, article") || document.body).innerText || "";
      const fallParagraphs = fallback.split("\n").map(s => s.trim()).filter(s => s.length > 50).map((p,i) => ({index:i, text:p}));
      chrome.runtime.sendMessage({ action: "capturePage", data: { url: window.location.href, title: document.title, timestamp: new Date().toISOString(), paragraphs: fallParagraphs } });
      return;
    }

    const paragraphs = article.textContent
      .split("\n")
      .map(p => p.trim())
      .filter(p => p.length > 50) // drop tiny lines
      .flatMap(p => {
        // Split paragraphs longer than 1000 characters
        if (p.length > 1000) {
          return p.match(/.{1,1000}/g) || [p];
        }
        return [p];
      })
      .map((p, i) => ({ index: i, text: p }));

    chrome.runtime.sendMessage({
      action: "capturePage",
      data: {
        url: window.location.href,
        title: article.title || document.title,
        timestamp: new Date().toISOString(),
        paragraphs: paragraphs
      }
    });
    console.log("DejaView: sent", paragraphs.length, "paragraphs for", window.location.href);
  } catch (err) {
    console.error("DejaView content extraction failed:", err);
  }
})();

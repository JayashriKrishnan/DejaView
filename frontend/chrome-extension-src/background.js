chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "capturePage") {
    const payload = message.data;
    // Adjust host if needed
    const url = "http://localhost:5000/capture";

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(j => console.log("DejaView backend response:", j))
    .catch(err => console.error("DejaView: failed to send capture:", err));
  }
});

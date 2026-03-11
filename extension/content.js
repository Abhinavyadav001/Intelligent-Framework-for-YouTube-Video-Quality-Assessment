console.log("YouTube Analyzer script loaded");
alert("YouTube Analyzer extension loaded!");

function insertPanel() {

    if (document.getElementById("yt-analyzer-panel")) return;

    const params = new URLSearchParams(window.location.search);
    const videoId = params.get("v");

    if (!videoId) return;

    const panel = document.createElement("div");
    panel.id = "yt-analyzer-panel";

    panel.innerHTML = `
        <h3>YouTube Analyzer</h3>
        <p>Video ID: ${videoId}</p>
        <p>Spam Detection: Ready</p>
        <p>Sentiment Analysis: Coming Soon</p>
    `;

    panel.style.position = "fixed";
    panel.style.top = "120px";
    panel.style.right = "20px";
    panel.style.width = "260px";
    panel.style.background = "#fff";
    panel.style.border = "1px solid #ccc";
    panel.style.padding = "12px";
    panel.style.zIndex = "9999";
    panel.style.boxShadow = "0 0 10px rgba(0,0,0,0.2)";
    panel.style.fontFamily = "Arial";

    document.body.appendChild(panel);
}

function observeUrlChange() {

    let lastUrl = location.href;

    new MutationObserver(() => {

        if (location.href !== lastUrl) {
            lastUrl = location.href;

            const oldPanel = document.getElementById("yt-analyzer-panel");
            if (oldPanel) oldPanel.remove();

            setTimeout(insertPanel, 1500);
        }

    }).observe(document, {subtree: true, childList: true});
}

insertPanel();
observeUrlChange();
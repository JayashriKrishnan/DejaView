document.addEventListener('DOMContentLoaded', function() {
  const openDashboardBtn = document.getElementById('openDashboard');

  openDashboardBtn.addEventListener('click', function() {
    // Open the dashboard served by Vite dev server
    chrome.tabs.create({
      url: 'http://localhost:5173/'
    });
  });
});

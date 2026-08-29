(function() {
    "use strict";

    const statusIcon = document.getElementById('statusIcon');
    const statusText = document.getElementById('statusText');
    const statusSub = document.getElementById('statusSub');
    const confidenceValue = document.getElementById('confidenceValue');
    const statusCard = document.getElementById('statusCard');
    const logsBody = document.getElementById('logsBody');
    const previewImg = document.getElementById('previewImg');
    const cameraError = document.getElementById('cameraError');

    // Handle stream errors
    previewImg.onerror = () => {
        previewImg.style.display = 'none';
        cameraError.style.display = 'flex';
    };

    previewImg.onload = () => {
        previewImg.style.display = 'block';
        cameraError.style.display = 'none';
    };

    function getStatusMeta(status) {
        let icon = 'fa-hourglass-half';
        let cssClass = 'waiting';
        let label = status || 'unknown';

        if (status === 'ALIVE') {
            icon = 'fa-heartbeat';
            cssClass = 'alive';
        } else if (status === 'DEAD') {
            icon = 'fa-skull';
            cssClass = 'dead';
        } else if (status === 'No Clam Detected') {
            icon = 'fa-eye-slash';
            cssClass = 'waiting';
        } else if (status === 'GoPro Error') {
            icon = 'fa-exclamation-triangle';
            cssClass = 'dead';
        }

        return { icon, cssClass, label };
    }

    async function fetchData() {
        try {
            // 1) Fetch Status
            const statusRes = await fetch('/api/status');
            const statusData = await statusRes.json();

            const rawStatus = statusData.status || 'Waiting...';
            const meta = getStatusMeta(rawStatus);
            
            statusIcon.innerHTML = `<i class="fas ${meta.icon}"></i>`;
            statusText.textContent = rawStatus;
            statusSub.innerHTML = `<i class="far fa-clock"></i> ${statusData.timestamp || '...'}`;
            
            let conf = statusData.confidence;
            confidenceValue.textContent = (typeof conf === 'number' && !isNaN(conf)) ? conf.toFixed(1) : '—';
            
            statusCard.className = `status-card ${meta.cssClass}`;

            // 2) Fetch Logs
            const logsRes = await fetch('/api/logs');
            const logsData = await logsRes.json();

            if (!Array.isArray(logsData) || logsData.length === 0) {
                logsBody.innerHTML = `<tr><td colspan="4" class="empty-state"><i class="fas fa-inbox"></i> No logs available yet.</td></tr>`;
                return;
            }

            let html = '';
            for (let row of logsData) {
                if (!Array.isArray(row) || row.length < 5) continue;
                const [timestamp, status, confidence, details, imgPath] = row;

                let badgeClass = 'badge-error';
                let badgeIcon = 'fa-circle';
                
                if (status === 'ALIVE') {
                    badgeClass = 'badge-alive';
                    badgeIcon = 'fa-heart';
                } else if (status === 'DEAD') {
                    badgeClass = 'badge-dead';
                    badgeIcon = 'fa-skull';
                } else if (status === 'No Clam Detected') {
                    badgeClass = 'badge-waiting';
                    badgeIcon = 'fa-eye-slash';
                } else if (status === 'GoPro Error') {
                    badgeClass = 'badge-error';
                    badgeIcon = 'fa-exclamation-triangle';
                }

                let confDisplay = (typeof confidence === 'number' && !isNaN(confidence)) 
                    ? confidence.toFixed(1) : '—';

                let imgHtml = `<span class="text-muted">—</span>`;
                if (imgPath && typeof imgPath === 'string' && imgPath.trim() !== '') {
                    const cleanPath = imgPath.startsWith('/') ? imgPath : '/' + imgPath;
                    imgHtml = `<a href="${cleanPath}" target="_blank"><img src="${cleanPath}" class="snapshot-img" alt="snapshot" loading="lazy"></a>`;
                }

                html += `<tr>
                    <td style="white-space:nowrap;">${timestamp || '—'}</td>
                    <td><span class="badge ${badgeClass}"><i class="fas ${badgeIcon}"></i> ${status || 'unknown'}</span></td>
                    <td><span class="pill">${confDisplay}%</span></td>
                    <td>${imgHtml}</td>
                </tr>`;
            }

            logsBody.innerHTML = html || `<tr><td colspan="4" class="empty-state">No records found.</td></tr>`;

        } catch (error) {
            console.error('Fetch error:', error);
            statusCard.className = 'status-card dead';
            statusText.textContent = 'Connection Error';
            statusSub.innerHTML = '<i class="fas fa-wifi"></i> Retrying...';
        }
    }

    // Initial Load + Refresh every 3 seconds for status/logs only
    fetchData();
    setInterval(fetchData, 3000);
})();
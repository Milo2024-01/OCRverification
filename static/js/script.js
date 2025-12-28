const form = document.getElementById("uploadForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const formData = new FormData(form);
    
    const res = await fetch("/analyze", {
        method: "POST",
        body: formData
    });
    
    const data = await res.json();
    
    if (data.error) {
        resultDiv.innerHTML = `<p style="color:red;">${data.error}</p>`;
    } else {
        resultDiv.innerHTML = `
            <p>Status: <b>${data.status || data['face_similarity']}</b></p>
            <p>Reason: ${data.reason || ''}</p>
            <p>Face Similarity: ${data.face_similarity ? data.face_similarity.toFixed(2) : 'N/A'}</p>
        `;
    }
});

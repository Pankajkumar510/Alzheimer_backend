const submitBtn = document.getElementById('submitBtn');
const resultEl = document.getElementById('result');
submitBtn.addEventListener('click', async () => {
  const fd = new FormData();
  const pid = document.getElementById('patientId').value || 'patient_' + Date.now();
  fd.append('patient_id', pid);
  const mri = document.getElementById('mriFile').files[0];
  const pet = document.getElementById('petFile').files[0];
  const cognitive = document.getElementById('cognitive').value;
  if (mri) fd.append('mri_file', mri);
  if (pet) fd.append('pet_file', pet);
  if (cognitive) fd.append('cognitive', cognitive);
  resultEl.textContent = 'Running inference...';
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const json = await res.json();
    resultEl.textContent = JSON.stringify(json, null, 2);
  } catch (e) {
    resultEl.textContent = 'Error: ' + e;
  }
});

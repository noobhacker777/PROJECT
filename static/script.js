const progText = document.getElementById('prog_text');
const resBox = document.getElementById('res');

async function poll(){
  try{
    const r = await fetch('/training-progress');
    const j = await r.json();
    progText.textContent = j.status || JSON.stringify(j);
  }catch(e){ progText.textContent = 'n/a' }
}
setInterval(poll,2000);

document.getElementById('send').onclick = async ()=>{
  const f = document.getElementById('fileinp').files[0];
  if(!f){resBox.textContent='choose file';return}
  const fd = new FormData(); fd.append('image', f);
  const r = await fetch('/predict', {method:'POST', body:fd});
  const j = await r.json(); resBox.textContent = JSON.stringify(j,null,2);
}

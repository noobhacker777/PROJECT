// HOLO Unified Master Dashboard JavaScript
const out = (id, text) => { const e = document.getElementById(id); if (e) e.textContent = text; };
const outAppend = (id, text) => { const e = document.getElementById(id); if (e) e.textContent += '\n' + text; };

// Track unsaved changes
let labelHasUnsavedChanges = false;
function markLabelDirty() { labelHasUnsavedChanges = true; }
function clearLabelDirty() { labelHasUnsavedChanges = false; }

// ============================================================================
// FORMAT CONVERSION: HOLO ↔ Label Studio (Zero-Drift)
// ============================================================================
// HOLO Format (Internal/Canonical):
//   [x_center, y_center, width, height] normalized to [0, 1], center-based
// Label Studio Format (Display/Export):
//   [x_top_left, y_top_left, width, height] in percentages [0, 100], top-left-based

function clamp(value, min_val = 0, max_val = 1) {
  return Math.max(min_val, Math.min(max_val, value));
}

// ============================================================
// DATASET MANAGER - RAW IMAGE MANAGEMENT
// ============================================================

async function loadDatasetImages() {
  try {
    out('datasetOut', '⏳ Loading raw images from dataset/images/...');
    
    const r = await fetch('/api/dataset/raw-images');
    const data = await r.json();
    
    if (!data.ok) {
      out('datasetOut', '❌ Error: ' + (data.error || 'Failed to load images'));
      return;
    }
    
    const images = data.images || [];
    
    // Build file explorer tree
    buildDatasetFileExplorer(images);
    
    // Populate image grid
    populateDatasetImageGrid(images);
    
    // Update total count
    document.getElementById('datasetTotalImages').textContent = images.length;
    
    out('datasetOut', `✅ Loaded ${images.length} raw image(s) from dataset/images/`);
    
  } catch (e) {
    out('datasetOut', '❌ Error: ' + e.message);
  }
}

function buildDatasetFileExplorer(images) {
  const treeContainer = document.getElementById('datasetFileExplorer');
  if (!treeContainer) return;
  
  if (images.length === 0) {
    treeContainer.innerHTML = '<div style="padding:10px; text-align:center; color:#999; font-size:11px;">No files uploaded</div>';
    document.getElementById('datasetFilesCount').textContent = '0';
    return;
  }
  
  // Group images by file extension
  const byExt = {};
  images.forEach(img => {
    const ext = img.name.split('.').pop().toUpperCase();
    if (!byExt[ext]) byExt[ext] = [];
    byExt[ext].push(img);
  });
  
  let html = '';
  const extensions = Object.keys(byExt).sort();
  
  extensions.forEach(ext => {
    const files = byExt[ext];
    html += `<div style="margin-bottom:8px;">
      <div onclick="toggleDatasetFileGroup(this)" style="cursor:pointer; padding:8px; background:#e8f4f8; border-radius:4px; font-weight:bold; font-size:12px; display:flex; align-items:center; gap:6px; user-select:none; transition:background 0.2s;" onmouseover="this.style.background='#d0e8f2'" onmouseout="this.style.background='#e8f4f8'">
        <span style="font-size:14px;">▼</span>
        <span>${ext}</span>
        <span style="color:#999; font-size:10px;">(${files.length})</span>
      </div>
      <div style="padding-left:12px; margin-top:4px; display:block;">`;
    
    files.forEach((img, idx) => {
      html += `<div onclick="selectDatasetFileFromExplorer('${img.name}')" style="cursor:pointer; padding:6px 8px; margin:3px 0; border-radius:3px; font-size:11px; background:white; border:1px solid #ddd; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; transition:all 0.2s;" onmouseover="this.style.background='#0dcaf0'; this.style.color='white'; this.style.fontWeight='bold';" onmouseout="this.style.background='white'; this.style.color='#333'; this.style.fontWeight='normal';">
        📄 ${img.name}
      </div>`;
    });
    
    html += `</div></div>`;
  });
  
  treeContainer.innerHTML = html;
  document.getElementById('datasetFilesCount').textContent = images.length;
}

function toggleDatasetFileGroup(element) {
  const group = element.nextElementSibling;
  if (group) {
    const isOpen = group.style.display !== 'none';
    group.style.display = isOpen ? 'none' : 'block';
    const arrow = element.querySelector('span:first-child');
    if (arrow) arrow.textContent = isOpen ? '▶' : '▼';
  }
}

function selectDatasetFileFromExplorer(imageName) {
  // Find and highlight image in grid
  const imageCards = document.querySelectorAll('#datasetImageGrid div[data-image-name]');
  imageCards.forEach(card => {
    if (card.getAttribute('data-image-name') === imageName) {
      card.style.border = '3px solid #0dcaf0';
      card.style.boxShadow = '0 0 12px rgba(13,202,240,0.6)';
      card.style.transform = 'scale(1.05)';
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
      card.style.border = '2px solid #ddd';
      card.style.boxShadow = 'none';
      card.style.transform = 'scale(1)';
    }
  });
}

function populateDatasetImageGrid(images) {
  const gridContainer = document.getElementById('datasetImageGrid');
  gridContainer.innerHTML = '';
  
  if (images.length === 0) {
    gridContainer.innerHTML = '<div style="grid-column:1/-1; padding:40px; text-align:center; color:#999; font-size:13px;">📁 No images uploaded yet. Use the upload button to add images.</div>';
    return;
  }
  
  images.forEach(img => {
    const imgCard = document.createElement('div');
    imgCard.setAttribute('data-image-name', img.name);
    imgCard.style.cssText = 'border:2px solid #ddd; border-radius:8px; overflow:hidden; background:#f9f9f9; transition:all 0.3s; cursor:pointer; display:flex; flex-direction:column;';
    imgCard.onmouseover = () => {
      imgCard.style.boxShadow = '0 6px 16px rgba(13,202,240,0.4)';
      imgCard.style.transform = 'translateY(-4px)';
    };
    imgCard.onmouseout = () => {
      imgCard.style.boxShadow = 'none';
      imgCard.style.transform = 'translateY(0)';
    };
    
    const imgElement = document.createElement('img');
    imgElement.src = img.url || img.path;
    imgElement.style.cssText = 'width:100%; height:120px; object-fit:cover; cursor:pointer; background:#e0e0e0;';
    imgElement.onclick = () => previewDatasetImage(img.url || img.path, img.name);
    imgElement.onerror = () => {
      imgElement.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 150"%3E%3Crect fill="%23ddd" width="200" height="150"/%3E%3Ctext x="50%" y="50%" text-anchor="middle" dy=".3em" fill="%23999" font-size="14"%3E❌ Error%3C/text%3E%3C/svg%3E';
    };
    
    const nameDiv = document.createElement('div');
    nameDiv.style.cssText = 'padding:8px; font-size:11px; font-weight:bold; color:#333; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;';
    nameDiv.textContent = img.name;
    nameDiv.title = img.name;
    
    const buttonDiv = document.createElement('div');
    buttonDiv.style.cssText = 'padding:6px; display:flex; gap:4px; background:#f0f0f0;';
    
    const previewBtn = document.createElement('button');
    previewBtn.textContent = '👁 Preview';
    previewBtn.style.cssText = 'flex:1; padding:6px; background:#0dcaf0; color:white; border:none; border-radius:3px; cursor:pointer; font-size:10px; font-weight:bold; transition:background 0.2s;';
    previewBtn.title = 'Preview image';
    previewBtn.onmouseover = () => previewBtn.style.background = '#0ab8e6';
    previewBtn.onmouseout = () => previewBtn.style.background = '#0dcaf0';
    previewBtn.onclick = (e) => {
      e.stopPropagation();
      previewDatasetImage(img.url || img.path, img.name);
    };
    
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '🗑 Delete';
    deleteBtn.style.cssText = 'flex:1; padding:6px; background:#dc3545; color:white; border:none; border-radius:3px; cursor:pointer; font-size:10px; font-weight:bold; transition:background 0.2s;';
    deleteBtn.title = 'Delete image';
    deleteBtn.onmouseover = () => deleteBtn.style.background = '#c82333';
    deleteBtn.onmouseout = () => deleteBtn.style.background = '#dc3545';
    deleteBtn.onclick = (e) => {
      e.stopPropagation();
      deleteDatasetImage(img.name);
    };
    
    buttonDiv.appendChild(previewBtn);
    buttonDiv.appendChild(deleteBtn);
    imgCard.appendChild(imgElement);
    imgCard.appendChild(nameDiv);
    imgCard.appendChild(buttonDiv);
    gridContainer.appendChild(imgCard);
  });
}

function previewDatasetImage(imagePath, imageName) {
  const modal = document.createElement('div');
  modal.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); display:flex; align-items:center; justify-content:center; z-index:6000;';
  
  const container = document.createElement('div');
  container.style.cssText = 'background:white; padding:20px; border-radius:12px; max-width:90vw; max-height:90vh; overflow:auto; position:relative; box-shadow:0 10px 40px rgba(0,0,0,0.3);';
  
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.style.cssText = 'position:absolute; top:10px; right:10px; background:#dc3545; color:white; border:none; border-radius:50%; width:40px; height:40px; font-size:24px; cursor:pointer; font-weight:bold; transition:background 0.2s;';
  closeBtn.onmouseover = () => closeBtn.style.background = '#c82333';
  closeBtn.onmouseout = () => closeBtn.style.background = '#dc3545';
  closeBtn.onclick = () => modal.remove();
  
  const img = document.createElement('img');
  img.src = imagePath;
  img.style.cssText = 'max-width:100%; max-height:80vh; border:2px solid #ddd; border-radius:8px; display:block; margin:0 auto;';
  
  const title = document.createElement('h3');
  title.textContent = imageName;
  title.style.cssText = 'margin:0 0 10px 0; color:#333; text-align:center;';
  
  container.appendChild(closeBtn);
  container.appendChild(title);
  container.appendChild(img);
  modal.appendChild(container);
  document.body.appendChild(modal);
  
  modal.onclick = (e) => {
    if (e.target === modal) modal.remove();
  };
}

function searchDatasetImages() {
  const searchTerm = document.getElementById('datasetSearchInput').value.toLowerCase().trim();
  
  if (!window.currentDatasetImages) return;
  
  const filtered = window.currentDatasetImages.filter(img => 
    img.name.toLowerCase().includes(searchTerm)
  );
  
  populateDatasetImageGrid(filtered);
}

function clearDatasetSearch() {
  document.getElementById('datasetSearchInput').value = '';
  searchDatasetImages();
}

async function uploadDatasetImages() {
  const fileInput = document.getElementById('datasetUploadInput');
  const files = fileInput.files;
  
  if (files.length === 0) {
    alert('⚠️ Please select images to upload');
    return;
  }
  
  try {
    out('datasetOut', `⏳ Uploading ${files.length} image(s) to dataset/images/...`);
    
    const formData = new FormData();
    for (let file of files) {
      formData.append('files', file);
    }
    
    // Show progress bar
    document.getElementById('datasetUploadProgress').style.display = 'block';
    const progressFill = document.querySelector('#datasetUploadProgress > div');
    
    const r = await fetch('/api/dataset/raw-upload', {
      method: 'POST',
      body: formData
    });
    
    const data = await r.json();
    
    if (data.ok) {
      progressFill.style.width = '100%';
      out('datasetOut', `✅ Uploaded ${data.count} image(s) to dataset/images/`);
      
      fileInput.value = '';
      
      setTimeout(() => {
        document.getElementById('datasetUploadProgress').style.display = 'none';
        loadDatasetImages();
      }, 500);
    } else {
      out('datasetOut', '❌ Error: ' + (data.error || 'Upload failed'));
      document.getElementById('datasetUploadProgress').style.display = 'none';
    }
  } catch (e) {
    out('datasetOut', '❌ Error: ' + e.message);
    document.getElementById('datasetUploadProgress').style.display = 'none';
  }
}

async function deleteDatasetImage(imageName) {
  if (!confirm(`Delete raw image: ${imageName}?`)) return;
  
  try {
    const r = await fetch('/api/dataset/raw-delete', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_name: imageName })
    });
    
    const data = await r.json();
    
    if (data.ok || r.ok) {
      out('datasetOut', `✅ Deleted: ${imageName}`);
      loadDatasetImages();
    } else {
      alert('❌ Error: ' + (data.error || 'Failed to delete'));
    }
  } catch (e) {
    alert('❌ Error: ' + e.message);
  }
}

function round6decimal(value) {
  return Math.round(value * 1000000) / 1000000;
}

function holoToLabelStudio(x_center, y_center, width, height) {
  // Convert HOLO center format to Label Studio top-left percentage format
  let x_tl = (x_center - width / 2) * 100;
  let y_tl = (y_center - height / 2) * 100;
  let width_pct = width * 100;
  let height_pct = height * 100;
  
  x_tl = clamp(x_tl, 0, 100);
  y_tl = clamp(y_tl, 0, 100);
  width_pct = clamp(width_pct, 0, 100);
  height_pct = clamp(height_pct, 0, 100);
  
  return [round6decimal(x_tl), round6decimal(y_tl), round6decimal(width_pct), round6decimal(height_pct)];
}

function labelstudioToHolo(x_tl, y_tl, width_pct, height_pct) {
  // Convert Label Studio top-left percentage format to HOLO center format
  let x_tl_norm = x_tl / 100;
  let y_tl_norm = y_tl / 100;
  let width_norm = width_pct / 100;
  let height_norm = height_pct / 100;
  
  let x_center = x_tl_norm + (width_norm / 2);
  let y_center = y_tl_norm + (height_norm / 2);
  
  x_center = clamp(x_center, 0, 1);
  y_center = clamp(y_center, 0, 1);
  width_norm = clamp(width_norm, 0, 1);
  height_norm = clamp(height_norm, 0, 1);
  
  return [round6decimal(x_center), round6decimal(y_center), round6decimal(width_norm), round6decimal(height_norm)];
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  // Ctrl+S to save labels AND crop/save images
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    // Crop and save boxes first, then save JSON
    cropAndSaveBoxes().then(() => {
      setTimeout(() => labelSave(), 500);
    }).catch(err => {
      console.error('Crop error:', err);
      labelSave();  // Still save JSON even if crops fail
    });
  }
});

let labelImg = new Image(), labelBoxes = [], labelDragging = false, labelCurrent = null;
let labelClassNames = ['class0']; // class names loaded from server
let labelExistingCount = 0; // track how many boxes were loaded from file
let labelSelectedBox = -1; // index of selected box for editing (-1 = none)

const labelCanvas = document.getElementById('labelCanvas');
const labelCtx = labelCanvas ? labelCanvas.getContext('2d') : null;
// interactive handles for resizing selected box
// handle sizes (corner square and edge bars)
let _lbl_cornerSize = 16; // corner square size in pixels
let _lbl_edgeThickness = 12; // edge bar thickness in pixels
let _lbl_handleDragging = false;
let _lbl_activeHandle = null; // {idx, corner}
let _lbl_handleOrig = null; // snapshot of box before drag
let classStatsCache = null;
let monitorInterval = null;
let _lbl_visualPad = 3; // visual + hit padding in pixels around handles
let _lbl_hoverHandle = null; // {idx, corner} when mouse near any handle

// Helper: return handle under a point (pixel coords relative to canvas)
function getHandleUnderPoint(px, py) {
  const canvas = document.getElementById('labelCanvas');
  if (!canvas) return null;
  // check all boxes so hovering any edge/corner shows resize cursor
  for (let bi = 0; bi < labelBoxes.length; bi++) {
    const b = labelBoxes[bi];
    if (!b) continue;
    const left = b.x * canvas.width - b.w * canvas.width / 2;
    const top = b.y * canvas.height - b.h * canvas.height / 2;
    const w = b.w * canvas.width, h = b.h * canvas.height;

    // Edge-first detection: edges get a wider hit band so moving along edges shows edge cursors
    const edgeHitW = Math.max(40, w * 0.25);
    const edgeHitH = Math.max(40, h * 0.25);
    const edges = [ {x:left + w/2, y: top, name:'t', w: edgeHitW, h: _lbl_edgeThickness + _lbl_visualPad * 2}, {x:left + w, y: top + h/2, name:'r', w: _lbl_edgeThickness + _lbl_visualPad * 2, h: edgeHitH}, {x:left + w/2, y: top + h, name:'b', w: edgeHitW, h: _lbl_edgeThickness + _lbl_visualPad * 2}, {x:left, y: top + h/2, name:'l', w: _lbl_edgeThickness + _lbl_visualPad * 2, h: edgeHitH} ];
    for (let i=0;i<edges.length;i++){
      const e = edges[i];
      if (Math.abs(px - e.x) <= (e.w/2) && Math.abs(py - e.y) <= (e.h/2)) return { idx: bi, corner: e.name };
    }

    // corners: hit radius should match visible corner hint size (larger for better UX)
    const cornerRadius = Math.max(12, Math.floor(_lbl_cornerSize * 0.75)) + _lbl_visualPad;
    const corners = [ {x:left,y:top,name:'tl'}, {x:left+w,y:top,name:'tr'}, {x:left+w,y:top+h,name:'br'}, {x:left,y:top+h,name:'bl'} ];
    for (let i=0;i<corners.length;i++){
      const c = corners[i];
      if (Math.abs(px - c.x) <= cornerRadius && Math.abs(py - c.y) <= cornerRadius) return { idx: bi, corner: c.name };
    }
  }
  return null;
}

function switchTab(tabName) {
  document.querySelectorAll('[id^="tab-"]').forEach(el => el.style.display = 'none');
  const tab = document.getElementById('tab-' + tabName);
  if (tab) tab.style.display = 'block';
  if (tabName === 'label') refreshLabelList();
}

// LABEL functions
let labelScale = 1; // scale factor for display vs actual coordinates
// UI-controlled drawing options
let labelShowGrid = true;
let labelGridLineWidth = 2;
let labelBorderLineWidth = 3;

async function refreshLabelList() {
  const list = document.getElementById('labelImageList');
  if (!list) return;
  try {
    const r = await fetch('/images');
    const j = await r.json();
    list.innerHTML = '';
    (j.images || []).forEach(n => {
      const opt = new Option(n, n);
      list.add(opt);
    });
    // Load class names
    await loadLabelClassNames();
    
    // Add change event handler to check for unsaved changes
    if (!list.dataset.changeHandlerAdded) {
      list.addEventListener('change', () => {
        labelLoad();
      });
      list.dataset.changeHandlerAdded = 'true';
    }
  } catch (e) {
    out('labelOut', 'Error loading images: ' + e.message);
  }
}

// Upload image into dataset/images via UI
async function uploadLabelImage() {
  const f = document.getElementById('labelUploadFile').files[0];
  if (!f) { out('labelSaveLog', 'Choose a file first'); document.getElementById('labelSaveLog').style.display='block'; return; }
  const fd = new FormData(); fd.append('image', f);
  try {
    const r = await fetch('/upload_image', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.name) {
      out('labelSaveLog', 'Uploaded: ' + (j.name || j.saved || '')); document.getElementById('labelSaveLog').style.display='block';
      await refreshLabelList();
      // select and load the new image
      const sel = document.getElementById('labelImageList');
      if (sel) {
        sel.value = j.name;
        labelLoad();
      }
    } else {
      out('labelSaveLog', 'Upload response: ' + JSON.stringify(j)); document.getElementById('labelSaveLog').style.display='block';
    }
  } catch (e) {
    out('labelSaveLog', 'Error: ' + e.message); document.getElementById('labelSaveLog').style.display='block';
  }
}

async function loadLabelClassNames() {
  try {
    const r = await fetch('/config');
    const j = await r.json();
    if (j.names) {
      labelClassNames = j.names;
      // Update class dropdown in dashboard
      const select = document.getElementById('labelClass');
      if (select) {
        select.innerHTML = '';
        j.names.forEach((name, idx) => {
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = idx + ': ' + name;
          select.appendChild(opt);
        });
      }
    }
  } catch(e) { /* use defaults */ }
}

// Navigate to previous image
function loadPreviousImage() {
  const select = document.getElementById('labelImageList');
  if (!select || select.options.length === 0) return;
  
  let currentIdx = select.selectedIndex;
  if (currentIdx > 0) {
    const nextImageName = select.options[currentIdx - 1].value;
    
    if (labelHasUnsavedChanges) {
      showUnsavedChangesDialog(nextImageName);
      return;
    }
    
    select.selectedIndex = currentIdx - 1;
    loadImageAndLabels(nextImageName);
  } else {
    out('labelOut', '⚠️ Already at first image');
  }
}

// Navigate to next image
function loadNextImage() {
  const select = document.getElementById('labelImageList');
  if (!select || select.options.length === 0) return;
  
  let currentIdx = select.selectedIndex;
  if (currentIdx < select.options.length - 1) {
    const nextImageName = select.options[currentIdx + 1].value;
    
    if (labelHasUnsavedChanges) {
      showUnsavedChangesDialog(nextImageName);
      return;
    }
    
    select.selectedIndex = currentIdx + 1;
    loadImageAndLabels(nextImageName);
  } else {
    out('labelOut', '⚠️ Already at last image');
  }
}

function labelLoad() {
  const name = document.getElementById('labelImageList').value;
  if (!name) { out('labelOut', 'Choose an image first'); return; }
  
  // Check for unsaved changes
  if (labelHasUnsavedChanges) {
    showUnsavedChangesDialog(name);
    return;
  }
  
  loadImageAndLabels(name);
}

function showUnsavedChangesDialog(nextImageName) {
  // Create overlay
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; z-index:3001;';
  
  // Create dialog
  const dialog = document.createElement('div');
  dialog.style.cssText = 'background:white; border:2px solid #ff9800; border-radius:8px; padding:20px; max-width:400px; box-shadow:0 4px 20px rgba(0,0,0,0.3);';
  
  dialog.innerHTML = `
    <h3 style="margin-top:0; color:#ff9800; display:flex; align-items:center;">
      ⚠️ Unsaved Changes
    </h3>
    <p style="color:#333; margin:15px 0; font-size:14px;">
      You have unsaved changes to the current image. Do you want to save them before switching?
    </p>
    <div style="display:flex; gap:8px; justify-content:flex-end;">
      <button id="unsavedSave" style="padding:8px 16px; background:#4CAF50; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold;">
        💾 Save
      </button>
      <button id="unsavedNext" style="padding:8px 16px; background:#2196F3; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold;">
        ➡️ Save & Next
      </button>
      <button id="unsavedDiscard" style="padding:8px 16px; background:#dc3545; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold;">
        🗑️ Discard
      </button>
    </div>
  `;
  
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  
  // Handle Save button
  document.getElementById('unsavedSave').onclick = () => {
    labelSave().then(() => {
      overlay.remove();
      setTimeout(() => loadImageAndLabels(nextImageName), 100);
    }).catch(e => {
      alert('Save failed: ' + e.message);
    });
  };
  
  // Handle Save & Next button
  document.getElementById('unsavedNext').onclick = () => {
    labelSave().then(() => {
      overlay.remove();
      setTimeout(() => loadImageAndLabels(nextImageName), 100);
    }).catch(e => {
      alert('Save failed: ' + e.message);
    });
  };
  
  // Handle Discard button
  document.getElementById('unsavedDiscard').onclick = () => {
    clearLabelDirty();
    overlay.remove();
    setTimeout(() => loadImageAndLabels(nextImageName), 100);
  };
  
  // Handle ESC key
  const handleEsc = (e) => {
    if (e.key === 'Escape') {
      overlay.remove();
      document.removeEventListener('keydown', handleEsc);
      // Reset image selector to previous image
      const select = document.getElementById('labelImageList');
      if (select.value !== nextImageName) {
        select.value = document.getElementById('imageNameInput').value;
      }
    }
  };
  document.addEventListener('keydown', handleEsc);
}

function loadImageAndLabels(name) {
  // Ensure canvas context is available
  const canvas = document.getElementById('labelCanvas');
  const ctx = canvas ? canvas.getContext('2d') : null;
  if (!canvas || !ctx) { out('labelOut', 'Canvas not available'); return; }
  
  labelImg.src = '/uploads/image/' + name;
  labelImg.onload = async () => {
    console.log('✅ Image loaded:', {name, naturalSize: labelImg.naturalWidth + 'x' + labelImg.naturalHeight});
    labelBoxes = [];
    // Set canvas to ACTUAL image dimensions for accurate coordinate calculation
    canvas.width = labelImg.naturalWidth || labelImg.width || 800;
    canvas.height = labelImg.naturalHeight || labelImg.height || 600;
    // Calculate display scale (canvas displays at max 600px width but actual dimensions are used for labels)
    const displayScale = canvas.width / Math.min(canvas.width, 600);
    labelScale = displayScale;
    console.log('⚙️  Canvas setup:', {canvasSize: canvas.width + 'x' + canvas.height, displaySize: Math.min(canvas.width, 600), labelScale});
    
    // Draw image immediately to canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(labelImg, 0, 0, canvas.width, canvas.height);
    
    // Load existing labels if they exist
    await loadExistingDashboardLabels(name);
    
    labelDraw();
    updateLabelBoxPreview();
    renderBoxList();
    out('labelOut', 'Image loaded: ' + name + ' (actual: ' + canvas.width + 'x' + canvas.height + ', display scale: ' + labelScale.toFixed(2) + 'x, boxes: ' + labelBoxes.length + ')');
  };
  labelImg.onerror = () => out('labelOut', 'Error loading image: ' + name);
}

// Clear all annotations and start fresh labeling on the same image
function labelClear() {
  if (labelBoxes.length === 0) {
    out('labelOut', '⚠️ No labels to clear');
    return;
  }
  
  if (!confirm('Delete ALL ' + labelBoxes.length + ' boxes and start fresh? (Cannot undo)')) {
    return;
  }
  
  // Clear all boxes
  labelBoxes = [];
  labelSelectedBox = -1;
  labelDragging = false;
  labelCurrent = null;
  markLabelDirty();  // Mark as unsaved
  
  // Redraw canvas
  labelDraw();
  updateLabelBoxPreview();
  renderBoxList();
  
  out('labelOut', '🔄 Cleared all labels - ready for new annotation!');
  console.log('🔄 All labels cleared, canvas reset for fresh annotation');
}

async function loadExistingDashboardLabels(imageName) {
  try {
    // Extract image stem without extension
    const imageStem = imageName.split('.')[0];
    
    // Try to load from JSON file in dataset/json_data/
    try {
      const jsonUrl = '/api/load_json_labels?image=' + encodeURIComponent(imageStem);
      const jsonResponse = await fetch(jsonUrl);
      
      if (jsonResponse.ok) {
        const jsonData = await jsonResponse.json();
        
        if (jsonData && jsonData.boxes && Array.isArray(jsonData.boxes)) {
          labelBoxes = [];
          jsonData.boxes.forEach(box => {
            if (box.bbox && box.bbox.length >= 5) {
              // bbox format: [0, x, y, w, h]
              labelBoxes.push({
                x: parseFloat(box.bbox[1]),
                y: parseFloat(box.bbox[2]),
                w: parseFloat(box.bbox[3]),
                h: parseFloat(box.bbox[4]),
                'class': 0,
                sku_id: box.sku || null,
                cropped: box.cropped || null
              });
            }
          });
          labelExistingCount = labelBoxes.length;
          console.log('✅ Loaded', labelBoxes.length, 'boxes from JSON:', imageStem);
          renderBoxList();
          return;
        }
      }
    } catch(jsonError) {
      console.log('ℹ️  No JSON labels found, trying legacy format...');
    }
    
    // Fallback: Use lightweight endpoint that only reads label .txt to avoid running inference/GPU work
    const url = '/label_contents?image=' + encodeURIComponent(imageName);
    const r = await fetch(url);
    const j = await r.json();
    // update API preview with raw JSON (label_contents object)
    try { updateApiPreview(j); } catch(e) {}
    if (j && j.label_contents) {
      const lines = j.label_contents.trim().split('\n').filter(l => l.trim());
      labelBoxes = [];
      lines.forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 5) {
          const cls = parseInt(parts[0]);
          const x = parseFloat(parts[1]);
          const y = parseFloat(parts[2]);
          const w = parseFloat(parts[3]);
          const h = parseFloat(parts[4]);
          labelBoxes.push({x, y, w, h, 'class': cls});
        }
      });
      labelExistingCount = labelBoxes.length;
    } else {
      labelExistingCount = 0;
    }
  } catch(e) { /* silent */ }
}

function updateApiPreview(json) {
  const apiDiv = document.getElementById('apiPreview');
  if (!apiDiv) return;
  try {
    apiDiv.style.display = 'block';
    apiDiv.innerHTML = '<pre>' + JSON.stringify(json, null, 2) + '</pre>';
  } catch (e) {
    apiDiv.textContent = String(json);
    apiDiv.style.display = 'block';
  }
}

function labelDraw() {
  const canvas = document.getElementById('labelCanvas');
  const ctx = canvas ? canvas.getContext('2d') : null;
  if (!canvas || !ctx) return;
  console.log('🎨 Drawing', labelBoxes.length, 'boxes');
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Draw background image
  if (labelImg.src && labelImg.complete) {
    ctx.drawImage(labelImg, 0, 0, canvas.width, canvas.height);
  }
  
  // Draw all boxes
  labelBoxes.forEach((b, idx) => {
    const x = b.x * canvas.width - b.w * canvas.width / 2;
    const y = b.y * canvas.height - b.h * canvas.height / 2;
    const w = b.w * canvas.width, h = b.h * canvas.height;
    // Highlight selected box (or hovered box) with different color and optionally thicker border
    const isSelected = (idx === labelSelectedBox);
    const isHovered = (_lbl_hoverHandle && _lbl_hoverHandle.idx === idx);
    if (isSelected || isHovered) {
      ctx.strokeStyle = isSelected ? 'blue' : '#ffd54f';
      // increase line width when hovered to make edit area prominent
      ctx.lineWidth = Math.max(3, labelBorderLineWidth + (isHovered ? 3 : 0));
      ctx.fillStyle = isSelected ? 'rgba(0,0,255,0.18)' : 'rgba(255,213,79,0.06)';
    } else {
      ctx.strokeStyle = 'red'; ctx.lineWidth = Math.max(1, Math.round(labelBorderLineWidth/1.2));
      ctx.fillStyle = 'rgba(255,0,0,0.12)';
    }
    ctx.strokeRect(x, y, w, h);
    ctx.fillRect(x, y, w, h);
    // Draw class name and index (support both b.class and legacy b.cls)
    const cid = (typeof b.class !== 'undefined') ? b.class : b.cls;
    const className = labelClassNames[cid] || ('class'+cid);
    ctx.fillStyle = idx === labelSelectedBox ? 'blue' : 'red';
    ctx.font = 'bold 14px Arial';
    ctx.fillText(idx + ': ' + className, x, Math.max(15, y-5));
    
    // Draw coordinates as Label Studio percentages at bottom-right of box (for selected box only)
    if (idx === labelSelectedBox && w > 50 && h > 50) {
      // Convert HOLO (center) to Label Studio (top-left percentage) for display
      const [x_pct, y_pct, w_pct, h_pct] = holoToLabelStudio(b.x, b.y, b.w, b.h);
      const coordText = `Label Studio: (${x_pct.toFixed(1)}, ${y_pct.toFixed(1)})`;
      const dimText = `${w_pct.toFixed(1)} × ${h_pct.toFixed(1)} %`;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.font = '11px Arial';
      const coordMetrics = ctx.measureText(coordText);
      const dimMetrics = ctx.measureText(dimText);
      const maxWidth = Math.max(coordMetrics.width, dimMetrics.width) + 6;
      // Draw background box
      ctx.fillRect(x + w - maxWidth - 4, y + h - 30, maxWidth + 4, 28);
      // Draw text
      ctx.fillStyle = '#ffff00';
      ctx.fillText(coordText, x + w - maxWidth - 2, y + h - 16);
      ctx.fillText(dimText, x + w - maxWidth - 2, y + h - 4);
    }
  });
  // Photoshop-like edit highlight: blue outline and bottom-right corner indicator for selected box
  try {
    if (labelSelectedBox >= 0 && labelBoxes[labelSelectedBox]) {
      const b = labelBoxes[labelSelectedBox];
      const left = b.x * canvas.width - b.w * canvas.width / 2;
      const top = b.y * canvas.height - b.h * canvas.height / 2;
      const w = b.w * canvas.width, h = b.h * canvas.height;
      ctx.save();
      // outer thicker blue stroke
      ctx.strokeStyle = 'rgba(30,144,255,0.95)';
      ctx.lineWidth = Math.max(3, labelBorderLineWidth + 2);
      ctx.strokeRect(left - 2, top - 2, w + 4, h + 4);
      // dashed bottom guide line
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.strokeStyle = 'rgba(30,144,255,0.65)'; ctx.lineWidth = 2; ctx.moveTo(left, top + h); ctx.lineTo(left + w, top + h); ctx.stroke();
      ctx.setLineDash([]);
      // Draw small hint squares at all 4 corners and 4 edge midpoints
      try {
        const cornerSq = Math.max(10, Math.round(_lbl_cornerSize * 0.75 * 3));
        const midSq = Math.max(10, Math.round(_lbl_cornerSize * 0.6 * 3));
        const cornersAll = [
          {x: left, y: top},
          {x: left + w, y: top},
          {x: left + w, y: top + h},
          {x: left, y: top + h}
        ];
        const mids = [
          {x: left + w/2, y: top},
          {x: left + w, y: top + h/2},
          {x: left + w/2, y: top + h},
          {x: left, y: top + h/2}
        ];
        ctx.save();
        // corner hints
        cornersAll.forEach(c => {
          const cx = Math.round(c.x - cornerSq/2);
          const cy = Math.round(c.y - cornerSq/2);
          ctx.fillStyle = 'rgba(30,144,255,0.95)';
          ctx.fillRect(cx, cy, cornerSq, cornerSq);
          ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.strokeRect(cx, cy, cornerSq, cornerSq);
        });
        // side/midpoint hints
        mids.forEach(m => {
          const mx = Math.round(m.x - midSq/2);
          const my = Math.round(m.y - midSq/2);
          ctx.fillStyle = 'rgba(30,144,255,0.85)';
          ctx.fillRect(mx, my, midSq, midSq);
          ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.strokeRect(mx, my, midSq, midSq);
        });
        ctx.restore();
      } catch (e) {}
      ctx.restore();
    }
  } catch (e) {}
  
  // Draw handles for selected box (inside labelDraw so canvas/ctx are available)
  if (labelSelectedBox >= 0 && labelBoxes[labelSelectedBox]) {
    const b = labelBoxes[labelSelectedBox];
    const left = b.x * canvas.width - b.w * canvas.width / 2;
    const top = b.y * canvas.height - b.h * canvas.height / 2;
    const w = b.w * canvas.width, h = b.h * canvas.height;
    const corners = [
      {cx: left, cy: top, name: 'tl'},
      {cx: left + w, cy: top, name: 'tr'},
      {cx: left + w, cy: top + h, name: 'br'},
      {cx: left, cy: top + h, name: 'bl'}
    ];
    const edgeHitW = Math.max(40, w * 0.25);
    const edgeHitH = Math.max(40, h * 0.25);
    const edges = [
      {cx: left + w/2, cy: top, name: 't', w: edgeHitW, h: _lbl_edgeThickness},
      {cx: left + w, cy: top + h/2, name: 'r', w: _lbl_edgeThickness, h: edgeHitH},
      {cx: left + w/2, cy: top + h, name: 'b', w: edgeHitW, h: _lbl_edgeThickness},
      {cx: left, cy: top + h/2, name: 'l', w: _lbl_edgeThickness, h: edgeHitH}
    ];
    ctx.save(); ctx.fillStyle='white'; ctx.strokeStyle='black'; ctx.lineWidth=2;
    corners.forEach(c => { ctx.beginPath(); ctx.rect(c.cx - _lbl_cornerSize/2 - _lbl_visualPad, c.cy - _lbl_cornerSize/2 - _lbl_visualPad, _lbl_cornerSize + _lbl_visualPad*2, _lbl_cornerSize + _lbl_visualPad*2); ctx.fill(); ctx.stroke(); });
    edges.forEach(e => { ctx.beginPath(); ctx.rect(e.cx - e.w/2 - _lbl_visualPad, e.cy - e.h/2 - _lbl_visualPad, e.w + _lbl_visualPad*2, e.h + _lbl_visualPad*2); ctx.fill(); ctx.stroke(); });
    ctx.restore();
  }

  // Draw hovered box handles (highlight specific handle)
  if (_lbl_hoverHandle) {
    const hb = labelBoxes[_lbl_hoverHandle.idx];
    if (hb) {
      const left = hb.x * canvas.width - hb.w * canvas.width / 2;
      const top = hb.y * canvas.height - hb.h * canvas.height / 2;
      const w = hb.w * canvas.width, h = hb.h * canvas.height;
      const corners = [
        {cx: left, cy: top, name: 'tl'},
        {cx: left + w, cy: top, name: 'tr'},
        {cx: left + w, cy: top + h, name: 'br'},
        {cx: left, cy: top + h, name: 'bl'}
      ];
      const edgeHitW = Math.max(40, w * 0.25);
      const edgeHitH = Math.max(40, h * 0.25);
      const edges = [
        {cx: left + w/2, cy: top, name: 't', w: edgeHitW, h: _lbl_edgeThickness},
        {cx: left + w, cy: top + h/2, name: 'r', w: _lbl_edgeThickness, h: edgeHitH},
        {cx: left + w/2, cy: top + h, name: 'b', w: edgeHitW, h: _lbl_edgeThickness},
        {cx: left, cy: top + h/2, name: 'l', w: _lbl_edgeThickness, h: edgeHitH}
      ];
      ctx.save(); ctx.fillStyle='rgba(255,255,255,0.9)'; ctx.strokeStyle='rgba(0,0,0,0.7)'; ctx.lineWidth=1;
      corners.forEach(c => { ctx.beginPath(); ctx.rect(c.cx - _lbl_cornerSize/2, c.cy - _lbl_cornerSize/2, _lbl_cornerSize, _lbl_cornerSize); ctx.fill(); ctx.stroke(); });
      edges.forEach(e => { ctx.beginPath(); ctx.rect(e.cx - e.w/2, e.cy - e.h/2, e.w, e.h); ctx.fill(); ctx.stroke(); });
      ctx.restore();

      ctx.save(); ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 3; ctx.fillStyle = 'rgba(255,213,79,0.12)';
      corners.forEach(c => {
        ctx.beginPath(); ctx.rect(c.cx - (_lbl_cornerSize/2 + _lbl_visualPad), c.cy - (_lbl_cornerSize/2 + _lbl_visualPad), _lbl_cornerSize + _lbl_visualPad*2, _lbl_cornerSize + _lbl_visualPad*2); ctx.stroke(); ctx.fillRect(c.cx - _lbl_cornerSize/2, c.cy - _lbl_cornerSize/2, _lbl_cornerSize, _lbl_cornerSize);
      });
      edges.forEach(e => {
        ctx.beginPath(); ctx.rect(e.cx - e.w/2 - _lbl_visualPad, e.cy - e.h/2 - _lbl_visualPad, e.w + _lbl_visualPad*2, e.h + _lbl_visualPad*2); ctx.stroke(); ctx.fillRect(e.cx - e.w/2, e.cy - e.h/2, e.w, e.h);
      });
      ctx.restore();
    }
  }
  
  // Show preview
  updateLabelBoxPreview();
}

function updateLabelBoxPreview() {
  const preview = document.getElementById('labelBoxPreview');
  const tbody = document.getElementById('labelBoxTable');
  
  // If these elements don't exist, skip the update (they're optional)
  if (!preview || !tbody) return;
  
  if (labelBoxes.length === 0) {
    preview.style.display = 'none';
    return;
  }
  preview.style.display = 'block';
  tbody.innerHTML = '';
  labelBoxes.forEach((b, idx) => {
    const cid = (typeof b.class !== 'undefined') ? b.class : b.cls;
    const skuId = b.sku_id || '';
    const tr = document.createElement('tr');
    const isSelected = (idx === labelSelectedBox);
    // Highlight selected row, alternate background for non-selected
    tr.style.backgroundColor = isSelected ? '#ffd700' : (idx % 2 === 0 ? '#f9f9f9' : '#ffffff');
    tr.style.borderBottom = '1px solid #eee';
    tr.style.fontWeight = isSelected ? 'bold' : 'normal';
    tr.style.boxShadow = isSelected ? '0 0 8px rgba(255, 215, 0, 0.8)' : 'none';
    // Set row ID for quick access
    tr.id = 'lbl_row_' + idx;
    const isEditing = !!window._labelRowEditSnapshots && window._labelRowEditSnapshots[idx];
    if (isEditing) {
      const s = window._labelRowEditSnapshots[idx];
      tr.innerHTML = `
        <td style="padding:8px; border:1px solid #ddd; text-align:center; font-weight:bold;">${idx}</td>
        <td style="padding:8px; border:1px solid #ddd;"><select id="lbl_edit_class_${idx}"></select></td>
        <td style="padding:8px; border:1px solid #ddd;"><input id="lbl_edit_sku_${idx}" type="text" placeholder="SKU_0001" value="${skuId}" style="width:100%"></td>
        <td style="padding:8px; border:1px solid #ddd; text-align:right;"><input id="lbl_edit_x_${idx}" type="number" step="0.000001" value="${s.x.toFixed(6)}" style="width:100%"></td>
        <td style="padding:8px; border:1px solid #ddd; text-align:right;"><input id="lbl_edit_y_${idx}" type="number" step="0.000001" value="${s.y.toFixed(6)}" style="width:100%"></td>
        <td style="padding:8px; border:1px solid #ddd; text-align:right;"><input id="lbl_edit_w_${idx}" type="number" step="0.000001" value="${s.w.toFixed(6)}" style="width:100%"></td>
        <td style="padding:8px; border:1px solid #ddd; text-align:right;"><input id="lbl_edit_h_${idx}" type="number" step="0.000001" value="${s.h.toFixed(6)}" style="width:100%"></td>
        <td style="padding:8px; border:1px solid #ddd; text-align:center;"><button onclick="saveLabelRow(${idx})">Save</button> <button onclick="cancelLabelRow(${idx})">Cancel</button></td>
      `;
      tbody.appendChild(tr);
      // populate class select
      const sel = document.getElementById(`lbl_edit_class_${idx}`);
      if (sel) { sel.innerHTML=''; labelClassNames.forEach((n,i)=>{const o=document.createElement('option'); o.value=i; o.textContent = i+': '+n; sel.appendChild(o);}); sel.value = cid; }
    } else {
      tr.innerHTML = `
        <td style="padding:6px; border:1px solid #cfcfcf; text-align:center; font-weight:600;">${idx}</td>
        <td style="padding:6px; border:1px solid #cfcfcf; text-align:center;">${cid}</td>
        <td style="padding:6px; border:1px solid #cfcfcf; text-align:right;">${b.x.toFixed(6)}</td>
        <td style="padding:6px; border:1px solid #cfcfcf; text-align:right;">${b.y.toFixed(6)}</td>
        <td style="padding:6px; border:1px solid #cfcfcf; text-align:right;">${b.w.toFixed(6)}</td>
        <td style="padding:6px; border:1px solid #cfcfcf; text-align:right;">${b.h.toFixed(6)}</td>
        <td style="padding:6px; border:1px solid #cfcfcf; text-align:center;"><button onclick="deleteLabelRow(${idx})" style="padding:4px 8px;">Delete</button></td>
      `;
      tr.onclick = function(e) {
        // Don't select box when clicking buttons or selects
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'SELECT') return;
        labelSelectedBox = idx;
        highlightBoxRow(idx);
        labelDraw();
      };
      tr.style.cursor = 'pointer';
      tbody.appendChild(tr);
    }
  });
}

// Select box for cropping and save SKU
window.selectBoxForCrop = function(idx) {
  labelSelectedBox = idx;
  highlightBoxRow(idx);
  document.getElementById('skuIdInput').focus();
  out('labelOut', 'Box ' + idx + ' selected. Enter SKU ID and click "Save Crop" or press Enter');
};

// Save crop for selected box with SKU ID
window.saveCropForSelectedBox = function() {
  if (labelSelectedBox < 0 || labelSelectedBox >= labelBoxes.length) {
    out('labelOut', '❌ No box selected');
    return;
  }
  const box = labelBoxes[labelSelectedBox];
  const skuInput = document.getElementById('skuIdInput');
  if (!skuInput.value.trim()) {
    out('labelOut', '❌ Enter SKU ID first');
    return;
  }
  const skuId = skuInput.value.trim();
  box.sku_id = skuId;
  
  // Save crop image
  if (!labelImg.src) {
    out('labelOut', '❌ No image loaded');
    return;
  }
  
  // Crop region from canvas
  const canvas = document.getElementById('labelCanvas');
  const left = Math.max(0, (box.x - box.w/2) * canvas.width);
  const top = Math.max(0, (box.y - box.h/2) * canvas.height);
  const width = Math.min(canvas.width - left, box.w * canvas.width);
  const height = Math.min(canvas.height - top, box.h * canvas.height);
  
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = width;
  cropCanvas.height = height;
  const cropCtx = cropCanvas.getContext('2d');
  cropCtx.drawImage(labelImg, left, top, width, height, 0, 0, width, height);
  
  // Send crop to server
  cropCanvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('image', blob, 'crop.jpg');
    formData.append('sku_id', skuId);
    const imageName = labelImg.src.split('/').pop().split('.')[0] || 'image';
    formData.append('image_name', imageName);
    
    fetch('/api/save_crop', {
      method: 'POST',
      body: formData
    }).then(r => r.json()).then(data => {
      if (data.ok) {
        out('labelOut', '✅ Crop saved: ' + skuId + ' (' + data.path + ')');
        skuInput.value = '';
        renderBoxList();  // Reload list when SKU is assigned
      } else {
        out('labelOut', '❌ Error: ' + (data.error || 'Unknown'));
      }
    }).catch(e => {
      out('labelOut', '❌ Network error: ' + e.message);
    });
  }, 'image/jpeg', 0.95);
};

// Allow Enter key to save crop
document.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && document.getElementById('skuIdInput') === document.activeElement) {
    e.preventDefault();
    saveCropForSelectedBox();
  }
});

// Highlight and scroll to selected box row in table
function highlightBoxRow(idx) {
  if (idx < 0) {
    // No box selected, remove all highlights
    const allRows = document.querySelectorAll('[id^="lbl_row_"]');
    allRows.forEach(row => {
      row.style.backgroundColor = row.id.match(/\d+/)[0] % 2 === 0 ? '#f9f9f9' : '#ffffff';
      row.style.fontWeight = 'normal';
      row.style.boxShadow = 'none';
    });
    return;
  }
  
  const selectedRow = document.getElementById('lbl_row_' + idx);
  if (!selectedRow) return;
  
  // Remove all highlights
  const allRows = document.querySelectorAll('[id^="lbl_row_"]');
  allRows.forEach(row => {
    const rowIdx = parseInt(row.id.match(/\d+/)[0]);
    row.style.backgroundColor = rowIdx % 2 === 0 ? '#f9f9f9' : '#ffffff';
    row.style.fontWeight = 'normal';
    row.style.boxShadow = 'none';
  });
  
  // Highlight selected row
  selectedRow.style.backgroundColor = '#ffd700';
  selectedRow.style.fontWeight = 'bold';
  selectedRow.style.boxShadow = '0 0 8px rgba(255, 215, 0, 0.8)';
  
  // Scroll to selected row
  selectedRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Snapshot map for dashboard label row edits
window._labelRowEditSnapshots = window._labelRowEditSnapshots || {};

window.startLabelRowEdit = function(idx) {
  if (!labelBoxes[idx]) return;
  window._labelRowEditSnapshots[idx] = { x: labelBoxes[idx].x, y: labelBoxes[idx].y, w: labelBoxes[idx].w, h: labelBoxes[idx].h, class: labelBoxes[idx].class };
  updateLabelBoxPreview();
}

window.cancelLabelRow = function(idx) {
  delete window._labelRowEditSnapshots[idx];
  updateLabelBoxPreview();
  labelDraw();
}

window.saveLabelRow = function(idx) {
  try {
    const nx = parseFloat(document.getElementById(`lbl_edit_x_${idx}`).value);
    const ny = parseFloat(document.getElementById(`lbl_edit_y_${idx}`).value);
    const nw = parseFloat(document.getElementById(`lbl_edit_w_${idx}`).value);
    const nh = parseFloat(document.getElementById(`lbl_edit_h_${idx}`).value);
    const nc = parseInt(document.getElementById(`lbl_edit_class_${idx}`).value) || 0;
    labelBoxes[idx].x = nx; labelBoxes[idx].y = ny; labelBoxes[idx].w = nw; labelBoxes[idx].h = nh; labelBoxes[idx].class = nc;
  } catch(e) { console.log('saveLabelRow', e); }
  delete window._labelRowEditSnapshots[idx];
  updateLabelBoxPreview();
  labelDraw();
  renderBoxList();  // Reload list when class changes
}

window.resizeLabelRow = function(idx, factor) {
  if (!labelBoxes[idx]) return;
  labelBoxes[idx].w = Math.max(0.000001, labelBoxes[idx].w * factor);
  labelBoxes[idx].h = Math.max(0.000001, labelBoxes[idx].h * factor);
  updateLabelBoxPreview(); labelDraw();
}

window.deleteLabelRow = function(idx) {
  if (!confirm('Delete box ' + idx + '?')) return;
  try {
    if (idx >= 0 && idx < labelBoxes.length) {
      labelBoxes.splice(idx, 1);
      if (labelSelectedBox === idx) labelSelectedBox = -1;
      updateLabelBoxPreview();
      labelDraw();
      out('labelOut', '🗑️ Box deleted. Saving changes...');
      
      // Auto-save the updated annotation to persist deletion
      labelSave();
    } else {
      out('labelOut', '❌ Invalid box index: ' + idx);
    }
  } catch(e) {
    out('labelOut', '❌ Delete error: ' + e.message);
  }
}

window.renameLabelRow = function(idx) {
  if (!labelBoxes[idx]) return;
  const cid = labelBoxes[idx].class || labelBoxes[idx].cls || 0;
  const current = labelClassNames[cid] || ('class'+cid);
  const name = prompt('Rename class (local only) for preview and dropdown:', current);
  if (!name) return;
  labelClassNames[cid] = name;
  updateLabelBoxPreview(); labelDraw();
}

if (labelCanvas) {
  // CANVAS DRAWING FUNCTIONALITY: Cursor edge-to-edge drawing + Resize handles
  // Boxes persist in memory - each draw adds a new box (not replace)
  // Click on box corners/edges to resize
  
  labelCanvas.onmousedown = (e) => { 
    const canvas = document.getElementById('labelCanvas');
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    
    // Convert display pixels to canvas pixels
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;
    const canvasX = displayX * (canvas.width / rect.width);
    const canvasY = displayY * (canvas.height / rect.height);
    
    // Store initial click position to detect drag vs click
    canvas._clickStartX = canvasX;
    canvas._clickStartY = canvasY;
    canvas._clickStartTime = Date.now();
    canvas._clickOnBoxIdx = -1;
    
    // Check if clicking on existing box handle (resize) first
    const handleRadius = 15;  // pixels for handle detection
    const edgeThickness = 12;  // thickness for edge detection
    
    for (let i = labelBoxes.length - 1; i >= 0; i--) {
      const b = labelBoxes[i];
      if (b.isPreview) continue;  // Skip preview boxes
      
      const left = b.x * canvas.width - b.w * canvas.width / 2;
      const top = b.y * canvas.height - b.h * canvas.height / 2;
      const right = left + b.w * canvas.width;
      const bottom = top + b.h * canvas.height;
      
      // Check corners
      const corners = [
        {cx: left, cy: top, name: 'tl'},
        {cx: right, cy: top, name: 'tr'},
        {cx: right, cy: bottom, name: 'br'},
        {cx: left, cy: bottom, name: 'bl'}
      ];
      
      for (let corner of corners) {
        if (Math.abs(canvasX - corner.cx) <= handleRadius && Math.abs(canvasY - corner.cy) <= handleRadius) {
          // Start resize from corner
          labelCanvas._resizing = true;
          labelCanvas._resizeBoxIdx = i;
          labelCanvas._resizeCorner = corner.name;
          labelCanvas._resizeOrig = {x: b.x, y: b.y, w: b.w, h: b.h};
          labelSelectedBox = i;
          highlightBoxRow(i);
          console.log('🔧 Resize started on box', i, 'corner:', corner.name);
          return;
        }
      }
      
      // Check edges (sides)
      const edges = [
        {x: left + (right - left) / 2, y: top, name: 't', type: 'h'},       // top
        {x: right, y: top + (bottom - top) / 2, name: 'r', type: 'v'},      // right
        {x: left + (right - left) / 2, y: bottom, name: 'b', type: 'h'},    // bottom
        {x: left, y: top + (bottom - top) / 2, name: 'l', type: 'v'}        // left
      ];
      
      for (let edge of edges) {
        const distX = Math.abs(canvasX - edge.x);
        const distY = Math.abs(canvasY - edge.y);
        const threshold = edge.type === 'h' ? [edgeThickness * 2, edgeThickness] : [edgeThickness, edgeThickness * 2];
        
        if (distX <= threshold[0] && distY <= threshold[1]) {
          // Start resize from edge
          labelCanvas._resizing = true;
          labelCanvas._resizeBoxIdx = i;
          labelCanvas._resizeCorner = edge.name;  // t, r, b, l
          labelCanvas._resizeOrig = {x: b.x, y: b.y, w: b.w, h: b.h};
          labelSelectedBox = i;
          highlightBoxRow(i);
          console.log('🔧 Resize started on box', i, 'edge:', edge.name);
          return;
        }
      }
      
      // Check if clicking inside box (but allow drag to draw new box)
      if (canvasX >= left && canvasX <= right && canvasY >= top && canvasY <= bottom) {
        canvas._clickOnBoxIdx = i;  // Remember which box we clicked on
        canvas._showSKUPopup = false;  // Flag: will show popup only on mouse up without dragging
        labelSelectedBox = i;
        highlightBoxRow(i);
        
        out('labelOut', '📦 Selected box ' + i + ' (drag to draw over it, or release for SKU)');
        labelDraw();
        return;
      }
    }
    
    // Not clicking on any box - prepare to start new drawing
    labelSelectedBox = -1;
    highlightBoxRow(-1);
    labelDragging = true;
    
    // Store normalized coordinates (0-1)
    const normX = canvasX / canvas.width;
    const normY = canvasY / canvas.height;
    labelCurrent = { x1: normX, y1: normY, x2: normX, y2: normY, isPreview: true };
    
    console.log('✏️ Start drawing from cursor:', {normalized: {x: normX, y: normY}, totalBoxes: labelBoxes.length});
  };
  
  labelCanvas.onmousemove = (e) => {
    const canvas = document.getElementById('labelCanvas');
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;
    const canvasX = displayX * (canvas.width / rect.width);
    const canvasY = displayY * (canvas.height / rect.height);
    
    // Handle resize dragging
    if (canvas._resizing && canvas._resizeBoxIdx !== undefined) {
      const idx = canvas._resizeBoxIdx;
      const corner = canvas._resizeCorner;
      const b = labelBoxes[idx];
      const orig = canvas._resizeOrig;
      
      // Convert to pixel coordinates
      const left = orig.x * canvas.width - orig.w * canvas.width / 2;
      const top = orig.y * canvas.height - orig.h * canvas.height / 2;
      const right = left + orig.w * canvas.width;
      const bottom = top + orig.h * canvas.height;
      
      // Clamp to canvas
      const clampX = Math.max(0, Math.min(canvas.width, canvasX));
      const clampY = Math.max(0, Math.min(canvas.height, canvasY));
      
      let newLeft = left, newTop = top, newRight = right, newBottom = bottom;
      
      // Corner resize
      if (corner === 'tl') { newLeft = clampX; newTop = clampY; canvas.style.cursor = 'nwse-resize'; }
      else if (corner === 'tr') { newRight = clampX; newTop = clampY; canvas.style.cursor = 'nesw-resize'; }
      else if (corner === 'br') { newRight = clampX; newBottom = clampY; canvas.style.cursor = 'nwse-resize'; }
      else if (corner === 'bl') { newLeft = clampX; newBottom = clampY; canvas.style.cursor = 'nesw-resize'; }
      // Edge resize
      else if (corner === 't') { newTop = clampY; canvas.style.cursor = 'ns-resize'; }
      else if (corner === 'b') { newBottom = clampY; canvas.style.cursor = 'ns-resize'; }
      else if (corner === 'l') { newLeft = clampX; canvas.style.cursor = 'ew-resize'; }
      else if (corner === 'r') { newRight = clampX; canvas.style.cursor = 'ew-resize'; }
      
      // Enforce minimum size
      if (newRight - newLeft < 10) newRight = newLeft + 10;
      if (newBottom - newTop < 10) newBottom = newTop + 10;
      
      // Update box with new dimensions
      const newW = (newRight - newLeft) / canvas.width;
      const newH = (newBottom - newTop) / canvas.height;
      const newX = (newLeft + newRight) / 2 / canvas.width;
      const newY = (newTop + newBottom) / 2 / canvas.height;
      
      b.x = newX;
      b.y = newY;
      b.w = newW;
      b.h = newH;
      
      out('labelOut', '🔧 Resizing box ' + idx + '... Size: ' + Math.round(newW * canvas.width) + 'x' + Math.round(newH * canvas.height) + 'px');
      markLabelDirty();  // Mark as unsaved
      labelDraw();
      updateLabelBoxPreview();
      renderBoxList();
      return;
    }
    
    // Check if we're dragging over a selected box (to allow overlay drawing)
    if (canvas._clickOnBoxIdx !== -1 && canvas._clickStartX !== undefined) {
      const dragDistance = Math.hypot(canvasX - canvas._clickStartX, canvasY - canvas._clickStartY);
      const dragThreshold = 8;  // pixels of movement to confirm drawing intent
      
      if (dragDistance > dragThreshold) {
        // User is dragging - start new box drawing (overlay mode)
        canvas._clickOnBoxIdx = -1;  // Clear the selection
        labelDragging = true;
        labelCurrent = {
          x1: canvas._clickStartX / canvas.width,
          y1: canvas._clickStartY / canvas.height,
          x2: canvasX / canvas.width,
          y2: canvasY / canvas.height,
          isPreview: true
        };
        out('labelOut', '🎨 Drawing new box (overlay mode)');
        labelDraw();
        return;
      }
    }
    
    // Check for resize cursor on hover (corners + edges)
    const handleRadius = 15;
    const edgeThickness = 12;
    let onHandle = false;
    
    for (let i = labelBoxes.length - 1; i >= 0; i--) {
      const b = labelBoxes[i];
      if (b.isPreview) continue;
      
      const left = b.x * canvas.width - b.w * canvas.width / 2;
      const top = b.y * canvas.height - b.h * canvas.height / 2;
      const right = left + b.w * canvas.width;
      const bottom = top + b.h * canvas.height;
      
      // Check corners
      const corners = [
        {cx: left, cy: top, cursor: 'nwse-resize'},
        {cx: right, cy: top, cursor: 'nesw-resize'},
        {cx: right, cy: bottom, cursor: 'nwse-resize'},
        {cx: left, cy: bottom, cursor: 'nesw-resize'}
      ];
      
      for (let corner of corners) {
        if (Math.abs(canvasX - corner.cx) <= handleRadius && Math.abs(canvasY - corner.cy) <= handleRadius) {
          canvas.style.cursor = corner.cursor;
          onHandle = true;
          break;
        }
      }
      
      if (onHandle) break;
      
      // Check edges
      const edges = [
        {x: left + (right - left) / 2, y: top, cursor: 'ns-resize', type: 'h'},      // top
        {x: right, y: top + (bottom - top) / 2, cursor: 'ew-resize', type: 'v'},     // right
        {x: left + (right - left) / 2, y: bottom, cursor: 'ns-resize', type: 'h'},   // bottom
        {x: left, y: top + (bottom - top) / 2, cursor: 'ew-resize', type: 'v'}       // left
      ];
      
      for (let edge of edges) {
        const distX = Math.abs(canvasX - edge.x);
        const distY = Math.abs(canvasY - edge.y);
        const threshold = edge.type === 'h' ? [edgeThickness * 2, edgeThickness] : [edgeThickness, edgeThickness * 2];
        
        if (distX <= threshold[0] && distY <= threshold[1]) {
          canvas.style.cursor = edge.cursor;
          onHandle = true;
          break;
        }
      }
      
      if (onHandle) break;
    }
    
    if (onHandle) return;
    
    // Normal drawing mode
    if (!labelDragging || !labelCurrent) {
      canvas.style.cursor = 'default';
      return;
    }
    
    // Clamp coordinates to canvas boundaries (0 to canvas.width/height)
    const clampedCanvasX = Math.max(0, Math.min(canvas.width, canvasX));
    const clampedCanvasY = Math.max(0, Math.min(canvas.height, canvasY));
    
    // Update end position (normalized and clamped to 0-1)
    const normX = clampedCanvasX / canvas.width;
    const normY = clampedCanvasY / canvas.height;
    labelCurrent.x2 = normX;
    labelCurrent.y2 = normY;
    
    // Calculate box dimensions
    const x1 = Math.min(labelCurrent.x1, labelCurrent.x2);
    const y1 = Math.min(labelCurrent.y1, labelCurrent.y2);
    const w = Math.abs(labelCurrent.x2 - labelCurrent.x1);
    const h = Math.abs(labelCurrent.y2 - labelCurrent.y1);
    const cx = x1 + w / 2;
    const cy = y1 + h / 2;
    
    // Display dimensions in pixels for feedback
    const wPx = w * canvas.width;
    const hPx = h * canvas.height;
    
    console.log('📐 Drawing preview (clamped to canvas):', {normalized: {x1, y1, w, h}, clamped: true, pixels: {w: Math.round(wPx), h: Math.round(hPx)}});
    
    // Remove old preview box if exists (only the current drawing preview)
    if (labelBoxes.length > 0 && labelBoxes[labelBoxes.length - 1].isPreview) {
      labelBoxes.pop();
    }
    
    const classElem = document.getElementById('labelClass');
    const cid = classElem ? parseInt(classElem.value) || 0 : 0;
    
    // Create preview box (marked as preview)
    const previewBox = { 
      x: cx,           // Center X normalized
      y: cy,           // Center Y normalized
      w: w,            // Width normalized
      h: h,            // Height normalized
      'class': cid,
      isPreview: true  // Mark as temporary preview
    };
    labelBoxes.push(previewBox);
    markLabelDirty();
    
    out('labelOut', '✏️ Drawing box... Size: ' + Math.round(wPx) + 'x' + Math.round(hPx) + 'px | Total boxes: ' + (labelBoxes.length - 1) + ' saved + 1 preview');
    canvas.style.cursor = 'crosshair';
    labelDraw();
    renderBoxList();
  };
  
  // Cursor leaving canvas: Keep drawing! Box stays at canvas edge (clamped)
  // No longer cancel drawing when mouse leaves
  labelCanvas.onmouseleave = () => { 
    // Just reset cursor, don't cancel drawing
    if (!labelDragging) {
      labelCanvas.style.cursor = 'default';
    }
  };
  
  labelCanvas.onmouseup = async () => { 
    const canvas = document.getElementById('labelCanvas');
    
    // Finish resizing
    if (canvas._resizing) {
      canvas._resizing = false;
      const idx = canvas._resizeBoxIdx;
      const b = labelBoxes[idx];
      out('labelOut', '✅ Box ' + idx + ' resized! Size: ' + Math.round(b.w * canvas.width) + 'x' + Math.round(b.h * canvas.height) + 'px');
      canvas._resizeBoxIdx = undefined;
      canvas._resizeCorner = undefined;
      canvas._resizeOrig = undefined;
      labelDraw();
      updateLabelBoxPreview();
      renderBoxList();
      return;
    }
    
    // If clicked on existing box without dragging, check SKU status
    if (canvas._clickOnBoxIdx !== -1 && !labelDragging) {
      const boxIdx = canvas._clickOnBoxIdx;
      const box = labelBoxes[boxIdx];
      canvas._clickOnBoxIdx = -1;
      
      // If box already has SKU assigned, don't show popup - just select it
      if (box.sku_id) {
        out('labelOut', `📦 Box ${boxIdx} selected - SKU: ${box.sku_id} (no change needed)`);
        return;
      }
      
      // Box has no SKU - show popup to select SKU
      showSKUSelectionPopup(boxIdx, (selectedSKU) => {
        if (selectedSKU) {
          labelBoxes[boxIdx].sku_id = selectedSKU;
          renderBoxList();
        }
      }, true);  // true = mandatory SKU selection
      
      return;
    }
    
    canvas._clickOnBoxIdx = -1;
    canvas._clickStartX = undefined;
    canvas._clickStartY = undefined;
    
    // Finish drawing
    if (!labelDragging || !labelCurrent) return;
    
    labelDragging = false;
    
    // Finalize: Convert preview to permanent box
    if (labelBoxes.length > 0 && labelBoxes[labelBoxes.length - 1].isPreview) {
      const lastBox = labelBoxes[labelBoxes.length - 1];
      delete lastBox.isPreview;  // Remove preview flag - now permanent
      markLabelDirty();  // Mark as unsaved changes
      
      const lastIdx = labelBoxes.length - 1;
      
      // Show SKU selection popup (new box - isMandatory=false)
      showSKUSelectionPopup(lastIdx, (selectedSKU) => {
        // If user cancelled (selectedSKU is null), box is already removed by showSKUSelectionPopup
        if (!selectedSKU) {
          out('labelOut', '⚠️ Label creation cancelled');
          console.log('❌ Box creation cancelled by user');
          return;
        }
        
        // User confirmed with SKU selection
        lastBox.sku_id = selectedSKU;
        console.log('✅ SKU assigned to box:', lastBox.sku_id);
        
        const cid = lastBox.class || 0;
        const className = labelClassNames[cid] || ('class'+cid);
        const skuDisplay = lastBox.sku_id ? `SKU: ${lastBox.sku_id}\n` : '';
        
        // Convert to Label Studio format for display
        const [x_pct, y_pct, w_pct, h_pct] = holoToLabelStudio(lastBox.x, lastBox.y, lastBox.w, lastBox.h);
        
        const boxOutput = `📦 Box ${lastIdx} SAVED!\n` +
          `Class: ${className}\n` +
          `${skuDisplay}` +
          `Tier1: ${tier1} | Tier2: ${tier2}\n` +
          `Normalized: x=${lastBox.x.toFixed(4)}, y=${lastBox.y.toFixed(4)}, w=${lastBox.w.toFixed(4)}, h=${lastBox.h.toFixed(4)}\n` +
          `Label Studio %: x=${x_pct.toFixed(1)}, y=${y_pct.toFixed(1)}, w=${w_pct.toFixed(1)}, h=${h_pct.toFixed(1)}\n` +
          `💾 Total boxes in memory: ${labelBoxes.length}`;
        
        out('labelOut', boxOutput);
        console.log('✋ Box finalized and saved:', {lastIdx, box: lastBox});
        
        // Force list refresh with a small delay to ensure DOM is ready
        setTimeout(() => {
          renderBoxList();
        }, 10);
        
        labelDraw();
        updateLabelBoxPreview();
        saveLabelsToJSON();  // Save to JSON in real-time
      });
      
      // Auto-select for editing
      labelSelectedBox = lastIdx;
      highlightBoxRow(lastIdx);
    }
    
    labelCurrent = null;
    labelDraw();
    updateLabelBoxPreview();
  };
}

// Populate global label dropdowns for dashboard
function populateDashboardClassDropdowns() {
  try {
    const sel = document.getElementById('labelClass'); if (sel) { sel.innerHTML=''; labelClassNames.forEach((n,i)=>{const o=document.createElement('option'); o.value=i; o.textContent = i+': '+n; sel.appendChild(o);}); }
    const ds = document.getElementById('dashboardClassSelect'); if (ds) { ds.innerHTML=''; labelClassNames.forEach((n,i)=>{const o=document.createElement('option'); o.value=i; o.textContent = i+': '+n; ds.appendChild(o);}); }
  } catch(e){}
}

// Add class to server (reuses /add_class endpoint)
async function addDashboardClassToServer(name) {
  const r = await fetch('/add_class', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name}) });
  const j = await r.json();
  if (j.names) return j.names; throw new Error(j.error || 'no response');
}

// Show the dashboard class chooser modal for newly created box
async function showDashboardClassChooserForBox(idx) {
  const modal = document.getElementById('dashboardClassChooser');
  const select = document.getElementById('dashboardClassSelect');
  const newBtn = document.getElementById('dashboardClassNewBtn');
  const newInput = document.getElementById('dashboardClassNewInput');
  const ok = document.getElementById('dashboardClassOk');
  const cancel = document.getElementById('dashboardClassCancel');
  if (!modal || !select) return;
  function populate() { select.innerHTML=''; labelClassNames.forEach((n,i)=>{const o=document.createElement('option'); o.value=i; o.textContent=i+': '+n; select.appendChild(o);}); }
  populate(); newInput.style.display='none'; modal.style.display='flex';
  
  // Show box info
  const box = labelBoxes[idx];
  out('labelOut', '📦 Box ' + idx + ' created!');
  
  const onNew = ()=>{ newInput.style.display = newInput.style.display==='none' ? 'block' : 'none'; if (newInput.style.display==='block') newInput.focus(); }
  newBtn.onclick = onNew;
  const cleanup = ()=>{ modal.style.display='none'; newBtn.onclick=null; ok.onclick=null; cancel.onclick=null; }
  ok.onclick = async ()=>{
    if (newInput.style.display==='block' && newInput.value && newInput.value.trim()) {
      try { const names = await addDashboardClassToServer(newInput.value.trim()); labelClassNames = names; } catch(e) { showAlert('Failed to add class: '+e.message, '❌ Add Class Failed'); }
      populate();
      labelBoxes[idx].class = labelClassNames.length - 1;
    } else {
      labelBoxes[idx].class = parseInt(select.value) || 0;
    }
    // persist to disk immediately for this image (no page reload)
    try {
      const imgName = document.getElementById('labelImageList').value;
      if (imgName) {
        await saveLabelsForImage(imgName);
      }
    } catch(e) { out('labelOut', 'Auto-save after create failed: '+e.message); }
    cleanup(); populateDashboardClassDropdowns(); updateLabelBoxPreview(); labelDraw(); renderBoxList();  // Reload list when class changes
  };
  cancel.onclick = ()=>{ if (idx>=0 && idx<labelBoxes.length) labelBoxes.splice(idx,1); cleanup(); populateDashboardClassDropdowns(); updateLabelBoxPreview(); labelDraw(); renderBoxList(); };  // Reload list when cancelled
}

// Save simple single-class annotation
async function saveLabelsToJSON() {
  const imageName = document.getElementById('labelImageList')?.value || document.getElementById('imageNameInput')?.value;
  if (!imageName) {
    console.log('No image name, skipping auto-save');
    return;
  }
  
  // Extract stem from image name (remove extension)
  const imageStem = imageName.split('.').slice(0, -1).join('.');
  
  if (labelBoxes.length === 0) {
    console.log('No boxes to save');
    return;
  }
  
  // Get image dimensions from canvas
  const canvas = document.getElementById('labelCanvas');
  const imageWidth = canvas ? canvas.width : 640;
  const imageHeight = canvas ? canvas.height : 480;
  
  // Build annotation for JSON save with SKU
  const annotation = {
    image_name: imageName,
    image_stem: imageStem,
    image_width: imageWidth,
    image_height: imageHeight,
    timestamp: new Date().toISOString(),
    class_name: "product",
    boxes: labelBoxes.map(function(box) {
      const normalizedX = Math.max(0, Math.min(1, box.x));
      const normalizedY = Math.max(0, Math.min(1, box.y));
      const normalizedW = Math.max(0, Math.min(1, box.w));
      const normalizedH = Math.max(0, Math.min(1, box.h));
      
      const x6 = parseFloat(normalizedX.toFixed(6));
      const y6 = parseFloat(normalizedY.toFixed(6));
      const w6 = parseFloat(normalizedW.toFixed(6));
      const h6 = parseFloat(normalizedH.toFixed(6));
      
      return {
        bbox: [0, x6, y6, w6, h6],
        class_name: "product",
        sku: box.sku_id || null,
        cropped: box.cropped || null
      };
    })
  };
  
  try {
    // Save JSON to dataset folder
    const r = await fetch('/api/save_simple_labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        boxes: annotation
      })
    });
    
    if (!r.ok) {
      console.error('Save failed with status:', r.status);
      const errData = await r.json();
      console.error('Server error:', errData);
      return;
    }
    
    const j = await r.json();
    
    if (j.success || j.saved) {
      console.log('✅ Auto-saved to JSON:', imageStem + '.json', j);
      out('labelOut', `✅ Saved ${j.boxes_count || labelBoxes.length} boxes to ${imageStem}.json`);
    } else {
      console.warn('Save returned but success=false:', j);
    }
  } catch(e) {
    console.error('❌ Auto-save failed:', e);
    out('labelOut', `❌ Save error: ${e.message}`);
  }
}

// Save simple single-class annotation
async function labelSave() {
  const imageName = document.getElementById('labelImageList').value;
  if (!imageName) {
    out('labelOut', '❌ Choose image first');
    return;
  }
  
  // Extract stem from image name (remove extension)
  const imageStem = imageName.split('.').slice(0, -1).join('.');
  
  if (labelBoxes.length === 0) {
    out('labelOut', '❌ No boxes to save');
    return;
  }
  
  // Get image dimensions from canvas
  const canvas = document.getElementById('labelCanvas');
  const imageWidth = canvas ? canvas.width : 640;
  const imageHeight = canvas ? canvas.height : 480;
  
  // Build annotation for JSON save with SKU
  const annotation = {
    image_name: imageName,
    image_stem: imageStem,
    image_width: imageWidth,
    image_height: imageHeight,
    timestamp: new Date().toISOString(),
    class_name: "product",
    boxes: labelBoxes.map(function(box) {
      const normalizedX = Math.max(0, Math.min(1, box.x));
      const normalizedY = Math.max(0, Math.min(1, box.y));
      const normalizedW = Math.max(0, Math.min(1, box.w));
      const normalizedH = Math.max(0, Math.min(1, box.h));
      
      const x6 = parseFloat(normalizedX.toFixed(6));
      const y6 = parseFloat(normalizedY.toFixed(6));
      const w6 = parseFloat(normalizedW.toFixed(6));
      const h6 = parseFloat(normalizedH.toFixed(6));
      
      return {
        bbox: [0, x6, y6, w6, h6],
        class_name: "product",
        sku: box.sku_id || null,
        cropped: box.cropped || null
      };
    })
  };
  
  try {
    // Save JSON to dataset folder
    const r = await fetch('/api/save_simple_labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        boxes: annotation
      })
    });
    const j = await r.json();
    clearLabelDirty();
    
    let msg = '✅ SAVED TO JSON\n' +
      '📦 Total boxes: ' + labelBoxes.length + '\n' +
      '📁 Location: dataset/json_data/' + imageStem + '.json\n';
    
    // Show SKU info if any boxes have SKUs
    const skuBoxes = labelBoxes.filter(b => b.sku_id);
    if (skuBoxes.length > 0) {
      msg += '🏷️ SKUs saved: ' + skuBoxes.map(b => b.sku_id).join(', ');
    }
    
    out('labelOut', msg);
  } catch(e) {
    out('labelOut', '❌ Error saving annotation: ' + e.message);
  }
}


// Crop and save individual box images
async function cropAndSaveBoxes() {
  if (labelBoxes.length === 0) {
    out('labelOut', '⚠️ No boxes to crop');
    return Promise.resolve();
  }
  
  const imageName = document.getElementById('labelImageList').value;
  if (!imageName) {
    out('labelOut', '❌ Choose image first');
    return Promise.resolve();
  }
  
  const imageStem = imageName.split('.').slice(0, -1).join('.');
  const canvas = document.getElementById('labelCanvas');
  
  if (!canvas || !labelImg.src) {
    out('labelOut', '❌ Image not loaded');
    return Promise.resolve();
  }
  
  return new Promise((resolve) => {
    try {
      let savedCount = 0;
      let errorCount = 0;
      let deletedCount = 0;
      const selectedSKU = getSelectedSKU();
      const cropPromises = [];
      
      // Crop each box and save as separate image
      for (let i = 0; i < labelBoxes.length; i++) {
        const box = labelBoxes[i];
        const cropFilename = `${imageStem}_box${i}.jpg`;
        
        // First: Delete old crop file if it exists (from previous SKU)
        if (box.cropped) {
          console.log(`🗑️ Deleting old crop file: ${box.cropped}`);
          // Parse old path to get old SKU and filename
          const pathParts = box.cropped.split('/');
          if (pathParts.length >= 3) {
            const oldSKU = pathParts[1];
            const oldFilename = pathParts[2];
            
            // Don't fetch delete if the old SKU is same as current
            const currentSKU = box.sku_id || selectedSKU;
            if (oldSKU !== currentSKU) {
              // Delete old file from old SKU folder
              fetch('/api/delete_crop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  skuId: oldSKU,
                  cropFilename: oldFilename
                })
              }).then(r => r.json()).then(data => {
                if (data.ok && data.deleted) {
                  console.log(`✅ Deleted old crop file from ${oldSKU}/`);
                  deletedCount++;
                }
              }).catch(e => console.error('Delete old crop error:', e));
            }
          }
        }
        
        // Convert normalized coordinates to pixel coordinates
        const left = Math.max(0, (box.x - box.w / 2) * canvas.width);
        const top = Math.max(0, (box.y - box.h / 2) * canvas.height);
        const width = Math.min(canvas.width - left, box.w * canvas.width);
        const height = Math.min(canvas.height - top, box.h * canvas.height);
        
        // Create temporary canvas for cropping
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = Math.round(width);
        cropCanvas.height = Math.round(height);
        const cropCtx = cropCanvas.getContext('2d');
        
        // Draw cropped portion
        cropCtx.drawImage(
          labelImg,
          left, top, width, height,
          0, 0, cropCanvas.width, cropCanvas.height
        );
        
        // Convert to blob and send to server using /api/save_crop endpoint
        const cropPromise = new Promise((cropResolve) => {
          cropCanvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, cropFilename);
            // Use selected SKU or box's SKU if available
            const skuId = box.sku_id || selectedSKU || `box_${i}`;
            formData.append('sku_id', skuId);
            formData.append('image_name', imageName);
            formData.append('image_stem', imageStem);
            formData.append('box_index', i);
            
            try {
              const response = await fetch('/api/save_crop', {
                method: 'POST',
                body: formData
              });
              
              const data = await response.json();
              if (data.ok || data.success) {
                // Use the actual filename returned by server (might have random suffix if duplicate)
                const actualFilename = data.filename || cropFilename;
                labelBoxes[i].cropped = `openclip_dataset/${skuId}/${actualFilename}`;
                savedCount++;
                console.log(`✅ Cropped box ${i} → ${skuId}/${actualFilename}`);
              } else {
                errorCount++;
                console.error(`❌ Failed to save box ${i}:`, data.error);
              }
            } catch(e) {
              errorCount++;
              console.error('Crop save error:', e.message);
            }
            cropResolve();
          }, 'image/jpeg', 0.9);
        });
        
        cropPromises.push(cropPromise);
      }
      
      out('labelOut', `🔄 Cropping ${labelBoxes.length} box(es) and saving to openclip_dataset/{sku}/...`);
      
      // Wait for all crops to complete
      Promise.all(cropPromises).then(() => {
        let msg = `✅ Cropped ${savedCount}/${labelBoxes.length} boxes`;
        if (deletedCount > 0) msg += ` | 🗑️ Deleted ${deletedCount} old crop files`;
        if (errorCount > 0) msg += ` | ⚠️ ${errorCount} errors`;
        out('labelOut', msg + ' - ready to save JSON');
        resolve();
      });
    } catch(e) {
      out('labelOut', '❌ Crop error: ' + e.message);
      resolve();
    }
  });
}

// SCAN functions
async function takeImageScan() {
  const fileInput = document.getElementById('scanImageFile');
  if (!fileInput || !fileInput.files[0]) {
    showAlert('Please select or capture an image first', '⚠️ No Image Selected');
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('image', file);
  
  const resultDiv = document.getElementById('scanResult');
  resultDiv.style.display = 'none';
  document.getElementById('scanOut').textContent = '⏳ Running detection on image...';
  
  try {
    // Call /api/detect endpoint
    const r = await fetch('/api/detect', { 
      method: 'POST', 
      body: formData 
    });
    const data = await r.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Detection failed');
    }
    
    // Update unified display
    const productCount = data.product_count || 0;
    const skuMatches = data.sku_matches || {};
    const uniqueSkuCount = Object.keys(skuMatches).length;
    
    document.getElementById('productCountBadge').textContent = `${productCount} Product${productCount !== 1 ? 's' : ''}`;
    document.getElementById('skuCountBadge').textContent = `${uniqueSkuCount} SKU${uniqueSkuCount !== 1 ? 's' : ''}`;
    
    // Update summary stats
    document.getElementById('detectionCount').textContent = productCount;
    document.getElementById('imageDimensions').textContent = data.image_size ? 
      `${data.image_size[0]} × ${data.image_size[1]}px` : 'Unknown';
    document.getElementById('uniqueSkuCount').textContent = uniqueSkuCount;
    
    // Build unified detection details
    let detailsHtml = '';
    if (data.detections && data.detections.length > 0) {
      detailsHtml = `Detected ${productCount} product(s):\n\n`;
      data.detections.forEach((det, idx) => {
        detailsHtml += `Product ${idx + 1}:\n`;
        detailsHtml += `  Confidence: ${(det.confidence * 100).toFixed(1)}%\n`;
        detailsHtml += `  Bounding Box: [${det.box.map(v => v.toFixed(3)).join(', ')}]\n`;
        if (det.matched_sku) {
          detailsHtml += `  Matched SKU: ${det.matched_sku}\n`;
          detailsHtml += `  SKU Similarity: ${(det.sku_similarity * 100).toFixed(1)}%\n`;
        }
        detailsHtml += '\n';
      });
      
      // Add statistics
      const avgConf = (data.detections.reduce((sum, d) => sum + d.confidence, 0) / data.detections.length * 100).toFixed(1);
      const maxConf = (Math.max(...data.detections.map(d => d.confidence)) * 100).toFixed(1);
      const minConf = (Math.min(...data.detections.map(d => d.confidence)) * 100).toFixed(1);
      
      detailsHtml += `Statistics:\n`;
      detailsHtml += `  Average Confidence: ${avgConf}%\n`;
      detailsHtml += `  Max Confidence: ${maxConf}%\n`;
      detailsHtml += `  Min Confidence: ${minConf}%\n`;
      
      // Add SKU matches summary
      if (Object.keys(skuMatches).length > 0) {
        detailsHtml += `\nSKU Matches:\n`;
        Object.entries(skuMatches).forEach(([sku, similarity]) => {
          detailsHtml += `  ${sku}: ${(similarity * 100).toFixed(1)}%\n`;
        });
      }
    } else {
      detailsHtml = 'No products detected in the image.';
    }
    
    document.getElementById('detectionDetails').textContent = detailsHtml;
    
    // Display raw JSON
    document.getElementById('scanRawJson').textContent = JSON.stringify(data, null, 2);
    
    // Store image URL for cleanup later
    if (data.image_url) {
      const imageFilename = data.image_url.split('/').pop();
      cleanupTmpFile(imageFilename);
    }
    
    resultDiv.style.display = 'block';
    document.getElementById('scanOut').textContent = `✅ Detection complete: ${productCount} product(s) found`;
    
  } catch (e) {
    document.getElementById('scanOut').textContent = `❌ Error: ${e.message}`;
    console.error('Scan error:', e);
  }
}

// Cleanup temporary scan image files
async function cleanupTmpFile(filename) {
  try {
    const response = await fetch('/api/tmp/cleanup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: filename })
    });
    const result = await response.json();
    if (result.success) {
      console.log(`✓ Cleaned up: ${filename}`);
    } else {
      console.warn(`Could not cleanup: ${result.error}`);
    }
  } catch (e) {
    console.error('Cleanup error:', e);
  }
}

// small helper to escape HTML when showing text
function escapeHtml(unsafe) {
  return unsafe.replace(/[&<>\"]/g, function(c) { return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; });
}


function showImagesForClass(classId) {
  if (!classStatsCache || !classStatsCache.classes || !classStatsCache.classes[classId]) {
    out('scanOut', 'No data for class ' + classId);
    return;
  }
  const classData = classStatsCache.classes[classId];
  const images = classData.images || [];
  const visDiv = document.getElementById('scanVis');
  if (!visDiv) return;
  
  visDiv.innerHTML = `<div style="padding:12px; background:#f9f9f9; border-radius:4px; margin-bottom:12px;">
    <h3>Images for class <strong>${classData.name}</strong> (ID: ${classId})</h3>
    <p>${images.length} images with this class label</p>
  </div>`;
  
  if (images.length === 0) {
    visDiv.innerHTML += '<p style="color:#999;">No images found.</p>';
    return;
  }
  
  const grid = document.createElement('div');
  grid.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; padding: 12px;';
  
  images.forEach(imgName => {
    const thumbDiv = document.createElement('div');
    thumbDiv.style.cssText = 'border: 1px solid #ddd; border-radius: 4px; overflow: hidden; background: #fafafa; text-align: center;';
    
    const img = document.createElement('img');
    img.src = '/uploads/image/' + encodeURIComponent(imgName);
    img.style.cssText = 'width: 100%; height: 120px; object-fit: cover; display: block;';
    img.onerror = () => { img.style.display = 'none'; };
    
    const label = document.createElement('div');
    label.textContent = imgName.substring(0, 20) + (imgName.length > 20 ? '...' : '');
    label.style.cssText = 'font-size: 11px; padding: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #666;';
    
    thumbDiv.appendChild(img);
    thumbDiv.appendChild(label);
    grid.appendChild(thumbDiv);
  });
  
  visDiv.appendChild(grid);
}

// TRAIN functions
// Update config preview when dropdowns change
document.addEventListener('DOMContentLoaded', function() {
  const updatePreview = () => {
    const epochs = document.getElementById('trainEpochs')?.value || '100';
    const imgsz = document.getElementById('trainImgsz')?.value || '640';
    const batch = document.getElementById('trainBatch')?.value || '8';
    const preview = document.getElementById('configPreview');
    if (preview) {
      preview.textContent = `${epochs}ep × ${imgsz}px × ${batch}batch`;
    }
  };
  
  document.getElementById('trainEpochs')?.addEventListener('change', updatePreview);
  document.getElementById('trainImgsz')?.addEventListener('change', updatePreview);
  document.getElementById('trainBatch')?.addEventListener('change', updatePreview);
  updatePreview();
});

async function startTrain() {
  const epochs = parseInt(document.getElementById('trainEpochs').value, 10) || 100;
  const imgsz = parseInt(document.getElementById('trainImgsz').value, 10) || 640;
  const batch = parseInt(document.getElementById('trainBatch').value, 10) || 8;
  const patience = parseInt(document.getElementById('trainPatience').value, 10) || 100;
  const device = document.getElementById('trainDevice').value || '0';
  
  const url = '/train?epochs=' + encodeURIComponent(epochs) + 
              '&device=' + encodeURIComponent(device) + 
              '&imgsz=' + encodeURIComponent(imgsz) +
              '&batch=' + encodeURIComponent(batch) +
              '&patience=' + encodeURIComponent(patience);
  
  const status = document.getElementById('trainStatus');
  const progressBar = document.getElementById('trainProgress');
  
  if (progressBar) {
    progressBar.style.width = '0%';
  }
  
  status.style.display = 'block';
  status.className = 'status info';
  
  status.textContent = `⏳ Starting training: ${epochs} epochs, ${imgsz}px, batch ${batch}, patience ${patience}...`;
  out('trainOut', `Starting training with:\n• Epochs: ${epochs}\n• Image size: ${imgsz}x${imgsz}\n• Batch size: ${batch}\n• Patience: ${patience}\n• Device: ${device}\n\nPlease wait...`);
  
  try {
    const r = await fetch(url, { method: 'POST' });
    const j = await r.json();
    
    if (j.started) {
      status.className = 'status ok';
      status.textContent = `✅ Training started!`;
      out('trainOut', `Training started\n${j.description}\n\n${JSON.stringify(j, null, 2)}`);
      startMonitoringTrain();
    } else {
      status.className = 'status error';
      status.textContent = `❌ Failed: ${j.error || 'Unknown error'}`;
      out('trainOut', JSON.stringify(j, null, 2));
    }
  } catch (e) {
    status.className = 'status error';
    status.textContent = `❌ Error: ${e.message}`;
    out('trainOut', `Error: ${e.message}`);
  }
}

function startMonitoringTrain() {
  // Training progress monitoring disabled - removed web requests
  const statusDiv = document.getElementById('trainStatus');
  if (statusDiv) {
    statusDiv.textContent = '⏳ Training started... (monitoring disabled)';
    setTimeout(() => {
      statusDiv.style.display = 'none';
    }, 3000);
  }
  out('trainOut', '⏳ Training started in background...\n(progress monitoring disabled)');
}

// Scan Model Upload Functions
async function handleScanModelUpload() {
  const fileInput = document.getElementById('scanModelUploadInput');
  const file = fileInput.files[0];
  
  if (!file) return;
  
  if (!file.name.endsWith('.pt')) {
    alert('⚠️ Please select a .pt file (YOLO model)');
    fileInput.value = '';
    return;
  }
  
  try {
    out('trainOut', `⏳ Uploading scan model: ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)...`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    document.getElementById('scanModelUploadProgress').style.display = 'block';
    const progressBar = document.querySelector('#scanModelUploadProgressBar');
    
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percentComplete = (e.loaded / e.total) * 100;
        progressBar.style.width = percentComplete + '%';
      }
    });
    
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        
        if (response.success) {
          document.getElementById('scanModelUploadStatus').innerHTML = 
            `✅ Scan model ready: <strong>${response.filename}</strong><br><small style="color:#666;">${response.size} MB · ${response.filepath}</small>`;
          out('trainOut', `✅ Scan model uploaded successfully: ${response.message}`);
        } else {
          throw new Error(response.error || 'Upload failed');
        }
      } else {
        throw new Error(`Server error: ${xhr.status}`);
      }
      document.getElementById('scanModelUploadProgress').style.display = 'none';
    });
    
    xhr.addEventListener('error', () => {
      alert('❌ Upload error: ' + xhr.statusText);
      document.getElementById('scanModelUploadProgress').style.display = 'none';
    });
    
    xhr.open('POST', '/api/upload-scan-model');
    xhr.send(formData);
    
  } catch (e) {
    alert('❌ Error: ' + e.message);
    document.getElementById('scanModelUploadProgress').style.display = 'none';
  }
}

function clearScanModelUpload() {
  document.getElementById('scanModelUploadInput').value = '';
  document.getElementById('scanModelUploadStatus').textContent = 'No scan model uploaded yet';
  out('trainOut', '🔄 Scan model upload cleared');
}

// SKU Bulk Upload Functions
async function handleSkuBulkUpload() {
  const fileInput = document.getElementById('skuBulkUploadInput');
  const file = fileInput.files[0];
  
  if (!file) return;
  
  if (!file.name.endsWith('.zip')) {
    alert('⚠️ Please select a .zip file');
    fileInput.value = '';
    return;
  }
  
  try {
    out('datasetOut', `⏳ Uploading SKU bulk data: ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)...`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    document.getElementById('skuBulkUploadProgress').style.display = 'block';
    const progressBar = document.querySelector('#skuBulkUploadProgressBar');
    const messageDiv = document.getElementById('skuBulkUploadMessage');
    
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percentComplete = (e.loaded / e.total) * 100;
        progressBar.style.width = percentComplete + '%';
        messageDiv.textContent = `Uploading... ${Math.round(percentComplete)}%`;
      }
    });
    
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        
        if (response.success) {
          messageDiv.textContent = `✅ Import complete: ${response.skus_imported} SKU(s), ${response.total_images} image(s) imported`;
          document.getElementById('skuBulkUploadStatus').innerHTML = 
            `✅ Data imported: <strong>${response.skus_imported} SKU(s)</strong> · ${response.total_images} images<br><small style="color:#666;">${response.message}</small>`;
          out('datasetOut', `✅ ${response.message}`);
          
          // Refresh SKU list
          setTimeout(() => {
            loadDatasetView();
            document.getElementById('skuBulkUploadProgress').style.display = 'none';
          }, 1000);
        } else {
          throw new Error(response.error || 'Import failed');
        }
      } else {
        throw new Error(`Server error: ${xhr.status}`);
      }
    });
    
    xhr.addEventListener('error', () => {
      alert('❌ Upload error: ' + xhr.statusText);
      document.getElementById('skuBulkUploadProgress').style.display = 'none';
    });
    
    xhr.open('POST', '/api/bulk-upload-skus');
    xhr.send(formData);
    
  } catch (e) {
    alert('❌ Error: ' + e.message);
    document.getElementById('skuBulkUploadProgress').style.display = 'none';
  }
}

function clearSkuBulkUpload() {
  document.getElementById('skuBulkUploadInput').value = '';
  document.getElementById('skuBulkUploadStatus').textContent = 'No file selected';
  document.getElementById('skuBulkUploadProgress').style.display = 'none';
  out('datasetOut', '🔄 SKU bulk upload cleared');
}

// DATASET functions
async function loadDatasetLabels() {
  try {
    const r = await fetch('/dataset-info');
    const j = await r.json();
    const labelsDiv = document.getElementById('datasetLabels');
    if (labelsDiv) {
      labelsDiv.innerHTML = '';
      if (j.classes && Object.keys(j.classes).length > 0) {
        const sorted = Object.entries(j.classes).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
        sorted.forEach(([id, data]) => {
          const btn = document.createElement('button');
            // Make label buttons visually square and center their content
            btn.style.cssText = 'padding:12px; background:#667eea; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold; transition:background 0.2s; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; aspect-ratio:1/1; min-width:140px;';
            btn.innerHTML = `<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px;"><strong style="font-size:15px; line-height:1.1;">${data.name}</strong><span style="font-size:12px; opacity:0.9">${data.count} images</span></div>`;
          btn.addEventListener('click', () => showDatasetImages(id, data));
          btn.addEventListener('mouseover', () => btn.style.background = '#764ba2');
          btn.addEventListener('mouseout', () => btn.style.background = '#667eea');
          labelsDiv.appendChild(btn);
        });
      }
    }
  } catch(e) {
    out('datasetOut', 'Error loading dataset: ' + e.message);
  }
}

async function deleteDatasetImage(imageName, classId, classData) {
  if (!confirm(`Delete image: ${imageName}?\n\nThis will remove the image and label from the dataset.`)) {
    return;
  }
  
  try {
    const r = await fetch('/delete_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageName })
    });
    
    const j = await r.json();
    
    if (j.deleted_image) {
      showAlert('✓ Image deleted successfully', '✅ Success');
      // Reload images for this class
      await showDatasetImages(classId, classData);
    } else {
      showAlert('❌ Error: ' + (j.error || 'Failed to delete image'), '❌ Delete Failed');
    }
  } catch(e) {
    showAlert('Error deleting image: ' + e.message, '❌ Error');
  }
}

async function loadAllDatasetImages() {
  const imageGrid = document.getElementById('datasetImageGrid');
  const imagesDiv = document.getElementById('datasetImages');
  const titleDiv = document.getElementById('datasetClassTitle');
  
  if (!imageGrid || !imagesDiv || !titleDiv) return;
  
  imageGrid.style.display = 'block';
  titleDiv.textContent = 'All Images in Dataset';
  imagesDiv.innerHTML = '<div style="grid-column:1/-1; text-align:center; padding:20px;">⏳ Loading all images...</div>';
  
  try {
    // Fetch all images from all classes
    const r = await fetch('/label_contents');
    const j = await r.json();
    
    imagesDiv.innerHTML = '';
    
    if (j.images && j.images.length > 0) {
      let imageCount = 0;
      j.images.forEach(imgData => {
        imageCount++;
        const imgDiv = document.createElement('div');
        imgDiv.style.cssText = 'border:1px solid #ddd; border-radius:4px; overflow:hidden; background:#fafafa; position:relative;';
        
        const imgEl = document.createElement('img');
        imgEl.src = imgData.display_url ? imgData.display_url : ('/uploads/image/' + encodeURIComponent(imgData.name));
        imgEl.style.cssText = 'width:100%; height:200px; object-fit:cover; display:block; background:#f0f0f0;';
        imgEl.title = imgData.name;
        
        const infoDiv = document.createElement('div');
        infoDiv.style.cssText = 'padding:8px; background:#f9f9f9; border-top:1px solid #eee; font-size:12px;';
        infoDiv.innerHTML = `<strong>${imgData.name}</strong><br/><span style="color:#666;">${imgData.box_count || 0} boxes</span>`;
        
        // Add delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.innerHTML = '🗑 Delete';
        deleteBtn.style.cssText = 'position:absolute; top:5px; right:5px; padding:4px 8px; background:rgba(255,0,0,0.9); color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px; font-weight:bold; opacity:0; transition:opacity 0.2s; z-index:10;';
        deleteBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          deleteDatasetImageAll(imgData.name);
        };
        
        imgDiv.onmouseover = () => { deleteBtn.style.opacity = '1'; };
        imgDiv.onmouseout = () => { deleteBtn.style.opacity = '0'; };
        
        imgDiv.appendChild(imgEl);
        imgDiv.appendChild(deleteBtn);
        imgDiv.appendChild(infoDiv);
        imagesDiv.appendChild(imgDiv);
      });
      
      // Update title with count
      titleDiv.textContent = `All Images in Dataset (${imageCount} images)`;
    } else {
      imagesDiv.innerHTML = '<p style="color:#999; grid-column:1/-1; text-align:center; padding:20px;">No images found in dataset</p>';
    }
  } catch(e) {
    out('datasetOut', 'Error loading all images: ' + e.message);
    imagesDiv.innerHTML = '<p style="color:#999; grid-column:1/-1; text-align:center; padding:20px;">Error loading images</p>';
  }
}

async function deleteDatasetImageAll(imageName) {
  if (!confirm(`Delete image: ${imageName}?\n\nThis will remove the image and label from the dataset.`)) {
    return;
  }
  
  try {
    const r = await fetch('/delete_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageName })
    });
    
    const j = await r.json();
    
    if (j.deleted_image) {
      showAlert('✓ Image deleted successfully', '✅ Success');
      // Reload all images
      await loadAllDatasetImages();
    } else {
      showAlert('❌ Error: ' + (j.error || 'Failed to delete image'), '❌ Delete Failed');
    }
  } catch(e) {
    showAlert('Error deleting image: ' + e.message, '❌ Error');
  }
}

async function showDatasetImages(classId, classData) {
  const imageGrid = document.getElementById('datasetImageGrid');
  const imagesDiv = document.getElementById('datasetImages');
  const titleDiv = document.getElementById('datasetClassTitle');
  
  if (!imageGrid || !imagesDiv || !titleDiv) return;
  
  imageGrid.style.display = 'block';
  titleDiv.textContent = `${classData.name} (${classData.count} images)`;
  imagesDiv.innerHTML = '';
  
  try {
    const r = await fetch('/dataset-images?class=' + encodeURIComponent(classId) + '&max_kb=100');
    const j = await r.json();
    
    if (j.images && j.images.length > 0) {
      j.images.forEach(imgData => {
        const imgDiv = document.createElement('div');
        imgDiv.style.cssText = 'border:1px solid #ddd; border-radius:4px; overflow:hidden; background:#fafafa; position:relative;';
        
        const imgEl = document.createElement('img');
        imgEl.src = imgData.display_url ? imgData.display_url : ('/uploads/image/' + encodeURIComponent(imgData.name));
        imgEl.style.cssText = 'width:100%; height:200px; object-fit:cover; display:block; background:#f0f0f0;';
        imgEl.title = imgData.name;
        
        const infoDiv = document.createElement('div');
        infoDiv.style.cssText = 'padding:8px; background:#f9f9f9; border-top:1px solid #eee; font-size:12px;';
        infoDiv.innerHTML = `<strong>${imgData.name}</strong><br/><span style="color:#666;">${imgData.box_count} boxes</span><br/><span style="color:#999; font-size:11px">${imgData.size_kb || '?'} KB</span>`;
        
        // Add delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.innerHTML = '🗑 Delete';
        deleteBtn.style.cssText = 'position:absolute; top:5px; right:5px; padding:4px 8px; background:rgba(255,0,0,0.9); color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px; font-weight:bold; opacity:0; transition:opacity 0.2s; z-index:10;';
        deleteBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          deleteDatasetImage(imgData.name, classId, classData);
        };
        
        imgDiv.onmouseover = () => { deleteBtn.style.opacity = '1'; };
        imgDiv.onmouseout = () => { deleteBtn.style.opacity = '0'; };
        
        // Add SVG canvas overlay for showing boxes if available
        if (imgData.boxes && imgData.boxes.length > 0) {
          const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          svg.style.cssText = 'position:absolute; width:100%; height:200px; top:0; left:0; pointer-events:none;';
          svg.setAttribute('viewBox', '0 0 ' + (imgData.img_width || 640) + ' ' + (imgData.img_height || 480));
          
          imgData.boxes.forEach(box => {
            const x1 = (box.x - box.w / 2) * (imgData.img_width || 640);
            const y1 = (box.y - box.h / 2) * (imgData.img_height || 480);
            const w = box.w * (imgData.img_width || 640);
            const h = box.h * (imgData.img_height || 480);
            
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x1);
            rect.setAttribute('y', y1);
            rect.setAttribute('width', w);
            rect.setAttribute('height', h);
            rect.setAttribute('fill', 'none');
            rect.setAttribute('stroke', '#00ff00');
            rect.setAttribute('stroke-width', '2');
            svg.appendChild(rect);
          });
          
          imgDiv.style.position = 'relative';
          imgDiv.appendChild(imgEl);
          imgDiv.appendChild(svg);
          imgDiv.appendChild(deleteBtn);
        } else {
          imgDiv.appendChild(imgEl);
          imgDiv.appendChild(deleteBtn);
        }
        
        imgDiv.appendChild(infoDiv);
        imagesDiv.appendChild(imgDiv);
      });
    } else {
      imagesDiv.innerHTML = '<p style="color:#999; grid-column:1/-1; text-align:center; padding:20px;">No images found for this class</p>';
    }
  } catch(e) {
    out('datasetOut', 'Error loading images: ' + e.message);
  }
}

// ============================================================================
// DATASET VIEWER - SKU BROWSER WITH PAGINATION AND SEARCH
// ============================================================================




// BACKUPS functions
async function loadBackups() {
  try {
    const outDiv = document.getElementById('backupOut');
    outDiv.style.display = 'block';
    out('backupOut', '⏳ Loading backups...');
    
    const r = await fetch('/backups');
    const j = await r.json();
    
    const table = document.getElementById('backupsTable');
    table.innerHTML = '';
    
    // Show current model info
    if (j.current_model) {
      const row = document.createElement('tr');
      row.style.background = '#d4edda';
      row.innerHTML = `
        <td style="padding:10px; border:1px solid #ddd; font-weight:bold;">⭐ ${j.current_model.name}</td>
        <td style="padding:10px; border:1px solid #ddd; text-align:center;">${j.current_model.size_mb}</td>
        <td style="padding:10px; border:1px solid #ddd;">${j.current_model.modified}</td>
        <td style="padding:10px; border:1px solid #ddd; text-align:center; color:#999;">Current</td>
      `;
      table.appendChild(row);
    }
    
    if (!j.backups || j.backups.length === 0) {
      if (!j.current_model) {
        table.innerHTML = '<tr><td colspan="4" style="padding:20px; text-align:center; color:#999;">No backups found. Start training to create backups.</td></tr>';
      }
      out('backupOut', 'No backups yet.');
      return;
    }
    
    j.backups.forEach((b, idx) => {
      const row = document.createElement('tr');
      row.style.background = idx % 2 === 0 ? '#fff' : '#f9f9f9';
      row.innerHTML = `
        <td style="padding:10px; border:1px solid #ddd;"><code style="font-size:12px;">${b.name}</code></td>
        <td style="padding:10px; border:1px solid #ddd; text-align:center;">${b.size_mb}</td>
        <td style="padding:10px; border:1px solid #ddd;">${b.modified}</td>
        <td style="padding:10px; border:1px solid #ddd; text-align:center;">
          <button onclick="restoreBackup('${b.name}')" style="padding:4px 8px; background:#27ae60; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px; margin-right:4px;">↩️ Restore</button>
          <button onclick="downloadBackup('${b.name}')" style="padding:4px 8px; background:#3498db; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px; margin-right:4px;">⬇️ Download</button>
          <button onclick="deleteBackup('${b.name}')" style="padding:4px 8px; background:#e74c3c; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px;">🗑 Delete</button>
        </td>
      `;
      table.appendChild(row);
    });
    
    out('backupOut', `✅ Loaded ${j.total} backup(s) + current model`);
  } catch (e) {
    out('backupOut', '❌ Error loading backups: ' + e.message);
    document.getElementById('backupOut').style.display = 'block';
  }
}

async function restoreBackup(backup_name) {
  if (!confirm(`Restore backup: ${backup_name}?\n\nCurrent scan.pt will be replaced.`)) return;
  
  try {
    const outDiv = document.getElementById('backupOut');
    outDiv.style.display = 'block';
    out('backupOut', `↩️ Restoring ${backup_name}...`);
    
    const r = await fetch('/backups', { 
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backup_name: backup_name })
    });
    const j = await r.json();
    
    if (r.ok && j.success) {
      out('backupOut', `✅ ${j.message}`);
      setTimeout(() => loadBackups(), 1000); // Reload list
    } else {
      out('backupOut', `❌ Error: ${j.error || j.message}`);
    }
  } catch (e) {
    out('backupOut', '❌ Error restoring backup: ' + e.message);
    document.getElementById('backupOut').style.display = 'block';
  }
}

async function downloadBackup(backup_name) {
  try {
    // Start download
    const url = `/backups/download/${encodeURIComponent(backup_name)}`;
    const link = document.createElement('a');
    link.href = url;
    link.download = backup_name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    const outDiv = document.getElementById('backupOut');
    outDiv.style.display = 'block';
    out('backupOut', `⬇️ Downloading ${backup_name}...`);
  } catch (e) {
    out('backupOut', '❌ Error downloading backup: ' + e.message);
    document.getElementById('backupOut').style.display = 'block';
  }
}

async function deleteBackup(filename) {
  if (!confirm(`Delete backup: ${filename}?`)) return;
  
  try {
    const outDiv = document.getElementById('backupOut');
    outDiv.style.display = 'block';
    out('backupOut', `🗑 Deleting ${filename}...`);
    
    const r = await fetch(`/backups?file=${encodeURIComponent(filename)}`, { method: 'DELETE' });
    const j = await r.json();
    
    if (r.ok) {
      out('backupOut', `✅ ${j.message}`);
      setTimeout(() => loadBackups(), 500); // Reload list
    } else {
      out('backupOut', `❌ Error: ${j.error}`);
    }
  } catch (e) {
    out('backupOut', '❌ Error deleting backup: ' + e.message);
    document.getElementById('backupOut').style.display = 'block';
  }
}

// ============================================================
// SKU MANAGEMENT FUNCTIONS
// ============================================================

// Add SKU to table
function addSKUToTable() {
  const input = document.getElementById('skuInput');
  const sku = input.value.trim().toUpperCase();
  
  if (!sku) {
    showAlert('Please enter a SKU ID', '⚠️ Empty SKU');
    return;
  }
  
  const list = getSKUList();
  if (list.includes(sku)) {
    showAlert('SKU already exists: ' + sku, '⚠️ Duplicate SKU');
    return;
  }
  
  list.push(sku);
  saveSKUList(list);
  input.value = '';
}

// Render SKU table
function renderSKUTable(filterText = '') {
  const tbody = document.getElementById('skuTableBody');
  const list = getSKUList();
  
  // Filter SKUs based on search text
  const filteredList = filterText 
    ? list.filter(sku => sku.toUpperCase().includes(filterText.toUpperCase()))
    : list;
  
  if (filteredList.length === 0) {
    if (filterText) {
      tbody.innerHTML = '<tr style="background:#f9f9f9;"><td colspan="3" style="padding:12px; text-align:center; color:#999; font-size:11px;">No SKUs found matching "' + filterText + '"</td></tr>';
    } else {
      tbody.innerHTML = '<tr style="background:#f9f9f9;"><td colspan="3" style="padding:12px; text-align:center; color:#999; font-size:11px;">No SKUs added yet</td></tr>';
    }
    return;
  }
  
  tbody.innerHTML = filteredList.map((sku, idx) => `
    <tr style="background:${idx % 2 === 0 ? '#ffffff' : '#f9f9f9'}; border-bottom:1px solid #eee;">
      <td style="padding:8px; border:1px solid #ddd; text-align:center; font-weight:bold;">${idx + 1}</td>
      <td style="padding:8px; border:1px solid #ddd;">${sku}</td>
      <td style="padding:8px; border:1px solid #ddd; text-align:center; gap:4px; display:flex; justify-content:center;">
        <button onclick="editSKU('${sku}')" style="padding:4px 8px; background:#0dcaf0; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px;">Edit</button>
        <button onclick="removeSKUFromTable('${sku}')" style="padding:4px 8px; background:#ff6b6b; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px;">Delete</button>
      </td>
    </tr>
  `).join('');
}

// Search/Filter SKU table
function searchSKU() {
  const searchInput = document.getElementById('skuSearchInput');
  const filterText = searchInput.value;
  renderSKUTable(filterText);
}

// Edit SKU
function editSKU(oldSKU) {
  showPrompt('Enter new SKU ID:', oldSKU, '✏️ Edit SKU', async (newSKU) => {
    const newSKUUpper = newSKU.trim().toUpperCase();
    
    if (!newSKUUpper) {
      showAlert('SKU cannot be empty', '⚠️ Empty SKU');
      return;
    }
    
    if (newSKUUpper === oldSKU) {
      return; // No change
    }
    
    const list = getSKUList();
    
    if (list.includes(newSKUUpper)) {
      showAlert('SKU already exists: ' + newSKUUpper, '⚠️ Duplicate SKU');
      return;
    }
    
    // Move crop files from old to new SKU folder
    try {
      const r = await fetch('/api/rename_sku', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ oldSKU: oldSKU, newSKU: newSKUUpper })
      });
      const result = await r.json();
      
      if (!result.ok) {
        showAlert('File move failed: ' + (result.error || 'Unknown error'), '❌ Error');
        return;
      }
      
      // Update SKU in list
      const idx = list.indexOf(oldSKU);
      if (idx >= 0) {
        list[idx] = newSKUUpper;
        saveSKUList(list);
        renderSKUTable();
        // Update all boxes that had the old SKU
        labelBoxes.forEach(box => {
          if (box.sku_id === oldSKU) {
            box.sku_id = newSKUUpper;
          }
        });
        renderBoxList();
        showAlert(`✅ SKU renamed: ${oldSKU} → ${newSKUUpper}\n📁 Moved ${result.movedCount} crop files`, '✅ Success');
      }
    } catch(e) {
      showAlert('Error: ' + e.message, '❌ Error');
    }
  });
}

// Remove SKU from table
function removeSKUFromTable(sku) {
  showDeleteSKUConfirmation(sku);
}

// Show delete SKU confirmation popup (requires typing "delete")
function showDeleteSKUConfirmation(sku) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; z-index:3001;';
  
  const popup = document.createElement('div');
  popup.style.cssText = 'background:white; padding:24px; border-radius:8px; box-shadow:0 4px 16px rgba(0,0,0,0.3); max-width:400px; width:90%;';
  
  const titleEl = document.createElement('h3');
  titleEl.textContent = '⚠️ Delete SKU ID';
  titleEl.style.cssText = 'margin:0 0 12px 0; font-size:16px; color:#d32f2f; text-align:center;';
  
  const messageEl = document.createElement('p');
  messageEl.textContent = `Are you sure you want to delete "${sku}"? Type "delete" to confirm:`;
  messageEl.style.cssText = 'margin:0 0 8px 0; font-size:13px; color:#666;';
  
  const warningEl = document.createElement('p');
  warningEl.textContent = '❌ This action cannot be undone. All boxes with this SKU will lose their SKU assignment.';
  warningEl.style.cssText = 'margin:0 0 16px 0; font-size:12px; color:#d32f2f; font-weight:bold; padding:8px; background:#ffe0e0; border-radius:4px;';
  
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'Type "delete" to confirm...';
  input.style.cssText = 'width:100%; padding:10px 12px; border:2px solid #ff6b6b; border-radius:4px; font-size:14px; box-sizing:border-box; margin-bottom:16px;';
  input.focus();
  
  const closePopup = () => {
    overlay.remove();
  };
  
  const confirmDelete = () => {
    if (input.value.toLowerCase() !== 'delete') {
      showAlert('You must type "delete" to confirm', '❌ Invalid Input');
      return;
    }
    
    // Perform the actual deletion
    const list = getSKUList().filter(s => s !== sku);
    saveSKUList(list);
    
    // Also remove SKU from any boxes that use it
    labelBoxes.forEach(box => {
      if (box.sku_id === sku) {
        box.sku_id = null;
      }
    });
    
    renderBoxList();
    renderSKUTable();
    
    closePopup();
    showAlert(`SKU "${sku}" has been deleted`, '✅ SKU Deleted');
  };
  
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      confirmDelete();
    }
    if (e.key === 'Escape') {
      closePopup();
    }
  });
  
  const buttonContainer = document.createElement('div');
  buttonContainer.style.cssText = 'display:flex; gap:8px; justify-content:flex-end;';
  
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.style.cssText = 'padding:8px 16px; background:#6c757d; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px;';
  cancelBtn.onclick = closePopup;
  buttonContainer.appendChild(cancelBtn);
  
  const deleteBtn = document.createElement('button');
  deleteBtn.textContent = 'Delete (Enter)';
  deleteBtn.style.cssText = 'padding:8px 24px; background:#dc3545; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px;';
  deleteBtn.onclick = confirmDelete;
  
  buttonContainer.appendChild(deleteBtn);
  
  popup.appendChild(titleEl);
  popup.appendChild(messageEl);
  popup.appendChild(warningEl);
  popup.appendChild(input);
  popup.appendChild(buttonContainer);
  overlay.appendChild(popup);
  document.body.appendChild(overlay);
}

// Render Box List with SKU Selector
function renderBoxList() {
  const tbody = document.getElementById('boxListTableBody');
  
  if (!tbody) {
    console.error('❌ boxListTableBody element not found!');
    return;
  }
  
  if (!labelBoxes || labelBoxes.length === 0) {
    tbody.innerHTML = '<tr style="background:#f9f9f9;"><td colspan="4" style="padding:12px; text-align:center; color:#999; font-size:11px;">No boxes detected</td></tr>';
    return;
  }
  
  const skuList = getSKUList();
  
  const rows = labelBoxes.map((box, idx) => {
    const xCenter = (box.x * 100).toFixed(2);
    const yCenter = (box.y * 100).toFixed(2);
    const width = (box.w * 100).toFixed(2);
    const height = (box.h * 100).toFixed(2);
    
    const isSelected = (idx === labelSelectedBox);
    const bgColor = isSelected ? '#ffd700' : (idx % 2 === 0 ? '#ffffff' : '#f9f9f9');
    const fontWeight = isSelected ? 'bold' : 'normal';
    const boxShadow = isSelected ? '0 0 8px rgba(255, 215, 0, 0.8)' : 'none';
    
    // Build SKU dropdown with "None" option
    const currentSKU = box.sku_id || '';
    const skuOptions = [
      `<option value="" ${!currentSKU ? 'selected' : ''}>None</option>`,
      ...skuList.map(sku => `<option value="${sku}" ${currentSKU === sku ? 'selected' : ''}>${sku}</option>`)
    ].join('');
    
    const skuDropdown = `
      <select onchange="assignSKUToBox(${idx}, this.value)" style="padding:4px; border:2px solid ${currentSKU ? '#28a745' : '#ddd'}; border-radius:3px; font-size:11px; width:100%; background-color:#fff; font-weight:${currentSKU ? 'bold' : 'normal'}; color:${currentSKU ? '#28a745' : '#333'};">
        ${skuOptions}
      </select>
    `;
    
    return `
      <tr id="lbl_row_${idx}" style="background:${bgColor}; border-bottom:1px solid #eee; font-weight:${fontWeight}; box-shadow:${boxShadow};">
        <td style="padding:8px; border:1px solid #ddd; text-align:center; font-weight:bold;">${idx}</td>
        <td style="padding:8px; border:1px solid #ddd; font-size:10px;">
          X:${xCenter}% Y:${yCenter}%<br>W:${width}% H:${height}%
        </td>
        <td style="padding:8px; border:1px solid #ddd;">
          ${skuDropdown}
        </td>
        <td style="padding:8px; border:1px solid #ddd; text-align:center;">
          <button onclick="deleteBoxLabel(${idx})" style="padding:4px 8px; background:#dc3545; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px;">Delete</button>
        </td>
      </tr>
    `;
  }).join('');
  
  // Update DOM with all rows at once
  tbody.innerHTML = rows;
  console.log('✅ Box list refreshed -', labelBoxes.length, 'boxes');
}

// Assign SKU to a box
function assignSKUToBox(boxIndex, sku) {
  if (boxIndex >= 0 && boxIndex < labelBoxes.length) {
    const box = labelBoxes[boxIndex];
    const imageStem = document.getElementById('imageNameInput')?.value?.split('.')[0] || '';
    const cropFilename = imageStem ? `${imageStem}_box${boxIndex}.jpg` : null;
    const oldSKU = box.sku_id;
    
    // Update box SKU immediately in memory
    labelBoxes[boxIndex].sku_id = sku || null;
    markLabelDirty();  // Mark as unsaved
    console.log(`📝 Box ${boxIndex} SKU changed: ${oldSKU || 'None'} → ${sku || 'None'}`);
    
    // Reload list IMMEDIATELY to update dropdown display with new SKU
    renderBoxList();
    console.log(`✅ UI Updated - dropdown shows: ${sku || 'None'}`);
    
    // Show immediate feedback
    const moveMsg = (oldSKU && oldSKU !== sku && cropFilename && box.cropped) ? ' (moving crop image...)' : '';
    out('labelOut', `✅ Box ${boxIndex} assigned to SKU: ${sku || 'None'}${moveMsg}`);
    
    // If changing SKU and crop file exists, move it in real-time
    if (oldSKU && oldSKU !== sku && cropFilename && sku && box.cropped) {
      console.log(`🔄 Moving crop image: ${cropFilename} from openclip_dataset/${oldSKU}/ → openclip_dataset/${sku}/`);
      
      fetch('/api/move_crop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          cropFilename: cropFilename,
          oldSKU: oldSKU,
          newSKU: sku 
        })
      }).then(r => r.json()).then(data => {
        if (data.ok) {
          if (data.moved) {
            console.log(`✅ Crop image moved successfully: openclip_dataset/${oldSKU}/${cropFilename} → openclip_dataset/${sku}/${cropFilename}`);
            // Update the cropped path to reflect new location
            labelBoxes[boxIndex].cropped = `openclip_dataset/${sku}/${cropFilename}`;
            out('labelOut', `✅ Box ${boxIndex}: SKU changed to ${sku} | 📁 Crop image moved to openclip_dataset/${sku}/`);
          } else {
            // Crop file didn't exist in old folder, no need to move
            console.log(`ℹ️ No crop image found in openclip_dataset/${oldSKU}/ - nothing to move`);
            out('labelOut', `✅ Box ${boxIndex}: SKU changed to ${sku}`);
          }
        } else {
          console.warn(`⚠️ Failed to move crop image:`, data.error || data.message);
          out('labelOut', `⚠️ Box ${boxIndex}: SKU changed, but crop move failed: ${data.error || data.message}`);
        }
      }).catch(e => {
        console.error('❌ Network error moving crop:', e.message);
        out('labelOut', `❌ Error moving crop image: ${e.message}`);
      });
    } else if (oldSKU && oldSKU !== sku && !sku) {
      // Removing SKU assignment
      console.log(`🗑️ SKU removed - if crop image exists in openclip_dataset/${oldSKU}/, it remains there`);
      out('labelOut', `✅ Box ${boxIndex}: SKU removed`);
    }
    
    // Save JSON to file in real-time
    saveLabelsToJSON().then(() => {
      console.log(`💾 JSON saved - Box ${boxIndex} now assigned to SKU: ${sku || 'None'}`);
    }).catch(e => {
      console.error('Failed to save JSON:', e);
      out('labelOut', `⚠️ JSON save failed: ${e.message}`);
    });
  }
}

// Delete a box label
function deleteBoxLabel(boxIndex) {
  if (boxIndex >= 0 && boxIndex < labelBoxes.length) {
    const box = labelBoxes[boxIndex];
    const imageStem = document.getElementById('imageNameInput')?.value?.split('.')[0] || '';
    const cropFilename = imageStem ? `${imageStem}_box${boxIndex}.jpg` : null;
    
    // Show confirmation dialog with Save/Cancel options
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; z-index:3001;';
    
    const dialog = document.createElement('div');
    dialog.style.cssText = 'background:white; padding:24px; border-radius:8px; box-shadow:0 4px 16px rgba(0,0,0,0.3); max-width:450px; width:90%;';
    
    const titleEl = document.createElement('h3');
    titleEl.textContent = '⚠️ Delete Box Confirmation';
    titleEl.style.cssText = 'margin:0 0 12px 0; font-size:16px; color:#d32f2f; text-align:center;';
    
    const messageEl = document.createElement('p');
    messageEl.textContent = `Are you sure you want to delete Box ${boxIndex}?`;
    messageEl.style.cssText = 'margin:0 0 8px 0; font-size:14px; color:#333;';
    
    const detailsEl = document.createElement('p');
    if (box.sku_id && cropFilename) {
      detailsEl.innerHTML = `<strong>This will:</strong><br>✓ Remove box from image<br>✓ Delete crop: <code style="background:#f0f0f0; padding:2px 4px;">${cropFilename}</code> from <code style="background:#f0f0f0; padding:2px 4px;">openclip_dataset/${box.sku_id}/</code><br>✓ Update JSON annotation`;
    } else {
      detailsEl.innerHTML = `<strong>This will:</strong><br>✓ Remove box from image<br>✓ Update JSON annotation`;
    }
    detailsEl.style.cssText = 'margin:0 0 16px 0; font-size:12px; color:#666; line-height:1.6;';
    
    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = 'display:flex; gap:8px; justify-content:flex-end;';
    
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = '❌ Cancel';
    cancelBtn.style.cssText = 'padding:10px 16px; background:#6c757d; color:white; border:none; border-radius:4px; cursor:pointer; font-size:14px; font-weight:bold;';
    
    const saveBtn = document.createElement('button');
    saveBtn.textContent = '✅ Delete & Save';
    saveBtn.style.cssText = 'padding:10px 16px; background:#d32f2f; color:white; border:none; border-radius:4px; cursor:pointer; font-size:14px; font-weight:bold;';
    
    const closeDialog = () => {
      if (overlay.parentElement) overlay.remove();
    };
    
    cancelBtn.onclick = () => {
      out('labelOut', '❌ Box deletion cancelled - no changes made');
      console.log(`ℹ️ Box ${boxIndex} deletion cancelled`);
      closeDialog();
    };
    
    saveBtn.onclick = async () => {
      closeDialog();
      
      // Delete crop file from filesystem if exists
      if (box.sku_id && cropFilename) {
        console.log(`🗑️ Deleting crop: ${cropFilename} from openclip_dataset/${box.sku_id}/`);
        try {
          const delResponse = await fetch('/api/delete_crop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              skuId: box.sku_id, 
              cropFilename: cropFilename 
            })
          });
          const delData = await delResponse.json();
          if (delData.ok) {
            console.log(`✅ Crop deleted: openclip_dataset/${box.sku_id}/${cropFilename}`);
          } else {
            console.warn(`⚠️ Failed to delete crop:`, delData.error);
          }
        } catch (e) {
          console.error('Error deleting crop file:', e);
        }
      }
      
      // Remove box from list
      labelBoxes.splice(boxIndex, 1);
      markLabelDirty();  // Mark as unsaved
      console.log(`✅ Box ${boxIndex} removed from annotation`);
      
      // Update UI
      renderBoxList();
      labelDraw();
      out('labelOut', `🗑️ Box ${boxIndex} deleted successfully - crop removed from openclip_dataset/`);
      
      // Save changes to JSON
      try {
        await saveLabelsToJSON();
        console.log(`💾 JSON updated after deletion`);
      } catch (e) {
        console.error('Failed to save JSON after deletion:', e);
      }
    };
    
    buttonContainer.appendChild(cancelBtn);
    buttonContainer.appendChild(saveBtn);
    
    dialog.appendChild(titleEl);
    dialog.appendChild(messageEl);
    dialog.appendChild(detailsEl);
    dialog.appendChild(buttonContainer);
    
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
  }
}

// Initialize SKU table on page load
document.addEventListener('DOMContentLoaded', () => {
  loadSKUListFromFile();
  
  // Add search input listener
  const searchInput = document.getElementById('skuSearchInput');
  if (searchInput) {
    searchInput.addEventListener('input', searchSKU);
  }
});

// Load SKU list from server (dataset/SKU.json)
async function loadSKUListFromFile() {
  try {
    const response = await fetch('/api/load_sku_list');
    const data = await response.json();
    
    if (data.ok && data.skus && data.skus.length > 0) {
      globalSKUList = data.skus;
      console.log('✅ Loaded ' + data.count + ' SKUs from dataset/SKU.json');
    } else {
      globalSKUList = [];
    }
    
    renderSKUTable();
  } catch(e) {
    console.error('Error loading SKU list:', e.message);
    renderSKUTable();
  }
}

// Global SKU list (session only, not saved in localStorage/cookies)
let globalSKUList = [];

function getSKUList() {
  // Return session-only SKU list (no localStorage)
  return globalSKUList;
}

function saveSKUList(list) {
  // Update session list only (no localStorage persistence)
  globalSKUList = [...new Set(list)].sort();
  renderSKUTable();
  // Save to server
  saveSKUListToFile(globalSKUList);
}

// Save SKU list to dataset/SKU.json on server
async function saveSKUListToFile(skuList) {
  try {
    const response = await fetch('/api/save_sku_list', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skuList: skuList })
    });
    const data = await response.json();
    if (data.ok) {
      console.log('✅ SKU list saved to dataset/SKU.json');
    }
  } catch(e) {
    console.error('Error saving SKU list:', e.message);
  }
}

function openAddSKUPopup() {
  document.getElementById('addSKUModal').style.display = 'flex';
  document.getElementById('newSKUInput').focus();
  document.getElementById('newSKUInput').value = '';
}

function closeAddSKUPopup() {
  document.getElementById('addSKUModal').style.display = 'none';
}

function openBulkSKUPopup() {
  document.getElementById('bulkSKUModal').style.display = 'flex';
  document.getElementById('bulkSKUInput').focus();
  document.getElementById('bulkSKUInput').value = '';
}

function closeBulkSKUPopup() {
  document.getElementById('bulkSKUModal').style.display = 'none';
}

function addSingleSKU() {
  const input = document.getElementById('newSKUInput');
  const sku = input.value.trim().toUpperCase();
  
  if (!sku) {
    showAlert('Please enter a SKU ID', '⚠️ Empty SKU');
    return;
  }
  
  if (!/^[A-Z0-9_-]+$/.test(sku)) {
    showAlert('SKU must contain only letters, numbers, underscores, and hyphens', '⚠️ Invalid Format');
    return;
  }
  
  const list = getSKUList();
  if (list.includes(sku)) {
    showAlert('SKU already exists: ' + sku, '⚠️ Duplicate SKU');
    return;
  }
  
  list.push(sku);
  saveSKUList(list);
  closeAddSKUPopup();
  showNotification('SKU added: ' + sku);
}

function addBulkSKUs() {
  const input = document.getElementById('bulkSKUInput');
  const text = input.value.trim();
  
  if (!text) {
    showAlert('Please paste SKUs', '⚠️ Empty Input');
    return;
  }
  
  // Parse both comma-separated and line-separated
  const skus = text
    .split(/[,\n]/)
    .map(s => s.trim().toUpperCase())
    .filter(s => s && /^[A-Z0-9_-]+$/.test(s));
  
  if (skus.length === 0) {
    showAlert('No valid SKUs found', '⚠️ Invalid SKUs');
    return;
  }
  
  const list = getSKUList();
  let added = 0;
  let duplicates = 0;
  
  skus.forEach(sku => {
    if (!list.includes(sku)) {
      list.push(sku);
      added++;
    } else {
      duplicates++;
    }
  });
  
  saveSKUList(list);
  closeBulkSKUPopup();
  showNotification(`Added ${added} SKU(s)` + (duplicates > 0 ? `, ${duplicates} duplicates skipped` : ''));
}

function refreshSKUList() {
  const list = getSKUList();
  const display = document.getElementById('skuListDisplay');
  
  // Skip if element doesn't exist (not used in current dashboard)
  if (!display) return;
  
  if (list.length === 0) {
    display.innerHTML = '<span style="color:#999; font-size:11px; padding:8px; text-align:center;">No SKUs added</span>';
  } else {
    display.innerHTML = list.map(sku => `
      <div style="background:white; padding:8px; border-radius:4px; border:1px solid #667eea; text-align:center; font-size:11px; font-weight:bold; color:#667eea; position:relative; cursor:pointer;" title="Click to remove">
        ${sku}
        <button onclick="removeSKU('${sku}')" style="position:absolute; top:-8px; right:-8px; background:#ff6b6b; color:white; border:none; border-radius:50%; width:18px; height:18px; font-size:12px; cursor:pointer; padding:0; display:flex; align-items:center; justify-content:center;">×</button>
      </div>
    `).join('');
  }
}

function removeSKU(sku) {
  const list = getSKUList().filter(s => s !== sku);
  saveSKUList(list);
  showNotification('SKU removed: ' + sku);
}

function updateSKUDropdown() {
  const list = getSKUList();
  const searchInput = document.getElementById('skuSearchInput');
  
  // Get current search term to re-filter if needed
  const currentSearch = searchInput ? searchInput.value.toLowerCase().trim() : '';
  
  // If there's a search term, filter the list; otherwise show all
  const filtered = currentSearch 
    ? list.filter(sku => sku.toLowerCase().includes(currentSearch))
    : list;
  
  renderSKUDropdownList(filtered);
  // Don't clear search input - let user keep their selection
}

function renderSKUDropdownList(items) {
  const dropdownList = document.getElementById('skuDropdownList');
  if (!dropdownList) return;
  
  if (items.length === 0) {
    dropdownList.innerHTML = '<div style="padding:10px; text-align:center; color:#999;">No SKUs available</div>';
  } else {
    dropdownList.innerHTML = items.map(sku => `
      <div onclick="selectSKUFromDropdown('${sku}')" style="padding:10px; border-bottom:1px solid #eee; cursor:pointer; font-size:12px; background:white; transition:background 0.2s;" 
           onmouseover="this.style.background='#f0f0f0'" onmouseout="this.style.background='white'">
        🏷️ ${sku}
      </div>
    `).join('');
  }
}

function filterSKUList() {
  const searchInput = document.getElementById('skuSearchInput');
  const searchTerm = searchInput.value.toLowerCase().trim();
  const list = getSKUList();
  
  // Filter SKUs based on search term
  const filtered = searchTerm 
    ? list.filter(sku => sku.toLowerCase().includes(searchTerm))
    : list;
  
  renderSKUDropdownList(filtered);
  showSKUDropdown();
}

function showSKUDropdown() {
  const dropdownList = document.getElementById('skuDropdownList');
  if (dropdownList) {
    dropdownList.style.display = 'block';
  }
}

function hideSKUDropdown() {
  const dropdownList = document.getElementById('skuDropdownList');
  if (dropdownList) {
    dropdownList.style.display = 'none';
  }
}

function selectSKUFromDropdown(sku) {
  const searchInput = document.getElementById('skuSearchInput');
  if (searchInput) {
    searchInput.value = sku;
  }
  hideSKUDropdown();
  showNotification('Selected: ' + sku);
}

function getSelectedSKU() {
  const searchInput = document.getElementById('skuSearchInput');
  return searchInput ? searchInput.value.trim() : '';
}

function showNotification(message) {
  // Create temporary notification
  const div = document.createElement('div');
  div.style.cssText = 'position:fixed; bottom:20px; right:20px; background:#20c997; color:white; padding:12px 20px; border-radius:4px; box-shadow:0 2px 8px rgba(0,0,0,0.2); z-index:2000; animation:slideIn 0.3s ease;';
  div.textContent = message;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 3000);
}

// Custom alert popup (no browser alerts)
function showAlert(message, title = '⚠️ Alert') {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; z-index:3000;';
  
  const popup = document.createElement('div');
  popup.style.cssText = 'background:white; padding:24px; border-radius:8px; box-shadow:0 4px 16px rgba(0,0,0,0.3); max-width:400px; text-align:center;';
  
  const titleEl = document.createElement('h3');
  titleEl.textContent = title;
  titleEl.style.cssText = 'margin:0 0 12px 0; font-size:16px; color:#333;';
  
  const messageEl = document.createElement('p');
  messageEl.textContent = message;
  messageEl.style.cssText = 'margin:0 0 16px 0; font-size:14px; color:#666; line-height:1.5;';
  
  const button = document.createElement('button');
  button.textContent = 'OK';
  button.style.cssText = 'padding:8px 24px; background:#0dcaf0; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px;';
  button.onclick = () => overlay.remove();
  
  popup.appendChild(titleEl);
  popup.appendChild(messageEl);
  popup.appendChild(button);
  overlay.appendChild(popup);
  document.body.appendChild(overlay);
}

// Custom prompt/input dialog
function showPrompt(message, defaultValue = '', title = '✏️ Input', onConfirm, onCancel) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; z-index:3000;';
  
  const popup = document.createElement('div');
  popup.style.cssText = 'background:white; padding:24px; border-radius:8px; box-shadow:0 4px 16px rgba(0,0,0,0.3); max-width:400px;';
  
  const titleEl = document.createElement('h3');
  titleEl.textContent = title;
  titleEl.style.cssText = 'margin:0 0 12px 0; font-size:16px; color:#333;';
  
  const messageEl = document.createElement('p');
  messageEl.textContent = message;
  messageEl.style.cssText = 'margin:0 0 12px 0; font-size:14px; color:#666;';
  
  const input = document.createElement('input');
  input.type = 'text';
  input.value = defaultValue;
  input.style.cssText = 'width:100%; padding:8px 12px; border:1px solid #ddd; border-radius:4px; font-size:14px; box-sizing:border-box; margin-bottom:16px;';
  input.focus();
  input.select();
  
  const buttonContainer = document.createElement('div');
  buttonContainer.style.cssText = 'display:flex; gap:8px; justify-content:flex-end;';
  
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.style.cssText = 'padding:8px 16px; background:#6c757d; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px;';
  cancelBtn.onclick = () => {
    overlay.remove();
    if (onCancel) onCancel();
  };
  
  const confirmBtn = document.createElement('button');
  confirmBtn.textContent = 'OK';
  confirmBtn.style.cssText = 'padding:8px 24px; background:#0dcaf0; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px;';
  confirmBtn.onclick = () => {
    overlay.remove();
    if (onConfirm) onConfirm(input.value);
  };
  
  // Allow Enter to confirm
  input.onkeydown = (e) => {
    if (e.key === 'Enter') confirmBtn.click();
    if (e.key === 'Escape') cancelBtn.click();
  };
  
  buttonContainer.appendChild(cancelBtn);
  buttonContainer.appendChild(confirmBtn);
  
  popup.appendChild(titleEl);
  popup.appendChild(messageEl);
  popup.appendChild(input);
  popup.appendChild(buttonContainer);
  overlay.appendChild(popup);
  document.body.appendChild(overlay);
}

// Show SKU selection popup for new box (or when selecting existing box)
function showSKUSelectionPopup(boxIndex, onConfirm, isMandatory = false) {
  const canvas = document.getElementById('labelCanvas');
  
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; z-index:3001;';
  
  const popup = document.createElement('div');
  popup.style.cssText = 'background:white; padding:24px; border-radius:8px; box-shadow:0 4px 16px rgba(0,0,0,0.3); max-width:450px; width:90%;';
  
  // Determine if this is a new box (popup called during draw finalization)
  const isNewBox = !isMandatory;  // New boxes call with isMandatory=false, existing boxes with isMandatory=true
  
  const titleEl = document.createElement('h3');
  titleEl.textContent = isNewBox ? '📌 Create Label - Assign SKU' : '📌 Assign SKU to Box ' + boxIndex;
  titleEl.style.cssText = 'margin:0 0 12px 0; font-size:16px; color:#333; text-align:center;';
  
  const messageEl = document.createElement('p');
  if (isNewBox) {
    messageEl.textContent = '✅ Label created. Select SKU to confirm or Cancel to discard:';
  } else {
    messageEl.textContent = '⚠️ SKU selection is required:';
  }
  messageEl.style.cssText = 'margin:0 0 8px 0; font-size:13px; color:' + (isNewBox ? '#28a745' : '#d32f2f') + ';';
  
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'Type to search or select from list...';
  input.style.cssText = 'width:100%; padding:10px 12px; border:2px solid #0dcaf0; border-radius:4px; font-size:14px; box-sizing:border-box; margin-bottom:12px;';
  input.focus();
  
  const skuList = getSKUList();
  const suggestionContainer = document.createElement('div');
  suggestionContainer.style.cssText = 'max-height:200px; overflow-y:auto; margin-bottom:12px; border:1px solid #ddd; border-radius:4px; background:#f9f9f9;';
  
  const updateSuggestions = () => {
    const searchText = input.value.toUpperCase();
    const filtered = searchText ? skuList.filter(s => s.includes(searchText)) : skuList;
    
    suggestionContainer.innerHTML = filtered.length > 0 
      ? filtered.map((sku, idx) => `
          <div onclick="this.parentElement._selectSKU('${sku}')" style="padding:8px 12px; cursor:pointer; border-bottom:1px solid #eee; background:white; transition:background 0.2s;">
            <span style="font-weight:bold; color:#0dcaf0;">${sku}</span>
          </div>
        `).join('')
      : `<div style="padding:12px; text-align:center; color:#999; font-size:12px;">No SKUs found</div>`;
  };
  
  suggestionContainer._selectSKU = (sku) => {
    input.value = sku;
    confirmSelection();
  };
  
  const closePopup = () => {
    if (overlay && overlay.parentElement) {
      overlay.remove();
    }
    if (canvas) {
      canvas.style.pointerEvents = 'auto';  // Re-enable canvas
      canvas.style.cursor = 'default';
    }
  };
  
  const confirmSelection = () => {
    const selectedSKU = input.value.trim();
    
    // For new boxes: require SKU selection, show error if empty
    if (isNewBox && !selectedSKU) {
      showAlert('Please select an SKU to create the label', '❌ SKU Required');
      return;
    }
    
    // For existing boxes (mandatory click): require SKU selection
    if (isMandatory && !selectedSKU) {
      showAlert('Please select an SKU', '⚠️ SKU Required');
      return;
    }
    
    closePopup();
    
    // Reload list and save JSON before calling callback
    setTimeout(() => {
      renderBoxList();
      saveLabelsToJSON();
    }, 10);
    
    if (onConfirm) onConfirm(selectedSKU);
  };
  
  const cancelSelection = () => {
    // If this is a new box, discard it
    if (isNewBox) {
      labelBoxes.splice(boxIndex, 1);
      renderBoxList();
      labelDraw();
    }
    
    closePopup();
    if (onConfirm) onConfirm(null);
    
    // Show message only after popup is closed
    if (isNewBox) {
      showAlert('Label discarded', '🗑️ Cancel');
    }
  };
  
  input.addEventListener('input', updateSuggestions);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      confirmSelection();
    }
    // Allow ESC to close popup and discard box
    if (e.key === 'Escape') {
      cancelSelection();
    }
  });
  
  const buttonContainer = document.createElement('div');
  buttonContainer.style.cssText = 'display:flex; gap:8px; justify-content:flex-end;';
  
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = isNewBox ? '❌ Cancel (Discard)' : '❌ Cancel';
  cancelBtn.style.cssText = 'padding:8px 16px; background:#dc3545; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px; transition: background 0.2s;';
  cancelBtn.onmouseenter = () => { cancelBtn.style.background = '#c82333'; };
  cancelBtn.onmouseleave = () => { cancelBtn.style.background = '#dc3545'; };
  cancelBtn.onclick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    cancelSelection();
    return false;
  };
  buttonContainer.appendChild(cancelBtn);
  
  // Also close popup when clicking overlay background
  overlay.onclick = (e) => {
    if (e.target === overlay && !isMandatory) {
      cancelSelection();
    }
  };
  
  const confirmBtn = document.createElement('button');
  confirmBtn.textContent = 'Confirm (Enter)';
  confirmBtn.style.cssText = 'padding:8px 24px; background:#0dcaf0; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px; transition: background 0.2s;';
  confirmBtn.onmousedown = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  confirmBtn.onclick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    confirmSelection();
    return false;
  };
  
  buttonContainer.appendChild(confirmBtn);
  
  updateSuggestions();
  
  popup.appendChild(titleEl);
  popup.appendChild(messageEl);
  popup.appendChild(input);
  popup.appendChild(suggestionContainer);
  popup.appendChild(buttonContainer);
  overlay.appendChild(popup);
  document.body.appendChild(overlay);
  
  // Disable canvas interaction while popup is open
  if (canvas) {
    canvas.style.pointerEvents = 'none';
  }
}

// ============================================================
// MODEL BACKUP MANAGEMENT
// ============================================================

async function loadBackups() {
  try {
    const response = await fetch('/api/models/info');
    const data = await response.json();
    
    if (!data) {
      out('backupOut', 'No backups available');
      return;
    }
    
    let html = '<table style="width:100%; border-collapse:collapse; font-size:13px;">';
    html += '<thead><tr style="background:#667eea; color:white; font-weight:bold;">';
    html += '<th style="padding:10px; text-align:left; border:1px solid #ddd;">Filename</th>';
    html += '<th style="padding:10px; text-align:center; border:1px solid #ddd;">Size (MB)</th>';
    html += '<th style="padding:10px; text-align:left; border:1px solid #ddd;">Modified</th>';
    html += '<th style="padding:10px; text-align:center; border:1px solid #ddd;">Action</th>';
    html += '</tr></thead><tbody>';
    
    // Current model
    if (data.current) {
      const sizeMB = (data.current.size / 1024 / 1024).toFixed(2);
      const date = new Date(data.current.modified).toLocaleString();
      html += `<tr style="background:#e8f5e9; border:1px solid #ddd;">
        <td style="padding:10px; border:1px solid #ddd;"><strong>✓ ${data.current.name}</strong> (Current)</td>
        <td style="padding:10px; text-align:center; border:1px solid #ddd;">${sizeMB}</td>
        <td style="padding:10px; border:1px solid #ddd;">${date}</td>
        <td style="padding:10px; text-align:center; border:1px solid #ddd;">—</td>
      </tr>`;
    }
    
    // Backup models
    if (data.backups && data.backups.length > 0) {
      for (const backup of data.backups) {
        const sizeMB = (backup.size / 1024 / 1024).toFixed(2);
        const date = new Date(backup.modified).toLocaleString();
        html += `<tr style="border:1px solid #ddd;">
          <td style="padding:10px; border:1px solid #ddd;">${backup.name}</td>
          <td style="padding:10px; text-align:center; border:1px solid #ddd;">${sizeMB}</td>
          <td style="padding:10px; border:1px solid #ddd;">${date}</td>
          <td style="padding:10px; text-align:center; border:1px solid #ddd;">
            <button onclick="restoreBackup('${backup.name}')" style="padding:4px 8px; background:#2196F3; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px;">Restore</button>
            <button onclick="deleteBackup('${backup.name}')" style="padding:4px 8px; background:#f44336; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px; margin-left:4px;">Delete</button>
          </td>
        </tr>`;
      }
    } else if (!data.current) {
      html += '<tr><td colspan="4" style="padding:20px; text-align:center; color:#999;">No backups available</td></tr>';
    }
    
    html += '</tbody></table>';
    const table = document.getElementById('backupsTable');
    if (table) {
      table.innerHTML = html;
    }
  } catch (e) {
    out('backupOut', `Error loading backups: ${e.message}`);
  }
}

async function restoreBackup(backupName) {
  if (!confirm(`Restore backup: ${backupName}?\n\nThis will replace the current model.`)) {
    return;
  }
  
  try {
    const response = await fetch('/api/models/restore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backup_name: backupName })
    });
    const data = await response.json();
    
    if (data.success) {
      out('backupOut', `✅ ${data.message}`);
      setTimeout(() => loadBackups(), 500);
    } else {
      out('backupOut', `❌ Error: ${data.error || 'Unknown error'}`);
    }
  } catch (e) {
    out('backupOut', `Error: ${e.message}`);
  }
}

async function deleteBackup(backupName) {
  if (!confirm(`Delete backup: ${backupName}?\n\nThis cannot be undone.`)) {
    return;
  }
  
  try {
    const response = await fetch('/api/models/delete-backup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backup_name: backupName })
    });
    const data = await response.json();
    
    if (data.success) {
      out('backupOut', `✅ ${data.message}`);
      setTimeout(() => loadBackups(), 500);
    } else {
      out('backupOut', `❌ Error: ${data.error || 'Unknown error'}`);
    }
  } catch (e) {
    out('backupOut', `Error: ${e.message}`);
  }
}

// ============================================================
// DATASET VIEWER - SHOW DATASET WITH SKU FILTERING
// ============================================================

async function loadDatasetView() {
  try {
    out('datasetOut', '⏳ Loading SKU list from openclip_dataset...');
    
    // Fetch all SKUs
    const r = await fetch('/api/dataset/skus');
    const data = await r.json();
    
    if (!data.ok) {
      out('datasetOut', '❌ Error: ' + (data.error || 'Failed to load SKUs'));
      return;
    }
    
    const skus = data.skus || [];
    
    // Update dropdown with all SKUs
    const dropdown = document.getElementById('skuSearchFilter');
    if (dropdown) {
      dropdown.innerHTML = '<option value="">-- All SKUs --</option>' +
        skus.map(sku => `<option value="${sku.name}">${sku.name}</option>`).join('');
    }
    
    // Populate SKU table
    populateSkuTable(skus);
    
    // Update summary - check which IDs exist
    let totalImages = 0;
    skus.forEach(sku => totalImages += (sku.count || 0));
    
    // Try to update old summary elements (dataset tab)
    if (document.getElementById('summaryImages')) {
      document.getElementById('summaryImages').textContent = totalImages;
    }
    if (document.getElementById('summarySkus')) {
      document.getElementById('summarySkus').textContent = skus.length;
    }
    
    // Update SKU Manager stats (if in that tab)
    if (document.getElementById('skuTotalCount')) {
      document.getElementById('skuTotalCount').textContent = skus.length;
    }
    
    out('datasetOut', `✅ Loaded ${skus.length} SKU(s) with ${totalImages} total image(s)`);
    
  } catch (e) {
    out('datasetOut', '❌ Error: ' + e.message);
  }
}

function populateSkuTable(skus) {
  const tbody = document.getElementById('datasetSkuTableBody');
  if (!tbody) return;
  
  if (skus.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="padding:20px; text-align:center; color:#999;">No SKUs found in openclip_dataset/</td></tr>`;
    out('datasetOut', '📊 No SKUs found. Create folders in openclip_dataset/ to organize images.');
    return;
  }
  
  tbody.innerHTML = skus.map((sku, idx) => `
    <tr style="background:${idx % 2 === 0 ? '#fff' : '#f9f9f9'}; border-bottom:1px solid #ddd; cursor:pointer;" onclick="openSkuImagesModal('${sku.name}')">
      <td style="padding:10px; border:1px solid #ddd; font-weight:bold; color:#0dcaf0;">${sku.name}</td>
      <td style="padding:10px; border:1px solid #ddd; text-align:center; font-weight:bold;">${sku.count || 0}</td>
      <td style="padding:10px; border:1px solid #ddd; text-align:center;">-</td>
      <td style="padding:10px; border:1px solid #ddd; text-align:center;">
        <button onclick="openSkuImagesModal('${sku.name}'); event.stopPropagation();" style="padding:6px 12px; background:#0dcaf0; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px; font-weight:bold; margin-right:5px;">👁 View</button>
        <button onclick="deleteSkuFolder('${sku.name}'); event.stopPropagation();" style="padding:6px 12px; background:#dc3545; color:white; border:none; border-radius:3px; cursor:pointer; font-size:11px; font-weight:bold;">🗑 Delete</button>
      </td>
    </tr>
  `).join('');
}

async function filterSkuList() {
  try {
    const selectedSku = document.getElementById('skuSearchFilter').value;
    
    if (!selectedSku) {
      // Show all SKUs
      out('datasetOut', '⏳ Loading all SKUs...');
      await loadDatasetView();
      return;
    }
    
    out('datasetOut', `⏳ Searching for SKU: ${selectedSku}...`);
    
    // Fetch all SKUs
    const r = await fetch('/api/dataset/skus');
    const data = await r.json();
    
    if (!data.ok) {
      out('datasetOut', '❌ Error: ' + (data.error || 'Failed to load SKUs'));
      return;
    }
    
    // Filter SKUs by selected value
    const skus = data.skus || [];
    const filtered = skus.filter(sku => sku.name === selectedSku);
    
    if (filtered.length === 0) {
      document.getElementById('datasetSkuTableBody').innerHTML = 
        `<tr><td colspan="4" style="padding:20px; text-align:center; color:#999;">SKU not found: ${selectedSku}</td></tr>`;
      out('datasetOut', `❌ SKU not found: ${selectedSku}`);
      return;
    }
    
    populateSkuTable(filtered);
    out('datasetOut', `✅ Found SKU: ${selectedSku} (${filtered[0].count} images)`);
    
  } catch (e) {
    out('datasetOut', '❌ Error: ' + e.message);
  }
}

function resetSkuFilter() {
  document.getElementById('skuSearchFilter').value = '';
  out('datasetOut', '🔄 Reset filter');
  loadDatasetView();
}

async function openSkuImagesModal(skuName) {
  try {
    // Fetch images for this SKU
    const r = await fetch(`/api/dataset/sku/${encodeURIComponent(skuName)}/images`);
    const data = await r.json();
    
    if (!data.ok) {
      alert('❌ Error loading images: ' + (data.error || 'Unknown error'));
      return;
    }
    
    const images = data.images || [];
    
    // Store original images for search filtering
    window.currentSkuImages = images;
    window.currentSkuName = skuName;
    
    // Hide SKU list, show image grid
    document.getElementById('skuListView').style.display = 'none';
    document.getElementById('imageGridView').style.display = 'block';
    
    // Clear search input
    document.getElementById('imageSearchInput').value = '';
    
    // Update title
    document.getElementById('imageGridTitle').textContent = `📁 ${skuName} - ${images.length} Image(s)`;
    
    // Populate image grid
    populateImageGrid(images);
    
  } catch (e) {
    alert('❌ Error: ' + e.message);
  }
}

// Build and display file explorer tree
function buildFileExplorerTree(images) {
  const treeContainer = document.getElementById('fileExplorerTree');
  if (!treeContainer) return;
  
  if (images.length === 0) {
    treeContainer.innerHTML = '<div style="padding:10px; text-align:center; color:#999; font-size:11px;">No files</div>';
    document.getElementById('filesCount').textContent = '0';
    return;
  }
  
  // Group images by extension
  const byExt = {};
  images.forEach(img => {
    const ext = img.name.split('.').pop().toUpperCase();
    if (!byExt[ext]) byExt[ext] = [];
    byExt[ext].push(img);
  });
  
  let html = '';
  const extensions = Object.keys(byExt).sort();
  
  extensions.forEach(ext => {
    const files = byExt[ext];
    html += `<div style="margin-bottom:8px;">
      <div onclick="toggleFileGroup(this)" style="cursor:pointer; padding:6px; background:#e8f4f8; border-radius:4px; font-weight:bold; font-size:11px; display:flex; align-items:center; gap:6px; user-select:none;">
        <span style="font-size:14px;">▼</span>
        <span>${ext}</span>
        <span style="color:#999; font-size:10px;">(${files.length})</span>
      </div>
      <div style="padding-left:12px; margin-top:4px; display:block;">`;
    
    files.forEach((img, idx) => {
      html += `<div onclick="selectFileFromExplorer('${img.name}')" style="cursor:pointer; padding:4px 6px; margin:2px 0; border-radius:3px; font-size:11px; background:white; border:1px solid #ddd; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; transition:all 0.2s;" onmouseover="this.style.background='#0dcaf0'; this.style.color='white'; this.style.fontWeight='bold';" onmouseout="this.style.background='white'; this.style.color='#333'; this.style.fontWeight='normal';">
        📄 ${img.name}
      </div>`;
    });
    
    html += `</div></div>`;
  });
  
  treeContainer.innerHTML = html;
  document.getElementById('filesCount').textContent = images.length;
}

// Toggle file group visibility
function toggleFileGroup(element) {
  const group = element.nextElementSibling;
  if (group) {
    const isOpen = group.style.display !== 'none';
    group.style.display = isOpen ? 'none' : 'block';
    const arrow = element.querySelector('span:first-child');
    if (arrow) arrow.textContent = isOpen ? '▶' : '▼';
  }
}

// Select file from explorer and highlight in grid
function selectFileFromExplorer(imageName) {
  // Find the image in the grid and highlight it
  const imageCards = document.querySelectorAll('#imageGridContainer > div');
  imageCards.forEach(card => {
    const nameDiv = card.querySelector('div:nth-child(2)');
    if (nameDiv && nameDiv.textContent === imageName) {
      // Highlight this card
      card.style.border = '3px solid #0dcaf0';
      card.style.boxShadow = '0 0 12px rgba(13,202,240,0.6)';
      card.style.transform = 'scale(1.05)';
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      
      // Remove highlight from others
      imageCards.forEach(c => {
        if (c !== card) {
          c.style.border = '2px solid #ddd';
          c.style.boxShadow = 'none';
          c.style.transform = 'scale(1)';
        }
      });
    }
  });
}

// Preview image in full view (modal)
function previewFullImage(imagePath, imageName) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); display:flex; align-items:center; justify-content:center; z-index:3001; cursor:pointer;';
  overlay.onclick = () => overlay.remove();
  
  const container = document.createElement('div');
  container.style.cssText = 'max-width:90%; max-height:90%; display:flex; flex-direction:column; background:white; border-radius:8px; overflow:hidden; box-shadow:0 4px 30px rgba(0,0,0,0.3);';
  
  const header = document.createElement('div');
  header.style.cssText = 'padding:15px; background:#0dcaf0; color:white; font-weight:bold; display:flex; justify-content:space-between; align-items:center;';
  header.innerHTML = `<span>👁 Preview: ${imageName}</span><button onclick="this.parentElement.parentElement.parentElement.remove()" style="padding:4px 12px; background:white; color:#0dcaf0; border:none; border-radius:3px; cursor:pointer; font-weight:bold;">✕ Close</button>`;
  
  const img = document.createElement('img');
  img.src = imagePath;
  img.style.cssText = 'max-width:100%; max-height:calc(90vh - 60px); object-fit:contain; padding:10px; background:#f9f9f9;';
  
  container.appendChild(header);
  container.appendChild(img);
  overlay.appendChild(container);
  document.body.appendChild(overlay);
}

function populateImageGrid(images) {
  const gridContainer = document.getElementById('imageGridContainer');
  gridContainer.innerHTML = '';
  
  if (images.length === 0) {
    gridContainer.innerHTML = '<div style="grid-column:1/-1; padding:40px; text-align:center; color:#999;">No images found</div>';
    buildFileExplorerTree([]);
    return;
  }
  
  // Build file explorer tree
  buildFileExplorerTree(images);
  
  // Update total images count
  document.getElementById('totalImagesCount').textContent = images.length;
  
  images.forEach(img => {
    const imgCard = document.createElement('div');
    imgCard.style.cssText = 'border:2px solid #ddd; border-radius:8px; overflow:hidden; background:#f9f9f9; transition:all 0.3s; cursor:pointer; display:flex; flex-direction:column;';
    imgCard.onmouseover = () => {
      imgCard.style.boxShadow = '0 6px 16px rgba(13,202,240,0.4)';
      imgCard.style.transform = 'translateY(-4px)';
    };
    imgCard.onmouseout = () => {
      imgCard.style.boxShadow = 'none';
      imgCard.style.transform = 'translateY(0)';
    };
    
    const imgElement = document.createElement('img');
    // Use url field from API response
    imgElement.src = img.url || img.path;
    imgElement.style.cssText = 'width:100%; height:100px; object-fit:cover; cursor:pointer; background:#e0e0e0;';
    imgElement.onclick = () => previewFullImage(img.url || img.path, img.name);
    imgElement.onerror = () => {
      imgElement.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 150"%3E%3Crect fill="%23ddd" width="200" height="150"/%3E%3Ctext x="50%" y="50%" text-anchor="middle" dy=".3em" fill="%23999" font-size="14"%3E❌ Error%3C/text%3E%3C/svg%3E';
    };
    
    const nameDiv = document.createElement('div');
    nameDiv.style.cssText = 'padding:8px; font-size:11px; font-weight:bold; color:#333; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;';
    nameDiv.textContent = img.name;
    nameDiv.title = img.name;
    
    const buttonDiv = document.createElement('div');
    buttonDiv.style.cssText = 'padding:6px; display:flex; gap:4px; background:#f0f0f0;';
    
    const previewBtn = document.createElement('button');
    previewBtn.textContent = '👁';
    previewBtn.style.cssText = 'flex:1; padding:5px; background:#0dcaf0; color:white; border:none; border-radius:3px; cursor:pointer; font-size:10px; font-weight:bold;';
    previewBtn.title = 'Preview image';
    previewBtn.onclick = (e) => {
      e.stopPropagation();
      previewFullImage(img.url || img.path, img.name);
    };
    
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '🗑';
    deleteBtn.style.cssText = 'flex:1; padding:5px; background:#dc3545; color:white; border:none; border-radius:3px; cursor:pointer; font-size:10px; font-weight:bold;';
    deleteBtn.title = 'Delete image';
    deleteBtn.onclick = (e) => {
      e.stopPropagation();
      deleteImage(window.currentSkuName, img.name, () => openSkuImagesModal(window.currentSkuName));
    };
    
    buttonDiv.appendChild(previewBtn);
    buttonDiv.appendChild(deleteBtn);
    imgCard.appendChild(imgElement);
    imgCard.appendChild(nameDiv);
    imgCard.appendChild(buttonDiv);
    gridContainer.appendChild(imgCard);
  });
}

function searchImagesRealtime() {
  const searchTerm = document.getElementById('imageSearchInput').value.toLowerCase().trim();
  
  if (!window.currentSkuImages) return;
  
  // Filter images by search term
  const filtered = window.currentSkuImages.filter(img => 
    img.name.toLowerCase().includes(searchTerm)
  );
  
  // Update title with search results count
  const title = document.getElementById('imageGridTitle');
  if (searchTerm) {
    title.textContent = `📁 ${window.currentSkuName} - ${filtered.length} Match(es) Found`;
  } else {
    title.textContent = `📁 ${window.currentSkuName} - ${window.currentSkuImages.length} Image(s)`;
  }
  
  // Populate grid with filtered results
  populateImageGrid(filtered);
}

function clearImageSearch() {
  document.getElementById('imageSearchInput').value = '';
  searchImagesRealtime();
}

async function uploadImages() {
  const fileInput = document.getElementById('imageUploadInput');
  const files = fileInput.files;
  
  if (files.length === 0) {
    alert('⚠️ Please select images to upload');
    return;
  }
  
  if (!window.currentSkuName) {
    alert('⚠️ No SKU selected');
    return;
  }
  
  try {
    out('datasetOut', `⏳ Uploading ${files.length} image(s)...`);
    
    const formData = new FormData();
    for (let file of files) {
      formData.append('files', file);
    }
    
    // Show progress bar
    document.getElementById('uploadProgress').style.display = 'block';
    const progressFill = document.querySelector('#uploadProgressBar > div');
    
    const r = await fetch(`/api/dataset/sku/${encodeURIComponent(window.currentSkuName)}/upload`, {
      method: 'POST',
      body: formData
    });
    
    const data = await r.json();
    
    progressFill.style.width = '100%';
    
    if (data.ok) {
      out('datasetOut', `✅ Uploaded ${data.count} image(s)`);
      
      // Clear file input
      fileInput.value = '';
      
      // Refresh image grid
      setTimeout(() => {
        document.getElementById('uploadProgress').style.display = 'none';
        openSkuImagesModal(window.currentSkuName);
      }, 500);
    } else {
      out('datasetOut', '❌ Error: ' + (data.error || 'Upload failed'));
      document.getElementById('uploadProgress').style.display = 'none';
    }
  } catch (e) {
    out('datasetOut', '❌ Error: ' + e.message);
    document.getElementById('uploadProgress').style.display = 'none';
  }
}

async function deleteImage(skuName, imageName, callback) {
  if (!confirm(`Delete image: ${imageName}?`)) return;
  
  try {
    const r = await fetch(`/api/dataset/sku/${encodeURIComponent(skuName)}/image`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_name: imageName })
    });
    
    const data = await r.json();
    
    if (data.ok || r.ok) {
      out('datasetOut', `✅ Deleted: ${imageName}`);
      if (callback) {
        setTimeout(callback, 300);
      }
    } else {
      alert('❌ Error: ' + (data.error || 'Failed to delete'));
    }
  } catch (e) {
    alert('❌ Error: ' + e.message);
  }
}

async function backToSkuList() {
  // Hide image grid, show SKU list
  document.getElementById('imageGridView').style.display = 'none';
  document.getElementById('skuListView').style.display = 'block';
  out('datasetOut', '✅ Returned to SKU list');
}

async function deleteSkuFolder(skuName) {
  if (!confirm(`Delete entire SKU folder: ${skuName}? This will delete all images in it.`)) return;
  
  try {
    const r = await fetch(`/api/dataset/sku/${encodeURIComponent(skuName)}`, {
      method: 'DELETE'
    });
    
    const data = await r.json();
    
    if (data.ok) {
      out('datasetOut', `✅ Deleted SKU folder: ${skuName}`);
      loadDatasetView();
    } else {
      alert('❌ Error: ' + (data.error || 'Failed to delete'));
    }
  } catch (e) {
    alert('❌ Error: ' + e.message);
  }
}

async function previewFullImage(imagePath, imageName) {
  const modal = document.createElement('div');
  modal.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); display:flex; align-items:center; justify-content:center; z-index:6000;';
  
  const container = document.createElement('div');
  container.style.cssText = 'background:white; padding:20px; border-radius:12px; max-width:90vw; max-height:90vh; overflow:auto; position:relative; box-shadow:0 10px 40px rgba(0,0,0,0.3);';
  
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.style.cssText = 'position:absolute; top:10px; right:10px; background:#dc3545; color:white; border:none; border-radius:50%; width:40px; height:40px; font-size:24px; cursor:pointer; font-weight:bold;';
  closeBtn.onclick = () => modal.remove();
  
  const img = document.createElement('img');
  img.src = imagePath;
  img.style.cssText = 'max-width:100%; max-height:80vh; border:2px solid #ddd; border-radius:8px;';
  
  const title = document.createElement('h3');
  title.textContent = imageName;
  title.style.cssText = 'margin:0 0 10px 0; color:#333; text-align:center;';
  
  container.appendChild(closeBtn);
  container.appendChild(title);
  container.appendChild(img);
  modal.appendChild(container);
  document.body.appendChild(modal);
  
  modal.onclick = (e) => {
    if (e.target === modal) modal.remove();
  };
}

async function loadDatasetSummary() {
  try {
    const r = await fetch('/api/dataset/stats');
    const data = await r.json();
    
    if (data.error) {
      console.error('Error loading dataset summary:', data.error);
      return;
    }
    
    // Update summary stats (only if elements exist)
    const summaryImages = document.getElementById('summaryImages');
    const summarySkus = document.getElementById('summarySkus');
    if (summaryImages) summaryImages.textContent = data.images || 0;
    if (summarySkus) summarySkus.textContent = data.unique_skus || 0;
    
  } catch (e) {
    console.error('Error loading dataset summary:', e);
  }
}

function clearDatasetFilters() {
  loadDatasetView();
}

// Load dataset when switching to Dataset tab
function switchTab(tabName) {
  const sections = document.querySelectorAll('[id^="tab-"]');
  sections.forEach(s => s.style.display = 'none');
  
  const tab = document.getElementById('tab-' + tabName);
  if (tab) tab.style.display = 'block';
  
  // Load dataset view when switching to dataset tab
  if (tabName === 'dataset') {
    loadDatasetSummary();
    setTimeout(() => loadDatasetView(), 100);
  }
  
  // Load SKU Manager when switching to sku-manager tab
  if (tabName === 'sku-manager') {
    setTimeout(() => loadDatasetView(), 100);
  }
  
  // Load backups when switching to backups tab
  if (tabName === 'backups') {
    setTimeout(() => loadBackups(), 100);
  }
  
  // Load training status when switching to train tab
  if (tabName === 'train') {
    setTimeout(() => startMonitoringTrain(), 100);
  }
  
  // Load label list when switching to label tab
  if (tabName === 'label') {
    setTimeout(() => refreshLabelList(), 100);
  }
}

// ============================================================
// PRODUCT DETECTION & SKU MATCHING via /api/detect
// ============================================================

let lastDetectionResults = null;

async function scanAndDetect() {
  const fileInput = document.getElementById('detectImageFile');
  if (!fileInput.files || fileInput.files.length === 0) {
    alert('Please select an image file');
    return;
  }

  const file = fileInput.files[0];
  
  // Validate file type
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file');
    return;
  }

  // Show progress
  document.getElementById('detectionProgress').style.display = 'block';
  document.getElementById('detectionResultsModal').style.display = 'none';

  try {
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (response.ok && data.success) {
      lastDetectionResults = data;
      displayDetectionResults(data);
    } else {
      alert('Detection failed: ' + (data.error || data.message || 'Unknown error'));
    }
  } catch (error) {
    console.error('Detection error:', error);
    alert('Error: ' + error.message);
  } finally {
    document.getElementById('detectionProgress').style.display = 'none';
  }
}

function displayDetectionResults(data) {
  // Show results modal
  document.getElementById('detectionResultsModal').style.display = 'block';

  // Update stats
  document.getElementById('detectionCount').textContent = data.product_count || 0;
  
  const skuMatches = data.sku_matches || {};
  document.getElementById('uniqueSkuCount').textContent = Object.keys(skuMatches).length;

  // Show uploaded image
  if (data.image_url) {
    document.getElementById('detectionImagePreview').src = data.image_url;
  }

  // Populate detections table
  const tableBody = document.getElementById('detectionTableBody');
  tableBody.innerHTML = '';

  if (data.detections && data.detections.length > 0) {
    data.detections.forEach(det => {
      const row = document.createElement('tr');
      row.style.backgroundColor = (data.detections.indexOf(det) % 2 === 0) ? '#fff' : '#f9f9f9';
      
      const box = det.box || [0, 0, 0, 0];
      const boxStr = `(${box[0]}, ${box[1]}, ${box[2]}, ${box[3]})`;
      
      row.innerHTML = `
        <td style="padding:8px; border:1px solid #ddd;">${det.id}</td>
        <td style="padding:8px; border:1px solid #ddd; color:#667eea; font-weight:bold;">${(det.confidence * 100).toFixed(1)}%</td>
        <td style="padding:8px; border:1px solid #ddd; font-family:monospace; font-size:10px;">${boxStr}</td>
        <td style="padding:8px; border:1px solid #ddd; font-weight:bold;">${det.matched_sku || '(not matched)'}</td>
        <td style="padding:8px; border:1px solid #ddd; color:#764ba2; font-weight:bold;">${det.sku_similarity ? (det.sku_similarity * 100).toFixed(1) + '%' : '-'}</td>
      `;
      tableBody.appendChild(row);
    });
  } else {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="5" style="padding:10px; text-align:center; color:#999;">No products detected</td>';
    tableBody.appendChild(row);
  }

  // Show SKU matches
  const skuMatchesList = document.getElementById('skuMatchesList');
  skuMatchesList.innerHTML = '';

  if (Object.keys(skuMatches).length > 0) {
    Object.entries(skuMatches).forEach(([sku, similarity]) => {
      const matchDiv = document.createElement('div');
      matchDiv.style.cssText = 'background:#f0f0f0; padding:10px; border-radius:6px; display:flex; justify-content:space-between; align-items:center;';
      
      const percent = (similarity * 100).toFixed(1);
      matchDiv.innerHTML = `
        <span style="font-weight:bold; color:#333;">${sku}</span>
        <div style="background:#667eea; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; font-size:12px;">${percent}%</div>
      `;
      skuMatchesList.appendChild(matchDiv);
    });
  } else {
    skuMatchesList.innerHTML = '<div style="padding:10px; color:#999;">No SKU matches found</div>';
  }

  // Show crops if available
  if (data.crops_url) {
    document.getElementById('cropsSection').style.display = 'block';
    document.getElementById('detectionCropsPreview').src = data.crops_url;
  } else {
    document.getElementById('cropsSection').style.display = 'none';
  }
}

function closeDetectionResults() {
  document.getElementById('detectionResultsModal').style.display = 'none';
}

function openDetectionInNewTab() {
  if (!lastDetectionResults) {
    alert('No detection results available');
    return;
  }

  const newTab = window.open('', '_blank');
  newTab.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Detection Results</title>
      <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; margin-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-label { font-size: 12px; opacity: 0.9; }
        .stat-value { font-size: 28px; font-weight: bold; margin-top: 10px; }
        .section { margin-bottom: 30px; }
        .section h2 { border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 15px; }
        th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
        th { background: #f0f0f0; font-weight: bold; }
        img { max-width: 100%; height: auto; margin: 10px 0; border-radius: 6px; }
        .sku-match { background: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #667eea; border-radius: 4px; }
        pre { background: #f5f5f5; padding: 15px; border-radius: 6px; overflow-x: auto; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>🔍 Product Detection Results</h1>
        
        <div class="stats">
          <div class="stat-box">
            <div class="stat-label">Products Detected</div>
            <div class="stat-value">${lastDetectionResults.product_count || 0}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Unique SKUs</div>
            <div class="stat-value">${Object.keys(lastDetectionResults.sku_matches || {}).length}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Image Size</div>
            <div class="stat-value">${lastDetectionResults.image_size ? lastDetectionResults.image_size.join('×') : '-'}</div>
          </div>
        </div>

        <div class="section">
          <h2>📸 Uploaded Image</h2>
          <img src="${lastDetectionResults.image_url || ''}" />
        </div>

        ${lastDetectionResults.crops_url ? `
          <div class="section">
            <h2>🖼️ Detected Crops</h2>
            <img src="${lastDetectionResults.crops_url}" />
          </div>
        ` : ''}

        <div class="section">
          <h2>📊 Detected Products</h2>
          <table>
            <tr>
              <th>ID</th>
              <th>Confidence</th>
              <th>Bounding Box (x, y, w, h)</th>
              <th>Matched SKU</th>
              <th>SKU Similarity</th>
            </tr>
            ${(lastDetectionResults.detections || []).map(det => `
              <tr>
                <td>${det.id}</td>
                <td style="color: #667eea; font-weight: bold;">${(det.confidence * 100).toFixed(1)}%</td>
                <td style="font-family: monospace; font-size: 12px;">(${det.box ? det.box.map(v => v.toFixed(1)).join(', ') : '-'})</td>
                <td style="font-weight: bold;">${det.matched_sku || '(not matched)'}</td>
                <td style="color: #764ba2; font-weight: bold;">${det.sku_similarity ? (det.sku_similarity * 100).toFixed(1) + '%' : '-'}</td>
              </tr>
            `).join('')}
          </table>
        </div>

        ${Object.keys(lastDetectionResults.sku_matches || {}).length > 0 ? `
          <div class="section">
            <h2>🎯 SKU Matches Summary</h2>
            ${Object.entries(lastDetectionResults.sku_matches).map(([sku, similarity]) => `
              <div class="sku-match">
                <strong>${sku}</strong>: <span style="color: #667eea; font-weight: bold;">${(similarity * 100).toFixed(1)}%</span>
              </div>
            `).join('')}
          </div>
        ` : ''}

        <div class="section">
          <h2>📋 Raw JSON Response</h2>
          <pre>${JSON.stringify(lastDetectionResults, null, 2)}</pre>
        </div>

        <div style="text-align: center; margin-top: 30px; color: #999; font-size: 12px;">
          Generated: ${new Date().toLocaleString()}
        </div>
      </div>
    </body>
    </html>
  `);
  newTab.document.close();
}

function downloadDetectionJSON() {
  if (!lastDetectionResults) {
    alert('No detection results available');
    return;
  }

  const jsonStr = JSON.stringify(lastDetectionResults, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `detection_results_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  switchTab('label');
  // Pre-load dataset labels when page loads
  setTimeout(() => loadDatasetLabels(), 100);
  // Load backups when page loads
  setTimeout(() => loadBackups(), 200);
  // Initialize SKU management
  setTimeout(() => { 
    refreshSKUList(); 
    updateSKUDropdown(); 
  }, 500);
});

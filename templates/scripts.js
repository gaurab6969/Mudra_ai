document.getElementById("dropzone-file").addEventListener("change", function (event) {
  const file = event.target.files[0];
  const errorMsg = document.getElementById("error-msg");
  const filePreview = document.getElementById("file-preview");
  const uploadBtn = document.getElementById("upload-btn");

  if (file && file.size > 100000000) {
    errorMsg.textContent = "Maximum file upload size is 100 MB";
    errorMsg.classList.remove("hidden");
    filePreview.innerHTML = '';
    uploadBtn.disabled = true;
    return;
  }

  errorMsg.classList.add("hidden");

  const reader = new FileReader();
  reader.onload = function () {
    const fileUrl = reader.result;
    filePreview.innerHTML = `
      <div class="text-center">
        <img src="${fileUrl}" alt="Preview" class="w-48 h-48 object-contain mx-auto" />
        <button class="mt-4 text-red-600" onclick="removeFile()">Remove</button>
      </div>
    `;
  };
  reader.readAsDataURL(file);

  uploadBtn.disabled = false;
});

function removeFile() {
  document.getElementById("dropzone-file").value = "";
  document.getElementById("error-msg").classList.add("hidden");
  document.getElementById("file-preview").innerHTML = '';
  document.getElementById("upload-btn").disabled = true;
}

function uploadFile(file) {
  const progressBarInner = document.getElementById("progress-bar-inner");
  const progressText = document.getElementById("progress-text");
  const progressBar = document.getElementById("progress-bar");

  let progress = 0;
  progressBar.classList.remove("hidden");

  const interval = setInterval(function () {
    if (progress < 100) {
      progress += 10;
      progressBarInner.style.width = `${progress}%`;
      progressText.textContent = `${progress}%`;
    } else {
      clearInterval(interval);
      alert("Upload complete!");
    }
  }, 300);
}

document.getElementById("upload-btn").addEventListener("click", function () {
  const file = document.getElementById("dropzone-file").files[0];
  if (file) {
    uploadFile(file);
  }
});

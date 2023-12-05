function copyToClipboard() {
    var ansText = document.getElementById('ans-text').innerText;
    var tempTextarea = document.createElement('textarea');
  
    tempTextarea.value = ansText;
    document.body.appendChild(tempTextarea);
    tempTextarea.select();
    document.execCommand('copy');
    document.body.removeChild(tempTextarea);
}
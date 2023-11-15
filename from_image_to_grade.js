let grade = null; // 전역 변수 선언

// 비디오 및 캔버스 요소 가져오기
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// 웹캠 접근 함수
if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
      video.srcObject = stream;
    })
    .catch(function(error) {
      console.error("웹캠 접근 에러:", error);
    });
}

// 1분마다 사진 촬영하는 함수
setInterval(() => {
  // 비디오의 현재 이미지를 캔버스에 그림
  context.drawImage(video, 0, 0, 1920, 1080);
  // 캔버스의 내용을 이미지 데이터로 변환
  let imageData = canvas.toDataURL('image/png');

  // 이미지를 Blob으로 변환하는 함수
  async function convertToBlob(dataURL) {
    let response = await fetch(dataURL);
    let blob = await response.blob();
    return blob;
  }

  // 서버에 이미지 데이터를 전송하는 함수
  async function sendImageToServer(blob) {
    let formData = new FormData();
    formData.append("file", blob, "image.png");

    try {
      const response = await fetch('https://clean.hees.academy/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      console.log('성공:', data);
      grade = data.predicted_class; // 전역 변수에 서버로부터 반환된 값을 저장
    } catch (error) {
      console.error('에러:', error);
    }
  }

  // 이미지 데이터를 Blob으로 변환하고 서버에 전송
  convertToBlob(imageData)
    .then(blob => sendImageToServer(blob))
    .catch(error => console.error('Blob 변환 에러:', error));

}, 10000); // 60000ms = 1분



<script>
  import {onMount} from 'svelte';

  let grade = 4; // 반응형 변수

    onMount(async ()  => {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');

        if (navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({video: true});
                video.srcObject = stream;
            } catch (error) {
                console.error("웹캠 접근 에러:", error);
            }
        }

        setInterval(async () => {
            context.drawImage(video, 0, 0, 640, 480);
            let imageData = canvas.toDataURL('image/png');

            try {
                const blob = await (await fetch(imageData)).blob();
                const formData = new FormData();
                formData.append("file", blob, "image.png");

                const response = await fetch('https://clean.hees.academy/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                grade = data.predicted_class;
            } catch (error) {
                console.error('에러:', error);
            }
        }, 60000);
    });

    function gradeImage(grade) {
        switch (grade) {
            case "1":
                return "star.jpeg";
            case "2":
                return "heart.jpeg";
            case "3":
                return "forehead.jpeg";
            case "4":
                return "angry.jpeg";
            default:
                return "error.jpeg"; // 오류 이미지
        }
    }

    function gradeMessage(grade) {
        switch (grade) {
            case "1":
                return "눈이 부시게 빛나는 청결함! 청결의 별이 여기에 있네요. 이 환상적인 상태를 유지해주세요! ✨🏆";
            case "2":
                return "잘하고 있어요! 여기는 깔끔하고 상쾌해요. 조금만 더 노력하면 최고 등급도 가능하겠어요! 👍🌿";
            case "3":
                return "음, 여긴 조금 정리가 필요하네요. 괜찮아요, 작은 노력으로 큰 변화를 만들 수 있어요. 청소를 시작해볼까요? 🧽🛠️";
            case "4":
                return "이런, 이곳은 확실히 청소가 필요해 보여요! 깨끗한 공간을 위해 약간의 정리정돈이 필요할 때입니다. 🧹🗑️";
            default:
                return "청결도 감지에 실패했습니다.";
        }
    }
</script>

<header class="box">
    <img class="logo" src="LOGO.jpg">
</header>

<main>
    <video id="video" width="640" height="480" style="display:none;"></video>
    <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>

    {#if grade !== null}
        <section class="box">
            <span class="gr">Grade </span><span class="g">{grade}</span>
        </section>

        <section class="box">
            <img class="emoticon" src={gradeImage(grade)} alt="Emoticon">
        </section>

        <section class="box">
            <q class="box">{gradeMessage(grade)}</q>
        </section>
    {:else}
        <p>청결도 감지에 실패했습니다.</p>
    {/if}
</main>

<footer class="box">
    <r>하나고등학교 팀 움파룸파(강민주, 노현종, 유승주)</r>
</footer>

<style>
    @import url('https://fonts.googleapis.com/css2?family=Gasoek+One&family=Patua+One&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@600&display=swap');


    .box {
        text-align: center;
        width: 100%;
        margin: 3% auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }


    .logo {
        width: 70%;
        height: auto;
    }


    p {
        font-size: xx-large;
        font-family: 'Patua One', serif;
    }

    .emoticon {
        width: 50%;
        height: auto;
    }

    q {
        font-size: x-large;
        font-family: 'Noto Sans KR', sans-serif;
    }

    r {
        background-color: black;
        color: white;
        font-size: large;
        font-family: 'Noto Sans KR', sans-serif;
    }

    .g {
        font-family: 'Patua One', serif;
        font-size: xx-large;
        font-weight: bold;
        color: red
    }

    .gr {
        text-align: center;
        font-family: 'Patua One', serif;
        font-size: xx-large;
    }
</style>
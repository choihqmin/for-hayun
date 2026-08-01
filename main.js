import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

const asset = (filename) =>
  `${import.meta.env.BASE_URL}${filename}`;

document.body.style.setProperty(
  "--background-image",
  `url("${asset("background.jpeg")}")`,
);

const video = document.querySelector("#webcam");
const canvas = document.querySelector("#outputCanvas");
const context = canvas.getContext("2d");

const startButton =
  document.querySelector("#startButton");

const statusText =
  document.querySelector("#status");

const cameraContainer =
  document.querySelector("#cameraContainer");

const controls =
  document.querySelector(".controls");

const guide =
  document.querySelector("#guide");

const ending =
  document.querySelector("#ending");

let handLandmarker = null;
let webcamRunning = false;
let lastVideoTime = -1;
let endingStarted = false;

/*
 * 화면에서 왼쪽에 보이는 손:
 * i / l / o / v / e
 *
 * 화면에서 오른쪽에 보이는 손:
 * u / ha / yun / so / much
 *
 * 배열 순서:
 * 엄지, 검지, 중지, 약지, 새끼
 */
const WORDS = {
  left: ["i", "l", "o", "v", "e"],
  right: ["u", "ha", "yun", "so", "much"],
};

const FINGERTIP_INDEXES = [
  4,
  8,
  12,
  16,
  20,
];

/*
 * 손가락 접힘 각도를 계산할 관절 번호.
 *
 * 각 배열:
 * 시작 관절, 중간 관절, 손가락 끝
 */
const FINGER_ANGLE_POINTS = [
  [1, 2, 4],
  [5, 6, 8],
  [9, 10, 12],
  [13, 14, 16],
  [17, 18, 20],
];

/*
 * 각도가 이 값보다 작아지면
 * 손가락이 접혔다고 판단한다.
 */
const FOLD_ANGLE_THRESHOLD = 138;

/*
 * 접힘 상태가 몇 프레임 연속 유지돼야
 * 확실하게 접힌 것으로 인정할지 결정한다.
 */
const FOLD_CONFIRM_FRAMES = 5;

/*
 * 펴진 상태도 몇 프레임 연속 확인한 뒤
 * 다시 접을 수 있는 상태로 초기화한다.
 */
const OPEN_CONFIRM_FRAMES = 4;

/*
 * 10개 음성을 전부 경험한 뒤,
 * 열 손가락을 모두 접은 상태를
 * 이 시간만큼 유지해야 엔딩이 시작된다.
 */
const END_HOLD_TIME = 1200;

let allFoldedStartedAt = null;

/*
 * 손가락별 음성 파일.
 *
 * public 폴더 안에 다음 파일이 있어야 한다.
 * i.m4a, l.m4a, o.m4a, v.m4a, e.m4a
 * u.m4a, ha.m4a, yun.m4a, so.m4a, much.m4a
 */
const sounds = {
  i: new Audio(asset("i.m4a")),
  l: new Audio(asset("l.m4a")),
  o: new Audio(asset("o.m4a")),
  v: new Audio(asset("v.m4a")),
  e: new Audio(asset("e.m4a")),
  u: new Audio(asset("u.m4a")),
  ha: new Audio(asset("ha.m4a")),
  yun: new Audio(asset("yun.m4a")),
  so: new Audio(asset("so.m4a")),
  much: new Audio(asset("much.m4a")),
};

for (const sound of Object.values(sounds)) {
  sound.preload = "auto";
  sound.volume = 1;
}

/*
 * 이전에 확정된 손가락 접힘 상태.
 */
const previousFoldedStates = {
  left: [false, false, false, false, false],
  right: [false, false, false, false, false],
};

/*
 * 손가락별 접힘 감지 연속 프레임 수.
 */
const foldFrameCounts = {
  left: [0, 0, 0, 0, 0],
  right: [0, 0, 0, 0, 0],
};

/*
 * 손가락별 펴짐 감지 연속 프레임 수.
 */
const openFrameCounts = {
  left: [0, 0, 0, 0, 0],
  right: [0, 0, 0, 0, 0],
};

/*
 * 한 번 이상 음성을 들은 단어들을 기록한다.
 */
const experiencedWords = new Set();

/**
 * MediaPipe 손 인식 모델을 준비한다.
 */
async function initializeHandLandmarker() {
  try {
    statusText.textContent =
      "MediaPipe 파일을 불러오는 중...";

    const vision =
      await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/" +
          "@mediapipe/tasks-vision@latest/wasm",
      );

    statusText.textContent =
      "손 인식 모델을 불러오는 중...";

    handLandmarker =
      await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/" +
              "mediapipe-models/" +
              "hand_landmarker/" +
              "hand_landmarker/" +
              "float16/1/" +
              "hand_landmarker.task",

            delegate: "CPU",
          },

          runningMode: "VIDEO",
          numHands: 2,

          minHandDetectionConfidence: 0.45,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        },
      );

    statusText.textContent =
      "준비 완료 — 카메라 시작 버튼을 눌러 주세요.";

    startButton.disabled = false;
  } catch (error) {
    console.error(error);

    statusText.textContent =
      "손 인식 모델을 불러오지 못했습니다.";
  }
}

/**
 * 브라우저 음성 자동재생 제한을 해제한다.
 */
async function unlockAudio() {
  for (const sound of Object.values(sounds)) {
    try {
      sound.muted = true;

      await sound.play();

      sound.pause();
      sound.currentTime = 0;
      sound.muted = false;
    } catch (error) {
      sound.muted = false;

      console.log(
        "음성 파일 준비 메시지:",
        error,
      );
    }
  }
}

/**
 * 카메라를 실행한다.
 */
async function startWebcam() {
  if (!handLandmarker) {
    statusText.textContent =
      "아직 손 인식 모델이 준비되지 않았습니다.";

    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    statusText.textContent =
      "이 브라우저에서는 카메라를 사용할 수 없습니다.";

    return;
  }

  try {
    startButton.disabled = true;

    statusText.textContent =
      "카메라 권한을 요청하는 중...";

    await unlockAudio();

    const stream =
      await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",

          width: {
            ideal: 1280,
          },

          height: {
            ideal: 720,
          },
        },

        audio: false,
      });

    video.srcObject = stream;

    video.addEventListener(
      "loadeddata",
      () => {
        webcamRunning = true;

        startButton.style.display = "none";

        statusText.textContent =
          "두 손을 펼친 뒤 손가락을 하나씩 접어 보세요.";

        predictWebcam();
      },
      {
        once: true,
      },
    );
  } catch (error) {
    console.error(error);

    startButton.disabled = false;

    if (error.name === "NotAllowedError") {
      statusText.textContent =
        "카메라 권한이 거부되었습니다.";
    } else {
      statusText.textContent =
        "카메라를 실행하지 못했습니다.";
    }
  }
}

/**
 * 카메라의 각 프레임에서 손을 감지한다.
 */
function predictWebcam() {
  if (
    !webcamRunning ||
    !handLandmarker ||
    endingStarted
  ) {
    return;
  }

  if (
    canvas.width !== video.videoWidth ||
    canvas.height !== video.videoHeight
  ) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const results =
      handLandmarker.detectForVideo(
        video,
        performance.now(),
      );

    drawScene(results);
  }

  window.requestAnimationFrame(predictWebcam);
}

/**
 * 세 점 사이 각도를 계산한다.
 */
function calculateAngle(
  pointA,
  pointB,
  pointC,
) {
  const vectorBA = {
    x: pointA.x - pointB.x,
    y: pointA.y - pointB.y,
    z: pointA.z - pointB.z,
  };

  const vectorBC = {
    x: pointC.x - pointB.x,
    y: pointC.y - pointB.y,
    z: pointC.z - pointB.z,
  };

  const dotProduct =
    vectorBA.x * vectorBC.x +
    vectorBA.y * vectorBC.y +
    vectorBA.z * vectorBC.z;

  const magnitudeBA = Math.hypot(
    vectorBA.x,
    vectorBA.y,
    vectorBA.z,
  );

  const magnitudeBC = Math.hypot(
    vectorBC.x,
    vectorBC.y,
    vectorBC.z,
  );

  if (
    magnitudeBA === 0 ||
    magnitudeBC === 0
  ) {
    return 180;
  }

  const cosine = Math.min(
    1,
    Math.max(
      -1,
      dotProduct /
        (magnitudeBA * magnitudeBC),
    ),
  );

  return (
    Math.acos(cosine) *
    (180 / Math.PI)
  );
}

/**
 * 손가락 하나가 접혔는지 판단한다.
 */
function isFingerFolded(
  landmarks,
  fingerIndex,
) {
  const [
    startIndex,
    middleIndex,
    tipIndex,
  ] = FINGER_ANGLE_POINTS[fingerIndex];

  const angle = calculateAngle(
    landmarks[startIndex],
    landmarks[middleIndex],
    landmarks[tipIndex],
  );

  /*
   * 엄지는 일반 손가락보다 구조가 달라서
   * 조금 넓은 기준을 적용한다.
   */
  const threshold =
    fingerIndex === 0
      ? FOLD_ANGLE_THRESHOLD + 8
      : FOLD_ANGLE_THRESHOLD;

  return angle < threshold;
}

/**
 * 해당 단어 음성을 재생한다.
 */
function playWord(word) {
  const sound = sounds[word];

  if (!sound) {
    return;
  }

  /*
   * 다른 음성이 재생 중이어도
   * 새 손가락 음성이 바로 들리게 한다.
   */
  sound.pause();
  sound.currentTime = 0;

  sound.play().catch((error) => {
    console.error(
      `${word} 음성 재생 실패:`,
      error,
    );
  });
}

/**
 * 한 손의 다섯 손가락 상태를 업데이트한다.
 */
function updateFingerStates(
  landmarks,
  handSide,
  words,
) {
  const confirmedStates = [];

  for (
    let fingerIndex = 0;
    fingerIndex < 5;
    fingerIndex++
  ) {
    const rawFolded =
      isFingerFolded(
        landmarks,
        fingerIndex,
      );

    if (rawFolded) {
      foldFrameCounts[handSide][fingerIndex] += 1;
      openFrameCounts[handSide][fingerIndex] = 0;
    } else {
      openFrameCounts[handSide][fingerIndex] += 1;
      foldFrameCounts[handSide][fingerIndex] = 0;
    }

    const previouslyFolded =
      previousFoldedStates[handSide][fingerIndex];

    let confirmedFolded =
      previouslyFolded;

    /*
     * 연속 접힘 프레임이 충분할 때만
     * 접힌 것으로 확정한다.
     */
    if (
      !previouslyFolded &&
      foldFrameCounts[handSide][fingerIndex] >=
        FOLD_CONFIRM_FRAMES
    ) {
      confirmedFolded = true;

      const word = words[fingerIndex];

      playWord(word);
      experiencedWords.add(word);
    }

    /*
     * 연속 펴짐 프레임이 충분할 때만
     * 다시 펴진 상태로 확정한다.
     */
    if (
      previouslyFolded &&
      openFrameCounts[handSide][fingerIndex] >=
        OPEN_CONFIRM_FRAMES
    ) {
      confirmedFolded = false;
    }

    previousFoldedStates[handSide][fingerIndex] =
      confirmedFolded;

    confirmedStates.push(
      confirmedFolded,
    );
  }

  return confirmedStates;
}

/**
 * 10개 단어 음성을 모두 들었는지 확인한다.
 */
function hasExperiencedAllWords() {
  const allWords = [
    ...WORDS.left,
    ...WORDS.right,
  ];

  return allWords.every((word) =>
    experiencedWords.has(word),
  );
}

/**
 * 엔딩 조건을 확인한다.
 */
function checkEnding(
  leftFoldedStates,
  rightFoldedStates,
) {
  const bothHandsAvailable =
    Array.isArray(leftFoldedStates) &&
    Array.isArray(rightFoldedStates);

  if (!bothHandsAvailable) {
    allFoldedStartedAt = null;
    return;
  }

  const allTenFingersFolded = [
    ...leftFoldedStates,
    ...rightFoldedStates,
  ].every(Boolean);

  const allWordsExperienced =
    hasExperiencedAllWords();

  if (
    !allTenFingersFolded ||
    !allWordsExperienced
  ) {
    allFoldedStartedAt = null;
    return;
  }

  if (allFoldedStartedAt === null) {
    allFoldedStartedAt =
      performance.now();

    statusText.textContent =
      "그대로 잠시 유지해 주세요...";

    return;
  }

  const heldTime =
    performance.now() -
    allFoldedStartedAt;

  if (heldTime >= END_HOLD_TIME) {
    startEnding();
  }
}

/**
 * 엔딩을 시작한다.
 */
function startEnding() {
  if (endingStarted) {
    return;
  }

  endingStarted = true;
  webcamRunning = false;

  statusText.textContent = "";

  /*
   * 먼저 카메라와 안내 문구가
   * 자연스럽게 사라진다.
   */
  cameraContainer.classList.add(
    "is-hidden",
  );

  controls.classList.add(
    "is-hidden",
  );

  guide.classList.add(
    "is-hidden",
  );

  /*
   * 약간의 여백 뒤에 엔딩 문구가 나타난다.
   * CSS에서 처음 3초 동안 커지고,
   * 이후 화면을 유영한다.
   */
  window.setTimeout(() => {
    ending.classList.add(
      "is-visible",
    );

    ending.setAttribute(
      "aria-hidden",
      "false",
    );
  }, 900);

  /*
   * 카메라가 완전히 사라진 뒤
   * 카메라 스트림을 종료한다.
   */
  const stream = video.srcObject;

  if (stream instanceof MediaStream) {
    window.setTimeout(() => {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }, 2200);
  }
}

/**
 * 손가락 위에 단어를 표시한다.
 */
function drawFingerWords(
  landmarks,
  words,
) {
  for (
    let fingerIndex = 0;
    fingerIndex <
    FINGERTIP_INDEXES.length;
    fingerIndex++
  ) {
    const fingertip =
      landmarks[
        FINGERTIP_INDEXES[fingerIndex]
      ];

    const word =
      words[fingerIndex];

    const x =
      fingertip.x * canvas.width;

    const y =
      fingertip.y * canvas.height;

    let fontSize = Math.max(
      18,
      canvas.width * 0.028,
    );

    if (word.length >= 4) {
      fontSize *= 0.72;
    } else if (word.length >= 3) {
      fontSize *= 0.84;
    }

    context.save();

    context.translate(x, y);

    /*
     * 캔버스가 좌우 반전되어 있으므로
     * 글자만 다시 정상 방향으로 뒤집는다.
     */
    context.scale(-1, 1);

    context.font =
      `200 ${fontSize}px ` +
      `"Helvetica Neue", Arial, sans-serif`;

    context.textAlign = "center";
    context.textBaseline = "bottom";

    /*
     * 이미 들은 단어는 옅게 표시한다.
     */
    context.fillStyle =
      experiencedWords.has(word)
        ? "rgba(255, 154, 210, 0.38)"
        : "#ff9ad2";

    context.fillText(
      word,
      0,
      -28,
    );

    context.restore();
  }
}

/**
 * 손 관절과 글자를 화면에 그린다.
 */
function drawScene(results) {
  context.clearRect(
    0,
    0,
    canvas.width,
    canvas.height,
  );

  if (!results.landmarks?.length) {
    statusText.textContent =
      "두 손을 카메라에 보여 주세요.";

    allFoldedStartedAt = null;

    return;
  }

  /*
   * 화면상의 x좌표 기준으로 정렬한다.
   *
   * 첫 번째 손:
   * 화면 왼쪽
   *
   * 두 번째 손:
   * 화면 오른쪽
   */
  const sortedHands =
    [...results.landmarks].sort(
      (handA, handB) =>
        handA[0].x - handB[0].x,
    );

  const drawingUtils =
    new DrawingUtils(context);

  let leftFoldedStates = null;
  let rightFoldedStates = null;

  for (
    let handIndex = 0;
    handIndex < sortedHands.length;
    handIndex++
  ) {
    const landmarks =
      sortedHands[handIndex];

    const handSide =
      handIndex === 0
        ? "left"
        : "right";

    const words =
      WORDS[handSide];

    /*
     * 손 관절 연결선.
     * 최종 화면에서 필요 없으면
     * 이 블록과 아래 drawLandmarks 블록을
     * 나중에 삭제하면 된다.
     */
    drawingUtils.drawConnectors(
      landmarks,
      HandLandmarker.HAND_CONNECTIONS,
      {
        color:
          "rgba(255, 255, 255, 0.28)",

        lineWidth: 2,
      },
    );

    drawingUtils.drawLandmarks(
      landmarks,
      {
        color:
          "rgba(255, 154, 210, 0.55)",

        fillColor:
          "rgba(255, 255, 255, 0.75)",

        lineWidth: 1,
        radius: 3,
      },
    );

    const foldedStates =
      updateFingerStates(
        landmarks,
        handSide,
        words,
      );

    if (handSide === "left") {
      leftFoldedStates =
        foldedStates;
    } else {
      rightFoldedStates =
        foldedStates;
    }

    drawFingerWords(
      landmarks,
      words,
    );
  }

  /*
   * 두 손을 모두 보여 주는 동안
   * 진행 상태를 표시한다.
   */
  if (sortedHands.length === 2) {
    statusText.textContent =
      `${experiencedWords.size}/10 음성을 들었어요`;
  } else {
    statusText.textContent =
      "두 손을 모두 보여 주세요.";
  }

  checkEnding(
    leftFoldedStates,
    rightFoldedStates,
  );
}

startButton.addEventListener(
  "click",
  startWebcam,
);

initializeHandLandmarker();
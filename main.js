import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

/*
 * For Hayun V4
 *
 * 기능별 구역:
 * 1. 설정
 * 2. 화면 요소
 * 3. 음성 관리자
 * 4. LOVE 콤보
 * 5. 손가락 상태 감지
 * 6. 카메라/MediaPipe
 * 7. 화면 그리기
 * 8. 사용법 팝업
 * 9. 엔딩
 */

const asset = (filename) =>
  `${import.meta.env.BASE_URL}${filename}`;

const CONFIG = {
  foldAngleThreshold: 138,
  thumbThresholdOffset: 8,
  foldConfirmFrames: 5,
  openConfirmFrames: 4,
  endingHoldMs: 1200,

  loveCombo: {
    sequence: ["l", "o", "v", "e"],
    maxTotalMs: 800,
    soundName: "love",
  },

  mediaPipe: {
    wasmPath:
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
    modelPath:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  },
};

const WORDS = {
  left: ["i", "l", "o", "v", "e"],
  right: ["u", "ha", "yun", "so", "much"],
};

const SOUND_FILES = {
  i: "i.m4a",
  l: "l.m4a",
  o: "o.m4a",
  v: "v.m4a",
  e: "e.m4a",
  u: "u.m4a",
  ha: "ha.m4a",
  yun: "yun.m4a",
  so: "so.m4a",
  much: "much.m4a",
  love: "love.m4a",
};

const FINGERTIP_INDEXES = [4, 8, 12, 16, 20];

const FINGER_ANGLE_POINTS = [
  [1, 2, 4],
  [5, 6, 8],
  [9, 10, 12],
  [13, 14, 16],
  [17, 18, 20],
];

const elements = {
  video: document.querySelector("#webcam"),
  canvas: document.querySelector("#outputCanvas"),
  startButton: document.querySelector("#startButton"),
  status: document.querySelector("#status"),
  cameraContainer: document.querySelector("#cameraContainer"),
  controls: document.querySelector(".controls"),
  guide: document.querySelector("#guide"),
  ending: document.querySelector("#ending"),

  helpButton: document.querySelector("#helpButton"),
  helpModal: document.querySelector("#helpModal"),
  helpBackdrop: document.querySelector("#helpBackdrop"),
  helpCloseButton: document.querySelector("#helpCloseButton"),
  helpConfirmButton: document.querySelector("#helpConfirmButton"),
};

const context = elements.canvas.getContext("2d");

document.body.style.setProperty(
  "--background-image",
  `url("${asset("background.jpeg")}")`,
);

const appState = {
  handLandmarker: null,
  webcamRunning: false,
  lastVideoTime: -1,
  endingStarted: false,
  allFoldedStartedAt: null,
  audioUnlocked: false,
  animationFrameId: null,

  experiencedWords: new Set(),

  previousFoldedStates: {
    left: [false, false, false, false, false],
    right: [false, false, false, false, false],
  },

  foldFrameCounts: {
    left: [0, 0, 0, 0, 0],
    right: [0, 0, 0, 0, 0],
  },

  openFrameCounts: {
    left: [0, 0, 0, 0, 0],
    right: [0, 0, 0, 0, 0],
  },
};

/* ------------------------------------------------------------------ */
/* 음성 관리자                                                        */
/* ------------------------------------------------------------------ */

function createSoundLibrary() {
  const sounds = {};

  for (const [name, filename] of Object.entries(SOUND_FILES)) {
    const audio = new Audio(asset(filename));
    audio.preload = "auto";
    audio.volume = 1;
    audio.playsInline = true;
    sounds[name] = audio;
  }

  return sounds;
}

const sounds = createSoundLibrary();

function stopAllSounds(exceptName = null) {
  for (const [name, sound] of Object.entries(sounds)) {
    if (name === exceptName) {
      continue;
    }

    sound.pause();

    try {
      sound.currentTime = 0;
    } catch {
      // Safari가 아직 파일 메타데이터를 읽지 못한 경우 무시한다.
    }
  }
}

/*
 * iPhone Safari에서는 카메라 버튼 클릭 같은 사용자 동작 안에서
 * 실제 play()가 한 번 실행되어야 이후 음성이 안정적으로 재생된다.
 *
 * 여러 파일을 차례로 재생하지 않고, 각 파일을 매우 짧게 무음 재생하여
 * 웹을 열자마자 소리가 나는 문제를 막는다.
 */
async function unlockAudioFromUserGesture() {
  if (appState.audioUnlocked) {
    return true;
  }

  const unlockTargets = Object.values(sounds);

  try {
    await Promise.all(
      unlockTargets.map(async (sound) => {
        const originalMuted = sound.muted;
        const originalVolume = sound.volume;

        sound.muted = true;
        sound.volume = 0;

        const playPromise = sound.play();

        if (playPromise) {
          await playPromise;
        }

        sound.pause();

        try {
          sound.currentTime = 0;
        } catch {
          // 메타데이터 로드 전이면 currentTime 변경이 막힐 수 있다.
        }

        sound.muted = originalMuted;
        sound.volume = originalVolume;
      }),
    );

    appState.audioUnlocked = true;
    return true;
  } catch (error) {
    console.warn("음성 잠금 해제 실패:", error);
    return false;
  }
}

async function playSound(name, { stopOthers = true } = {}) {
  const sound = sounds[name];

  if (!sound || !appState.audioUnlocked) {
    return false;
  }

  if (stopOthers) {
    stopAllSounds(name);
  }

  sound.pause();

  try {
    sound.currentTime = 0;
  } catch {
    // Safari 메타데이터 로드 전 예외 방지
  }

  try {
    await sound.play();
    return true;
  } catch (error) {
    console.error(`${name} 음성 재생 실패:`, error);
    return false;
  }
}

/* ------------------------------------------------------------------ */
/* LOVE 콤보                                                          */
/* ------------------------------------------------------------------ */

function createSequenceCombo({
  sequence,
  maxTotalMs,
  onComplete,
}) {
  let nextIndex = 0;
  let startedAt = null;

  function reset() {
    nextIndex = 0;
    startedAt = null;
  }

  function start(timestamp) {
    nextIndex = 1;
    startedAt = timestamp;
  }

  function add(
    value,
    timestamp = performance.now(),
  ) {
    const firstValue = sequence[0];

    /*
     * 아직 LOVE 입력이 시작되지 않았다면
     * 첫 글자인 l만 시작으로 인정한다.
     */
    if (nextIndex === 0) {
      if (value === firstValue) {
        start(timestamp);
      }

      return false;
    }

    /*
     * l을 접은 뒤 0.8초가 넘었으면
     * 기존 입력을 취소한다.
     */
    if (
      startedAt === null ||
      timestamp - startedAt > maxTotalMs
    ) {
      reset();

      /*
       * 시간 초과 순간 들어온 글자가 l이라면
       * 그 l부터 새롭게 시작한다.
       */
      if (value === firstValue) {
        start(timestamp);
      }

      return false;
    }

    /*
     * 지금 기다리고 있는 글자와
     * 다른 글자가 들어오면 초기화한다.
     */
    if (value !== sequence[nextIndex]) {
      reset();

      /*
       * 틀린 입력이 l이라면
       * 새로운 LOVE 입력으로 바로 시작한다.
       */
      if (value === firstValue) {
        start(timestamp);
      }

      return false;
    }

    /*
     * 올바른 다음 글자가 들어왔으므로
     * 다음 순서로 이동한다.
     */
    nextIndex += 1;

    /*
     * l → o → v → e를 전부 완료했다.
     */
    if (nextIndex === sequence.length) {
      onComplete();
      reset();

      return true;
    }

    return false;
  }

  return {
    add,
    reset,
  };
}

const loveCombo = createSequenceCombo({
  sequence: CONFIG.loveCombo.sequence,
  maxTotalMs: CONFIG.loveCombo.maxTotalMs,
  onComplete: () => {
    playSound(CONFIG.loveCombo.soundName);
  },
});

/*
 * 단어가 확정되었을 때 호출되는 단일 진입점.
 * 앞으로 새로운 콤보를 추가할 때 이 함수만 연결하면 된다.
 */
function handleWordTriggered(word) {
  appState.experiencedWords.add(word);

  const comboCompleted = loveCombo.add(word);

  if (!comboCompleted) {
    playSound(word);
  }
}

/* ------------------------------------------------------------------ */
/* 손가락 상태 계산                                                   */
/* ------------------------------------------------------------------ */

function calculateAngle(pointA, pointB, pointC) {
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

  if (magnitudeBA === 0 || magnitudeBC === 0) {
    return 180;
  }

  const cosine = Math.min(
    1,
    Math.max(
      -1,
      dotProduct / (magnitudeBA * magnitudeBC),
    ),
  );

  return Math.acos(cosine) * (180 / Math.PI);
}

function isFingerFolded(landmarks, fingerIndex) {
  const [startIndex, middleIndex, tipIndex] =
    FINGER_ANGLE_POINTS[fingerIndex];

  const angle = calculateAngle(
    landmarks[startIndex],
    landmarks[middleIndex],
    landmarks[tipIndex],
  );

  const threshold =
    fingerIndex === 0
      ? CONFIG.foldAngleThreshold +
        CONFIG.thumbThresholdOffset
      : CONFIG.foldAngleThreshold;

  return angle < threshold;
}

function updateFingerStates(
  landmarks,
  handSide,
  words,
) {
  const confirmedStates = [];

  for (let fingerIndex = 0; fingerIndex < 5; fingerIndex++) {
    const rawFolded = isFingerFolded(
      landmarks,
      fingerIndex,
    );

    if (rawFolded) {
      appState.foldFrameCounts[handSide][fingerIndex] += 1;
      appState.openFrameCounts[handSide][fingerIndex] = 0;
    } else {
      appState.openFrameCounts[handSide][fingerIndex] += 1;
      appState.foldFrameCounts[handSide][fingerIndex] = 0;
    }

    const previouslyFolded =
      appState.previousFoldedStates[handSide][fingerIndex];

    let confirmedFolded = previouslyFolded;

    if (
      !previouslyFolded &&
      appState.foldFrameCounts[handSide][fingerIndex] >=
        CONFIG.foldConfirmFrames
    ) {
      confirmedFolded = true;
      handleWordTriggered(words[fingerIndex]);
    }

    if (
      previouslyFolded &&
      appState.openFrameCounts[handSide][fingerIndex] >=
        CONFIG.openConfirmFrames
    ) {
      confirmedFolded = false;
    }

    appState.previousFoldedStates[handSide][fingerIndex] =
      confirmedFolded;

    confirmedStates.push(confirmedFolded);
  }

  return confirmedStates;
}

/* ------------------------------------------------------------------ */
/* MediaPipe와 카메라                                                 */
/* ------------------------------------------------------------------ */

async function initializeHandLandmarker() {
  try {
    elements.status.textContent =
      "MediaPipe 파일을 불러오는 중...";

    const vision = await FilesetResolver.forVisionTasks(
      CONFIG.mediaPipe.wasmPath,
    );

    elements.status.textContent =
      "손 인식 모델을 불러오는 중...";

    appState.handLandmarker =
      await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: CONFIG.mediaPipe.modelPath,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });

    elements.status.textContent =
      "준비 완료 — 카메라 시작 버튼을 눌러 주세요.";

    elements.startButton.disabled = false;
  } catch (error) {
    console.error(error);

    elements.status.textContent =
      "손 인식 모델을 불러오지 못했습니다.";
  }
}

async function startWebcam() {
  if (!appState.handLandmarker) {
    elements.status.textContent =
      "아직 손 인식 모델이 준비되지 않았습니다.";
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    elements.status.textContent =
      "이 브라우저에서는 카메라를 사용할 수 없습니다.";
    return;
  }

  try {
    elements.startButton.disabled = true;
    elements.status.textContent =
      "음성과 카메라를 준비하는 중...";

    /*
     * 반드시 클릭 이벤트 흐름 안에서 먼저 음성을 해제한다.
     * iPhone Safari 대응의 핵심 부분이다.
     */
    const audioReady = await unlockAudioFromUserGesture();

    if (!audioReady) {
      elements.status.textContent =
        "음성을 준비하지 못했습니다. 버튼을 다시 눌러 주세요.";
      elements.startButton.disabled = false;
      return;
    }

    const stream =
      await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

    elements.video.srcObject = stream;

    await elements.video.play();

    appState.webcamRunning = true;
    elements.startButton.style.display = "none";

    elements.status.textContent =
      "두 손을 펼친 뒤 손가락을 하나씩 접어 보세요.";

    predictWebcam();
  } catch (error) {
    console.error(error);

    elements.startButton.disabled = false;

    if (error.name === "NotAllowedError") {
      elements.status.textContent =
        "카메라 권한이 거부되었습니다.";
    } else {
      elements.status.textContent =
        "카메라를 실행하지 못했습니다.";
    }
  }
}

function predictWebcam() {
  if (
    !appState.webcamRunning ||
    !appState.handLandmarker ||
    appState.endingStarted
  ) {
    return;
  }

  if (
    elements.canvas.width !== elements.video.videoWidth ||
    elements.canvas.height !== elements.video.videoHeight
  ) {
    elements.canvas.width = elements.video.videoWidth;
    elements.canvas.height = elements.video.videoHeight;
  }

  if (elements.video.currentTime !== appState.lastVideoTime) {
    appState.lastVideoTime = elements.video.currentTime;

    const results =
      appState.handLandmarker.detectForVideo(
        elements.video,
        performance.now(),
      );

    drawScene(results);
  }

  appState.animationFrameId =
    window.requestAnimationFrame(predictWebcam);
}

/* ------------------------------------------------------------------ */
/* 화면 그리기                                                        */
/* ------------------------------------------------------------------ */

function drawFingerWords(landmarks, words) {
  for (
    let fingerIndex = 0;
    fingerIndex < FINGERTIP_INDEXES.length;
    fingerIndex++
  ) {
    const fingertip =
      landmarks[FINGERTIP_INDEXES[fingerIndex]];

    const word = words[fingerIndex];
    const x = fingertip.x * elements.canvas.width;
    const y = fingertip.y * elements.canvas.height;

    let fontSize = Math.max(
      18,
      elements.canvas.width * 0.028,
    );

    if (word.length >= 4) {
      fontSize *= 0.72;
    } else if (word.length >= 3) {
      fontSize *= 0.84;
    }

    context.save();
    context.translate(x, y);
    context.scale(-1, 1);

    context.font =
      `200 ${fontSize}px "Helvetica Neue", Arial, sans-serif`;

    context.textAlign = "center";
    context.textBaseline = "bottom";

    context.fillStyle =
      appState.experiencedWords.has(word)
        ? "rgba(255, 154, 210, 0.38)"
        : "#ff9ad2";

    context.fillText(word, 0, -28);
    context.restore();
  }
}

function drawScene(results) {
  context.clearRect(
    0,
    0,
    elements.canvas.width,
    elements.canvas.height,
  );

  if (!results.landmarks?.length) {
    elements.status.textContent =
      "두 손을 카메라에 보여 주세요.";

    appState.allFoldedStartedAt = null;
    return;
  }

  const sortedHands = [...results.landmarks].sort(
    (handA, handB) => handA[0].x - handB[0].x,
  );

  const drawingUtils = new DrawingUtils(context);

  let leftFoldedStates = null;
  let rightFoldedStates = null;

  for (
    let handIndex = 0;
    handIndex < sortedHands.length;
    handIndex++
  ) {
    const landmarks = sortedHands[handIndex];

    const handSide =
      handIndex === 0 ? "left" : "right";

    const words = WORDS[handSide];

    drawingUtils.drawConnectors(
      landmarks,
      HandLandmarker.HAND_CONNECTIONS,
      {
        color: "rgba(255, 255, 255, 0.28)",
        lineWidth: 2,
      },
    );

    drawingUtils.drawLandmarks(landmarks, {
      color: "rgba(255, 154, 210, 0.55)",
      fillColor: "rgba(255, 255, 255, 0.75)",
      lineWidth: 1,
      radius: 3,
    });

    const foldedStates = updateFingerStates(
      landmarks,
      handSide,
      words,
    );

    if (handSide === "left") {
      leftFoldedStates = foldedStates;
    } else {
      rightFoldedStates = foldedStates;
    }

    drawFingerWords(landmarks, words);
  }

  if (sortedHands.length === 2) {
    elements.status.textContent =
      `${appState.experiencedWords.size}/10 음성을 들었어요`;
  } else {
    elements.status.textContent =
      "두 손을 모두 보여 주세요.";
  }

  checkEnding(leftFoldedStates, rightFoldedStates);
}

/* ------------------------------------------------------------------ */
/* 사용법 팝업                                                        */
/* ------------------------------------------------------------------ */

function openHelp() {
  elements.helpModal.classList.add("is-visible");
  elements.helpModal.setAttribute("aria-hidden", "false");
  elements.helpButton.setAttribute("aria-expanded", "true");
  elements.helpCloseButton.focus();
}

function closeHelp() {
  elements.helpModal.classList.remove("is-visible");
  elements.helpModal.setAttribute("aria-hidden", "true");
  elements.helpButton.setAttribute("aria-expanded", "false");
  elements.helpButton.focus();
}

function initializeHelpModal() {
  elements.helpButton.addEventListener("click", openHelp);
  elements.helpBackdrop.addEventListener("click", closeHelp);
  elements.helpCloseButton.addEventListener("click", closeHelp);
  elements.helpConfirmButton.addEventListener("click", closeHelp);

  document.addEventListener("keydown", (event) => {
    if (
      event.key === "Escape" &&
      elements.helpModal.classList.contains("is-visible")
    ) {
      closeHelp();
    }
  });
}

/* ------------------------------------------------------------------ */
/* 엔딩                                                               */
/* ------------------------------------------------------------------ */

function hasExperiencedAllWords() {
  const allWords = [
    ...WORDS.left,
    ...WORDS.right,
  ];

  return allWords.every((word) =>
    appState.experiencedWords.has(word),
  );
}

function checkEnding(
  leftFoldedStates,
  rightFoldedStates,
) {
  const bothHandsAvailable =
    Array.isArray(leftFoldedStates) &&
    Array.isArray(rightFoldedStates);

  if (!bothHandsAvailable) {
    appState.allFoldedStartedAt = null;
    return;
  }

  const allTenFingersFolded = [
    ...leftFoldedStates,
    ...rightFoldedStates,
  ].every(Boolean);

  if (
    !allTenFingersFolded ||
    !hasExperiencedAllWords()
  ) {
    appState.allFoldedStartedAt = null;
    return;
  }

  if (appState.allFoldedStartedAt === null) {
    appState.allFoldedStartedAt = performance.now();

    elements.status.textContent =
      "그대로 잠시 유지해 주세요...";

    return;
  }

  const heldTime =
    performance.now() -
    appState.allFoldedStartedAt;

  if (heldTime >= CONFIG.endingHoldMs) {
    startEnding();
  }
}

function startEnding() {
  if (appState.endingStarted) {
    return;
  }

  appState.endingStarted = true;
  appState.webcamRunning = false;

  if (appState.animationFrameId !== null) {
    cancelAnimationFrame(appState.animationFrameId);
  }

  stopAllSounds();
  loveCombo.reset();

  elements.status.textContent = "";

  elements.cameraContainer.classList.add("is-hidden");
  elements.controls.classList.add("is-hidden");
  elements.guide.classList.add("is-hidden");
  elements.helpButton.style.display = "none";
  closeHelpWithoutFocus();

  window.setTimeout(() => {
    elements.ending.classList.add("is-visible");
    elements.ending.setAttribute("aria-hidden", "false");
  }, 900);

  const stream = elements.video.srcObject;

  if (stream instanceof MediaStream) {
    window.setTimeout(() => {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }, 2200);
  }
}

function closeHelpWithoutFocus() {
  elements.helpModal.classList.remove("is-visible");
  elements.helpModal.setAttribute("aria-hidden", "true");
  elements.helpButton.setAttribute("aria-expanded", "false");
}

/* ------------------------------------------------------------------ */
/* 시작                                                               */
/* ------------------------------------------------------------------ */

elements.startButton.addEventListener(
  "click",
  startWebcam,
);

initializeHelpModal();
initializeHandLandmarker();

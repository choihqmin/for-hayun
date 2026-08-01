import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

const asset = (filename) =>
  `${import.meta.env.BASE_URL}${filename}`;

const CONFIG = {
  foldAngleThreshold: 138,
  thumbThresholdOffset: 8,
  foldConfirmFrames: 5,
  openConfirmFrames: 4,
  readyHoldMs: 700,
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

  cameraContainer:
    document.querySelector("#cameraContainer"),

  controls:
    document.querySelector(".controls"),

  guide:
    document.querySelector("#guide"),

  ending:
    document.querySelector("#ending"),

  helpButton:
    document.querySelector("#helpButton"),

  helpModal:
    document.querySelector("#helpModal"),

  helpBackdrop:
    document.querySelector("#helpBackdrop"),

  helpCloseButton:
    document.querySelector("#helpCloseButton"),

  helpConfirmButton:
    document.querySelector("#helpConfirmButton"),
};

const context =
  elements.canvas.getContext("2d");

document.body.style.setProperty(
  "--background-image",
  `url("${asset("background.jpeg")}")`,
);

function createFingerMatrix(value = false) {
  return {
    left: Array(5).fill(value),
    right: Array(5).fill(value),
  };
}

function createCounterMatrix() {
  return {
    left: Array(5).fill(0),
    right: Array(5).fill(0),
  };
}

const appState = {
  handLandmarker: null,
  webcamRunning: false,
  lastVideoTime: -1,
  animationFrameId: null,

  inputEnabled: false,
  readyStartedAt: null,

  endingStarted: false,
  allFoldedStartedAt: null,

  experiencedWords: new Set(),

  stableFoldedStates:
    createFingerMatrix(false),

  foldFrameCounts:
    createCounterMatrix(),

  openFrameCounts:
    createCounterMatrix(),
};

/* --------------------------------------------------------------- */
/* AudioManager                                                    */
/* --------------------------------------------------------------- */

class AudioManager {
  constructor(files) {
    this.files = files;

    this.context = null;

    this.buffers =
      new Map();

    this.loadingPromises =
      new Map();

    this.queue = [];

    this.currentSource = null;

    this.isPlaying = false;
    this.unlocked = false;
  }

  getContext() {
    if (this.context) {
      return this.context;
    }

    const AudioContextClass =
      window.AudioContext ||
      window.webkitAudioContext;

    if (!AudioContextClass) {
      return null;
    }

    this.context =
      new AudioContextClass();

    return this.context;
  }

  async unlock() {
    if (this.unlocked) {
      return true;
    }

    const audioContext =
      this.getContext();

    if (!audioContext) {
      return false;
    }

    try {
      if (
        audioContext.state ===
        "suspended"
      ) {
        await audioContext.resume();
      }

      this.unlocked =
        audioContext.state ===
        "running";

      if (this.unlocked) {
        /*
         * 음성은 재생하지 않고
         * 파일 데이터만 미리 불러온다.
         */
        void this.preloadAll();
      }

      return this.unlocked;
    } catch (error) {
      console.warn(
        "음성 시스템 활성화 실패:",
        error,
      );

      return false;
    }
  }

  async preloadAll() {
    await Promise.allSettled(
      Object.keys(this.files).map(
        (name) => this.load(name),
      ),
    );
  }

  async load(name) {
    if (this.buffers.has(name)) {
      return this.buffers.get(name);
    }

    if (
      this.loadingPromises.has(name)
    ) {
      return this.loadingPromises.get(name);
    }

    const filename =
      this.files[name];

    const audioContext =
      this.getContext();

    if (
      !filename ||
      !audioContext
    ) {
      return null;
    }

    const loadingPromise =
      fetch(asset(filename), {
        cache: "force-cache",
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(
              `${filename}: ${response.status}`,
            );
          }

          return response.arrayBuffer();
        })
        .then((arrayBuffer) =>
          audioContext.decodeAudioData(
            arrayBuffer,
          ),
        )
        .then((buffer) => {
          this.buffers.set(
            name,
            buffer,
          );

          this.loadingPromises.delete(
            name,
          );

          return buffer;
        })
        .catch((error) => {
          this.loadingPromises.delete(
            name,
          );

          console.error(
            `${name} 음성 준비 실패:`,
            error,
          );

          return null;
        });

    this.loadingPromises.set(
      name,
      loadingPromise,
    );

    return loadingPromise;
  }

  enqueue(name) {
    if (
      !this.unlocked ||
      !this.files[name]
    ) {
      return;
    }

    /*
     * 같은 음성이 대기열 끝에 이미 있으면
     * 중복 추가하지 않는다.
     */
    if (
      this.queue[
        this.queue.length - 1
      ] === name
    ) {
      return;
    }

    this.queue.push(name);

    void this.drainQueue();
  }

  playPriority(name) {
    if (
      !this.unlocked ||
      !this.files[name]
    ) {
      return;
    }

    this.clear({
      stopCurrent: true,
    });

    this.queue.push(name);

    void this.drainQueue();
  }

  clear({
    stopCurrent = false,
  } = {}) {
    this.queue = [];

    if (stopCurrent) {
      this.stopCurrent();
    }
  }

  stopCurrent() {
    if (!this.currentSource) {
      this.isPlaying = false;
      return;
    }

    const source =
      this.currentSource;

    this.currentSource = null;
    this.isPlaying = false;

    try {
      source.onended = null;
      source.stop();
    } catch {
      // 이미 종료된 경우
    }

    try {
      source.disconnect();
    } catch {
      // 이미 해제된 경우
    }
  }

  async drainQueue() {
    if (
      this.isPlaying ||
      this.queue.length === 0
    ) {
      return;
    }

    const name =
      this.queue.shift();

    const audioContext =
      this.getContext();

    if (!audioContext) {
      this.queue = [];
      return;
    }

    try {
      if (
        audioContext.state ===
        "suspended"
      ) {
        await audioContext.resume();
      }

      const buffer =
        await this.load(name);

      if (!buffer) {
        void this.drainQueue();
        return;
      }

      /*
       * 파일을 불러오는 동안 다른 재생이 시작됐다면
       * 현재 작업은 다시 대기시킨다.
       */
      if (this.isPlaying) {
        this.queue.unshift(name);
        return;
      }

      const source =
        audioContext.createBufferSource();

      source.buffer = buffer;

      source.connect(
        audioContext.destination,
      );

      this.currentSource = source;
      this.isPlaying = true;

      source.onended = () => {
        if (
          this.currentSource === source
        ) {
          this.currentSource = null;
          this.isPlaying = false;
        }

        try {
          source.disconnect();
        } catch {
          // 이미 해제된 경우
        }

        void this.drainQueue();
      };

      source.start(0);
    } catch (error) {
      this.currentSource = null;
      this.isPlaying = false;

      console.error(
        `${name} 음성 재생 실패:`,
        error,
      );

      void this.drainQueue();
    }
  }
}

const audioManager =
  new AudioManager(SOUND_FILES);

/* --------------------------------------------------------------- */
/* LOVE 콤보                                                       */
/* --------------------------------------------------------------- */

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
    const firstValue =
      sequence[0];

    if (nextIndex === 0) {
      if (value === firstValue) {
        start(timestamp);
      }

      return false;
    }

    if (
      startedAt === null ||
      timestamp - startedAt >
        maxTotalMs
    ) {
      reset();

      if (value === firstValue) {
        start(timestamp);
      }

      return false;
    }

    if (
      value !==
      sequence[nextIndex]
    ) {
      reset();

      if (value === firstValue) {
        start(timestamp);
      }

      return false;
    }

    nextIndex += 1;

    if (
      nextIndex ===
      sequence.length
    ) {
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

const loveCombo =
  createSequenceCombo({
    sequence:
      CONFIG.loveCombo.sequence,

    maxTotalMs:
      CONFIG.loveCombo.maxTotalMs,

    onComplete: () => {
      audioManager.playPriority(
        CONFIG.loveCombo.soundName,
      );
    },
  });

function handleWordTriggered(word) {
  if (!appState.inputEnabled) {
    return;
  }

  appState.experiencedWords.add(
    word,
  );

  const loveCompleted =
    loveCombo.add(word);

  if (!loveCompleted) {
    audioManager.enqueue(word);
  }
}

/* --------------------------------------------------------------- */
/* 손가락 상태                                                     */
/* --------------------------------------------------------------- */

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

  const magnitudeBA =
    Math.hypot(
      vectorBA.x,
      vectorBA.y,
      vectorBA.z,
    );

  const magnitudeBC =
    Math.hypot(
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
        (magnitudeBA *
          magnitudeBC),
    ),
  );

  return (
    Math.acos(cosine) *
    (180 / Math.PI)
  );
}

function isFingerFolded(
  landmarks,
  fingerIndex,
) {
  const [
    startIndex,
    middleIndex,
    tipIndex,
  ] =
    FINGER_ANGLE_POINTS[
      fingerIndex
    ];

  const angle =
    calculateAngle(
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

function getRawFoldedStates(
  landmarks,
) {
  return FINGER_ANGLE_POINTS.map(
    (_, fingerIndex) =>
      isFingerFolded(
        landmarks,
        fingerIndex,
      ),
  );
}

function resetFingerTracking() {
  appState.stableFoldedStates =
    createFingerMatrix(false);

  appState.foldFrameCounts =
    createCounterMatrix();

  appState.openFrameCounts =
    createCounterMatrix();
}

function updateFingerStates(
  rawStates,
  handSide,
  words,
) {
  const confirmedStates = [];

  for (
    let fingerIndex = 0;
    fingerIndex < 5;
    fingerIndex += 1
  ) {
    const rawFolded =
      rawStates[fingerIndex];

    if (rawFolded) {
      appState.foldFrameCounts[
        handSide
      ][fingerIndex] += 1;

      appState.openFrameCounts[
        handSide
      ][fingerIndex] = 0;
    } else {
      appState.openFrameCounts[
        handSide
      ][fingerIndex] += 1;

      appState.foldFrameCounts[
        handSide
      ][fingerIndex] = 0;
    }

    const previousState =
      appState.stableFoldedStates[
        handSide
      ][fingerIndex];

    let nextState =
      previousState;

    if (
      !previousState &&
      appState.foldFrameCounts[
        handSide
      ][fingerIndex] >=
        CONFIG.foldConfirmFrames
    ) {
      nextState = true;

      handleWordTriggered(
        words[fingerIndex],
      );
    }

    if (
      previousState &&
      appState.openFrameCounts[
        handSide
      ][fingerIndex] >=
        CONFIG.openConfirmFrames
    ) {
      nextState = false;
    }

    appState.stableFoldedStates[
      handSide
    ][fingerIndex] =
      nextState;

    confirmedStates.push(
      nextState,
    );
  }

  return confirmedStates;
}
/* --------------------------------------------------------------- */
/* MediaPipe / 카메라                                              */
/* --------------------------------------------------------------- */

async function initializeHandLandmarker() {
  try {
    elements.status.textContent =
      "MediaPipe 파일을 불러오는 중...";

    const vision =
      await FilesetResolver.forVisionTasks(
        CONFIG.mediaPipe.wasmPath,
      );

    elements.status.textContent =
      "손 인식 모델을 불러오는 중...";

    appState.handLandmarker =
      await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              CONFIG.mediaPipe.modelPath,

            delegate: "CPU",
          },

          runningMode: "VIDEO",
          numHands: 2,

          minHandDetectionConfidence:
            0.45,

          minHandPresenceConfidence:
            0.45,

          minTrackingConfidence:
            0.45,
        },
      );

    elements.status.textContent =
      "준비 완료 — 카메라 시작 버튼을 눌러 주세요.";

    elements.startButton.disabled =
      false;
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

  if (
    !navigator.mediaDevices
      ?.getUserMedia
  ) {
    elements.status.textContent =
      "이 브라우저에서는 카메라를 사용할 수 없습니다.";

    return;
  }

  try {
    elements.startButton.disabled =
      true;

    elements.status.textContent =
      "음성과 카메라를 준비하는 중...";

    /*
     * 소리를 재생하지 않고
     * AudioContext만 활성화한다.
     */
    const audioReady =
      await audioManager.unlock();

    if (!audioReady) {
      elements.status.textContent =
        "음성 시스템을 준비하지 못했습니다.";

      elements.startButton.disabled =
        false;

      return;
    }

    const stream =
      await navigator.mediaDevices
        .getUserMedia({
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

    elements.video.srcObject =
      stream;

    await elements.video.play();

    appState.webcamRunning = true;
    appState.inputEnabled = false;
    appState.readyStartedAt = null;
    appState.allFoldedStartedAt =
      null;

    resetFingerTracking();
    loveCombo.reset();

    audioManager.clear({
      stopCurrent: true,
    });

    elements.startButton.style.display =
      "none";

    elements.status.textContent =
      "두 손과 열 손가락을 모두 펼쳐 주세요.";

    predictWebcam();
  } catch (error) {
    console.error(error);

    elements.startButton.disabled =
      false;

    if (
      error.name ===
      "NotAllowedError"
    ) {
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
    elements.canvas.width !==
      elements.video.videoWidth ||
    elements.canvas.height !==
      elements.video.videoHeight
  ) {
    elements.canvas.width =
      elements.video.videoWidth;

    elements.canvas.height =
      elements.video.videoHeight;
  }

  if (
    elements.video.currentTime !==
    appState.lastVideoTime
  ) {
    appState.lastVideoTime =
      elements.video.currentTime;

    const results =
      appState.handLandmarker
        .detectForVideo(
          elements.video,
          performance.now(),
        );

    drawScene(results);
  }

  appState.animationFrameId =
    window.requestAnimationFrame(
      predictWebcam,
    );
}

/* --------------------------------------------------------------- */
/* 화면 표시                                                       */
/* --------------------------------------------------------------- */

function drawFingerWords(
  landmarks,
  words,
) {
  for (
    let fingerIndex = 0;
    fingerIndex <
    FINGERTIP_INDEXES.length;
    fingerIndex += 1
  ) {
    const fingertip =
      landmarks[
        FINGERTIP_INDEXES[
          fingerIndex
        ]
      ];

    const word =
      words[fingerIndex];

    const x =
      fingertip.x *
      elements.canvas.width;

    const y =
      fingertip.y *
      elements.canvas.height;

    let fontSize =
      Math.max(
        18,
        elements.canvas.width *
          0.028,
      );

    if (word.length >= 4) {
      fontSize *= 0.72;
    } else if (
      word.length >= 3
    ) {
      fontSize *= 0.84;
    }

    context.save();

    context.translate(x, y);
    context.scale(-1, 1);

    context.font =
      `200 ${fontSize}px "Helvetica Neue", Arial, sans-serif`;

    context.textAlign =
      "center";

    context.textBaseline =
      "bottom";

    context.fillStyle =
      appState.experiencedWords.has(
        word,
      )
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

function updateReadyState(
  leftRawStates,
  rightRawStates,
) {
  const allTenOpen = [
    ...leftRawStates,
    ...rightRawStates,
  ].every(
    (folded) => !folded,
  );

  if (!allTenOpen) {
    appState.readyStartedAt =
      null;

    elements.status.textContent =
      "두 손과 열 손가락을 모두 펼쳐 주세요.";

    return false;
  }

  if (
    appState.readyStartedAt ===
    null
  ) {
    appState.readyStartedAt =
      performance.now();
  }

  const heldMs =
    performance.now() -
    appState.readyStartedAt;

  elements.status.textContent =
    "두 손을 편 채 잠시 유지해 주세요.";

  if (
    heldMs <
    CONFIG.readyHoldMs
  ) {
    return false;
  }

  /*
   * 준비 과정에서 쌓인 카운트를
   * 모두 초기화한다.
   */
  resetFingerTracking();

  appState.inputEnabled = true;
  appState.readyStartedAt = null;

  elements.status.textContent =
    "준비 완료! 손가락을 하나씩 접어 보세요.";

  return true;
}

function drawScene(results) {
  context.clearRect(
    0,
    0,
    elements.canvas.width,
    elements.canvas.height,
  );

  if (
    !results.landmarks?.length
  ) {
    elements.status.textContent =
      appState.inputEnabled
        ? "두 손을 카메라에 보여 주세요."
        : "두 손과 열 손가락을 모두 펼쳐 주세요.";

    appState.readyStartedAt =
      null;

    appState.allFoldedStartedAt =
      null;

    return;
  }

  const sortedHands = [
    ...results.landmarks,
  ].sort(
    (handA, handB) =>
      handA[0].x -
      handB[0].x,
  );

  const drawingUtils =
    new DrawingUtils(context);

  const handData = [];

  for (
    let handIndex = 0;
    handIndex <
    sortedHands.length;
    handIndex += 1
  ) {
    const landmarks =
      sortedHands[handIndex];

    const handSide =
      handIndex === 0
        ? "left"
        : "right";

    const words =
      WORDS[handSide];

    const rawStates =
      getRawFoldedStates(
        landmarks,
      );

    drawingUtils.drawConnectors(
      landmarks,
      HandLandmarker.HAND_CONNECTIONS,
      {
        color:
          "rgba(255,255,255,0.28)",

        lineWidth: 2,
      },
    );

    drawingUtils.drawLandmarks(
      landmarks,
      {
        color:
          "rgba(255,154,210,0.55)",

        fillColor:
          "rgba(255,255,255,0.75)",

        lineWidth: 1,
        radius: 3,
      },
    );

    drawFingerWords(
      landmarks,
      words,
    );

    handData.push({
      handSide,
      rawStates,
    });
  }

  if (
    handData.length !== 2
  ) {
    appState.readyStartedAt =
      null;

    appState.allFoldedStartedAt =
      null;

    elements.status.textContent =
      appState.inputEnabled
        ? "두 손을 모두 보여 주세요."
        : "두 손과 열 손가락을 모두 펼쳐 주세요.";

    return;
  }

  const leftData =
    handData.find(
      (item) =>
        item.handSide === "left",
    );

  const rightData =
    handData.find(
      (item) =>
        item.handSide === "right",
    );

  if (
    !leftData ||
    !rightData
  ) {
    return;
  }

  if (!appState.inputEnabled) {
    updateReadyState(
      leftData.rawStates,
      rightData.rawStates,
    );

    return;
  }

  const leftStableStates =
    updateFingerStates(
      leftData.rawStates,
      "left",
      WORDS.left,
    );

  const rightStableStates =
    updateFingerStates(
      rightData.rawStates,
      "right",
      WORDS.right,
    );

  elements.status.textContent =
    `${appState.experiencedWords.size}/10 음성을 들었어요`;

  checkEnding(
    leftStableStates,
    rightStableStates,
  );
}

/* --------------------------------------------------------------- */
/* 사용법 팝업                                                     */
/* --------------------------------------------------------------- */

function openHelp() {
  elements.helpModal.classList.add(
    "is-visible",
  );

  elements.helpModal.setAttribute(
    "aria-hidden",
    "false",
  );

  elements.helpButton.setAttribute(
    "aria-expanded",
    "true",
  );

  elements.helpCloseButton.focus();
}

function closeHelp({
  restoreFocus = true,
} = {}) {
  elements.helpModal.classList.remove(
    "is-visible",
  );

  elements.helpModal.setAttribute(
    "aria-hidden",
    "true",
  );

  elements.helpButton.setAttribute(
    "aria-expanded",
    "false",
  );

  if (restoreFocus) {
    elements.helpButton.focus();
  }
}

function initializeHelpModal() {
  elements.helpButton.addEventListener(
    "click",
    openHelp,
  );

  elements.helpBackdrop.addEventListener(
    "click",
    () => closeHelp(),
  );

  elements.helpCloseButton.addEventListener(
    "click",
    () => closeHelp(),
  );

  elements.helpConfirmButton.addEventListener(
    "click",
    () => closeHelp(),
  );

  document.addEventListener(
    "keydown",
    (event) => {
      if (
        event.key === "Escape" &&
        elements.helpModal.classList
          .contains("is-visible")
      ) {
        closeHelp();
      }
    },
  );
}

/* --------------------------------------------------------------- */
/* 엔딩                                                            */
/* --------------------------------------------------------------- */

function hasExperiencedAllWords() {
  return [
    ...WORDS.left,
    ...WORDS.right,
  ].every(
    (word) =>
      appState.experiencedWords
        .has(word),
  );
}

function checkEnding(
  leftFoldedStates,
  rightFoldedStates,
) {
  const allTenFolded = [
    ...leftFoldedStates,
    ...rightFoldedStates,
  ].every(Boolean);

  if (
    !allTenFolded ||
    !hasExperiencedAllWords()
  ) {
    appState.allFoldedStartedAt =
      null;

    return;
  }

  if (
    appState.allFoldedStartedAt ===
    null
  ) {
    appState.allFoldedStartedAt =
      performance.now();

    elements.status.textContent =
      "그대로 잠시 유지해 주세요...";

    return;
  }

  const heldMs =
    performance.now() -
    appState.allFoldedStartedAt;

  if (
    heldMs >=
    CONFIG.endingHoldMs
  ) {
    startEnding();
  }
}

function startEnding() {
  if (appState.endingStarted) {
    return;
  }

  appState.endingStarted = true;
  appState.webcamRunning = false;

  if (
    appState.animationFrameId !==
    null
  ) {
    cancelAnimationFrame(
      appState.animationFrameId,
    );
  }

  audioManager.clear({
    stopCurrent: true,
  });

  loveCombo.reset();

  elements.status.textContent =
    "";

  elements.cameraContainer
    .classList.add("is-hidden");

  elements.controls
    .classList.add("is-hidden");

  elements.guide
    .classList.add("is-hidden");

  elements.helpButton.style.display =
    "none";

  closeHelp({
    restoreFocus: false,
  });

  window.setTimeout(() => {
    elements.ending.classList.add(
      "is-visible",
    );

    elements.ending.setAttribute(
      "aria-hidden",
      "false",
    );
  }, 900);

  const stream =
    elements.video.srcObject;

  if (
    stream instanceof MediaStream
  ) {
    window.setTimeout(() => {
      for (
        const track of
        stream.getTracks()
      ) {
        track.stop();
      }
    }, 2200);
  }
}

/* --------------------------------------------------------------- */
/* 시작                                                            */
/* --------------------------------------------------------------- */

elements.startButton.addEventListener(
  "click",
  startWebcam,
);

initializeHelpModal();
initializeHandLandmarker();
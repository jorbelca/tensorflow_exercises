const diceData = await fetch("./data.json").then((res) => res.json());
const numDice = 150;
const preSize = numDice * 10;

const cutData = async () => {
  const img = document.getElementById("image");
  const imgTensor = tf.browser.fromPixels(img, 1);
  const resized = tf.image.resizeNearestNeighbor(imgTensor, [preSize, preSize]);
  const cutSize = numDice;
  const heightCuts = tf.split(resized, cutSize);
  const grid = heightCuts.map((sliver) => tf.split(sliver, cutSize, 1));

  return grid;
};

const predictResults = (model, tGrid) => {
  return tf.tidy(() => {
    const patchStack = [];

    // Create array of properly sized tensors
    tGrid.forEach((row) => {
      patchStack.push(tf.image.resizeNearestNeighbor(tf.stack(row), [12, 12]));
    });

    // Let the Model find the answers
    const imageGroup = tf.concat(patchStack);
    console.log("Ready to predict shape ", imageGroup.shape);
    const answers = model.predict(imageGroup);
    console.log("Predicted shape ", answers.shape);
    return answers;
  });
};

const displayPredictions = async (answers) => {
  tf.tidy(() => {
    const diceArray = Array.isArray(diceData) ? diceData : diceData?.data;
    if (!Array.isArray(diceArray)) {
      throw new TypeError(
        "Invalid diceData format: expected an array or an object with a 'data' array"
      );
    }
    const diceTensors = diceArray.map((dt) => tf.tensor(dt));
    const { indices } = tf.topk(answers);
    const answerIndices = indices.dataSync();

    const tColumns = [];
    for (let y = 0; y < numDice; y++) {
      const tRow = [];
      for (let x = 0; x < numDice; x++) {
        const curIndex = y * numDice + x;
        tRow.push(diceTensors[answerIndices[curIndex]]);
      }
      const oneRow = tf.concat(tRow, 1);
      tColumns.push(oneRow);
    }
    const diceConstruct = tf.concat(tColumns);
    console.log(tColumns);
    // Print the reconstruction to the canvas
    const can = document.getElementById("display");
    tf.browser.toPixels(diceConstruct, can);
  });
};

const dicify = async () => {
  // Reemplaza la lectura de .path por el uso de FileList
  const input = document.getElementById("model_path");
  const files = input?.files;
  if (!files || files.length === 0) {
    alert("Selecciona el model.json y los .bin del modelo.");
    return;
  }
  const fileArr = Array.from(files);
  const hasJson = fileArr.some((f) => f.name.endsWith(".json"));
  if (!hasJson) {
    alert("Falta el archivo model.json. Selecciona model.json y los .bin.");
    return;
  }

  console.log(
    "Cargando modelo desde File(s):",
    fileArr.map((f) => f.name)
  );
  const dModel = await tf.loadLayersModel(tf.io.browserFiles(fileArr));

  const grid = await cutData();
  const predictions = await predictResults(dModel, grid);
  await displayPredictions(predictions);
  console.log("Done Dicifying");

  tf.dispose([dModel, predictions]);
  tf.dispose(grid);
};
export { dicify };

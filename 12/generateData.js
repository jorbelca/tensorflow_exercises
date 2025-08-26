const pixelShift = async (inputTensor, mutations = []) => {
  // Add 1px white padding to height and width
  const padded = inputTensor.pad(
    [
      [1, 1],
      [1, 1],
    ],
    1
  );
  const cutSize = inputTensor.shape;
  for (let h = 0; h < 3; h++) {
    for (let w = 0; w < 3; w++) {
      mutations.push(padded.slice([h, w], cutSize));
    }
  }
  padded.dispose();
  return mutations;
};

// Creates combinations take any two from array (like Py itertools.combinations)
const combos = async (tensorArray) => {
  const startSize = tensorArray.length;
  for (let i = 0; i < startSize - 1; i++) {
    for (let j = i + 1; j < startSize; j++) {
      const overlay = tf.tidy(() => {
        return tf.where(
          tf.less(tensorArray[i], tensorArray[j]),
          tensorArray[i],
          tensorArray[j]
        );
      });
      tensorArray.push(overlay);
    }
  }
};

// Remove duplicates and stack into a 4D tensor
const consolidate = async (tensorArray) => {
  const groupedData = tf.stack(tensorArray);
  // Needs to switch processing to CPU for `tf.unique` on Node
  // See: https://github.com/tensorflow/tfjs/issues/4595
  const { values, _indices } = tf.unique(groupedData);
  tf.dispose([groupedData, _indices]);
  tf.dispose(tensorArray);
  return values;
};

// Adds shades to dice depending on idx, slowly darkens
const gradiate = (tensorArray, idx) => {
  const shade = 1 / 9; // all possible possible
  const startShade = shade * idx;
  const endShade = shade * (idx + 1);
  const stepSpeed = 0.05;

  for (let x = startShade; x < endShade; x += stepSpeed) {
    const shadeDie = tf.fill([12, 12], 1 - x);
    tensorArray.push(shadeDie);
  }
};

const runAugmentation = async (aTensor, idx) => {
  const mutes = await pixelShift(aTensor);
  await combos(mutes);
  await combos(mutes);
  await gradiate(mutes, idx); // a little bonus for shades of gray
  return await consolidate(mutes);
};

const createDataObject = async () => {
  const response = await fetch("./data.json");
  const inDice = await response.json();
  const diceData = {};

  for (let idx = 0; idx < inDice.data.length; idx++) {
    const die = inDice.data[idx];
    const imgTensor = tf.tensor(die);
    const results = await runAugmentation(imgTensor, idx);
    console.log("Unique Results:", idx, results.shape);
    // Store results
    diceData[idx] = results.arraySync();
    // clean
    tf.dispose([results, imgTensor]);
  }

  const jsonString = JSON.stringify(diceData);
  const blob = new Blob([jsonString], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "dice_data.json";
  link.click();
};

export { createDataObject };

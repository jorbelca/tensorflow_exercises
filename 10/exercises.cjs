const tf = require("@tensorflow/tfjs-node");
const glob = require("glob");
const fs = require("fs");

const main = async () => {
  const [X, Y] = await folderToTensors("files");

  const model = getModel(X);

  await model.fit(X, Y, {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.1,
    suffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`
        );
      },
    },
    earlyStopping: tf.callbacks.earlyStopping({
      monitor: "val_loss",
      patience: 5,
      mode: "max",
    }),
  });

  //Save the model
  await model.save("file://model");

  //Cleaning up
  tf.dispose([X, Y, model]);
  console.log("Tensors in memory cleaned up.", tf.memory().numTensors);
};

const folderToTensors = async (filePath) => {
  const XS = [];
  const YS = [];

  let pattern = filePath;
  if (!filePath.includes("*") && !filePath.includes(".")) {
    // Si es un directorio, agregar patrón para imágenes
    pattern = `${filePath}/**/*.{jpg,jpeg,png,gif,bmp,webp}`;
  }

  const files = glob.sync(pattern, { nodir: true }); // nodir excluye directorios

  console.log(`Encontrados ${files.length} archivos:`, files.slice(0, 5)); // Mostrar primeros 5

  // Read the CSV file
  files.forEach((file) => {
    const imageData = fs.readFileSync(file);
    const answer = encodeDir(file);
    const imageTensor = tf.node.decodeImage(imageData, 1);
    // Store in memory
    YS.push(answer);
    XS.push(imageTensor);
  });

  // Verificar que tenemos datos antes de hacer stack
  if (XS.length === 0) {
    throw new Error("No se encontraron imágenes válidas para procesar");
  }

  const X = tf.stack(XS);
  const Y = tf.oneHot(YS, 10);

  console.log("Images all converted to tensors:");
  console.log("X", X.shape);
  console.log("Y", Y.shape);

  // Normalize X to values 0 - 1
  const XNORM = X.div(255);
  // cleanup - corregir la limpieza
  XS.forEach((tensor) => tensor.dispose()); // Limpiar tensores individuales
  tf.dispose([X]); // X ya no se necesita si devuelves XNORM

  return [XNORM, Y]; // Devolver la versión normalizada
};

const getModel = (X) => {
  const model = tf.sequential();

  // Conv + Pool combo
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      strides: 1,
      padding: "same",
      activation: "relu",
      kernelInitializer: "heNormal",
      inputShape: [28, 28, 1],
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Conv + Pool combo
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      strides: 1,
      padding: "same",
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  // Conv + Pool combo
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      strides: 1,
      padding: "same",
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  // Flatten for connecting to deep layers
  model.add(tf.layers.flatten());
  // One hidden deep layer
  model.add(
    tf.layers.dense({
      units: 128,
      activation: "tanh",
    })
  );
  // Output
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "softmax",
    })
  );
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};

function encodeDir(filePath) {
  if (filePath.includes("bird")) return 0;
  if (filePath.includes("lion")) return 1;
  if (filePath.includes("owl")) return 2;
  if (filePath.includes("parrot")) return 3;
  if (filePath.includes("raccoon")) return 4;
  if (filePath.includes("skull")) return 5;
  if (filePath.includes("snail")) return 6;
  if (filePath.includes("snake")) return 7;
  if (filePath.includes("squirrel")) return 8;
  if (filePath.includes("tiger")) return 9;
  // Should never get here
  console.error("Unrecognized folder");
  process.exit(1);
}
main().catch(console.error);

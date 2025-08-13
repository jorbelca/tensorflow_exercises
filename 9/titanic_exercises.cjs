const dfd = require("danfojs-node");
const tf = require("@tensorflow/tfjs-node");

async function main() {
  const df = await dfd.readCSV("titanic_data/train.csv");
  const dft = await dfd.readCSV("titanic_data/test.csv");
  // df.head().print();

  // const empty_spots = df.isNa().sum();
  // empty_spots.print();
  // // Find the average
  // const empty_rate = empty_spots.div(df.isNa().count());
  // empty_rate.print();

  // // Verificar valores nulos
  // console.log("Conteo de valores nulos por columna:");
  // df.isNa().sum().print(); // Cambiado de isNull() a isNa()

  // Rellenar valores nulos con 0 antes de llamar a describe()
  df.fillNa(0);
  dft.fillNa(0);

  console.log("Train Size", df.columns.length);
  console.log("Test Size", dft.columns.length);
  const mega = dfd.concat({ dfList: [df, dft], axis: 0 });
  mega.dropNa();
  //Exporting the concatenated DataFrame to CSV
  //await dfd.toCSV(mega, { filePath: "titanic_data/mega.csv" });

  const clean = mega.drop({
    columns: ["Name", "PassengerId", "Ticket", "Cabin"],
  });
  //clean.head().print();

  //Enondering categorical data
  let onlyFull = clean.dropNa();
  console.log(`After mega-clean the row-count is now ${onlyFull.shape[0]}`);

  const encode = new dfd.LabelEncoder();
  encode.fit(onlyFull["Embarked"]);
  onlyFull["Embarked"] = encode.transform(onlyFull["Embarked"].values);
  encode.fit(onlyFull["Sex"]);
  onlyFull["Sex"] = encode.transform(onlyFull["Sex"].values);

  onlyFull = onlyFull.resetIndex({ inplace: false });

  // Añadir una columna con el índice original para hacer el split de forma segura
  const rid = Array.from({ length: onlyFull.shape[0] }, (_, i) => i);
  onlyFull.addColumn("_rid", rid, { inplace: true });

  //onlyFull.head().print();

  if (onlyFull.shape[0] >= 800) {
    const newTrain = await onlyFull.sample(800);
    console.log(`newTrain row count: ${newTrain.shape[0]}`);

    // Obtener los identificadores de las filas seleccionadas
    const trainIds = newTrain["_rid"].values;
    const trainSet = new Set(trainIds);

    // Calcular el complemento (filas para test) usando los _rid
    const allIds = onlyFull["_rid"].values;
    const testIdx = allIds.filter((id) => !trainSet.has(id));
    const newTest = onlyFull.iloc({ rows: testIdx });

    // Eliminar la columna auxiliar antes de exportar
    const newTrainOut = newTrain.drop({ columns: ["_rid"] });
    const newTestOut = newTest.drop({ columns: ["_rid"] });

    console.log(`newTest row count: ${newTestOut.shape[0]}`);
    await dfd.toCSV(newTrainOut, {
      filePath: "titanic_data/cleaned/newTrain.csv",
    });
    await dfd.toCSV(newTestOut, {
      filePath: "titanic_data/cleaned/newTest.csv",
    });
    console.log("Files written!");
  } else {
    console.error("No hay suficientes filas para el muestreo.");
  }
}
//main().catch(console.error);

async function mainTrain() {
  // Get cleaned data
  const df = await dfd.readCSV("titanic_data/cleaned/newTrain.csv");
  console.log("Train Size", df.shape[0]);
  const dft = await dfd.readCSV("titanic_data/cleaned/newTest.csv");
  console.log("Test Size", dft.shape[0]);
  // Split train into X/Y
  const trainX = df.iloc({ columns: [`1:`] }).tensor;
  const trainY = df["Survived"].tensor;
  // Split test into X/Y
  const testX = dft.iloc({ columns: [`1:`] }).tensor;
  const testY = dft["Survived"].tensor;

  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [trainX.shape[1]],
      units: 120,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
    })
  );
  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainX, trainY, {
    epochs: 100,
    batchSize: 32,
    validationData: [testX, testY],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`
        );
      },
    },
  });
}

mainTrain().catch(console.error);

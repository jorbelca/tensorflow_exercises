const dfd = require("danfojs-node");

async function main() {
  const df = await dfd.readCSV("titanic_data/train.csv");
  df.head().print();

  const empty_spots = df.isNa().sum();
  empty_spots.print();
  // Find the average
  const empty_rate = empty_spots.div(df.isNa().count());
  empty_rate.print();

  // Verificar valores nulos
  console.log("Conteo de valores nulos por columna:");
  df.isNa().sum().print(); // Cambiado de isNull() a isNa()

  // Rellenar valores nulos con 0 antes de llamar a describe()
  df.fillNa(0).describe().print();
}
main().catch(console.error);

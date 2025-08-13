var df;
load_csv(
  "https://raw.githubusercontent.com/GantMan/learn-tfjs/master/chapter9/extra/cleaned/newTrain.csv"
).then((d) => {
  df = d;
  console.log("df cargado");
});

var dft;
load_csv(
  "https://raw.githubusercontent.com/GantMan/learn-tfjs/refs/heads/master/chapter9/extra/cleaned/newTest.csv"
).then((d) => {
  dft = d;
  console.log("dft cargado");
});

var mega_df = dfd.concat({ df_list: [df, dft], axis: 0 });

table(mega_df.head());

if (df && dft) {
  const mega_df = dfd.concat({ df_list: [df, dft], axis: 0 }); // dfList, no df_list
  console.log("Concatenación exitosa");
  console.log("Forma del DataFrame:", mega_df.shape);
  mega_df.head(); // Esto debería mostrar la tabla
  table(mega_df.head());
} else {
  console.log(
    "Los DataFrames aún no están cargados. Espera y vuelve a ejecutar."
  );
}

grp = mega_df.groupby(["Sex"]);
table(grp.col(["Survived"]).mean());

survivors = mega_df.query({ column: "Survived", is: "==", to: 1 });
table(survivors);
viz(`agehist`, (x) => survivors["Age"].plot(x).hist());

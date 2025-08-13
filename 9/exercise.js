// use Danfo.js to identify the honorifics used on the Titanic
// and their associated survival rates. This is an excellent opportunity for you to get
// comfortable with Dnotebooks.

var df;
load_csv(
  "https://raw.githubusercontent.com/GantMan/learn-tfjs/refs/heads/master/chapter9/extra/titanic%20data/train.csv"
).then((d) => {
  df = d;
  console.log("df cargado");
});

var dft;
load_csv(
  "https://raw.githubusercontent.com/GantMan/learn-tfjs/refs/heads/master/chapter9/extra/titanic%20data/test.csv"
).then((d) => {
  dft = d;
  console.log("dft cargado");
});

var mega_df = dfd.concat({ df_list: [df, dft], axis: 0 });

function nameToNum(x) {
  if (x.includes("Mr")) {
    return 1;
  } else if (x.includes("Mrs")) {
    return 2;
  } else if (x.includes("Ms")) {
    return 3;
  } else if (x.includes("Rev")) {
    return 4; // Cambiado de 3 a 4 para ser único
  } else {
    return 0;
  }
}

const nameBuckets = mega_df["Name"].apply(nameToNum);
// Sintaxis corregida para addColumn en Danfo.js
mega_df.addColumn({ column: "Honorifics", value: nameBuckets });

hon = mega_df.groupby(["Honorifics"]);
table(hon.col(["Honorifics"]).count());

// Honorifics	Honorifics_count
// 0	0	340
// 1	1	959
// 2	3	2
// 3	4	8

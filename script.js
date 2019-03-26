console.log('Hello TensorFlow');

async function getData() {
  const houseDataReq = await fetch('https://raw.githubusercontent.com/meetnandu05/ml1/master/house.json');  
  const houseData = await houseDataReq.json();  
  const cleaned = houseData.map(house => ({
    price: house.Price,
    rooms: house.AvgAreaNumberofRooms,
  }))
  .filter(house => (house.price != null && house.rooms != null));
  
  return cleaned;
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.rooms,
    y: d.price,
  }));

  tfvis.render.scatterplot(
    {name: 'No.of rooms v Price'},
    {values}, 
    {
      xLabel: 'No. of rooms',
      yLabel: 'Price',
      height: 300
    }
  );

  // Create the model
const model = createModel();  
tfvis.show.modelSummary({name: 'Model Summary'}, model);

const tensorData = convertToTensor(data);
const {inputs, labels} = tensorData;
    
// Train the model  
await trainModel(model, inputs, labels);
console.log('Done Training');
  
testModel(model, data, tensorData);
  // More code will be added below
}

document.addEventListener('DOMContentLoaded', run);

function createModel() {
  // Create a sequential model
  const model = tf.sequential(); 
  
  // Add a single hidden layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  
  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

function convertToTensor(data) {
  
  return tf.tidy(() => {
    // Step 1. Shuffle the data    
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.rooms)
    const labels = data.map(d => d.price);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });  
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });
  
  const batchSize = 28;
  const epochs = 50;
  
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });

}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
  
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    
    const xs = tf.linspace(0, 1, 100);      
    const preds = model.predict(xs.reshape([100, 1]));      
    
    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    
    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });
  
 
  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });
  
  const originalPoints = inputData.map(d => ({
    x: d.rooms, y: d.price,
  }));
  
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'No. of rooms',
      yLabel: 'Price',
      height: 300
    }
  );
}
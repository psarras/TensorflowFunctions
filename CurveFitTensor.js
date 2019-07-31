let dataSetIn = [];
let dataSetOut = [];

const TestDataIn = [];
let TestDataOut = [];

let mouseDataIn = [];
let mouseDataOut = [];

const LEARNING_RATE = 0.5;
let model;

let functionSlider;
let errorTextbox;
let epochTextbox;
let startButton;
let stopButton;
let functionTextbox;

function setup() {
    let canvas = createCanvas(400, 400);

    for (let i = 0; i < 100; i++) {
        TestDataIn.push(map(i, 0, 100, 0, 1));
    }

    canvas.parent('canvascontainer');
    functionSlider = select('#functionSlider')
    errorTextbox = select('#error');
    epochsTextbox = select('#Epochs');
    functionTextbox = select('#FunctionSelection');
    startButton = select('#Start');
    stopButton = select('#Stop');

    startButton.mousePressed(Run_Optimization);
    stopButton.mousePressed(Stop_Optimization);

    let cycles = functionSlider.value();
}

function draw() {
    background(220);
    stroke(0);
    strokeWeight(3);
    for (let i = 0; i < dataSetIn.length; i++) {
        var x = dataSetIn[i];
        var y = dataSetOut[i];
        point(map(x, 0, 1, 0, width), map(y, 0, 1, 0, height));
    }

    stroke(255, 0, 0);
    for (let i = 0; i < TestDataOut.length; i++) {
        var x = TestDataIn[i];
        var y = TestDataOut[i];
        point(map(x, 0, 1, 0, width), map(y, 0, 1, 0, height));
    }

    functionTextbox.html(functionSlider.value());

    // if (mousePressed == true) {
    // }

    if (functionSlider.value() == 4) {
        if (mouseDataIn.length > 0) {
            for (let i = 0; i < mouseDataIn.length; i++) {
                var x = map(mouseDataIn[i], 0, 1, 0, width);
                var y = map(mouseDataOut[i], 0, 1, 0, height);
                ellipse(x, y, 5, 5);
            }
        }
    }

}

function mouseDragged() {

    mouseDataIn[parseInt(map(mouseX, 0, width, 0, 50))] = map(mouseX, 0, width, 0, 1)

    mouseDataOut[parseInt(map(mouseX, 0, width, 0, 50))] = map(mouseY, 0, height, 0, 1)

    // console.log("Adding Points: " + mouseX);
}

function keyReleased() {
    console.log(mouseDataIn);
}

function Stop_Optimization() {
    model = null;
}

async function Run_Optimization() {
    model = NeuralNetwork.createModel(1, [20, 10, 5], 1);

    dataSetIn = []
    dataSetOut = []
    var data_size = 200;

    var funcToUse;
    var val = functionSlider.value();
    if (val == 0) {
        funcToUse = Sine;
    } else if (val == 1) {
        funcToUse = Square;
    } else if (val == 2) {
        funcToUse = SquareRoot;
    } else if (val == 3) {
        funcToUse = Cos;
    }

    if (val == 4) {
        dataSetIn = mouseDataIn;
        dataSetOut = mouseDataOut;
    } else {
        for (let i = 0; i <= data_size; i++) {
            var x = map(i, 0, data_size, 0, 1);
            dataSetIn.push(x);
            dataSetOut.push(funcToUse(x));
        }

    }

    console.log(dataSetIn);
    console.log(min(dataSetOut));

    var dx = tf.tensor(dataSetIn);
    var dy = tf.tensor(dataSetOut);
    //OR
    // var dx = tf.tensor2d(dataSetIn, [data_size + 1, 1]);
    // var dy = tf.tensor2d(dataSetOut, [data_size + 1, 1]);

    console.log(dx.print());
    console.log(dy.print());

    await model.fit(dx, dy, {
        epochs: 2000,
        batch: 1,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // console.log(epoch + ":" + logs.loss);
                errorTextbox.html(logs.loss);
                epochsTextbox.html(epoch);
                updateTestPoints();
            }
        }
    });
}

function updateTestPoints() {
    tf.tidy(() => {
        let xs = tf.tensor(TestDataIn);
        const ys = model.predict(xs);
        TestDataOut = ys.dataSync();
        // console.log(TestDataOut);
    });
}


class NeuralNetwork {
    static createModel(INPUTS, HIDDEN, OUTPUTS) {
        const model = tf.sequential();

        let activationInside = 'relu';
        let activationLast = 'sigmoid';

        if (HIDDEN.length > 0) {

            let hidden0 = tf.layers.dense({
                inputShape: [INPUTS],
                units: HIDDEN[0],
                activation: activationInside,
                // kernelInitializer: 'glorotNormal'
            });
            model.add(hidden0);

            if (HIDDEN.length > 1) {
                for (let i = 1; i < HIDDEN.length; i++) {
                    var hidden1 = tf.layers.dense({
                        units: HIDDEN[i],
                        activation: activationInside,
                    });
                    model.add(hidden1);
                }
            }
            var output = tf.layers.dense({
                units: OUTPUTS,
                activation: activationLast
            });
        } else {
            var output = tf.layers.dense({
                inputShape: [INPUTS],
                units: OUTPUTS,
                activation: activationLast
            });
        }

        model.add(output);
        const optimizer = tf.train.sgd(LEARNING_RATE);

        model.compile({
            loss: 'meanSquaredError',
            optimizer: optimizer
        });
        console.log(model.summary());
        return model;
    }
}


//Get a values between [0, 1]
function Sine(x) {
    return map(sin(x * 2 * PI), -1, 1, 0, 1);
}

function Square(x) {
    return sq(x);
}

function SquareRoot(x) {
    return sqrt(x)
}

function Cos(x) {
    return map(cos(x * 2 * PI), -1, 1, 0, 1);
}
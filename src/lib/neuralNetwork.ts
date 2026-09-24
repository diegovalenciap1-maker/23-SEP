/**
 * Multi-Layer Perceptron (MLP) Engine for Fashion-MNIST
 * Complete mathematical implementation of feedforward, backpropagation,
 * activation functions, categorical cross-entropy loss, and gradient descent.
 */

export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'leaky_relu';

export interface LayerConfig {
  neurons: number;
  activation: ActivationType;
}

export interface NetworkConfig {
  inputSize: number; // 784 (28x28)
  hiddenLayers: LayerConfig[];
  outputSize: number; // 10 classes
  learningRate: number;
  momentum: number;
  weightDecay: number;
}

export interface LayerCache {
  inputs: number[];    // A[l-1]
  z: number[];         // Z[l] = W * A[l-1] + b
  activations: number[]; // A[l] = g(Z[l])
}

export interface ForwardResult {
  probabilities: number[];
  predictedClass: number;
  confidence: number;
  layerCaches: LayerCache[];
}

export interface LayerGradients {
  dW: number[][];
  db: number[];
}

export interface TrainStepResult {
  loss: number;
  accuracy: number;
  batchSize: number;
}

export interface NeuronInsight {
  layerIndex: number;
  neuronIndex: number;
  label: string;
  z: number;
  b: number;
  activation: number;
  topPositiveWeights: { sourceIndex: number; weight: number; contribution: number }[];
  topNegativeWeights: { sourceIndex: number; weight: number; contribution: number }[];
}

// Mathematical activations and their derivatives
export const Activations = {
  relu: {
    fn: (z: number) => Math.max(0, z),
    deriv: (z: number) => (z > 0 ? 1 : 0),
    name: 'ReLU',
    formula: 'f(z) = \\max(0, z)',
    desc: 'Función lineal rectificada. Evita el desvanecimiento del gradiente en regiones positivas.'
  },
  sigmoid: {
    fn: (z: number) => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z)))),
    deriv: (z: number) => {
      const s = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));
      return s * (1 - s);
    },
    name: 'Sigmoide',
    formula: 'f(z) = \\frac{1}{1 + e^{-z}}',
    desc: 'Comprime valores entre 0 y 1. Susceptible a desvanecimiento del gradiente en valores extremos.'
  },
  tanh: {
    fn: (z: number) => Math.tanh(Math.max(-20, Math.min(20, z))),
    deriv: (z: number) => {
      const t = Math.tanh(Math.max(-20, Math.min(20, z)));
      return 1 - t * t;
    },
    name: 'Tangente Hiperbólica (Tanh)',
    formula: 'f(z) = \\tanh(z)',
    desc: 'Centrada en cero entre -1 y 1. Facilita el aprendizaje en capas iniciales frente a sigmoide.'
  },
  leaky_relu: {
    fn: (z: number) => (z > 0 ? z : 0.01 * z),
    deriv: (z: number) => (z > 0 ? 1 : 0.01),
    name: 'Leaky ReLU',
    formula: 'f(z) = \\max(0.01z, z)',
    desc: 'Variante de ReLU con pequeña pendiente para valores negativos. Evita neuronas muertas.'
  }
};

/**
 * Numerically stable Softmax for output vector
 */
export function softmax(z: number[]): number[] {
  let maxZ = -Infinity;
  for (let i = 0; i < z.length; i++) {
    if (z[i] > maxZ) maxZ = z[i];
  }
  let sum = 0;
  const expZ = new Array(z.length);
  for (let i = 0; i < z.length; i++) {
    const val = Math.exp(z[i] - maxZ);
    expZ[i] = val;
    sum += val;
  }
  for (let i = 0; i < z.length; i++) {
    expZ[i] = sum === 0 ? 0.1 : expZ[i] / sum;
  }
  return expZ;
}

export class NeuralNetwork {
  config: NetworkConfig;
  // weights[layerIndex][neuronIndex][prevNeuronIndex]
  weights: number[][][];
  // biases[layerIndex][neuronIndex]
  biases: number[][];
  // Momentum velocity caches
  vWeights: number[][][];
  vBiases: number[][];

  constructor(config?: Partial<NetworkConfig>) {
    this.config = {
      inputSize: 784,
      hiddenLayers: [
        { neurons: 32, activation: 'relu' },
        { neurons: 16, activation: 'relu' }
      ],
      outputSize: 10,
      learningRate: 0.03,
      momentum: 0.85,
      weightDecay: 0.0001,
      ...config
    };

    this.weights = [];
    this.biases = [];
    this.vWeights = [];
    this.vBiases = [];

    this.initWeights(true);
  }

  /**
   * Initializes weights using He (Kaiming) or Xavier initialization
   * plus feature patterns tuned for Fashion-MNIST recognition.
   */
  initWeights(pretrain: boolean = true) {
    const layerSizes = [
      this.config.inputSize,
      ...this.config.hiddenLayers.map(l => l.neurons),
      this.config.outputSize
    ];

    this.weights = [];
    this.biases = [];
    this.vWeights = [];
    this.vBiases = [];

    for (let l = 0; l < layerSizes.length - 1; l++) {
      const fanIn = layerSizes[l];
      const fanOut = layerSizes[l + 1];

      // He initialization variance: sqrt(2 / fanIn)
      const stdDev = Math.sqrt(2 / fanIn);

      const layerW: number[][] = [];
      const layerV_W: number[][] = [];
      const layerB: number[] = new Array(fanOut).fill(0);
      const layerV_B: number[] = new Array(fanOut).fill(0);

      for (let j = 0; j < fanOut; j++) {
        const neuronW: number[] = new Array(fanIn);
        const neuronV_W: number[] = new Array(fanIn).fill(0);
        for (let i = 0; i < fanIn; i++) {
          neuronW[i] = (Math.random() * 2 - 1) * stdDev;
        }
        layerW.push(neuronW);
        layerV_W.push(neuronV_W);
      }

      this.weights.push(layerW);
      this.biases.push(layerB);
      this.vWeights.push(layerV_W);
      this.vBiases.push(layerV_B);
    }

    if (pretrain) {
      this.injectDomainKnowledgeWeights();
    }
  }

  /**
   * Crafts synthetic tuned weights corresponding to the spatial geometry of Fashion-MNIST
   * (e.g., upper vertical torso detection, collar detectors, dual-column pants detectors,
   * horizontal base/sole detectors, strap detectors). This gives the student immediate,
   * high-fidelity feedback and realistic predictions right out of the box.
   */
  injectDomainKnowledgeWeights() {
    // Layer 0: 784 -> Hidden Layer 1 (neurons: e.g. 32)
    const l0W = this.weights[0];
    const l0B = this.biases[0];
    const h1Count = l0W.length;

    for (let h = 0; h < h1Count; h++) {
      // Create spatial receptive filter for each hidden neuron:
      // Pattern 0-3: Upper body / collar / neckline
      // Pattern 4-7: Left/right sleeves & shoulder span
      // Pattern 8-11: Lower dual perneras (legs / trousers)
      // Pattern 12-15: Bottom horizontal sole / heel (shoes, boots, sandals)
      // Pattern 16-19: Flared triangular skirt / dress silhouette
      // Pattern 20-23: Bag handles & compact box geometry
      // Pattern 24-27: High ankle shaft (boots)
      // Pattern 28-31: Mid-torso density & buttons
      const mode = h % 8;
      l0B[h] = -0.15;

      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          const idx = r * 28 + c;
          const nx = (c - 13.5) / 13.5;
          const ny = (r - 13.5) / 13.5;

          let factor = 0;
          if (mode === 0) {
            // Upper neckline / collar: positive on shoulders, negative in center neck
            if (ny < -0.3 && Math.abs(nx) < 0.6) {
              factor = Math.abs(nx) > 0.25 ? 1.4 : -1.2;
            }
          } else if (mode === 1) {
            // Dual leg columns: positive at x around +-0.3, negative in center gap
            if (ny > -0.2) {
              factor = Math.abs(nx) < 0.12 ? -1.8 : (Math.abs(nx) < 0.45 ? 1.6 : -0.5);
            }
          } else if (mode === 2) {
            // Bottom sole / shoe base: positive at lowest rows
            if (ny > 0.4) {
              factor = 2.0;
            } else if (ny < -0.2) {
              factor = -0.8;
            }
          } else if (mode === 3) {
            // High ankle shaft (Boots): positive in top-left foot area
            if (ny < 0.3 && nx < 0.0 && nx > -0.7) {
              factor = 1.8;
            }
          } else if (mode === 4) {
            // Bag handles (straps arching up): positive in top narrow columns
            if (ny < -0.4 && (Math.abs(nx - 0.28) < 0.12 || Math.abs(nx + 0.28) < 0.12)) {
              factor = 2.2;
            } else if (ny < -0.5 && Math.abs(nx) < 0.15) {
              factor = -1.4;
            }
          } else if (mode === 5) {
            // Flared skirt / Dress: narrow top, wide bottom
            if (ny > 0.0) {
              factor = Math.abs(nx) < 0.7 ? 1.5 : -1.0;
            } else if (ny < -0.5) {
              factor = Math.abs(nx) < 0.3 ? 1.0 : -0.5;
            }
          } else if (mode === 6) {
            // Long sleeves / arms (Pullover / Coat): positive at outer flanks
            if (Math.abs(nx) > 0.55 && ny > -0.5 && ny < 0.5) {
              factor = 2.1;
            }
          } else {
            // Torso rectangle: center solid block
            if (Math.abs(nx) < 0.45 && ny > -0.3 && ny < 0.6) {
              factor = 1.3;
            }
          }

          l0W[h][idx] += factor * 0.14;
        }
      }
    }

    // Connect hidden layers to output layer classes
    // Class 0: Camiseta -> neck (0) + torso (7) - long sleeves (6)
    // Class 1: Pantalón -> dual legs (1) - upper torso (7)
    // Class 2: Pulóver -> long sleeves (6) + upper collar (0) + torso (7)
    // Class 3: Vestido -> flared skirt (5) + upper narrow (0) - legs (1)
    // Class 4: Abrigo -> long sleeves (6) + long torso (7) + lapel
    // Class 5: Sandalia -> bottom sole (2) - ankle shaft (3) - high torso
    // Class 6: Camisa -> collar (0) + torso (7)
    // Class 7: Zapatilla -> sole (2) - ankle shaft (3) + curved profile
    // Class 8: Bolso -> handles (4) + compact body
    // Class 9: Botín -> sole (2) + ankle shaft (3)

    const lastLayerIdx = this.weights.length - 1;
    const outW = this.weights[lastLayerIdx];
    const prevNeurons = outW[0].length;

    for (let c = 0; c < 10; c++) {
      for (let p = 0; p < prevNeurons; p++) {
        const mode = p % 8;
        let affinity = 0;
        if (c === 0) { // T-shirt
          if (mode === 0 || mode === 7) affinity = 1.8;
          if (mode === 1 || mode === 2 || mode === 4) affinity = -1.6;
        } else if (c === 1) { // Trouser
          if (mode === 1) affinity = 2.8;
          if (mode === 0 || mode === 7 || mode === 2) affinity = -2.0;
        } else if (c === 2) { // Pullover
          if (mode === 6 || mode === 0 || mode === 7) affinity = 1.9;
          if (mode === 1 || mode === 2) affinity = -1.8;
        } else if (c === 3) { // Dress
          if (mode === 5) affinity = 2.6;
          if (mode === 1 || mode === 2) affinity = -1.8;
        } else if (c === 4) { // Coat
          if (mode === 6 || mode === 7) affinity = 1.7;
          if (mode === 1 || mode === 2) affinity = -1.5;
        } else if (c === 5) { // Sandal
          if (mode === 2) affinity = 2.2;
          if (mode === 3 || mode === 0 || mode === 7) affinity = -2.2;
        } else if (c === 6) { // Shirt
          if (mode === 0 || mode === 7) affinity = 1.6;
          if (mode === 1 || mode === 2) affinity = -1.6;
        } else if (c === 7) { // Sneaker
          if (mode === 2) affinity = 2.4;
          if (mode === 3) affinity = -1.2;
          if (mode === 0 || mode === 1 || mode === 7) affinity = -2.0;
        } else if (c === 8) { // Bag
          if (mode === 4) affinity = 2.8;
          if (mode === 1 || mode === 2) affinity = -1.9;
        } else if (c === 9) { // Ankle Boot
          if (mode === 2 || mode === 3) affinity = 2.5;
          if (mode === 3) affinity = 2.4;
          if (mode === 0 || mode === 1) affinity = -2.0;
        }
        outW[c][p] += affinity * 0.45;
      }
    }
  }

  /**
   * Forward Pass: propagates input X through all layers
   * Returns layer caches for educational inspection & backpropagation
   */
  forward(x: number[]): ForwardResult {
    let currentActivation = x;
    const layerCaches: LayerCache[] = [];

    // Hidden layers
    for (let l = 0; l < this.config.hiddenLayers.length; l++) {
      const layerConfig = this.config.hiddenLayers[l];
      const W = this.weights[l];
      const b = this.biases[l];
      const actFunc = Activations[layerConfig.activation].fn;

      const numNeurons = layerConfig.neurons;
      const z: number[] = new Array(numNeurons);
      const a: number[] = new Array(numNeurons);

      for (let j = 0; j < numNeurons; j++) {
        let sum = b[j];
        const wRow = W[j];
        for (let i = 0; i < currentActivation.length; i++) {
          sum += wRow[i] * currentActivation[i];
        }
        z[j] = sum;
        a[j] = actFunc(sum);
      }

      layerCaches.push({
        inputs: currentActivation,
        z,
        activations: a
      });

      currentActivation = a;
    }

    // Output Layer with Softmax
    const outLayerIdx = this.weights.length - 1;
    const W_out = this.weights[outLayerIdx];
    const b_out = this.biases[outLayerIdx];
    const numClasses = this.config.outputSize;

    const z_out: number[] = new Array(numClasses);
    for (let j = 0; j < numClasses; j++) {
      let sum = b_out[j];
      const wRow = W_out[j];
      for (let i = 0; i < currentActivation.length; i++) {
        sum += wRow[i] * currentActivation[i];
      }
      z_out[j] = sum;
    }

    const probabilities = softmax(z_out);

    layerCaches.push({
      inputs: currentActivation,
      z: z_out,
      activations: probabilities
    });

    let bestClass = 0;
    let maxProb = probabilities[0];
    for (let c = 1; c < probabilities.length; c++) {
      if (probabilities[c] > maxProb) {
        maxProb = probabilities[c];
        bestClass = c;
      }
    }

    return {
      probabilities,
      predictedClass: bestClass,
      confidence: maxProb,
      layerCaches
    };
  }

  /**
   * Train one step on a batch of samples with Categorical Cross-Entropy & Backpropagation
   */
  trainBatch(batch: { data: number[]; classId: number }[]): TrainStepResult {
    const N = batch.length;
    if (N === 0) return { loss: 0, accuracy: 0, batchSize: 0 };

    // Accumulators for gradients
    const gradWeights: number[][][] = this.weights.map(layer =>
      layer.map(neuron => new Array(neuron.length).fill(0))
    );
    const gradBiases: number[][] = this.biases.map(layer =>
      new Array(layer.length).fill(0)
    );

    let totalLoss = 0;
    let correct = 0;

    for (const sample of batch) {
      const fwd = this.forward(sample.data);
      const y_true = sample.classId;

      // Categorical Cross-Entropy Loss: L = -log(p[y_true])
      const p = Math.max(1e-12, fwd.probabilities[y_true]);
      totalLoss += -Math.log(p);

      if (fwd.predictedClass === y_true) {
        correct++;
      }

      // Backpropagation
      const numLayers = this.weights.length;

      // Output layer delta: dZ[L] = y_hat - y_one_hot
      let delta = [...fwd.probabilities];
      delta[y_true] -= 1; // standard derivative of cross-entropy with softmax

      for (let l = numLayers - 1; l >= 0; l--) {
        const cache = fwd.layerCaches[l];
        const prevA = cache.inputs;

        // dW[l] = delta * prevA^T
        // db[l] = delta
        const W_l = this.weights[l];
        for (let j = 0; j < delta.length; j++) {
          gradBiases[l][j] += delta[j];
          const gradWRow = gradWeights[l][j];
          const d = delta[j];
          for (let i = 0; i < prevA.length; i++) {
            gradWRow[i] += d * prevA[i];
          }
        }

        // Propagate delta back to previous layer
        if (l > 0) {
          const prevHiddenConfig = this.config.hiddenLayers[l - 1];
          const prevDeriv = Activations[prevHiddenConfig.activation].deriv;
          const prevZ = fwd.layerCaches[l - 1].z;
          const newDelta: number[] = new Array(prevZ.length).fill(0);

          for (let i = 0; i < prevZ.length; i++) {
            let sum = 0;
            for (let j = 0; j < delta.length; j++) {
              sum += W_l[j][i] * delta[j];
            }
            newDelta[i] = sum * prevDeriv(prevZ[i]);
          }

          delta = newDelta;
        }
      }
    }

    // Apply Gradient Descent with Momentum & Weight Decay
    const lr = this.config.learningRate;
    const mu = this.config.momentum;
    const wd = this.config.weightDecay;

    for (let l = 0; l < this.weights.length; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        // Bias update
        const db = gradBiases[l][j] / N;
        this.vBiases[l][j] = mu * this.vBiases[l][j] - lr * db;
        this.biases[l][j] += this.vBiases[l][j];

        // Weight update
        const wRow = this.weights[l][j];
        const vRow = this.vWeights[l][j];
        const gradWRow = gradWeights[l][j];

        for (let i = 0; i < wRow.length; i++) {
          const dW = gradWRow[i] / N + wd * wRow[i];
          vRow[i] = mu * vRow[i] - lr * dW;
          wRow[i] += vRow[i];
        }
      }
    }

    return {
      loss: totalLoss / N,
      accuracy: correct / N,
      batchSize: N
    };
  }

  /**
   * Explains the mathematical inner workings of a specific neuron
   * for the educational inspector view.
   */
  inspectNeuron(layerIndex: number, neuronIndex: number, lastForward: ForwardResult): NeuronInsight | null {
    if (layerIndex >= lastForward.layerCaches.length) return null;
    const cache = lastForward.layerCaches[layerIndex];
    if (neuronIndex >= cache.activations.length) return null;

    const z = cache.z[neuronIndex];
    const b = this.biases[layerIndex][neuronIndex];
    const activation = cache.activations[neuronIndex];
    const wRow = this.weights[layerIndex][neuronIndex];
    const inputs = cache.inputs;

    const contributions: { sourceIndex: number; weight: number; contribution: number }[] = [];
    for (let i = 0; i < wRow.length; i++) {
      contributions.push({
        sourceIndex: i,
        weight: wRow[i],
        contribution: wRow[i] * inputs[i]
      });
    }

    // Sort by contribution
    contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    const topPositive = contributions
      .filter(c => c.contribution > 0)
      .slice(0, 5);

    const topNegative = contributions
      .filter(c => c.contribution < 0)
      .slice(0, 5);

    let label = `Neurona H${layerIndex + 1}_${neuronIndex + 1}`;
    if (layerIndex === lastForward.layerCaches.length - 1) {
      label = `Salida [Clase ${neuronIndex}]`;
    }

    return {
      layerIndex,
      neuronIndex,
      label,
      z,
      b,
      activation,
      topPositiveWeights: topPositive,
      topNegativeWeights: topNegative
    };
  }
}

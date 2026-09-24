import React, { useState, useEffect, useRef } from 'react';
import { NeuralNetwork, TrainStepResult, ActivationType } from '../lib/neuralNetwork';
import { generateTrainingBatch, FASHION_CLASSES } from '../data/fashionMnistData';
import { Play, Pause, RotateCcw, FastForward, Activity, CheckCircle, BarChart3 } from 'lucide-react';

interface TrainingStudioProps {
  network: NeuralNetwork;
  onNetworkUpdated: () => void;
  onResetWeights: () => void;
}

export const TrainingStudio: React.FC<TrainingStudioProps> = ({
  network,
  onNetworkUpdated,
  onResetWeights,
}) => {
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  const [totalBatches, setTotalBatches] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([1.85]);
  const [accHistory, setAccHistory] = useState<number[]>([0.22]);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [learningRate, setLearningRate] = useState<number>(0.03);
  const [momentum, setMomentum] = useState<number>(0.85);

  // 10x10 Confusion Matrix [trueClass][predClass]
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>(() =>
    Array.from({ length: 10 }, () => new Array(10).fill(0))
  );

  const trainingTimer = useRef<number | null>(null);

  // Apply hyperparameter updates to network
  useEffect(() => {
    network.config.learningRate = learningRate;
    network.config.momentum = momentum;
  }, [learningRate, momentum, network]);

  const runSingleBatch = () => {
    const batch = generateTrainingBatch(batchSize);
    const result = network.trainBatch(batch);

    // Update confusion matrix with this batch
    setConfusionMatrix((prev) => {
      const copy = prev.map((row) => [...row]);
      for (const sample of batch) {
        const fwd = network.forward(sample.data);
        copy[sample.classId][fwd.predictedClass] += 1;
      }
      return copy;
    });

    setTotalBatches((b) => b + 1);
    setLossHistory((h) => [...h.slice(-40), Math.round(result.loss * 1000) / 1000]);
    setAccHistory((h) => [...h.slice(-40), Math.round(result.accuracy * 1000) / 1000]);

    if ((totalBatches + 1) % 10 === 0) {
      setEpoch((e) => e + 1);
    }

    onNetworkUpdated();
  };

  const runEpoch = () => {
    for (let i = 0; i < 6; i++) {
      runSingleBatch();
    }
    setEpoch((e) => e + 1);
  };

  // Continuous training loop
  useEffect(() => {
    if (isTraining) {
      trainingTimer.current = window.setInterval(() => {
        runSingleBatch();
      }, 140);
    } else {
      if (trainingTimer.current) {
        clearInterval(trainingTimer.current);
        trainingTimer.current = null;
      }
    }
    return () => {
      if (trainingTimer.current) {
        clearInterval(trainingTimer.current);
      }
    };
  }, [isTraining, batchSize]);

  const handleReset = () => {
    setIsTraining(false);
    onResetWeights();
    setEpoch(0);
    setTotalBatches(0);
    setLossHistory([2.3]);
    setAccHistory([0.1]);
    setConfusionMatrix(Array.from({ length: 10 }, () => new Array(10).fill(0)));
  };

  const currentLoss = lossHistory[lossHistory.length - 1] || 0;
  const currentAcc = accHistory[accHistory.length - 1] || 0;

  return (
    <div className="space-y-6">
      {/* Top Training Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-5 bg-slate-900 border border-slate-800 rounded-xl">
        <div className="flex items-center gap-6">
          <div>
            <span className="text-xs text-slate-400 block">Pérdida Actual (Cross-Entropy):</span>
            <span className="text-xl font-bold font-mono text-cyan-400 tabular-nums">
              {currentLoss.toFixed(4)}
            </span>
          </div>
          <div className="border-l border-slate-800 pl-6">
            <span className="text-xs text-slate-400 block">Precisión en Batch:</span>
            <span className="text-xl font-bold font-mono text-emerald-400 tabular-nums">
              {(currentAcc * 100).toFixed(1)}%
            </span>
          </div>
          <div className="border-l border-slate-800 pl-6">
            <span className="text-xs text-slate-400 block">Épocas / Batches:</span>
            <span className="text-sm font-semibold font-mono text-white tabular-nums">
              Época {epoch} <span className="text-slate-500">·</span> {totalBatches} batches
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setIsTraining(!isTraining)}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all shadow-lg ${
              isTraining
                ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-900/30'
                : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-cyan-900/30'
            }`}
          >
            {isTraining ? (
              <>
                <Pause className="w-4 h-4" />
                <span>Pausar</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Entrenar en Vivo</span>
              </>
            )}
          </button>

          <button
            onClick={runEpoch}
            disabled={isTraining}
            className="flex items-center gap-1.5 px-3.5 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
            title="Avanzar una época completa de mini-batches"
          >
            <FastForward className="w-4 h-4 text-cyan-400" />
            <span>+1 Época</span>
          </button>

          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 text-xs font-medium rounded-lg transition-colors"
            title="Reiniciar parámetros de la red a valores aleatorios"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reiniciar</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Hyperparameters & Graphs (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Hyperparameters Card */}
          <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-sm font-semibold text-white">Hiperparámetros de Optimización</h3>

            <div className="space-y-4 text-xs">
              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Tasa de Aprendizaje (η):</span>
                  <span className="font-mono text-cyan-400 font-semibold">{learningRate.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.005"
                  max="0.1"
                  step="0.005"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Tamaño de Mini-Batch (N):</span>
                  <span className="font-mono text-cyan-400 font-semibold">{batchSize} muestras</span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {[8, 16, 32, 64].map((sz) => (
                    <button
                      key={sz}
                      onClick={() => setBatchSize(sz)}
                      className={`py-1 text-xs font-mono rounded border transition-colors ${
                        batchSize === sz
                          ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 font-bold'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      {sz}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Momento (Momentum):</span>
                  <span className="font-mono text-cyan-400 font-semibold">{momentum.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.95"
                  step="0.05"
                  value={momentum}
                  onChange={(e) => setMomentum(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Real-time Loss and Accuracy Graphs */}
          <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h3 className="text-sm font-semibold text-white">Curvas de Aprendizaje</h3>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="text-cyan-400">― Pérdida L(t)</span>
                <span className="text-emerald-400">― Precisión Acc(t)</span>
              </div>
            </div>

            {/* SVG Learning Curves */}
            <div className="w-full h-44 bg-slate-950 rounded-xl border border-slate-800 relative p-2">
              <svg className="w-full h-full" viewBox="0 0 240 100" preserveAspectRatio="none">
                {/* Horizontal reference lines */}
                <line x1="0" y1="25" x2="240" y2="25" stroke="#1e293b" strokeWidth="0.8" />
                <line x1="0" y1="50" x2="240" y2="50" stroke="#1e293b" strokeWidth="0.8" />
                <line x1="0" y1="75" x2="240" y2="75" stroke="#1e293b" strokeWidth="0.8" />

                {/* Loss Curve */}
                {lossHistory.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#06b6d4"
                    strokeWidth="2"
                    points={lossHistory
                      .map((val, idx) => {
                        const x = (idx / (lossHistory.length - 1)) * 240;
                        const y = Math.max(5, Math.min(95, (val / 2.5) * 90));
                        return `${x},${y}`;
                      })
                      .join(' ')}
                  />
                )}

                {/* Accuracy Curve */}
                {accHistory.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2"
                    points={accHistory
                      .map((val, idx) => {
                        const x = (idx / (accHistory.length - 1)) * 240;
                        const y = 95 - val * 90;
                        return `${x},${y}`;
                      })
                      .join(' ')}
                  />
                )}
              </svg>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Al descender la pérdida categórica cruzada, la precisión sobre el conjunto de prendas aumenta conforme los gradientes ajustan los pesos en cada capa.
            </p>
          </div>
        </div>

        {/* Right: 10x10 Confusion Matrix (7 cols) */}
        <div className="lg:col-span-7 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-sm font-semibold text-white">Matriz de Confusión (10 × 10)</h3>
              <p className="text-xs text-slate-400">
                Filas = Clase Real $\cdot$ Columnas = Clase Predicha por la Red
              </p>
            </div>
            <BarChart3 className="w-4 h-4 text-cyan-400" />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-center text-[10px] font-mono border-collapse">
              <thead>
                <tr>
                  <th className="p-1 text-slate-500 font-sans text-left">Real \ Pred</th>
                  {FASHION_CLASSES.map((fc) => (
                    <th key={fc.id} className="p-1 text-slate-400 font-semibold" title={fc.nameEn}>
                      {fc.id}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {FASHION_CLASSES.map((rowClass, r) => (
                  <tr key={r}>
                    <td className="p-1 text-left text-slate-400 font-medium truncate max-w-[90px]" title={rowClass.nameEn}>
                      {r}. {rowClass.nameEn.split('/')[0]}
                    </td>
                    {FASHION_CLASSES.map((colClass, c) => {
                      const count = confusionMatrix[r][c];
                      const isDiagonal = r === c;
                      return (
                        <td
                          key={c}
                          className="p-1 border border-slate-800/80 transition-colors"
                          style={{
                            backgroundColor:
                              count > 0
                                ? isDiagonal
                                  ? `rgba(16, 185, 129, ${Math.min(0.85, 0.15 + count * 0.08)})`
                                  : `rgba(244, 63, 94, ${Math.min(0.7, 0.15 + count * 0.1)})`
                                : '#020617',
                            color: count > 0 ? '#ffffff' : '#475569',
                          }}
                          title={`Real: ${rowClass.nameEn} -> Predicho: ${colClass.nameEn} (${count} veces)`}
                        >
                          {count}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pedagogical insight on confusions */}
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs space-y-1.5">
            <span className="font-semibold text-slate-200 block">
              ¿Por qué se producen confusiones típicas en Fashion-MNIST?
            </span>
            <p className="text-slate-400 leading-relaxed">
              Las prendas como <strong>Camisa (6)</strong> y <strong>Camiseta (0)</strong> comparten el mismo cuerpo rectangular y mangas, diferenciándose únicamente en el corte fino del cuello y botones centrales. Las <strong>Sandalias (5)</strong> frente a <strong>Zapatillas (7)</strong> comparten la suela baja, pero la red distingue los espacios vacíos del empeine.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

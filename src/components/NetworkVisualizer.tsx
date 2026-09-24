import React, { useState, useEffect } from 'react';
import { ForwardResult, NeuralNetwork, NeuronInsight, Activations } from '../lib/neuralNetwork';
import { FASHION_CLASSES, FashionSample } from '../data/fashionMnistData';
import { Play, Sparkles, HelpCircle, Layers, ArrowRight, Activity, Zap } from 'lucide-react';

interface NetworkVisualizerProps {
  network: NeuralNetwork;
  currentSample: FashionSample;
  forwardResult: ForwardResult | null;
  onRunForward: () => void;
  onSelectSample: (sample: FashionSample) => void;
  samples: FashionSample[];
}

export const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({
  network,
  currentSample,
  forwardResult,
  onRunForward,
  onSelectSample,
  samples,
}) => {
  const [selectedNeuron, setSelectedNeuron] = useState<{ layer: number; index: number } | null>(null);
  const [isAnimatingForward, setIsAnimatingForward] = useState<boolean>(false);
  const [pulseStage, setPulseStage] = useState<number>(-1); // -1: idle, 0: input, 1: h1, 2: h2, 3: output

  const runAnimatedForward = () => {
    setIsAnimatingForward(true);
    setPulseStage(0);
    setTimeout(() => setPulseStage(1), 220);
    setTimeout(() => setPulseStage(2), 440);
    setTimeout(() => {
      setPulseStage(3);
      onRunForward();
      setTimeout(() => {
        setIsAnimatingForward(false);
        setPulseStage(-1);
      }, 350);
    }, 660);
  };

  const neuronInsight: NeuronInsight | null =
    selectedNeuron && forwardResult
      ? network.inspectNeuron(selectedNeuron.layer, selectedNeuron.index, forwardResult)
      : null;

  // Derive visual nodes for layout
  // Layer 0: 784 inputs (we visualize 16 representative spatial sensors from the 28x28 grid)
  const inputSensorPoints = [
    { label: 'Cuello Superior', r: 4, c: 14, idx: 4 * 28 + 14 },
    { label: 'Hombro Izq', r: 5, c: 5, idx: 5 * 28 + 5 },
    { label: 'Hombro Der', r: 5, c: 22, idx: 5 * 28 + 22 },
    { label: 'Torso Central', r: 14, c: 14, idx: 14 * 28 + 14 },
    { label: 'Manga / Lateral Izq', r: 12, c: 4, idx: 12 * 28 + 4 },
    { label: 'Manga / Lateral Der', r: 12, c: 23, idx: 12 * 28 + 23 },
    { label: 'Pierna / Dobladillo Izq', r: 22, c: 8, idx: 22 * 28 + 8 },
    { label: 'Pierna / Dobladillo Der', r: 22, c: 19, idx: 22 * 28 + 19 },
    { label: 'Suela / Base Izq', r: 25, c: 5, idx: 25 * 28 + 5 },
    { label: 'Suela / Base Der', r: 25, c: 22, idx: 25 * 28 + 22 },
    { label: 'Empeine / Caña', r: 16, c: 8, idx: 16 * 28 + 8 },
    { label: 'Asa Superior', r: 2, c: 10, idx: 2 * 28 + 10 },
  ];

  const h1Count = Math.min(16, network.weights[0]?.length || 16);
  const h2Count = Math.min(12, network.weights[1]?.length || 12);
  const outCount = 10;

  // Selected sample info
  const trueClass = FASHION_CLASSES[currentSample.classId];
  const predictedClassId = forwardResult ? forwardResult.predictedClass : -1;
  const isCorrect = predictedClassId === currentSample.classId;

  return (
    <div className="space-y-6">
      {/* Top Banner / Sample Selection */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-slate-900 border border-slate-800 rounded-xl">
        <div className="flex items-center gap-4">
          {/* Mini 28x28 preview of current sample */}
          <div className="relative group">
            <canvas
              ref={(canvas) => {
                if (canvas && currentSample) {
                  const ctx = canvas.getContext('2d');
                  if (ctx) {
                    const imgData = ctx.createImageData(28, 28);
                    for (let i = 0; i < 784; i++) {
                      const v = Math.floor(currentSample.data[i] * 255);
                      imgData.data[i * 4] = v;
                      imgData.data[i * 4 + 1] = v;
                      imgData.data[i * 4 + 2] = v;
                      imgData.data[i * 4 + 3] = 255;
                    }
                    ctx.putImageData(imgData, 0, 0);
                  }
                }
              }}
              width={28}
              height={28}
              className="w-14 h-14 bg-black rounded-lg border border-slate-700 image-rendering-pixelated shadow-inner"
              style={{ imageRendering: 'pixelated' }}
            />
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Muestra Activa</span>
              <span className="text-slate-600">·</span>
              <span className="text-xs font-semibold text-cyan-400">{trueClass.name}</span>
            </div>
            <p className="text-sm text-slate-300">
              {currentSample.label} ({trueClass.category})
            </p>
          </div>
        </div>

        {/* Quick Sample Selector buttons */}
        <div className="flex flex-wrap items-center gap-1.5 max-w-xl">
          {samples.slice(0, 8).map((s) => {
            const isCurrent = s.id === currentSample.id;
            return (
              <button
                key={s.id}
                onClick={() => onSelectSample(s)}
                className={`px-2.5 py-1 text-xs rounded transition-colors ${
                  isCurrent
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 font-medium'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                }`}
              >
                {s.label.split(' ')[0]}
              </button>
            );
          })}
        </div>

        {/* Forward Action Button */}
        <button
          onClick={runAnimatedForward}
          disabled={isAnimatingForward}
          className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-cyan-600 hover:bg-cyan-500 rounded-lg shadow-lg shadow-cyan-900/30 transition-all disabled:opacity-50"
        >
          <Zap className="w-4 h-4 text-cyan-200" />
          <span>{isAnimatingForward ? 'Propagando...' : 'Propagar Forward Pass'}</span>
        </button>
      </div>

      {/* Main Architecture & Interactive Network Graph */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Network Graph (3 Cols) */}
        <div className="xl:col-span-3 p-5 bg-slate-900/80 border border-slate-800 rounded-xl relative overflow-hidden">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span className="font-semibold text-slate-200">Arquitectura Perceptrón Multicapa (MLP)</span>
              <span>·</span>
              <span>784 Entradas $\to$ 32 Oculta 1 $\to$ 16 Oculta 2 $\to$ 10 Salidas</span>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block" /> Peso +
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-400 inline-block" /> Peso -
              </span>
            </div>
          </div>

          {/* SVG Diagram Canvas */}
          <div className="relative w-full h-[520px] mt-4 flex items-center justify-between px-2 select-none">
            {/* SVG Lines between layers */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <defs>
                <linearGradient id="cyanLine" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.6" />
                  <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.9" />
                </linearGradient>
                <linearGradient id="pulseGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38bdf8" />
                  <stop offset="50%" stopColor="#ffffff" />
                  <stop offset="100%" stopColor="#38bdf8" />
                </linearGradient>
              </defs>

              {/* Sample connections between Input -> H1 */}
              {inputSensorPoints.map((inp, i) => {
                const y1 = 40 + i * (440 / inputSensorPoints.length);
                return Array.from({ length: 4 }).map((_, j) => {
                  const targetH = (i + j * 3) % h1Count;
                  const y2 = 30 + targetH * (460 / h1Count);
                  const isHighlighted = pulseStage === 0 || pulseStage === 1;
                  return (
                    <line
                      key={`l0-${i}-${targetH}`}
                      x1="18%"
                      y1={y1}
                      x2="42%"
                      y2={y2}
                      stroke={isHighlighted ? '#38bdf8' : '#334155'}
                      strokeWidth={isHighlighted ? '1.5' : '0.6'}
                      strokeOpacity={isHighlighted ? '0.7' : '0.2'}
                      strokeDasharray={isHighlighted ? '3,3' : undefined}
                    />
                  );
                });
              })}

              {/* Sample connections between H1 -> H2 */}
              {Array.from({ length: h1Count }).map((_, i) => {
                const y1 = 30 + i * (460 / h1Count);
                return Array.from({ length: 2 }).map((_, j) => {
                  const targetH = (i + j * 4) % h2Count;
                  const y2 = 50 + targetH * (420 / h2Count);
                  const isHighlighted = pulseStage === 1 || pulseStage === 2;
                  return (
                    <line
                      key={`l1-${i}-${targetH}`}
                      x1="44%"
                      y1={y1}
                      x2="68%"
                      y2={y2}
                      stroke={isHighlighted ? '#22d3ee' : '#334155'}
                      strokeWidth={isHighlighted ? '1.5' : '0.6'}
                      strokeOpacity={isHighlighted ? '0.8' : '0.2'}
                    />
                  );
                });
              })}

              {/* Sample connections between H2 -> Output */}
              {Array.from({ length: h2Count }).map((_, i) => {
                const y1 = 50 + i * (420 / h2Count);
                return Array.from({ length: 3 }).map((_, j) => {
                  const targetOut = (i * 2 + j) % outCount;
                  const y2 = 30 + targetOut * (460 / outCount);
                  const isHighlighted = pulseStage === 2 || pulseStage === 3;
                  const isWinner = forwardResult?.predictedClass === targetOut;
                  return (
                    <line
                      key={`l2-${i}-${targetOut}`}
                      x1="70%"
                      y1={y1}
                      x2="88%"
                      y2={y2}
                      stroke={isWinner ? '#34d399' : isHighlighted ? '#38bdf8' : '#334155'}
                      strokeWidth={isWinner ? '2' : isHighlighted ? '1.2' : '0.6'}
                      strokeOpacity={isWinner ? '0.9' : isHighlighted ? '0.6' : '0.15'}
                    />
                  );
                });
              })}
            </svg>

            {/* Column 1: Input Layer (Sensores 28x28) */}
            <div className="z-10 flex flex-col items-center justify-between h-full w-44 py-2">
              <div className="text-center mb-1">
                <span className="text-xs font-semibold text-slate-300 block">Capa de Entrada</span>
                <span className="text-[11px] text-slate-400 font-mono">784 píxeles (X)</span>
              </div>
              <div className="flex flex-col justify-around h-[440px] w-full">
                {inputSensorPoints.map((pt, i) => {
                  const pixelVal = currentSample.data[pt.idx] || 0;
                  const isActive = pixelVal > 0.4;
                  return (
                    <div
                      key={`inp-${i}`}
                      className="flex items-center gap-2 group cursor-pointer"
                      title={`${pt.label} (Píxel ${pt.idx}): ${pixelVal.toFixed(2)}`}
                    >
                      <div
                        className={`w-4 h-4 rounded-full border flex items-center justify-center transition-all ${
                          pulseStage === 0
                            ? 'ring-2 ring-cyan-400 scale-125'
                            : ''
                        }`}
                        style={{
                          backgroundColor: `rgb(${Math.round(pixelVal * 255)}, ${Math.round(
                            pixelVal * 255
                          )}, ${Math.round(pixelVal * 255)})`,
                          borderColor: isActive ? '#38bdf8' : '#475569',
                        }}
                      />
                      <span className="text-[11px] text-slate-400 truncate max-w-[110px] group-hover:text-slate-200">
                        {pt.label}
                      </span>
                    </div>
                  );
                })}
              </div>
              <span className="text-[10px] text-slate-400 italic">Muestreo de 12 nodos representativos</span>
            </div>

            {/* Column 2: Hidden Layer 1 (ReLU) */}
            <div className="z-10 flex flex-col items-center justify-between h-full w-28 py-2">
              <div className="text-center mb-1">
                <span className="text-xs font-semibold text-slate-300 block">Capa Oculta 1</span>
                <span className="text-[11px] text-cyan-400 font-mono">32 Neuronas (ReLU)</span>
              </div>
              <div className="flex flex-col justify-around h-[460px]">
                {Array.from({ length: h1Count }).map((_, i) => {
                  const cache = forwardResult?.layerCaches[0];
                  const actVal = cache ? cache.activations[i] || 0 : 0;
                  const isSelected = selectedNeuron?.layer === 0 && selectedNeuron?.index === i;
                  const brightness = Math.min(1, actVal * 0.7);

                  return (
                    <button
                      key={`h1-${i}`}
                      onClick={() => setSelectedNeuron({ layer: 0, index: i })}
                      className={`w-5 h-5 rounded-full border transition-all flex items-center justify-center ${
                        isSelected
                          ? 'ring-2 ring-amber-400 scale-125 border-amber-300'
                          : pulseStage === 1
                          ? 'ring-2 ring-cyan-400 scale-110'
                          : 'border-slate-700 hover:scale-110'
                      }`}
                      style={{
                        backgroundColor:
                          brightness > 0.05
                            ? `rgba(6, 182, 212, ${Math.max(0.2, brightness)})`
                            : '#0f172a',
                      }}
                      title={`H1 Neurona ${i + 1} | Activación: ${actVal.toFixed(3)}`}
                    >
                      <span className="text-[8px] font-mono text-slate-300 opacity-60">{i + 1}</span>
                    </button>
                  );
                })}
              </div>
              <span className="text-[10px] text-slate-400 font-mono">a = max(0, z)</span>
            </div>

            {/* Column 3: Hidden Layer 2 (ReLU) */}
            <div className="z-10 flex flex-col items-center justify-between h-full w-28 py-2">
              <div className="text-center mb-1">
                <span className="text-xs font-semibold text-slate-300 block">Capa Oculta 2</span>
                <span className="text-[11px] text-cyan-400 font-mono">16 Neuronas (ReLU)</span>
              </div>
              <div className="flex flex-col justify-around h-[420px]">
                {Array.from({ length: h2Count }).map((_, i) => {
                  const cache = forwardResult?.layerCaches[1];
                  const actVal = cache ? cache.activations[i] || 0 : 0;
                  const isSelected = selectedNeuron?.layer === 1 && selectedNeuron?.index === i;
                  const brightness = Math.min(1, actVal * 0.8);

                  return (
                    <button
                      key={`h2-${i}`}
                      onClick={() => setSelectedNeuron({ layer: 1, index: i })}
                      className={`w-5 h-5 rounded-full border transition-all flex items-center justify-center ${
                        isSelected
                          ? 'ring-2 ring-amber-400 scale-125 border-amber-300'
                          : pulseStage === 2
                          ? 'ring-2 ring-cyan-400 scale-110'
                          : 'border-slate-700 hover:scale-110'
                      }`}
                      style={{
                        backgroundColor:
                          brightness > 0.05
                            ? `rgba(56, 189, 248, ${Math.max(0.25, brightness)})`
                            : '#0f172a',
                      }}
                      title={`H2 Neurona ${i + 1} | Activación: ${actVal.toFixed(3)}`}
                    >
                      <span className="text-[8px] font-mono text-slate-300 opacity-60">{i + 1}</span>
                    </button>
                  );
                })}
              </div>
              <span className="text-[10px] text-slate-400 font-mono">a = max(0, z)</span>
            </div>

            {/* Column 4: Output Layer (Softmax) */}
            <div className="z-10 flex flex-col items-center justify-between h-full w-56 py-2">
              <div className="text-center mb-1">
                <span className="text-xs font-semibold text-slate-300 block">Capa de Salida</span>
                <span className="text-[11px] text-emerald-400 font-mono">10 Clases (Softmax)</span>
              </div>
              <div className="flex flex-col justify-around h-[460px] w-full">
                {FASHION_CLASSES.map((fc) => {
                  const prob = forwardResult ? forwardResult.probabilities[fc.id] || 0 : 0;
                  const isWinner = forwardResult?.predictedClass === fc.id;
                  const isTarget = currentSample.classId === fc.id;
                  const isSelected =
                    selectedNeuron?.layer === (forwardResult?.layerCaches.length ? forwardResult.layerCaches.length - 1 : 2) &&
                    selectedNeuron?.index === fc.id;

                  return (
                    <div
                      key={`out-${fc.id}`}
                      onClick={() =>
                        setSelectedNeuron({
                          layer: network.weights.length - 1,
                          index: fc.id,
                        })
                      }
                      className={`flex items-center justify-between p-1 px-2 rounded-lg cursor-pointer transition-all border ${
                        isSelected
                          ? 'bg-amber-950/40 border-amber-400/80 ring-1 ring-amber-400'
                          : isWinner
                          ? 'bg-emerald-950/40 border-emerald-500/70'
                          : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-2 overflow-hidden">
                        <span
                          className={`w-3.5 h-3.5 rounded-full flex items-center justify-center text-[8px] font-bold ${
                            isWinner
                              ? 'bg-emerald-400 text-slate-950'
                              : isTarget
                              ? 'bg-cyan-500 text-slate-950'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {fc.id}
                        </span>
                        <span className="text-xs text-slate-200 truncate">{fc.nameEn}</span>
                      </div>

                      {/* Probability bar & percentage */}
                      <div className="flex items-center gap-1.5 shrink-0">
                        <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all duration-300 ${
                              isWinner ? 'bg-emerald-400' : 'bg-slate-600'
                            }`}
                            style={{ width: `${Math.round(prob * 100)}%` }}
                          />
                        </div>
                        <span className="text-[10px] font-mono tabular-nums text-slate-300 w-8 text-right">
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
              <span className="text-[10px] text-slate-400 font-mono">P(y=c|x) = exp(z_c)/∑</span>
            </div>
          </div>
        </div>

        {/* Math & Neuron Inspector Card (1 Col) */}
        <div className="p-5 bg-slate-900/90 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-white">Inspección Matemática</h3>
            </div>
            <span className="text-[11px] text-slate-400">Clic en cualquier neurona</span>
          </div>

          {neuronInsight ? (
            <div className="space-y-4 text-xs">
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-cyan-300">{neuronInsight.label}</span>
                  <span className="text-[11px] text-slate-400">Capa {neuronInsight.layerIndex + 1}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 font-mono text-[11px]">
                  <div>
                    <span className="text-slate-400 block">Suma Ponderada (z):</span>
                    <span className="text-white font-medium">{neuronInsight.z.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Sesgo (b):</span>
                    <span className="text-slate-300">{neuronInsight.b.toFixed(4)}</span>
                  </div>
                  <div className="col-span-2 pt-1 border-t border-slate-800">
                    <span className="text-slate-400 block">Activación a = f(z):</span>
                    <span className="text-emerald-400 font-semibold text-sm">
                      {neuronInsight.activation.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Formula breakdown */}
              <div>
                <span className="text-slate-400 block mb-1 font-medium">Ecuación de la neurona:</span>
                <div className="p-2 bg-slate-950 font-mono text-[11px] text-cyan-200 rounded border border-slate-800/80">
                  z = ∑(w_i · x_i) + b<br />
                  a = {neuronInsight.layerIndex === network.weights.length - 1 ? 'Softmax(z)' : 'ReLU(z)'}
                </div>
              </div>

              {/* Major Positive Influences */}
              <div>
                <span className="text-slate-300 block mb-1 font-semibold text-[11px]">
                  Mayores Aportes Positivos (+)
                </span>
                <div className="space-y-1">
                  {neuronInsight.topPositiveWeights.map((w, idx) => (
                    <div
                      key={`pos-${idx}`}
                      className="flex items-center justify-between p-1.5 bg-slate-950/60 rounded border border-slate-800/50 font-mono text-[10px]"
                    >
                      <span className="text-slate-400">Entrada #{w.sourceIndex}</span>
                      <span className="text-emerald-400 font-medium">+{w.contribution.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Major Negative Influences */}
              <div>
                <span className="text-slate-300 block mb-1 font-semibold text-[11px]">
                  Mayores Aportes Negativos (-)
                </span>
                <div className="space-y-1">
                  {neuronInsight.topNegativeWeights.map((w, idx) => (
                    <div
                      key={`neg-${idx}`}
                      className="flex items-center justify-between p-1.5 bg-slate-950/60 rounded border border-slate-800/50 font-mono text-[10px]"
                    >
                      <span className="text-slate-400">Entrada #{w.sourceIndex}</span>
                      <span className="text-rose-400 font-medium">{w.contribution.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-6 text-center text-slate-400 space-y-2">
              <HelpCircle className="w-8 h-8 text-slate-600 mx-auto" />
              <p className="text-xs">
                Selecciona cualquier nodo en el gráfico (Capa Oculta 1, Capa Oculta 2 o Salida) para descomponer su cálculo exacto de pesos y sesgos.
              </p>
            </div>
          )}

          {/* Quick Inference Verdict Card */}
          {forwardResult && (
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs space-y-1.5 mt-4">
              <span className="text-slate-400 block font-medium">Veredicto de Clasificación</span>
              <div className="flex items-center justify-between">
                <span className="text-white font-semibold">
                  {FASHION_CLASSES[forwardResult.predictedClass].name}
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                    isCorrect
                      ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                      : 'bg-amber-950 text-amber-300 border border-amber-800'
                  }`}
                >
                  {isCorrect ? 'Correcto' : 'Error'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Confianza: {(forwardResult.confidence * 100).toFixed(1)}%
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

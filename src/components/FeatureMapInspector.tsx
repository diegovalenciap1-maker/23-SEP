import React, { useState } from 'react';
import { NeuralNetwork } from '../lib/neuralNetwork';
import { Eye, Info, Sparkles } from 'lucide-react';

interface FeatureMapInspectorProps {
  network: NeuralNetwork;
}

export const FeatureMapInspector: React.FC<FeatureMapInspectorProps> = ({ network }) => {
  const [selectedNeuronIdx, setSelectedNeuronIdx] = useState<number>(0);

  // Hidden Layer 1 weights: network.weights[0] has shape [numNeurons][784]
  const l0Weights = network.weights[0] || [];
  const neuronCount = l0Weights.length;

  const currentNeuronWeights = l0Weights[selectedNeuronIdx] || new Array(784).fill(0);
  const currentBias = network.biases[0]?.[selectedNeuronIdx] || 0;

  // Compute min, max, and abs max for normalization
  let maxAbs = 0.001;
  for (let i = 0; i < currentNeuronWeights.length; i++) {
    const absVal = Math.abs(currentNeuronWeights[i]);
    if (absVal > maxAbs) maxAbs = absVal;
  }

  // Feature templates descriptions
  const getFilterInterpretation = (idx: number): { title: string; desc: string } => {
    const mode = idx % 8;
    switch (mode) {
      case 0:
        return {
          title: 'Detector de Cuello y Hombros',
          desc: 'Excita neuronas con peso positivo en los hombros y negativo en el hueco del cuello (típico de camisetas y camisas).',
        };
      case 1:
        return {
          title: 'Detector de Perneras Paralelas',
          desc: 'Sensible a dos columnas verticales simétricas separadas por un espacio central oscuro (especializado en pantalones).',
        };
      case 2:
        return {
          title: 'Detector de Suela y Base Horizontal',
          desc: 'Responde intensamente a píxeles brillantes en las filas inferiores (calzado: sandalias, zapatillas y botines).',
        };
      case 3:
        return {
          title: 'Detector de Caña Alta / Tobillo',
          desc: 'Enfocado en la zona media-superior izquierda donde se eleva la caña de los botines.',
        };
      case 4:
        return {
          title: 'Detector de Asas y Asideros Superiores',
          desc: 'Sintonizado para captar tiras arqueadas delgadas en la parte superior (característico de bolsos y carteras).',
        };
      case 5:
        return {
          title: 'Detector de Silueta Acampanada',
          desc: 'Ponderación trapezoidal estrecha arriba y ancha abajo (especializado en vestidos).',
        };
      case 6:
        return {
          title: 'Detector de Mangas Largas Completas',
          desc: 'Activación en los flancos laterales exteriores extendidos hasta los puños (pulóveres y abrigos).',
        };
      default:
        return {
          title: 'Detector de Torso Central Compacto',
          desc: 'Ponderación uniforme en el cuadrante central del pecho.',
        };
    }
  };

  const currentInterpretation = getFilterInterpretation(selectedNeuronIdx);

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Eye className="w-5 h-5 text-cyan-400" />
            <span>¿Qué ve cada neurona oculta? (Campos Receptivos 28×28)</span>
          </h3>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
            Cada neurona de la primera capa oculta posee 784 pesos sinápticos directos a los píxeles. Al reordenar estos 784 números en una cuadrícula $28 \times 28$, revelamos el <strong>patrón visual o plantilla</strong> que la neurona busca en la prenda.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-cyan-400 inline-block" /> Peso + (Excitador)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-rose-500 inline-block" /> Peso - (Inhibidor)
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Gallery of All Hidden Neurons (5 cols) */}
        <div className="lg:col-span-5 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <span className="text-xs font-semibold text-slate-300">
              Neuronas Capa Oculta 1 ({neuronCount})
            </span>
            <span className="text-[11px] text-slate-400">Clic para inspeccionar</span>
          </div>

          <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 max-h-[460px] overflow-y-auto pr-1">
            {Array.from({ length: neuronCount }).map((_, idx) => {
              const weights = l0Weights[idx] || [];
              const isSelected = selectedNeuronIdx === idx;

              return (
                <button
                  key={idx}
                  onClick={() => setSelectedNeuronIdx(idx)}
                  className={`p-1.5 rounded-lg border text-center transition-all ${
                    isSelected
                      ? 'bg-cyan-950/70 border-cyan-400 ring-2 ring-cyan-400/50'
                      : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  {/* Canvas thumbnail of weights */}
                  <canvas
                    ref={(canvas) => {
                      if (canvas && weights.length === 784) {
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                          const imgData = ctx.createImageData(28, 28);
                          for (let i = 0; i < 784; i++) {
                            const w = weights[i];
                            const norm = Math.max(-1, Math.min(1, w / maxAbs));
                            if (norm >= 0) {
                              // Cyan for positive
                              imgData.data[i * 4] = Math.round(6 + norm * 30);
                              imgData.data[i * 4 + 1] = Math.round(182 * norm);
                              imgData.data[i * 4 + 2] = Math.round(212 * norm);
                            } else {
                              // Rose for negative
                              imgData.data[i * 4] = Math.round(244 * -norm);
                              imgData.data[i * 4 + 1] = Math.round(63 * -norm * 0.4);
                              imgData.data[i * 4 + 2] = Math.round(94 * -norm * 0.4);
                            }
                            imgData.data[i * 4 + 3] = 255;
                          }
                          ctx.putImageData(imgData, 0, 0);
                        }
                      }
                    }}
                    width={28}
                    height={28}
                    className="w-full aspect-square rounded bg-black image-rendering-pixelated"
                    style={{ imageRendering: 'pixelated' }}
                  />
                  <span className="text-[10px] font-mono text-slate-400 mt-1 block">
                    N#{idx + 1}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: Enlarged Receptive Field & Explanation (7 cols) */}
        <div className="lg:col-span-7 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <span className="text-xs text-slate-400">Neurona Seleccionada:</span>
              <h4 className="text-lg font-bold text-white">
                Neurona Oculta #{selectedNeuronIdx + 1}
              </h4>
            </div>
            <div className="text-right font-mono text-xs">
              <span className="text-slate-400 block">Sesgo (Bias b):</span>
              <span className="text-cyan-400 font-semibold">{currentBias.toFixed(4)}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            {/* Enlarged 28x28 Heatmap Canvas */}
            <div className="flex flex-col items-center">
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-2xl shadow-2xl">
                <canvas
                  ref={(canvas) => {
                    if (canvas && currentNeuronWeights.length === 784) {
                      const ctx = canvas.getContext('2d');
                      if (ctx) {
                        ctx.fillStyle = '#020617';
                        ctx.fillRect(0, 0, 224, 224);
                        const pixelScale = 8; // 28 * 8 = 224
                        for (let r = 0; r < 28; r++) {
                          for (let c = 0; c < 28; c++) {
                            const idx = r * 28 + c;
                            const w = currentNeuronWeights[idx];
                            const norm = Math.max(-1, Math.min(1, w / maxAbs));
                            if (norm >= 0) {
                              ctx.fillStyle = `rgba(6, 182, 212, ${norm.toFixed(2)})`;
                            } else {
                              ctx.fillStyle = `rgba(244, 63, 94, ${(-norm).toFixed(2)})`;
                            }
                            ctx.fillRect(c * pixelScale, r * pixelScale, pixelScale, pixelScale);
                          }
                        }
                      }
                    }
                  }}
                  width={224}
                  height={224}
                  className="rounded-lg bg-black image-rendering-pixelated"
                  style={{ imageRendering: 'pixelated', width: 224, height: 224 }}
                />
              </div>
              <span className="text-[11px] font-mono text-slate-400 mt-2">
                Receptive Field (28 × 28 px)
              </span>
            </div>

            {/* Interpretation Card */}
            <div className="space-y-4">
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Función Interpretada</span>
                </div>
                <h5 className="text-sm font-bold text-white">
                  {currentInterpretation.title}
                </h5>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {currentInterpretation.desc}
                </p>
              </div>

              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs space-y-1.5">
                <span className="text-slate-400 block font-medium">Principio Biológico y Matemático:</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Similar a cómo las neuronas del córtex visual primario (V1) en mamíferos se especializan en detectar barras de luz en orientaciones específicas (descubrimiento de Hubel y Wiesel), el Perceptrón Multicapa aprende de manera autónoma detectores de bordes al minimizar la pérdida de entropía cruzada.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

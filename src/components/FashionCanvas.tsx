import React, { useState, useRef, useEffect } from 'react';
import { FashionSample, FASHION_CLASSES, FashionClass } from '../data/fashionMnistData';
import { ForwardResult, NeuralNetwork } from '../lib/neuralNetwork';
import { Eraser, Pencil, RotateCcw, Check, Sparkles, Sliders, Eye } from 'lucide-react';

interface FashionCanvasProps {
  currentSample: FashionSample;
  onUpdateSample: (sample: FashionSample) => void;
  samples: FashionSample[];
  network: NeuralNetwork;
  forwardResult: ForwardResult | null;
  onRunForward: () => void;
}

export const FashionCanvas: React.FC<FashionCanvasProps> = ({
  currentSample,
  onUpdateSample,
  samples,
  network,
  forwardResult,
  onRunForward,
}) => {
  const [brushSize, setBrushSize] = useState<number>(2);
  const [isErasing, setIsErasing] = useState<boolean>(false);
  const [hoveredPixel, setHoveredPixel] = useState<{ r: number; c: number; index: number; val: number } | null>(null);
  const [selectedFilterClass, setSelectedFilterClass] = useState<number | 'all'>('all');
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDrawing = useRef<boolean>(false);

  // Sync canvas display when currentSample changes
  useEffect(() => {
    drawSampleToCanvas(currentSample.data);
  }, [currentSample]);

  const drawSampleToCanvas = (data: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 280, 280);

    const pixelSize = 10; // 28 * 10 = 280px
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const val = data[r * 28 + c];
        if (val > 0) {
          const intensity = Math.round(val * 255);
          ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
          ctx.fillRect(c * pixelSize, r * pixelSize, pixelSize, pixelSize);
        }
      }
    }
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    isDrawing.current = true;
    applyPaint(e);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = Math.floor(x / 10);
    const r = Math.floor(y / 10);

    if (r >= 0 && r < 28 && c >= 0 && c < 28) {
      const idx = r * 28 + c;
      setHoveredPixel({
        r,
        c,
        index: idx,
        val: currentSample.data[idx] || 0,
      });
    }

    if (isDrawing.current) {
      applyPaint(e);
    }
  };

  const handlePointerUp = () => {
    if (isDrawing.current) {
      isDrawing.current = false;
      onRunForward();
    }
  };

  const applyPaint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerC = Math.floor(x / 10);
    const centerR = Math.floor(y / 10);

    const newData = [...currentSample.data];
    const targetVal = isErasing ? 0 : 0.95;

    for (let dr = -brushSize + 1; dr <= brushSize - 1; dr++) {
      for (let dc = -brushSize + 1; dc <= brushSize - 1; dc++) {
        const r = centerR + dr;
        const c = centerC + dc;
        if (r >= 0 && r < 28 && c >= 0 && c < 28) {
          const dist = Math.sqrt(dr * dr + dc * dc);
          if (dist <= brushSize) {
            const idx = r * 28 + c;
            newData[idx] = Math.max(0, Math.min(1, targetVal));
          }
        }
      }
    }

    onUpdateSample({
      ...currentSample,
      id: `custom-${Date.now()}`,
      label: 'Prenda Personalizada',
      data: newData,
    });
  };

  const handleClear = () => {
    const emptyData = new Array(784).fill(0);
    onUpdateSample({
      ...currentSample,
      id: `clear-${Date.now()}`,
      label: 'Lienzo Vacío',
      data: emptyData,
    });
    setTimeout(onRunForward, 50);
  };

  const handleInvert = () => {
    const invData = currentSample.data.map((v) => (v > 0.05 ? 0 : 0.9));
    onUpdateSample({
      ...currentSample,
      data: invData,
    });
    setTimeout(onRunForward, 50);
  };

  const filteredSamples =
    selectedFilterClass === 'all'
      ? samples
      : samples.filter((s) => s.classId === selectedFilterClass);

  return (
    <div className="space-y-6">
      {/* Category Filter Pills (Functional Buttons) */}
      <div className="flex flex-wrap items-center gap-1.5 p-2 bg-slate-900 border border-slate-800 rounded-xl">
        <span className="text-xs text-slate-400 px-2 font-medium">Clase:</span>
        <button
          onClick={() => setSelectedFilterClass('all')}
          className={`px-3 py-1 text-xs rounded-lg transition-colors whitespace-nowrap ${
            selectedFilterClass === 'all'
              ? 'bg-cyan-500 text-slate-950 font-semibold'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          Todas (10 Clases)
        </button>
        {FASHION_CLASSES.map((fc) => (
          <button
            key={fc.id}
            onClick={() => setSelectedFilterClass(fc.id)}
            className={`px-2.5 py-1 text-xs rounded-lg transition-colors whitespace-nowrap ${
              selectedFilterClass === fc.id
                ? 'bg-cyan-500 text-slate-950 font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            {fc.nameEn}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Interactive 28x28 Canvas */}
        <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-sm font-semibold text-white">Lienzo 28×28 Píxeles</h3>
              <p className="text-xs text-slate-400">
                Dibuja un artículo o elige una muestra de prueba
              </p>
            </div>
            <span className="text-xs font-mono text-cyan-400">784 píxeles escala de grises</span>
          </div>

          {/* Canvas Wrapper */}
          <div className="flex flex-col items-center">
            <div className="relative p-2 bg-slate-950 border border-slate-800 rounded-xl shadow-2xl">
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={() => {
                  isDrawing.current = false;
                  setHoveredPixel(null);
                }}
                className="cursor-crosshair rounded-lg touch-none bg-black image-rendering-pixelated"
                style={{ imageRendering: 'pixelated', width: 280, height: 280 }}
              />

              {/* Grid overlay toggle or hint */}
              <div className="absolute bottom-3 right-3 px-2 py-0.5 bg-slate-900/80 backdrop-blur rounded text-[10px] font-mono text-slate-400">
                28 × 28 px
              </div>
            </div>

            {/* Canvas Tooling Controls */}
            <div className="flex items-center justify-between w-full max-w-[280px] mt-3">
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setIsErasing(false)}
                  className={`p-1.5 rounded transition-colors ${
                    !isErasing ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-white'
                  }`}
                  title="Pincel"
                >
                  <Pencil className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setIsErasing(true)}
                  className={`p-1.5 rounded transition-colors ${
                    isErasing ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-white'
                  }`}
                  title="Borrador"
                >
                  <Eraser className="w-4 h-4" />
                </button>
              </div>

              {/* Brush size buttons */}
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                {[1, 2, 3].map((size) => (
                  <button
                    key={size}
                    onClick={() => setBrushSize(size)}
                    className={`w-6 h-6 rounded flex items-center justify-center font-mono ${
                      brushSize === size
                        ? 'bg-slate-800 text-cyan-400 font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {size}p
                  </button>
                ))}
              </div>

              {/* Clear & Invert buttons */}
              <div className="flex items-center gap-1">
                <button
                  onClick={handleClear}
                  className="px-2.5 py-1.5 text-xs text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
                  title="Limpiar lienzo"
                >
                  Limpiar
                </button>
              </div>
            </div>
          </div>

          {/* Hovered Pixel Inspector */}
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs font-mono flex items-center justify-between">
            {hoveredPixel ? (
              <>
                <span className="text-slate-400">
                  Fila {hoveredPixel.r}, Col {hoveredPixel.c} (Índice #{hoveredPixel.index})
                </span>
                <span className="text-cyan-400 font-semibold">
                  Valor X[{hoveredPixel.index}]: {hoveredPixel.val.toFixed(2)}
                </span>
              </>
            ) : (
              <span className="text-slate-400 italic">Pasa el cursor sobre la cuadrícula para inspeccionar</span>
            )}
          </div>
        </div>

        {/* Center Column: 2D to 1D Flattening Matrix Visualizer */}
        <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-sm font-semibold text-white">Aplanamiento (Flattening 2D $\to$ 1D)</h3>
              <p className="text-xs text-slate-400">
                Transformación de matriz (28, 28) a vector columna (784, 1)
              </p>
            </div>
            <Eye className="w-4 h-4 text-cyan-400" />
          </div>

          {/* Visual unrolling diagram */}
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs space-y-2">
            <p className="text-slate-300 leading-relaxed">
              Un Perceptrón Multicapa (MLP) convencional no comprende geometría bidimensional nativa: requiere que cada imagen se <strong className="text-cyan-300">aplane</strong> en una tira continua de números en el rango $[0, 1]$:
            </p>
            <div className="p-2 bg-slate-900 rounded font-mono text-[11px] text-cyan-200">
              X = [x_0, x_1, x_2, ..., x_783]^T
            </div>
          </div>

          {/* Continuous Strip Stream Representation */}
          <div className="space-y-1.5">
            <span className="text-xs text-slate-400 font-medium block">
              Tira de 784 píxeles normalizados:
            </span>
            <div className="flex flex-wrap gap-0.5 p-2 bg-black rounded-lg border border-slate-800 max-h-52 overflow-y-auto">
              {currentSample.data.map((val, idx) => {
                const isHovered = hoveredPixel?.index === idx;
                const brightness = Math.round(val * 255);
                return (
                  <div
                    key={idx}
                    onMouseEnter={() => {
                      const r = Math.floor(idx / 28);
                      const c = idx % 28;
                      setHoveredPixel({ r, c, index: idx, val });
                    }}
                    className={`w-2.5 h-2.5 transition-all cursor-pointer ${
                      isHovered ? 'ring-2 ring-cyan-400 scale-150 z-10' : ''
                    }`}
                    style={{
                      backgroundColor: `rgb(${brightness}, ${brightness}, ${brightness})`,
                    }}
                    title={`Píxel #${idx} (Fila ${Math.floor(idx / 28)}, Col ${idx % 28}): ${val.toFixed(2)}`}
                  />
                );
              })}
            </div>
          </div>

          {/* Presets Gallery */}
          <div className="space-y-2">
            <span className="text-xs text-slate-400 font-medium block">
              Muestras precargadas ({filteredSamples.length}):
            </span>
            <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto pr-1">
              {filteredSamples.map((s) => (
                <button
                  key={s.id}
                  onClick={() => {
                    onUpdateSample(s);
                    setTimeout(onRunForward, 50);
                  }}
                  className={`flex items-center gap-2 p-2 rounded-lg text-left border transition-colors ${
                    s.id === currentSample.id
                      ? 'bg-cyan-950/60 border-cyan-500/80 text-white'
                      : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="w-8 h-8 rounded bg-black shrink-0 border border-slate-700 flex items-center justify-center font-mono text-[9px] text-cyan-400">
                    {s.classId}
                  </div>
                  <div className="overflow-hidden">
                    <span className="text-xs font-medium block truncate">{s.label}</span>
                    <span className="text-[10px] text-slate-400 block truncate">
                      {FASHION_CLASSES[s.classId].nameEn}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Live Classification Leaderboard */}
        <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-sm font-semibold text-white">Probabilidades de Salida</h3>
              <p className="text-xs text-slate-400">Distribución calculada por función Softmax</p>
            </div>
            <Sparkles className="w-4 h-4 text-emerald-400" />
          </div>

          {/* Top Prediction Highlight Card */}
          {forwardResult && (
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Predicción Principal:</span>
                <span className="text-xs font-mono text-emerald-400 font-bold">
                  {(forwardResult.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <h4 className="text-lg font-bold text-white">
                {FASHION_CLASSES[forwardResult.predictedClass].name}
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                {FASHION_CLASSES[forwardResult.predictedClass].description}
              </p>
            </div>
          )}

          {/* 10 Class Softmax Bars */}
          <div className="space-y-2">
            {FASHION_CLASSES.map((fc) => {
              const prob = forwardResult ? forwardResult.probabilities[fc.id] || 0 : 0;
              const isWinner = forwardResult?.predictedClass === fc.id;
              const isTarget = currentSample.classId === fc.id;

              return (
                <div key={fc.id} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span
                      className={`truncate ${
                        isWinner ? 'text-emerald-400 font-semibold' : 'text-slate-300'
                      }`}
                    >
                      {fc.id}. {fc.nameEn}
                    </span>
                    <span className="font-mono text-slate-400 tabular-nums">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800/80">
                    <div
                      className={`h-full transition-all duration-300 rounded-full ${
                        isWinner
                          ? 'bg-gradient-to-r from-emerald-500 to-cyan-400'
                          : 'bg-slate-700'
                      }`}
                      style={{ width: `${Math.max(1, prob * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

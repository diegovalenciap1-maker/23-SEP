import React, { useState } from 'react';
import { Activations, ActivationType } from '../lib/neuralNetwork';
import { Lightbulb, BookOpen, ChevronRight, Sliders, CheckCircle, AlertTriangle } from 'lucide-react';

export const ConceptExplainer: React.FC = () => {
  const [selectedConcept, setSelectedConcept] = useState<number>(1);

  // Concept 1 interactive states (Single perceptron 2D boundary)
  const [w1, setW1] = useState<number>(1.2);
  const [w2, setW2] = useState<number>(-0.8);
  const [bias, setBias] = useState<number>(0.2);

  // Concept 4 interactive states (Activation curve explorer)
  const [selectedAct, setSelectedAct] = useState<ActivationType>('relu');
  const [testZ, setTestZ] = useState<number>(1.5);

  // Concept 5 interactive states (Learning rate slider & gradient step)
  const [learningRateSim, setLearningRateSim] = useState<number>(0.2);
  const [simStep, setSimStep] = useState<number>(0);

  // Activation plot generation
  const actInfo = Activations[selectedAct];
  const plotPoints: { z: number; fz: number; dfz: number }[] = [];
  for (let z = -4; z <= 4; z += 0.2) {
    plotPoints.push({
      z: Math.round(z * 10) / 10,
      fz: actInfo.fn(z),
      dfz: actInfo.deriv(z),
    });
  }

  return (
    <div className="space-y-6">
      {/* Concept Selector Navigation */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { id: 1, title: '1. El Perceptrón', sub: 'Anatomía de una neurona' },
          { id: 2, title: '2. ¿Por qué Multicapa?', sub: 'Separabilidad y capas ocultas' },
          { id: 3, title: '3. Forward Pass', sub: 'Cálculo matricial hacia adelante' },
          { id: 4, title: '4. Activaciones & Softmax', sub: 'No linealidad y probabilidades' },
          { id: 5, title: '5. Retropropagación', sub: 'Regla de la cadena y gradientes' },
        ].map((c) => {
          const isActive = selectedConcept === c.id;
          return (
            <button
              key={c.id}
              onClick={() => setSelectedConcept(c.id)}
              className={`p-3 rounded-xl border text-left transition-all ${
                isActive
                  ? 'bg-cyan-950/70 border-cyan-500/80 shadow-lg shadow-cyan-950/40 ring-1 ring-cyan-500'
                  : 'bg-slate-900 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <span className={`text-xs font-semibold block ${isActive ? 'text-cyan-300' : 'text-slate-300'}`}>
                {c.title}
              </span>
              <span className="text-[11px] text-slate-400 block truncate mt-0.5">
                {c.sub}
              </span>
            </button>
          );
        })}
      </div>

      {/* Concept 1: La Neurona Artificial */}
      {selectedConcept === 1 && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-cyan-400" />
              <span>1. La Neurona Artificial: Fundamento Matemático</span>
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              El <strong>Perceptrón</strong> (propuesto por Frank Rosenblatt en 1958) es la unidad mínima de cómputo en una red neuronal. Recibe un vector de entradas $X = [x_1, x_2, \dots, x_n]$, pondera cada señal por un peso sináptico $w_i$, suma un sesgo (<em>bias</em>) $b$, y pasa el resultado por una función de activación no lineal:
            </p>

            {/* Formula Block */}
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 font-mono text-xs space-y-2 text-cyan-300">
              <div className="text-slate-400">// Paso 1: Combinación lineal ponderada</div>
              <div>z = (w₁ · x₁) + (w₂ · x₂) + ... + (wₙ · xₙ) + b = ∑(wᵢ · xᵢ) + b</div>
              <div className="text-slate-400 pt-2">// Paso 2: Activación no lineal</div>
              <div>a = σ(z)</div>
            </div>

            <div className="space-y-2 text-xs text-slate-300">
              <h4 className="font-semibold text-white">¿Qué representa cada elemento en Fashion-MNIST?</h4>
              <ul className="list-disc pl-5 space-y-1.5 text-slate-400">
                <li>
                  <strong className="text-slate-200">Entradas ($x_i$):</strong> Cada uno de los 784 píxeles de la prenda (0.0 = fondo negro, 1.0 = tejido blanco).
                </li>
                <li>
                  <strong className="text-slate-200">Pesos ($w_i$):</strong> La fuerza e importancia que la neurona asigna a ese píxel. Un peso positivo indica que la presencia de brillo refuerza la activación; un peso negativo indica que el brillo en esa zona la desactiva (por ejemplo, brillo en el cuello desactiva la clase "Pulóver").
                </li>
                <li>
                  <strong className="text-slate-200">Sesgo ($b$):</strong> El umbral de disparo intrínseco de la neurona, independiente de los píxeles.
                </li>
              </ul>
            </div>
          </div>

          {/* Interactive Perceptron 2D Boundary Sandbox */}
          <div className="lg:col-span-5 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h4 className="text-sm font-semibold text-white">Frontera de Decisión Lineal</h4>
              <span className="text-xs font-mono text-cyan-400">w₁x₁ + w₂x₂ + b = 0</span>
            </div>

            {/* SVG 2D Decision Plot */}
            <div className="relative w-full h-56 bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center">
              <svg className="w-full h-full" viewBox="-120 -120 240 240">
                {/* Axes */}
                <line x1="-110" y1="0" x2="110" y2="0" stroke="#334155" strokeWidth="1" />
                <line x1="0" y1="-110" x2="0" y2="110" stroke="#334155" strokeWidth="1" />
                
                {/* Decision Line: w1*x + w2*y + bias*50 = 0 -> y = (-w1*x - bias*50) / w2 */}
                {(() => {
                  const safeW2 = Math.abs(w2) < 0.05 ? 0.05 : w2;
                  const xA = -100;
                  const yA = (-w1 * xA - bias * 40) / safeW2;
                  const xB = 100;
                  const yB = (-w1 * xB - bias * 40) / safeW2;
                  return (
                    <line
                      x1={xA}
                      y1={-yA}
                      x2={xB}
                      y2={-yB}
                      stroke="#06b6d4"
                      strokeWidth="2.5"
                    />
                  );
                })()}

                {/* Sample items: Sandalias (cyan) vs Botines (rose) */}
                <circle cx="-50" cy="-40" r="5" fill="#38bdf8" />
                <circle cx="-60" cy="-20" r="5" fill="#38bdf8" />
                <circle cx="-35" cy="-60" r="5" fill="#38bdf8" />
                <text x="-65" y="-50" fill="#38bdf8" fontSize="9" fontFamily="monospace">Sandalia</text>

                <circle cx="50" cy="40" r="5" fill="#fb7185" />
                <circle cx="65" cy="20" r="5" fill="#fb7185" />
                <circle cx="35" cy="55" r="5" fill="#fb7185" />
                <text x="45" y="70" fill="#fb7185" fontSize="9" fontFamily="monospace">Botín</text>
              </svg>
            </div>

            {/* Sliders for w1, w2, bias */}
            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between text-slate-400">
                  <span>Peso w₁ (Anchura de prenda):</span>
                  <span className="font-mono text-cyan-400">{w1.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-3"
                  max="3"
                  step="0.1"
                  value={w1}
                  onChange={(e) => setW1(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400">
                  <span>Peso w₂ (Altura de suela/caña):</span>
                  <span className="font-mono text-cyan-400">{w2.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-3"
                  max="3"
                  step="0.1"
                  value={w2}
                  onChange={(e) => setW2(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400">
                  <span>Sesgo b (Desplazamiento del umbral):</span>
                  <span className="font-mono text-cyan-400">{bias.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={bias}
                  onChange={(e) => setBias(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Concept 2: ¿Por qué Multicapa? */}
      {selectedConcept === 2 && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-cyan-400" />
              <span>2. ¿Por qué "Multicapa"? Superando la Barrera Lineal (XOR)</span>
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              En 1969, Marvin Minsky y Seymour Papert publicaron el célebre libro <em>Perceptrons</em>, demostrando que un perceptrón simple es incapaz de resolver problemas no linealmente separables (como la función lógica <strong>XOR</strong>).
            </p>
            <p className="text-sm text-slate-300 leading-relaxed">
              En Fashion-MNIST, clasificar un <em>Vestido</em> frente a una <em>Camisa</em> o un <em>Bolso</em> exige combinar múltiples condiciones lógicas:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <span className="font-semibold text-rose-400 block mb-1">
                  Perceptrón Simple (1 Capa)
                </span>
                <p className="text-slate-400">
                  Solo puede trazar <strong>un hiperplano recto</strong>. Falla estrepitosamente si la clase requiere detectar que hay tejido arriba Y ausencia de tejido al centro pero presencia abajo.
                </p>
              </div>

              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <span className="font-semibold text-emerald-400 block mb-1">
                  Perceptrón Multicapa (MLP)
                </span>
                <p className="text-slate-400">
                  Al interconectar <strong>capas ocultas con funciones de activación no lineales</strong>, la red compone múltiples hiperplanos formando regiones de decisión convexas y no convexas arbitrarias (Teorema de Aproximación Universal).
                </p>
              </div>
            </div>

            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs space-y-2">
              <h4 className="font-semibold text-cyan-300">Jerarquía de Representación en las Capas:</h4>
              <div className="space-y-1.5 font-mono text-[11px] text-slate-300">
                <div>• Capa 1 (Oculta): Detecta bordes primitivos (líneas verticales, tiras, suelas).</div>
                <div>• Capa 2 (Oculta): Combina bordes en partes anatómicas (mangas, solapas, perneras).</div>
                <div>• Capa 3 (Salida): Ensambla las partes en categorías de moda (Vestido, Pantalón, Botín).</div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h4 className="text-sm font-semibold text-white">Visualización del Problema XOR</h4>
            <div className="w-full h-56 bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-center p-4">
              <svg className="w-48 h-48" viewBox="-10 -10 120 120">
                {/* Grid */}
                <line x1="0" y1="100" x2="100" y2="100" stroke="#334155" strokeWidth="2" />
                <line x1="0" y1="100" x2="0" y2="0" stroke="#334155" strokeWidth="2" />

                {/* XOR Points */}
                {/* (0,0) -> 0 */}
                <circle cx="10" cy="90" r="7" fill="#64748b" />
                <text x="7" y="93" fill="white" fontSize="8" fontWeight="bold">0</text>

                {/* (1,1) -> 0 */}
                <circle cx="90" cy="10" r="7" fill="#64748b" />
                <text x="87" y="13" fill="white" fontSize="8" fontWeight="bold">0</text>

                {/* (0,1) -> 1 */}
                <circle cx="10" cy="10" r="7" fill="#06b6d4" />
                <text x="7" y="13" fill="black" fontSize="8" fontWeight="bold">1</text>

                {/* (1,0) -> 1 */}
                <circle cx="90" cy="90" r="7" fill="#06b6d4" />
                <text x="87" y="93" fill="black" fontSize="8" fontWeight="bold">1</text>

                {/* Attempted single line (failing) */}
                <line x1="-5" y1="60" x2="105" y2="60" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="3,3" />
              </svg>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Es imposible separar los puntos <strong className="text-cyan-400">1</strong> de los puntos <strong className="text-slate-400">0</strong> con una única línea recta continua. Se requieren al menos 2 líneas coordinadas por una capa oculta.
            </p>
          </div>
        </div>
      )}

      {/* Concept 3: Propagación hacia Adelante */}
      {selectedConcept === 3 && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-cyan-400" />
              <span>3. Propagación hacia Adelante (Forward Pass)</span>
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              El <strong>Forward Pass</strong> es el proceso determinista mediante el cual los píxeles de entrada fluyen a través de las matrices de pesos para generar una predicción de moda.
            </p>

            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3 font-mono text-xs text-cyan-300">
              <div className="text-slate-400">// Para cada capa l = 1 ... L:</div>
              <div>Z^[l] = W^[l] · A^[l-1] + b^[l]</div>
              <div>A^[l] = g^[l](Z^[l])</div>
              <div className="pt-2 text-slate-400">// Capa final L con función Softmax:</div>
              <div>ŷ = Softmax(Z^[L])</div>
            </div>

            <div className="space-y-2 text-xs text-slate-300">
              <h4 className="font-semibold text-white">Dimensiones en nuestra arquitectura:</h4>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <div className="p-2.5 bg-slate-950 rounded border border-slate-800">
                  <span className="text-slate-400 block">Capa 1:</span>
                  <span className="font-mono text-white text-[11px]">W₁: (32 × 784)</span>
                  <span className="font-mono text-slate-400 block text-[10px]">25,088 pesos</span>
                </div>
                <div className="p-2.5 bg-slate-950 rounded border border-slate-800">
                  <span className="text-slate-400 block">Capa 2:</span>
                  <span className="font-mono text-white text-[11px]">W₂: (16 × 32)</span>
                  <span className="font-mono text-slate-400 block text-[10px]">512 pesos</span>
                </div>
                <div className="p-2.5 bg-slate-950 rounded border border-slate-800">
                  <span className="text-slate-400 block">Capa Salida:</span>
                  <span className="font-mono text-white text-[11px]">W_out: (10 × 16)</span>
                  <span className="font-mono text-slate-400 block text-[10px]">160 pesos</span>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h4 className="text-sm font-semibold text-white">Flujo de Señales en Tiempo Real</h4>
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs space-y-3 font-mono">
              <div className="flex items-center justify-between text-slate-300">
                <span>Entrada 28×28:</span>
                <span className="text-cyan-400">784 valores ∈ [0, 1]</span>
              </div>
              <div className="text-center text-slate-500">↓ Multiplicación Matricial</div>
              <div className="flex items-center justify-between text-slate-300">
                <span>Pre-activación Z₁:</span>
                <span className="text-cyan-400">32 valores reales</span>
              </div>
              <div className="text-center text-slate-500">↓ ReLU: max(0, z)</div>
              <div className="flex items-center justify-between text-slate-300">
                <span>Activación A₁:</span>
                <span className="text-emerald-400">32 valores no negativos</span>
              </div>
              <div className="text-center text-slate-500">↓ Capa 2 & Salida Softmax</div>
              <div className="flex items-center justify-between text-slate-300 font-bold">
                <span>Probabilidades ŷ:</span>
                <span className="text-cyan-300">∑ p_i = 1.000</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Concept 4: Activaciones y Softmax */}
      {selectedConcept === 4 && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-cyan-400" />
              <span>4. Funciones de Activación y Softmax</span>
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              Sin funciones de activación no lineales, múltiples capas colapsarían matemáticamente en una sola transformación afín lineal (W₂ · W₁ · X = W_eq · X), perdiendo toda capacidad de aprender formas complejas de prendas.
            </p>

            {/* Activation Selector Tabs */}
            <div className="flex flex-wrap gap-2">
              {(['relu', 'sigmoid', 'tanh', 'leaky_relu'] as ActivationType[]).map((act) => (
                <button
                  key={act}
                  onClick={() => setSelectedAct(act)}
                  className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                    selectedAct === act
                      ? 'bg-cyan-500 text-slate-950 font-bold'
                      : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-white'
                  }`}
                >
                  {Activations[act].name}
                </button>
              ))}
            </div>

            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-cyan-300 text-sm">{actInfo.name}</span>
                <span className="font-mono text-xs text-slate-400">{actInfo.formula}</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">{actInfo.desc}</p>
            </div>

            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2 text-xs">
              <h4 className="font-semibold text-emerald-400">¿Por qué Softmax en la Capa de Salida?</h4>
              <p className="text-slate-300 leading-relaxed">
                A diferencia de la Sigmoide (que trata cada neurona independientemente), <strong>Softmax</strong> normaliza el vector exponencialmente obligando a que la suma total sea exactamente <strong>1.0 (100%)</strong>. Esto representa una verdadera distribución de probabilidad multiclase donde las prendas compiten entre sí.
              </p>
            </div>
          </div>

          {/* Interactive Curve & Derivative Viewer */}
          <div className="lg:col-span-5 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h4 className="text-sm font-semibold text-white">Curva f(z) y Derivada f'(z)</h4>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-cyan-400">― f(z)</span>
                <span className="text-rose-400">·· f'(z)</span>
              </div>
            </div>

            {/* SVG Plot */}
            <div className="w-full h-52 bg-slate-950 rounded-xl border border-slate-800 relative flex items-center justify-center">
              <svg className="w-full h-full" viewBox="-50 -50 100 100">
                {/* Axes */}
                <line x1="-45" y1="0" x2="45" y2="0" stroke="#334155" strokeWidth="0.8" />
                <line x1="0" y1="-45" x2="0" y2="45" stroke="#334155" strokeWidth="0.8" />

                {/* Function Curve f(z) */}
                <polyline
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2"
                  points={plotPoints
                    .map((p) => `${p.z * 10},${-Math.min(45, Math.max(-45, p.fz * 20))}`)
                    .join(' ')}
                />

                {/* Derivative Curve f'(z) */}
                <polyline
                  fill="none"
                  stroke="#fb7185"
                  strokeWidth="1.5"
                  strokeDasharray="2,2"
                  points={plotPoints
                    .map((p) => `${p.z * 10},${-Math.min(45, Math.max(-45, p.dfz * 20))}`)
                    .join(' ')}
                />

                {/* Current probe point */}
                <circle
                  cx={testZ * 10}
                  cy={-Math.min(45, Math.max(-45, actInfo.fn(testZ) * 20))}
                  r="3.5"
                  fill="#ffffff"
                  stroke="#06b6d4"
                  strokeWidth="1.5"
                />
              </svg>
            </div>

            {/* Slider to probe z */}
            <div className="space-y-2 text-xs">
              <div className="flex justify-between text-slate-400">
                <span>Valor de prueba (z):</span>
                <span className="font-mono text-white">{testZ.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-4"
                max="4"
                step="0.1"
                value={testZ}
                onChange={(e) => setTestZ(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <div className="grid grid-cols-2 gap-2 pt-1 font-mono text-[11px]">
                <div className="p-2 bg-slate-950 rounded border border-slate-800">
                  <span className="text-slate-400 block">f(z):</span>
                  <span className="text-cyan-400 font-bold">{actInfo.fn(testZ).toFixed(4)}</span>
                </div>
                <div className="p-2 bg-slate-950 rounded border border-slate-800">
                  <span className="text-slate-400 block">Derivada f'(z):</span>
                  <span className="text-rose-400 font-bold">{actInfo.deriv(testZ).toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Concept 5: Retropropagación y Descenso del Gradiente */}
      {selectedConcept === 5 && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-cyan-400" />
              <span>5. Retropropagación (Backpropagation) y Descenso del Gradiente</span>
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              ¿Cómo sabe la red qué peso modificar cuando confunde una <em>Camisa</em> con una <em>Camiseta</em>? Mediante la <strong>Regla de la Cadena</strong> del cálculo diferencial.
            </p>

            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 font-mono text-xs space-y-2 text-cyan-300">
              <div className="text-slate-400">// Regla de la cadena para actualizar un peso w_ij:</div>
              <div>∂L/∂w_ij = (∂L/∂a) · (∂a/∂z) · (∂z/∂w_ij)</div>
              <div className="text-slate-400 pt-2">// Actualización con Descenso del Gradiente:</div>
              <div>w_ij ← w_ij - η · (∂L/∂w_ij)</div>
            </div>

            <div className="space-y-2 text-xs text-slate-300">
              <h4 className="font-semibold text-white">Componentes Clave:</h4>
              <ul className="list-disc pl-5 space-y-1.5 text-slate-400">
                <li>
                  <strong className="text-slate-200">Función de Pérdida (Cross-Entropy):</strong> Cuantifica el error entre la predicción y la etiqueta real: L = -log(ŷ_real).
                </li>
                <li>
                  <strong className="text-slate-200">Tasa de Aprendizaje ($\eta$):</strong> Determina el tamaño del paso hacia el mínimo del error. Si es muy grande, oscila y diverge; si es muy pequeña, tarda horas en converger.
                </li>
                <li>
                  <strong className="text-slate-200">Momento (Momentum):</strong> Acelera el descenso en pendientes pronunciadas y amortigua oscilaciones, acumulando una fracción de la velocidad previa.
                </li>
              </ul>
            </div>
          </div>

          {/* Interactive Learning Rate Simulator */}
          <div className="lg:col-span-5 p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h4 className="text-sm font-semibold text-white">Simulador de Tasa de Aprendizaje (η)</h4>
              <span className="text-xs font-mono text-cyan-400">L(w) = w²</span>
            </div>

            {/* SVG Parabola and Descent Steps */}
            <div className="w-full h-52 bg-slate-950 rounded-xl border border-slate-800 relative flex items-center justify-center">
              <svg className="w-full h-full" viewBox="-60 -10 120 70">
                {/* Parabola: y = 0.04 * x^2 */}
                <path
                  d="M -50 50 Q 0 0 50 50"
                  fill="none"
                  stroke="#334155"
                  strokeWidth="2"
                />
                <circle cx="0" cy="0" r="3" fill="#34d399" />
                <text x="-15" y="-3" fill="#34d399" fontSize="6" fontFamily="monospace">Mínimo Global</text>

                {/* Simulate steps starting from w = 40 */}
                {(() => {
                  let currentW = 40;
                  const points: { x: number; y: number }[] = [];
                  points.push({ x: currentW, y: 0.025 * currentW * currentW });

                  for (let s = 0; s < 5; s++) {
                    const grad = 2 * currentW;
                    currentW = currentW - learningRateSim * grad;
                    points.push({ x: currentW, y: 0.025 * currentW * currentW });
                  }

                  return (
                    <>
                      {points.map((p, i) => {
                        if (i === 0) return null;
                        const prev = points[i - 1];
                        return (
                          <line
                            key={i}
                            x1={prev.x}
                            y1={prev.y}
                            x2={p.x}
                            y2={p.y}
                            stroke="#38bdf8"
                            strokeWidth="1.5"
                            strokeDasharray="2,2"
                          />
                        );
                      })}
                      {points.map((p, i) => (
                        <circle
                          key={`pt-${i}`}
                          cx={p.x}
                          cy={p.y}
                          r={i === 0 ? '4' : '2.5'}
                          fill={i === points.length - 1 ? '#f59e0b' : '#38bdf8'}
                        />
                      ))}
                    </>
                  );
                })()}
              </svg>
            </div>

            {/* Learning Rate Slider */}
            <div className="space-y-2 text-xs">
              <div className="flex justify-between text-slate-400">
                <span>Tasa de Aprendizaje (η):</span>
                <span className="font-mono text-cyan-400 font-bold">{learningRateSim.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.1"
                step="0.05"
                value={learningRateSim}
                onChange={(e) => setLearningRateSim(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <div className="p-2.5 bg-slate-950 rounded border border-slate-800 text-[11px] leading-relaxed">
                {learningRateSim < 0.15 ? (
                  <span className="text-slate-400">
                    <strong>Convergencia lenta:</strong> Pasos muy diminutos, se requieren demasiadas épocas.
                  </span>
                ) : learningRateSim <= 0.45 ? (
                  <span className="text-emerald-400">
                    <strong>Óptima:</strong> Desciende de manera firme y rápida hacia el mínimo sin rebotar.
                  </span>
                ) : learningRateSim <= 0.85 ? (
                  <span className="text-amber-400">
                    <strong>Alta:</strong> Oscila de un lado a otro del valle antes de asentarse.
                  </span>
                ) : (
                  <span className="text-rose-400">
                    <strong>Divergencia (Explosión):</strong> Los pasos son más grandes que el propio valle, alejándose del mínimo.
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

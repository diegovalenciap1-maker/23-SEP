import React, { useState } from 'react';
import { CheckCircle2, XCircle, Award, RotateCcw, ArrowRight, HelpCircle } from 'lucide-react';

interface QuizQuestion {
  id: number;
  question: string;
  context: string;
  options: { id: string; text: string }[];
  correctId: string;
  explanation: string;
}

const QUESTIONS: QuizQuestion[] = [
  {
    id: 1,
    question: '¿Qué ocurriría si elimináramos todas las funciones de activación no lineales (como ReLU o Sigmoide) entre las capas del MLP?',
    context: 'Arquitectura y No Linealidad',
    options: [
      { id: 'a', text: 'La red aprendería mucho más rápido y alcanzaría una precisión superior.' },
      { id: 'b', text: 'La red colapsaría matemáticamente en una sola transformación lineal afín, perdiendo la capacidad de aprender formas complejas de prendas.' },
      { id: 'c', text: 'Los gradientes explotarían a infinito en la primera época.' },
      { id: 'd', text: 'El vector de píxeles se reduciría automáticamente de 784 a 10.' },
    ],
    correctId: 'b',
    explanation: 'La composición de múltiples transformaciones lineales consecutivas es siempre otra transformación lineal: W₂ · (W₁ · X) = W_eq · X. Sin funciones no lineales intermedias, tener 100 capas ocultas equivale matemáticamente a tener un solo perceptrón simple, incapaz de resolver fronteras no convexas como el problema XOR.',
  },
  {
    id: 2,
    question: 'En Fashion-MNIST, ¿por qué la capa de salida usa Softmax en lugar de 10 funciones Sigmoides independientes?',
    context: 'Capa de Salida y Probabilidades',
    options: [
      { id: 'a', text: 'Porque Softmax es computacionalmente más simple que la sigmoide.' },
      { id: 'b', text: 'Porque Softmax normaliza las salidas exponencialmente obligando a que la suma total sea exactamente 1.0 (100%), modelando clases mutuamente excluyentes.' },
      { id: 'c', text: 'Porque la sigmoide no produce números entre 0 y 1.' },
      { id: 'd', text: 'Porque la función Softmax solo funciona con imágenes en blanco y negro.' },
    ],
    correctId: 'b',
    explanation: 'En clasificación multiclase donde cada imagen pertenece a una única categoría (una prenda no puede ser simultáneamente 100% zapatilla y 100% abrigo), Softmax hace competir a los logits entre sí: exp(z_c) / ∑ exp(z_j), asegurando una distribución de probabilidad válida y calibrada.',
  },
  {
    id: 3,
    question: 'Durante el entrenamiento, observas que la pérdida (Loss) oscila de manera caótica con valores gigantescos y diverge sin converger. ¿Qué ajuste soluciona este comportamiento?',
    context: 'Optimización y Descenso del Gradiente',
    options: [
      { id: 'a', text: 'Aumentar drásticamente la tasa de aprendizaje (Learning Rate η).' },
      { id: 'b', text: 'Reducir la tasa de aprendizaje (η), ya que pasos demasiado amplios hacen rebotar al optimizador fuera del valle de pérdida.' },
      { id: 'c', text: 'Eliminar el sesgo (bias) de todas las neuronas.' },
      { id: 'd', text: 'Invertir los píxeles de todas las prendas de blanco a negro.' },
    ],
    correctId: 'b',
    explanation: 'Una tasa de aprendizaje excesiva (η > 1.0) ocasiona que el paso de actualización supere el gradiente de descenso, rebotando en las paredes de la función de coste y provocando explosión de pesos (gradient explosion). Reducir η permite un descenso controlado hacia el mínimo local.',
  },
  {
    id: 4,
    question: '¿Cuál es el propósito geométrico fundamental del término de sesgo (bias b) en la ecuación z = ∑(wᵢ · xᵢ) + b?',
    context: 'Geometría del Perceptrón',
    options: [
      { id: 'a', text: 'Permite desplazar la frontera de decisión espacialmente sin estar restringida a pasar forzosamente por el origen (0, 0).' },
      { id: 'b', text: 'Garantiza que la activación resultante sea siempre un número entero par.' },
      { id: 'c', text: 'Sirve para multiplicar la resolución de la imagen 28x28.' },
      { id: 'd', text: 'Normaliza la desviación estándar de los píxeles a cero.' },
    ],
    correctId: 'a',
    explanation: 'Sin el término b, la ecuación w₁x₁ + w₂x₂ + ... = 0 siempre pasaría por el origen (0,0,...,0). El sesgo b actúa como un término independiente que traslada el hiperplano en el espacio n-dimensional para ajustarse al umbral óptimo de activación de cada prenda.',
  },
  {
    id: 5,
    question: 'Al entrenar con Fashion-MNIST, ¿por qué la red confunde con frecuencia la "Camiseta" (Clase 0) con la "Camisa" (Clase 6)?',
    context: 'Espacio de Características y Confusiones',
    options: [
      { id: 'a', text: 'Porque comparten una distribución espacial de píxeles muy similar (torso rectangular superior y mangas), diferenciándose solo por detalles finos en el cuello y botones.' },
      { id: 'b', text: 'Porque la etiqueta numérica 0 y 6 son matemáticamente idénticas en binario.' },
      { id: 'c', text: 'Porque la función ReLU borra los cuellos de las prendas.' },
      { id: 'd', text: 'Porque el algoritmo confunde el calzado con la ropa superior.' },
    ],
    correctId: 'a',
    explanation: 'El vector aplanado de 784 píxeles de una Camiseta y una Camisa es altamente correlacionado: ambas tienen brillo en el pecho y hombros y fondo negro alrededor. Las diferencias sutiles (botones centrales, cuello rígido en V o solapa) requieren que la red ajuste pesos de alta frecuencia en esa región específica.',
  },
];

export const QuizChallenges: React.FC = () => {
  const [currentIdx, setCurrentIdx] = useState<number>(0);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, string>>({});
  const [isAnswered, setIsAnswered] = useState<Record<number, boolean>>({});

  const q = QUESTIONS[currentIdx];
  const userChoice = selectedAnswers[q.id];
  const answered = isAnswered[q.id];
  const isCorrect = userChoice === q.correctId;

  const handleSelectOption = (optionId: string) => {
    if (answered) return;
    setSelectedAnswers((prev) => ({ ...prev, [q.id]: optionId }));
    setIsAnswered((prev) => ({ ...prev, [q.id]: true }));
  };

  const handleNext = () => {
    if (currentIdx < QUESTIONS.length - 1) {
      setCurrentIdx((i) => i + 1);
    }
  };

  const handlePrev = () => {
    if (currentIdx > 0) {
      setCurrentIdx((i) => i - 1);
    }
  };

  const handleResetQuiz = () => {
    setSelectedAnswers({});
    setIsAnswered({});
    setCurrentIdx(0);
  };

  // Score count
  const score = Object.entries(selectedAnswers).filter(
    ([qId, choice]) => choice === QUESTIONS.find((item) => item.id === Number(qId))?.correctId
  ).length;

  const totalAnswered = Object.keys(isAnswered).length;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Quiz Progress & Score Card */}
      <div className="flex items-center justify-between p-5 bg-slate-900 border border-slate-800 rounded-xl">
        <div>
          <span className="text-xs text-slate-400">Desafíos Conceptuales MLP</span>
          <h3 className="text-base font-bold text-white flex items-center gap-2 mt-0.5">
            <Award className="w-5 h-5 text-amber-400" />
            <span>Puntuación: {score} / {QUESTIONS.length} Correctas</span>
          </h3>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            {QUESTIONS.map((item, idx) => {
              const ans = isAnswered[item.id];
              const correct = selectedAnswers[item.id] === item.correctId;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentIdx(idx)}
                  className={`w-7 h-7 rounded-lg text-xs font-mono font-bold transition-all ${
                    currentIdx === idx
                      ? 'ring-2 ring-cyan-400 scale-110'
                      : ''
                  } ${
                    !ans
                      ? 'bg-slate-800 text-slate-400'
                      : correct
                      ? 'bg-emerald-500 text-slate-950'
                      : 'bg-rose-500 text-white'
                  }`}
                >
                  {item.id}
                </button>
              );
            })}
          </div>

          <button
            onClick={handleResetQuiz}
            className="p-2 text-slate-400 hover:text-white bg-slate-800 rounded-lg transition-colors"
            title="Reiniciar cuestionario"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Question Card */}
      <div className="p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
            {q.context}
          </span>
          <span className="text-xs text-slate-400 font-mono">
            Pregunta {currentIdx + 1} de {QUESTIONS.length}
          </span>
        </div>

        <h4 className="text-base font-semibold text-white leading-relaxed">
          {q.question}
        </h4>

        {/* Options */}
        <div className="space-y-3">
          {q.options.map((opt) => {
            const isSelected = userChoice === opt.id;
            const isTheCorrectOption = opt.id === q.correctId;

            let btnStyle = 'bg-slate-950 border-slate-800 hover:border-slate-700 text-slate-300';
            if (answered) {
              if (isTheCorrectOption) {
                btnStyle = 'bg-emerald-950/60 border-emerald-500 text-emerald-200 font-medium';
              } else if (isSelected && !isTheCorrectOption) {
                btnStyle = 'bg-rose-950/60 border-rose-500 text-rose-200';
              } else {
                btnStyle = 'bg-slate-950/40 border-slate-900 text-slate-500 opacity-60';
              }
            }

            return (
              <button
                key={opt.id}
                disabled={answered}
                onClick={() => handleSelectOption(opt.id)}
                className={`w-full p-4 rounded-xl border text-left text-xs transition-all flex items-start gap-3 ${btnStyle}`}
              >
                <span className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center font-mono font-bold shrink-0 text-slate-300 mt-0.5">
                  {opt.id.toUpperCase()}
                </span>
                <span className="leading-relaxed flex-1">{opt.text}</span>
                {answered && isTheCorrectOption && (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                )}
                {answered && isSelected && !isTheCorrectOption && (
                  <XCircle className="w-5 h-5 text-rose-400 shrink-0" />
                )}
              </button>
            );
          })}
        </div>

        {/* Feedback / Explanation Box */}
        {answered && (
          <div
            className={`p-4 rounded-xl border text-xs space-y-2 ${
              isCorrect
                ? 'bg-emerald-950/40 border-emerald-800 text-emerald-200'
                : 'bg-rose-950/40 border-rose-800 text-rose-200'
            }`}
          >
            <div className="flex items-center gap-2 font-bold">
              {isCorrect ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>¡Respuesta Correcta!</span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-rose-400" />
                  <span>Respuesta Incorrecta</span>
                </>
              )}
            </div>
            <p className="text-slate-300 leading-relaxed text-[11px]">
              {q.explanation}
            </p>
          </div>
        )}

        {/* Bottom Pagination Controls */}
        <div className="flex items-center justify-between pt-4 border-t border-slate-800">
          <button
            onClick={handlePrev}
            disabled={currentIdx === 0}
            className="px-4 py-2 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors disabled:opacity-30"
          >
            Anterior
          </button>

          {currentIdx < QUESTIONS.length - 1 ? (
            <button
              onClick={handleNext}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-cyan-600 hover:bg-cyan-500 rounded-lg transition-colors"
            >
              <span>Siguiente</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button
              onClick={handleResetQuiz}
              className="px-4 py-2 text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors"
            >
              Reiniciar Cuestionario
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

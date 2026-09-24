/**
 * Perceptrón Multicapa (MLP) - Fashion-MNIST Explorer
 * Interactive Educational Laboratory
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { TopNav, ActiveTab } from './components/TopNav';
import { NetworkVisualizer } from './components/NetworkVisualizer';
import { FashionCanvas } from './components/FashionCanvas';
import { ConceptExplainer } from './components/ConceptExplainer';
import { TrainingStudio } from './components/TrainingStudio';
import { FeatureMapInspector } from './components/FeatureMapInspector';
import { QuizChallenges } from './components/QuizChallenges';
import { NeuralNetwork, ForwardResult } from './lib/neuralNetwork';
import { generateFashionSamples, FashionSample, FASHION_CLASSES } from './data/fashionMnistData';
import { Sparkles, Brain, BookOpen, Layers, Award, BarChart2 } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('red');
  const [isPretrained, setIsPretrained] = useState<boolean>(true);

  // Initialize network instance
  const network = useMemo(() => new NeuralNetwork(), []);

  // Initialize fashion samples
  const allSamples = useMemo(() => generateFashionSamples(), []);
  const [currentSample, setCurrentSample] = useState<FashionSample>(allSamples[0]);
  const [forwardResult, setForwardResult] = useState<ForwardResult | null>(null);
  const [networkVersion, setNetworkVersion] = useState<number>(0);

  // Run forward inference
  const runInference = useCallback(() => {
    if (currentSample && network) {
      const res = network.forward(currentSample.data);
      setForwardResult(res);
    }
  }, [currentSample, network, networkVersion]);

  // Run inference whenever sample or network updates
  useEffect(() => {
    runInference();
  }, [runInference]);

  const handleResetWeights = () => {
    network.initWeights(false);
    setIsPretrained(false);
    setNetworkVersion((v) => v + 1);
  };

  const handleLoadPretrained = () => {
    network.initWeights(true);
    setIsPretrained(true);
    setNetworkVersion((v) => v + 1);
  };

  const handleNetworkUpdated = () => {
    setNetworkVersion((v) => v + 1);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-200">
      {/* Top Navigation Bar with strict 3-zone contract */}
      <TopNav
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onResetWeights={handleResetWeights}
        onLoadPretrained={handleLoadPretrained}
        isPretrained={isPretrained}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 lg:p-8 space-y-6">
        {/* Intro Hero Section with Clean Typography */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
              <span>Redes Neuronales Artificiales</span>
              <span aria-hidden="true">·</span>
              <span>Dataset Fashion-MNIST (Zalando Research)</span>
              <span aria-hidden="true">·</span>
              <span>784 Entradas a 10 Clases</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white" style={{ textWrap: 'balance' }}>
              Laboratorio Interactivo del Perceptrón Multicapa (MLP)
            </h1>
            <p className="text-sm text-slate-300 mt-1.5 max-w-3xl leading-relaxed">
              Explora visual y matemáticamente cómo las capas ocultas y las funciones de activación no lineales permiten a una red neuronal aprender a clasificar prendas de vestir a partir de matrices bidimensionales de 28×28 píxeles.
            </p>
          </div>

          {/* Quick Metrics Bar without Pills */}
          <div className="flex items-center gap-5 text-xs text-slate-300 shrink-0">
            <div>
              <span className="text-slate-400 block text-[11px]">Entradas</span>
              <span className="font-mono text-cyan-400 font-bold text-sm">784 px</span>
            </div>
            <div className="h-6 w-px bg-slate-800" />
            <div>
              <span className="text-slate-400 block text-[11px]">Capas Ocultas</span>
              <span className="font-mono text-white font-bold text-sm">32 + 16</span>
            </div>
            <div className="h-6 w-px bg-slate-800" />
            <div>
              <span className="text-slate-400 block text-[11px]">Clases Salida</span>
              <span className="font-mono text-emerald-400 font-bold text-sm">10 Modas</span>
            </div>
          </div>
        </div>

        {/* Dynamic Tab Content */}
        {activeTab === 'red' && (
          <NetworkVisualizer
            network={network}
            currentSample={currentSample}
            forwardResult={forwardResult}
            onRunForward={runInference}
            onSelectSample={setCurrentSample}
            samples={allSamples}
          />
        )}

        {activeTab === 'datos' && (
          <FashionCanvas
            currentSample={currentSample}
            onUpdateSample={setCurrentSample}
            samples={allSamples}
            network={network}
            forwardResult={forwardResult}
            onRunForward={runInference}
          />
        )}

        {activeTab === 'conceptos' && <ConceptExplainer />}

        {activeTab === 'entrenamiento' && (
          <TrainingStudio
            network={network}
            onNetworkUpdated={handleNetworkUpdated}
            onResetWeights={handleResetWeights}
          />
        )}

        {activeTab === 'filtros' && <FeatureMapInspector network={network} />}

        {activeTab === 'desafios' && <QuizChallenges />}
      </main>

      {/* Subdued Editorial Footer */}
      <footer className="mt-12 border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span>Perceptrón Multicapa (MLP)</span>
            <span aria-hidden="true">·</span>
            <span>Fashion-MNIST Benchmark</span>
            <span aria-hidden="true">·</span>
            <span>Entropía Cruzada y Backpropagation</span>
          </div>
          <div className="text-slate-400 text-[11px]">
            Laboratorio de Inteligencia Artificial y Aprendizaje Profundo
          </div>
        </div>
      </footer>
    </div>
  );
}

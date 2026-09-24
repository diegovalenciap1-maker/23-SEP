import React from 'react';
import { Play, RotateCcw, Sparkles } from 'lucide-react';

export type ActiveTab = 'red' | 'datos' | 'conceptos' | 'entrenamiento' | 'filtros' | 'desafios';

interface TopNavProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  onResetWeights: () => void;
  onLoadPretrained: () => void;
  isPretrained: boolean;
}

export const TopNav: React.FC<TopNavProps> = ({
  activeTab,
  onTabChange,
  onResetWeights,
  onLoadPretrained,
  isPretrained,
}) => {
  return (
    <header className="sticky top-0 z-40 flex items-center justify-between px-6 py-3.5 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
      {/* Zone 1: Single text element wordmark */}
      <a
        href="#red"
        onClick={(e) => {
          e.preventDefault();
          onTabChange('red');
        }}
        className="text-lg font-bold tracking-tight text-white hover:text-cyan-400 transition-colors"
      >
        MLP Fashion Lab
      </a>

      {/* Zone 2: Navigation links */}
      <nav className="hidden lg:flex items-center gap-6 text-sm font-medium text-slate-300">
        <button
          onClick={() => onTabChange('red')}
          className={`hover:text-cyan-400 transition-colors relative py-1 ${
            activeTab === 'red' ? 'text-cyan-400 font-semibold' : 'text-slate-300'
          }`}
        >
          Red & Inferencia
          {activeTab === 'red' && (
            <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400" />
          )}
        </button>
        <button
          onClick={() => onTabChange('datos')}
          className={`hover:text-cyan-400 transition-colors relative py-1 ${
            activeTab === 'datos' ? 'text-cyan-400 font-semibold' : 'text-slate-300'
          }`}
        >
          Dataset & Dibujo
          {activeTab === 'datos' && (
            <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400" />
          )}
        </button>
        <button
          onClick={() => onTabChange('conceptos')}
          className={`hover:text-cyan-400 transition-colors relative py-1 ${
            activeTab === 'conceptos' ? 'text-cyan-400 font-semibold' : 'text-slate-300'
          }`}
        >
          5 Conceptos Teóricos
          {activeTab === 'conceptos' && (
            <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400" />
          )}
        </button>
        <button
          onClick={() => onTabChange('entrenamiento')}
          className={`hover:text-cyan-400 transition-colors relative py-1 ${
            activeTab === 'entrenamiento' ? 'text-cyan-400 font-semibold' : 'text-slate-300'
          }`}
        >
          Estudio de Entrenamiento
          {activeTab === 'entrenamiento' && (
            <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400" />
          )}
        </button>
        <button
          onClick={() => onTabChange('filtros')}
          className={`hover:text-cyan-400 transition-colors relative py-1 ${
            activeTab === 'filtros' ? 'text-cyan-400 font-semibold' : 'text-slate-300'
          }`}
        >
          ¿Qué ve cada neurona?
          {activeTab === 'filtros' && (
            <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400" />
          )}
        </button>
        <button
          onClick={() => onTabChange('desafios')}
          className={`hover:text-cyan-400 transition-colors relative py-1 ${
            activeTab === 'desafios' ? 'text-cyan-400 font-semibold' : 'text-slate-300'
          }`}
        >
          Desafíos
          {activeTab === 'desafios' && (
            <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400" />
          )}
        </button>
      </nav>

      {/* Zone 3: Primary Actions */}
      <div className="flex items-center gap-2.5">
        <button
          onClick={onLoadPretrained}
          title="Carga pesos calibrados para clasificar Fashion-MNIST de inmediato"
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-cyan-300 bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-800/80 rounded-lg transition-colors whitespace-nowrap"
        >
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>{isPretrained ? 'Pesos Calibrados' : 'Cargar Pesos'}</span>
        </button>
        <button
          onClick={onResetWeights}
          title="Reinicializa los pesos con distribución He / Xavier aleatoria"
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors whitespace-nowrap"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
          <span>Reiniciar</span>
        </button>
      </div>
    </header>
  );
};

// Entry point for the minimal Aletheia Agent Compiler frontend
// React + TypeScript, strict minimal UI

import React from 'react';
import ModeSelector from './components/ModeSelector';
import FileDropzone from './components/FileDropzone';
import LogPanel from './components/LogPanel';
import TrainControls from './components/TrainControls';

const App: React.FC = () => {
  // State and logic will be added in implementation steps
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 bg-gray-800 border-b border-gray-700">
        <h1 className="text-lg font-bold tracking-wide">Aletheia Agent Compiler</h1>
        <TrainControls />
      </header>
      {/* Main body */}
      <main className="flex flex-1 overflow-hidden">
        {/* Left: Mode Selection */}
        <section className="w-1/5 min-w-[180px] border-r border-gray-800 p-4">
          <ModeSelector />
        </section>
        {/* Center: File/Link Input */}
        <section className="w-2/5 border-r border-gray-800 p-4 flex flex-col">
          <FileDropzone />
        </section>
        {/* Right: Logs/Output */}
        <section className="flex-1 p-4 flex flex-col">
          <LogPanel />
        </section>
      </main>
    </div>
  );
};

export default App;

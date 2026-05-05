// TrainControls.tsx
// Minimal train button, disabled logic to be added
import React from 'react';

const TrainControls: React.FC = () => {
  // State and handlers will be added in implementation steps
  return (
    <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white font-semibold disabled:opacity-50" disabled>
      Train
    </button>
  );
};

export default TrainControls;

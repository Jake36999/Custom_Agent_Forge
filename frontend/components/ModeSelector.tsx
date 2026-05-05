// ModeSelector.tsx
// Minimal radio button group for mode selection
import React from 'react';

const MODES = ["theorist", "coder", "advocate", "veteran"];

const ModeSelector: React.FC = () => {
  // State and handlers will be added in implementation steps
  return (
    <div>
      <h2 className="text-md font-semibold mb-3">Mode</h2>
      <div className="flex flex-col gap-2">
        {MODES.map((mode) => (
          <label key={mode} className="flex items-center gap-2 cursor-pointer hover:text-blue-400">
            <input type="radio" name="mode" value={mode} className="accent-blue-500" />
            <span className="capitalize">{mode}</span>
          </label>
        ))}
      </div>
    </div>
  );
};

export default ModeSelector;

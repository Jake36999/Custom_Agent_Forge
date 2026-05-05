// LogPanel.tsx
// Minimal scrollable log panel, monospace font
import React from 'react';

const LogPanel: React.FC = () => {
  // Log lines and output controls will be added in implementation steps
  return (
    <div className="flex flex-col h-full">
      <h2 className="text-md font-semibold mb-3">Logs</h2>
      <div className="flex-1 overflow-y-auto bg-gray-950 rounded p-3 font-mono text-xs text-green-300 border border-gray-800">
        {/* Log lines will be appended here */}
        <div>Pipeline ready.</div>
      </div>
      {/* Output controls (download/open) will go here after completion */}
    </div>
  );
};

export default LogPanel;

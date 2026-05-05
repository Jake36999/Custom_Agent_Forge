// FileDropzone.tsx
// Minimal drag-and-drop file upload and link input
import React from 'react';

const FileDropzone: React.FC = () => {
  // State and handlers will be added in implementation steps
  return (
    <div>
      <h2 className="text-md font-semibold mb-3">Sources</h2>
      <div className="border-2 border-dashed border-gray-600 rounded p-6 flex flex-col items-center justify-center bg-gray-800 hover:bg-gray-700 transition-colors">
        <span className="text-gray-400 mb-2">Drag & drop files here</span>
        <input type="file" multiple className="hidden" />
      </div>
      <div className="mt-4">
        <input type="text" placeholder="Paste link (optional)" className="w-full px-2 py-1 rounded bg-gray-700 text-gray-100 border border-gray-600 focus:outline-none focus:border-blue-500" />
      </div>
      {/* Uploaded files list will go here */}
      <ul className="mt-4 text-sm text-gray-300">
        {/* Example: <li>example.py</li> */}
      </ul>
    </div>
  );
};

export default FileDropzone;

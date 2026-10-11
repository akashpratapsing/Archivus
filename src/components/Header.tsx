import React, { useRef } from 'react';
import { BookOpen } from 'lucide-react';
import { SystemConfig } from '../types';

interface HeaderProps {
  config: SystemConfig | null;
  onProviderChange: (provider: 'gemini' | 'groq') => void;
  documentCount: number;
  totalChunks: number;
  onToggleSidebar?: () => void;
  onUploadFile?: (file: File) => void;
}

export const Header: React.FC<HeaderProps> = ({
  config,
  onProviderChange,
  documentCount,
  onToggleSidebar,
  onUploadFile,
}) => {
  const currentProvider = config?.current_provider || 'gemini';
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0] && onUploadFile) {
      onUploadFile(e.target.files[0]);
      e.target.value = '';
    }
  };

  return (
    <header className="h-13 sm:h-14 border-b border-[#DDD6C8] bg-[#F7F5F0] px-4 sm:px-6 flex items-center justify-between z-20 sticky top-0">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown,.py,.ts,.tsx,.js,.json,.csv,.sql,.html,.sh"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Brand title: Archivus with Index icon button on mobile */}
      <div className="flex items-center space-x-2.5">
        {onToggleSidebar && (
          <button
            onClick={onToggleSidebar}
            className="lg:hidden p-1.5 -ml-1 text-[#8B8578] hover:text-[#1C1C1A] transition-colors"
            title="Open Index"
            aria-label="Open Index"
          >
            <BookOpen className="w-4 h-4" strokeWidth={1.5} />
          </button>
        )}
        <span className="font-serif text-base tracking-tight font-bold text-[#1C1C1A]">
          Archivus
        </span>
      </div>

      {/* Right controls: ultra-minimal */}
      <div className="flex items-center space-x-4 text-xs font-serif text-[#8B8578]">
        {/* Minimal Document counter */}
        <span className="hidden sm:inline">
          <span className="text-[#1C1C1A]">{documentCount}</span> docs
        </span>

        {/* Minimal Engine Switch */}
        <div className="inline-flex border border-[#DDD6C8] bg-[#FDFCFA]">
          <button
            onClick={() => onProviderChange('gemini')}
            className={`px-2 py-0.5 text-xs transition-colors ${
              currentProvider === 'gemini'
                ? 'bg-[#2C3E50] text-[#FDFCFA]'
                : 'text-[#8B8578] hover:text-[#1C1C1A]'
            }`}
          >
            Gemini
          </button>
          <button
            onClick={() => onProviderChange('groq')}
            className={`px-2 py-0.5 text-xs transition-colors border-l border-[#DDD6C8] ${
              currentProvider === 'groq'
                ? 'bg-[#2C3E50] text-[#FDFCFA]'
                : 'text-[#8B8578] hover:text-[#1C1C1A]'
            }`}
          >
            Groq
          </button>
        </div>

        {/* Upload Action */}
        <button
          onClick={() => fileInputRef.current?.click()}
          className="text-[#2C3E50] hover:underline"
        >
          Upload
        </button>
      </div>
    </header>
  );
};

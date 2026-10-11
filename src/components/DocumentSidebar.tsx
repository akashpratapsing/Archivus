import React, { useRef, useState } from 'react';
import { BookOpen, X } from 'lucide-react';
import { IndexedDocument } from '../types';

interface DocumentSidebarProps {
  documents: IndexedDocument[];
  isLoading: boolean;
  uploading: boolean;
  uploadProgress: string | null;
  error: string | null;
  onUpload: (file: File) => Promise<void>;
  onDelete: (filename: string) => Promise<void>;
  onSeedSample: (sampleKey: string) => Promise<void>;
  onRefresh: () => void;
  isOpen: boolean;
  onToggle: () => void;
}

export const DocumentSidebar: React.FC<DocumentSidebarProps> = ({
  documents,
  isLoading,
  uploading,
  uploadProgress,
  error,
  onUpload,
  onDelete,
  onSeedSample,
  onRefresh,
  isOpen,
  onToggle,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await onUpload(e.target.files[0]);
      e.target.value = '';
    }
  };

  const filteredDocs = documents.filter((doc) =>
    doc.filename.toLowerCase().includes(searchFilter.toLowerCase())
  );

  return (
    <aside
      className={`
        fixed inset-y-0 left-0 z-30 w-64 bg-[#F7F5F0] border-r border-[#DDD6C8] flex flex-col transition-transform duration-200 ease-in-out font-serif
        lg:static lg:translate-x-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown,.py,.ts,.tsx,.js,.json,.csv,.sql,.html,.sh"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Header with icon instead of text */}
      <div className="p-3.5 border-b border-[#DDD6C8] flex items-center justify-between">
        <div className="flex items-center space-x-2 text-[#8B8578]">
          <BookOpen className="w-4 h-4" strokeWidth={1.5} />
          <span className="text-[11px] uppercase tracking-wider">Index</span>
        </div>
        <button
          onClick={onToggle}
          className="lg:hidden p-1 text-[#8B8578] hover:text-[#1C1C1A]"
          title="Close"
          aria-label="Close"
        >
          <X className="w-3.5 h-3.5" strokeWidth={1.5} />
        </button>
      </div>

      {/* Upload Affordance */}
      <div className="p-3 border-b border-[#DDD6C8]">
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            p-2.5 border border-dashed transition-colors cursor-pointer text-center text-xs
            ${
              dragActive
                ? 'border-[#2C3E50] bg-[#EFEAE0]'
                : 'border-[#DDD6C8] hover:border-[#8B8578] bg-[#FDFCFA]'
            }
          `}
        >
          <span className="text-[#1C1C1A]">
            {uploading ? 'Indexing…' : '+ Deposit'}
          </span>
        </div>

        {uploading && (
          <div className="mt-2">
            <div className="w-full bg-[#EBE5D8] h-[2px] overflow-hidden relative">
              <div className="bg-[#2C3E50] h-full w-1/3 animate-paper-line" />
            </div>
          </div>
        )}

        {error && (
          <div className="mt-2 p-1.5 border border-[#7A2E2E]/30 text-[#7A2E2E] text-[11px]">
            {error}
          </div>
        )}
      </div>

      {/* Minimal Filter */}
      {documents.length > 3 && (
        <div className="px-3 py-1.5 border-b border-[#DDD6C8]">
          <input
            type="text"
            placeholder="Search index…"
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            className="w-full bg-transparent text-xs text-[#1C1C1A] placeholder-[#8B8578] focus:outline-none py-0.5"
          />
        </div>
      )}

      {/* Index entries */}
      <div className="flex-1 overflow-y-auto divide-y divide-[#DDD6C8]/60">
        {filteredDocs.length === 0 ? (
          <div className="p-6 text-center text-[#8B8578] text-xs italic">
            No documents indexed yet
          </div>
        ) : (
          filteredDocs.map((doc) => (
            <div
              key={doc.filename}
              className="p-3 hover:bg-[#FDFCFA] transition-colors flex items-baseline justify-between group text-xs"
            >
              <div className="min-w-0 pr-2">
                <div className="text-[#1C1C1A] truncate" title={doc.filename}>
                  {doc.filename}
                </div>
                <div className="text-[11px] text-[#8B8578] mt-0.5">
                  {doc.chunk_count} passages
                </div>
              </div>

              <div>
                {deleteTarget === doc.filename ? (
                  <div className="flex items-center space-x-1 text-[11px]">
                    <button
                      onClick={() => {
                        onDelete(doc.filename);
                        setDeleteTarget(null);
                      }}
                      className="text-[#7A2E2E] hover:underline"
                    >
                      Delete
                    </button>
                    <span className="text-[#8B8578]">/</span>
                    <button
                      onClick={() => setDeleteTarget(null)}
                      className="text-[#8B8578]"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setDeleteTarget(doc.filename)}
                    className="text-[#8B8578] hover:text-[#7A2E2E] text-[11px] opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    ×
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
};

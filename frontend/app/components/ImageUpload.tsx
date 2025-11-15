"use client";

import { useCallback, useState } from "react";
import Image from "next/image";

interface ImageUploadProps {
  onImagesUpload: (files: File[]) => void;
  uploadedImages: File[];
  onRemoveImage: (index: number) => void;
}

export default function ImageUpload({
  onImagesUpload,
  uploadedImages,
  onRemoveImage,
}: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files).filter((file) =>
        file.type.startsWith("image/")
      );
      onImagesUpload(files);
    },
    [onImagesUpload]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        const files = Array.from(e.target.files);
        onImagesUpload(files);
      }
    },
    [onImagesUpload]
  );

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
          isDragging
            ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
            : "border-slate-300 dark:border-slate-600 hover:border-emerald-400 dark:hover:border-emerald-500"
        }`}
      >
        <input
          type="file"
          id="image-upload"
          multiple
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
        />
        <label
          htmlFor="image-upload"
          className="cursor-pointer flex flex-col items-center"
        >
          <svg
            className="w-12 h-12 text-slate-400 dark:text-slate-500 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <p className="text-slate-700 dark:text-slate-300 font-medium mb-1">
            Drag and drop images here, or click to select
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Supports multiple images (PNG, JPG, JPEG, WEBP)
          </p>
        </label>
      </div>

      {/* Uploaded Images Preview */}
      {uploadedImages.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Uploaded Images ({uploadedImages.length})
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {uploadedImages.map((file, index) => (
              <div
                key={index}
                className="relative group rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900"
              >
                <div className="aspect-square relative">
                  <img
                    src={URL.createObjectURL(file)}
                    alt={`Upload ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                  <button
                    onClick={() => onRemoveImage(index)}
                    className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-1.5 opacity-0 group-hover:opacity-100 transition-opacity"
                    aria-label="Remove image"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </div>
                <div className="p-2">
                  <p className="text-xs text-slate-600 dark:text-slate-400 truncate">
                    {file.name}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-500">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

